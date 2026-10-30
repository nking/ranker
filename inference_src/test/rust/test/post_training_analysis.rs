use std::{fs, io};
use std::path::PathBuf;
use polars::prelude::{LazyFrame, PolarsResult};

// Assuming your UserRequest is accessible here
pub mod helper {
    // Tell Rust to literally include the code from helper.rs here
    include!("helper.rs");
    include!("helper_users.rs");
}
use inference_engine::app_config::AppConfig;
use inference_engine::orchestrator::Orchestrator;
use inference_engine::movie_db::MovieDb;
use helper::{get_train_val_test_liked_uris, DataSize, ensure_plotters_font_registered};
use inference_engine::tensorflow::serving::model_spec::VersionChoice;
use crate::helper::{get_bin_dir, get_config_json_uri, get_movie_tiers_df, get_user_tiers_df, load_and_concat_parquet};

struct TestHarness {
    orchestrator: Orchestrator,
    #[allow(dead_code)]
    test_uris: Vec<String>,
    #[allow(dead_code)]
    ratings_uris: Vec<String>,
    #[allow(dead_code)]
    train_history_df: LazyFrame,
    pos_test_df: LazyFrame,
    #[allow(dead_code)]
    movies_df: LazyFrame,
    #[allow(dead_code)]
    movies_offset: usize,
    num_catalog_users: usize,
    #[allow(dead_code)]
    num_catalog_movies: usize,
    embed_len: usize,
    movie_tiers_df: LazyFrame,
    user_tiers_df: LazyFrame,
    candidate_uri: String,
    movies_path: String
}

impl TestHarness {
    async fn new() -> Self {
        println!("\n[SETUP]: Initializing test resource...");
        let config_path = get_config_json_uri().to_string();
        let config = AppConfig::load_from_file(&config_path)
            .expect("Could not load tiny config json file");

        let query_uri = config.query_uri.clone();
        let ranker_uri = config.ranker_uri.clone();
        let candidate_uri = config.candidate_uri.clone();
        let ranker_n_local_devices = config.ranker_n_local_devices;
        let top_k = config.top_k;
        let user_db_path: PathBuf = config.user_db_path.clone();
        let persisted_index_path : PathBuf = config.persisted_index_path.clone();
        let _params_json_uri : String = config.params_json_path;

        let movie_embeddings_uri : String = config.movie_embeddings_path;

        // we want to be able to test against this recommender, so don't include the test uris
        let ratings_map = get_train_val_test_liked_uris(DataSize::Full, false);
        let ratings_uris: Vec<&str> = vec![
            ratings_map.get("train_liked").unwrap(),
            ratings_map.get("train_3").unwrap(),
            ratings_map.get("train_disliked").unwrap(),
            ratings_map.get("val_liked").unwrap(),
            ratings_map.get("val_3").unwrap(),
            ratings_map.get("val_disliked").unwrap(),
        ];
        let test_ratings_uris = vec![
            ratings_map.get("test_liked").unwrap().clone(),
        ];

        let orchestrator = Orchestrator::new(
            query_uri,
            ranker_uri,
            config.query_saved_models_uri,
            config.ranker_saved_models_uri,
            config.ranker_serving_is_batched,
            &movie_embeddings_uri,
            ratings_uris,
            ranker_n_local_devices,
            top_k,
            persisted_index_path,
            user_db_path,
        ).await.unwrap();

        let ratings_uris: Vec<String> = vec![
            ratings_map.get("train_liked").unwrap().clone(),
            ratings_map.get("train_3").unwrap().clone(),
            ratings_map.get("train_disliked").unwrap().clone(),
            ratings_map.get("val_liked").unwrap().clone(),
            ratings_map.get("val_3").unwrap().clone(),
            ratings_map.get("val_disliked").unwrap().clone(),
        ];

        // read in data frames
        let pos_test_df: LazyFrame = load_and_concat_parquet(&[&ratings_map.get("test_liked").unwrap().clone()]).expect("error reading test ratings into df");
        let uri_slices: Vec<&str> = ratings_uris.iter().map(|s| s.as_str()).collect();
        let train_history_df : LazyFrame = load_and_concat_parquet(&uri_slices).expect("error reading ratings into df");
        let movies_df: LazyFrame = load_and_concat_parquet(&[&config.movies_path]).expect("error reading movie catalog into df");// Print the first 5 rows of an eager DataFrame
        println!("{}", movies_df.clone().collect().expect("error w/ movies_df").head(Some(5)));
        //["movie_id", "movie_tier"]
        let movie_tiers_df = get_movie_tiers_df(&train_history_df, &movies_df);
        //print_lazyframe_columns(&mut movie_tiers_df).expect("Failed to resolve movie_tiers LazyFrame schema!");

        let user_parquet = config.user_db_path.to_string_lossy().replace("bin", "parquet");
        let users_df : LazyFrame = load_and_concat_parquet(&[&user_parquet]).expect("error reading users into df");
        //["user_id", "user_tier"]
        let user_tiers_df : LazyFrame = get_user_tiers_df(&train_history_df, &users_df);
        //print_lazyframe_columns(&mut user_tiers_df).expect("Failed to resolve user_tiers LazyFrame schema!");

        //let pos_test_df = pos_test_df.join(movie_tiers_df.clone(), [col("movie_id")],
        //    [col("movie_id")], JoinArgs::new(JoinType::Inner));
        //"user_id", "movie_id", "rating", "timestamp", "tier", "movie_tier", "user_tier"
        //let pos_test_df = pos_test_df.join(user_tiers_df.clone(), [col("movie_id")],
        //    [col("user_id")], JoinArgs::new(JoinType::Inner));
        //print_lazyframe_columns(&mut pos_test_df).expect("Failed to resolve pos_test_df LazyFrame schema!");

        let ranker_metadata =
            orchestrator.get_or_fetch_ranker_metadata(Some(VersionChoice::Version(1)))
                .expect("error fetching ranker metadata");
        let movies_offset = ranker_metadata.num_catalog_users; // when subtracted from min(movie_ids), will be 1
        let num_catalog_movies = ranker_metadata.num_catalog_movies;
        let embed_len = ranker_metadata.embed_len;

        Self {
            orchestrator: orchestrator,
            test_uris: test_ratings_uris,
            ratings_uris: ratings_uris,
            movies_df: movies_df,
            movies_offset: movies_offset,
            num_catalog_users: ranker_metadata.num_catalog_users,
            num_catalog_movies: num_catalog_movies,
            embed_len: embed_len,
            train_history_df: train_history_df,
            pos_test_df: pos_test_df,
            movie_tiers_df: movie_tiers_df,
            user_tiers_df: user_tiers_df,
            candidate_uri: candidate_uri,
            movies_path: config.movies_path
        }
    }
}

fn recreate_directory(dir_path: &str) -> io::Result<()> {
    // Remove the directory if it exists, ignoring NotFound errors
    match fs::remove_dir_all(dir_path) {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {} // Safe to ignore if missing
        Err(e) => return Err(e), // Propagate any real permissions/system errors
    }

    // Create the directory (and any missing parent directories) fresh
    fs::create_dir_all(dir_path)?;

    Ok(())
}

#[allow(dead_code)]
fn print_lazyframe_columns(lf: &mut LazyFrame) -> PolarsResult<()> {
    // Resolve the schema of the lazy computation graph
    let schema = lf.collect_schema()?;

    // Extract the column names as a Vec<&str> or &String
    let column_names: Vec<_> = schema.iter_names().collect();

    // Print them all at once as an array
    println!("\n\nLazyFrame Columns: {:?}", column_names);

    // OR iterate through them one by one
    for name in schema.iter_names() {
        println!("- {}", name);
    }

    Ok(())
}

// --- TEARDOWN LOGIC ---
impl Drop for TestHarness {
    fn drop(&mut self) {
        println!("[TEARDOWN]");
    }
}

#[cfg(test)]
mod post_training_analysis {
    use std::collections::HashMap;
    use std::error::Error;
    use std::fs::File;
    use std::sync::Arc;
    use polars::df;
    use polars::prelude::*;
    use rayon::prelude::*; // Required for parallel slate comparisons in _diversity_metrics
    use tonic::{Request, Response, Status};
    use inference_engine::embeddings_ann::Searcher;
    use inference_engine::model_client::CandidateModelClient;
    // Bring everything from the outer scope (TestHarness, helper functions, etc.) into the test module
    use super::*;

    use plotters::prelude::*;
    use rustc_hash::FxHashMap;
    use std::cmp::Ordering;
    use std::path::{Path, PathBuf};
    use serde::{Deserialize, Serialize};
    use inference_engine::tensorflow::serving::model_spec::VersionChoice;
    use inference_engine::pb::{ApproxNearestNeighborsResponse, RankedMoviesAndRepr, UsersRequest, MoviesRequest, RankedMovies};
    use inference_engine::pb::recommender_service_server::RecommenderService;
    use inference_engine::ranker_model_metadata::RankerModelMetadata;
    use inference_engine::util::{get_top_k_desc_scores};
    use crate::helper::get_unique_user_and_first_timestamp;


    /// Summary directional statistics on the unit hypersphere S^(d-1) quantifying
    /// how faithfully the L2-normalized centroid represents an entity's multi-context
    /// GATv2 embedding distribution.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SphericalFidelitySummary {
        pub n_unique_entities: usize,
        pub n_multi_occurrence_entities: usize,
        /// Mean Resultant Length R_bar = || (1/N) sum_k v_k ||_2 in [0, 1].
        /// Mathematically equals the exact mean cosine similarity <v_k, mu> to the L2-normalized centroid!
        pub mean_resultant_length_avg: f64,
        pub mean_resultant_length_p10: f64,
        /// Spherical Variance S = 1.0 - R_bar in [0, 1].
        pub spherical_variance_avg: f64,
        /// Median von Mises-Fisher concentration parameter kappa = R_bar * (d - R_bar^2) / (1 - R_bar^2).
        pub vmf_kappa_median: f64,
        /// Mean geodesic angular deviation arccos(<v_k, mu>) in degrees.
        pub mean_angle_deg_avg: f64,
        /// 90th-percentile geodesic angular deviation in degrees.
        pub p90_angle_deg_avg: f64,
        /// Robust spherical MAD: 1.4826 * median(|theta_k - median(theta)|) in degrees.
        pub robust_mad_angle_deg_avg: f64,
        /// Fraction of multi-occurrence entities with R_bar >= 0.90 (mean angle <= 25.8 deg).
        pub pct_high_fidelity_r_ge_090: f64,
        /// Fraction of multi-occurrence entities with R_bar < 0.70 (high context sensitivity / polysemy).
        pub pct_context_polysemous_r_lt_070: f64,
    }

    #[tokio::test(flavor = "multi_thread")]
    pub async fn test_analysis() ->  Result<(), Box<dyn std::error::Error + Send + Sync>>{

        let harness = TestHarness::new().await;

        //NOTE:  edit these for the models to analyze
        let q_ver: usize = 1;
        let r_ver: usize = 2;

        let query_model_version = Some(VersionChoice::Version(q_ver as i64));
        let ranker_model_version = Some(VersionChoice::Version(r_ver as i64));

        let output_base_dir = get_bin_dir().unwrap().to_string_lossy().into_owned();
        let summary_output_dir = format!(
            "{}/post_training_analysis_q{:02}_r{:02}",
            output_base_dir.trim_end_matches('/'),
            q_ver,
            r_ver
        );
        let _ = recreate_directory(summary_output_dir.as_str())?;
        let parquet_output_dir = format!(
            "{}/parquet_metrics_q{:02}_r{:02}",
            output_base_dir.trim_end_matches('/'),
            q_ver,
            r_ver
        );
        let _ = recreate_directory(parquet_output_dir.as_str())?;


        // calculate @k metrics for retrieval and ranking for given model versions
        let _ = calc_metrics_at_k(&harness, query_model_version.clone(), ranker_model_version.clone(),
            &summary_output_dir, &parquet_output_dir).await;

        Ok(())
        // harness destructs when goes out of scope when method frame is done
    }

    async fn calc_metrics_at_k(harness: &TestHarness, query_model_version:Option<VersionChoice>,
        ranker_model_version:Option<VersionChoice>, summary_output_dir: &str, parquet_output_dir: &str)
        ->  Result<(), Box<dyn std::error::Error + Send + Sync>> {

        let ranker_metadata = harness.orchestrator.get_or_fetch_ranker_metadata(
            ranker_model_version.clone()
        ).expect("error fetching ranker metadata");
        let ranker_batch_size = ranker_metadata.batch_size;
        let num_candidates = ranker_metadata.num_candidates;

        // calculate top_k=20 for retrieval then ranker
        let top_k = harness.orchestrator.top_k;
        let embed_len = ranker_metadata.embed_len;

        // calc retrieval metrics @20 and ranker metrics @20
        // calculate kendall tau and spearman rank correlation between the ranked results too
        // write the parquet files out for pairwise comparisons

        let pos_test_movie_tiered = harness.pos_test_df.clone()
            .left_join(
                harness.movie_tiers_df.clone(),
                col("movie_id"),
                col("movie_id")
            );

        //Group by user_id and calculate the ground truth counts
        let user_gt_counts = pos_test_movie_tiered
            .group_by([col("user_id")])
            .agg([
                len().alias("total_positives"),
                col("movie_tier").eq(lit(0)).sum().alias("gt_pos_tier_0"),
                col("movie_tier").eq(lit(1)).sum().alias("gt_pos_tier_1"),
                col("movie_tier").eq(lit(2)).sum().alias("gt_pos_tier_2"),
            ]);

        // to make requests, need user_ids and their first timestamps
        let (user_ids, timestamps) = get_unique_user_and_first_timestamp(
            harness.pos_test_df.clone()
        ).expect("error fetching user_ids and first timestamps");

        let n_users = user_ids.len();
        println!("test n_unique_users={}", n_users);


        // =========== RETRIEVAL ===============================

        // no batch_size constraints for the query model or ANN requests
        let tonic_request: Request<UsersRequest>  =
            harness.orchestrator.get_users_request(&user_ids, &timestamps,
                query_model_version.clone(), ranker_model_version.clone()).await?;
        let mut users_request : UsersRequest = tonic_request.into_inner();
        users_request.k = Some(num_candidates as u32);
        let ann_reqs = Request::new(users_request.clone());
        let ann_res: ApproxNearestNeighborsResponse = harness.orchestrator
            ._approx_nearest_neighbors(ann_reqs).await?.into_inner();
        // length: n_users * num_candidates
        let candidate_movie_ids: Vec<i32> = ann_res.candidate_ids;
        assert_eq!(candidate_movie_ids.len(), n_users * num_candidates);
        let mut top_k_movie_ids: Vec<i32> = Vec::with_capacity(n_users * top_k);

        for i in (0..n_users).step_by(1){
            let i0 = i * num_candidates;
            let i1 = i0 + top_k;
            let top = candidate_movie_ids[i0..i1].to_vec();
            top_k_movie_ids.extend(top)
        }
        assert_eq!(top_k_movie_ids.len(), n_users * top_k);

        let n_users = user_ids.len();

        let mut ranker_movie_ids: Vec<i32> = Vec::with_capacity(candidate_movie_ids.len());

        // put through ranker
        // a request should be batch_size
        for i0 in (0..n_users).step_by(ranker_batch_size) {

            let i1 = std::cmp::min(i0 + ranker_batch_size, n_users);

            //println!("about to rank movie_ids for users: {}-{}", i0, i1);

            let mut ranked_movies = match harness.orchestrator._make_ranker_request(
                &user_ids[i0..i1],
                &timestamps[i0..i1],
                &ann_res.user_embeddings[i0*embed_len..i1*embed_len],
                &candidate_movie_ids[i0*num_candidates..i1*num_candidates],
                ranker_model_version.clone()
            ).await {
                Ok(response) => response,
                Err(e) => {
                    eprintln!("\n[ERROR] Ranker request failed for user batch indices {} to {}", i0, i1);
                    eprintln!("   - Number of users in batch: {}", i1 - i0);
                    eprintln!("   - Number of candidates sent: {}", (i1 - i0) * num_candidates);
                    eprintln!("   - Range of embedding elements sent: {}", (i1 - i0) * embed_len);
                    eprintln!("   - Error Details: {:#?}", e);

                    // Bubble the error up to the function's return type
                    return Err(e.into());
                }
            };

            //truncate to top_k movies for each user
            let (top_movie_ids, _top_scores) = get_top_k_desc_scores(
                & mut ranked_movies.movie_ids, &mut ranked_movies.scores, num_candidates, top_k
            );

            ranker_movie_ids.extend(top_movie_ids);
        }

        // =======================================================================================
        // ======= at this point we have all the retrievals and ranked movie ids for all test users ======

        //polars is already spinning up threads to use all cores available, so we run these
        //   sequentially rather than in parallel.

        let base_metrics = calc_and_write(
            harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &ann_res.user_embeddings,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "stratified_metrics.json".as_ref(),
            "metrics".as_ref(),
            _metrics
        ).await?;

        let coverage = calc_and_write(
            harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &ann_res.user_embeddings,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "coverage.json".as_ref(),
            "coverage".as_ref(),
            _coverage_and_gini
        ).await?;

        let pop_bias = calc_and_write(
            harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &ann_res.user_embeddings,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "popularity_bias.json".as_ref(),
            "popularity_bias".as_ref(),
            _popularity_bias
        ).await?;

        let diversity = calc_and_write(
            harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &ann_res.user_embeddings,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "diversity_metrics.json".as_ref(),
            "diversity_metrics".as_ref(),
            _diversity_metrics
        ).await?;

        let emb_hubness = calc_and_write(
            harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &ann_res.user_embeddings,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "embedding_hubness.json".as_ref(),
            "embedding_hubness".as_ref(),
            _embedding_hubness
        ).await?;

        let ctx_hubness= contextual_hubness(
            harness,
            query_model_version.clone(),
            ranker_model_version.clone(),
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_ids,
            &timestamps,
            &parquet_output_dir,
            &summary_output_dir,
        ).await?;

        let rank_agreement = rank_agreement_and_churn(
            top_k,
            &user_ids,
            &top_k_movie_ids,  // Retriever's Top 20
            &ranker_movie_ids, // Ranker's Top 20
            parquet_output_dir,
            summary_output_dir
        ).await?;

        let cold_start_movie_metrics = calc_cold_start_movie_metrics(
            harness,
            query_model_version.clone(),
            ranker_model_version.clone(),
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_ids,
            &timestamps,
            &ann_res.user_embeddings,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            parquet_output_dir,
            summary_output_dir,
        ).await?;


        let cold_start_user_metrics = calc_cold_start_user_metrics(
            harness,
            query_model_version.clone(),
            ranker_model_version.clone(),
            &ranker_metadata,
            top_k,
            parquet_output_dir,
            summary_output_dir,
        ).await?;

        // adding t-SNE, UMAP diagnostics
        let embedding_diagnostics = calc_embedding_diagnostics(
            harness,
            query_model_version.clone(),
            ranker_model_version.clone(),
            &ranker_metadata,
            summary_output_dir,
        ).await?;

        let cross_metrics_concl = cross_metric_analysis(
            &base_metrics, &coverage, &pop_bias, &diversity, &emb_hubness, &ctx_hubness,
            &rank_agreement, &cold_start_movie_metrics, &cold_start_user_metrics,
            num_candidates, top_k, summary_output_dir
        ).await?;

        // Synthesize and write master `summary_conclusions.json` and `summary_metrics.json`
        let _ = write_master_summary_and_conclusions(
            &base_metrics,
            &coverage,
            &pop_bias,
            &diversity,
            &emb_hubness,
            &ctx_hubness,
            &rank_agreement,
            &cold_start_movie_metrics,
            &cold_start_user_metrics,
            &embedding_diagnostics,
            &cross_metrics_concl,
            num_candidates,
            top_k,
            summary_output_dir,
        )?;

        Ok(())
    }

    async fn calc_cold_start_movie_metrics(
        harness: &TestHarness,
        query_model_version: Option<VersionChoice>,
        ranker_model_version: Option<VersionChoice>,
        ranker_model_metadata: &Arc<RankerModelMetadata>,
        num_candidates: usize,
        top_k: usize,
        user_ids: &[i32],
        timestamps: &[i64],
        user_embeddings: &[f32],
        _candidate_movie_ids: &[i32],
        _top_k_movie_ids: &[i32],
        _ranker_movie_ids: &[i32],
        parquet_output_dir: &str,
        summary_output_dir: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {

        let embed_len = ranker_model_metadata.embed_len;
        let ranker_batch_size = ranker_model_metadata.batch_size;
        let candidate_model_version = query_model_version.clone();
        let movies_offset = harness.movies_offset;

        // Sample 167 movies per tier (501 total) and generate OOV=0 embeddings
        let (old_movie_ids, new_movie_ids, new_movie_embeddings) = get_cold_start_movies(
            harness,
            candidate_model_version,
            embed_len,
        )
            .await
            .expect("Error while simulating cold start movies");

        // Copy original movie embeddings and append new_movie_embeddings onto the end
        let _original_movie_catalog: Vec<f32> = harness.orchestrator._get_movies_embedding_catalog();
        let mut new_movies_catalog: Vec<f32> = Vec::with_capacity(
            _original_movie_catalog.len() + new_movie_embeddings.len()
        );
        new_movies_catalog.extend_from_slice(&_original_movie_catalog);
        new_movies_catalog.extend_from_slice(&new_movie_embeddings);

        // Build USearch index of catalog that includes cold start movies
        let new_searcher: Searcher = Searcher::new_detached(
            &new_movies_catalog,
            embed_len,
            num_candidates,
        ).expect("Error while constructing Searcher");

        let n_users = user_ids.len();
        let user_history = harness.orchestrator._get_user_history();

        // ANN Retrieval @ num_candidates (e.g. 100)
        let new_candidate_movie_ids: Vec<i32> = harness.orchestrator
            ._get_approx_nearest_neighbors(
                user_ids,
                timestamps,
                user_embeddings,
                &user_history,
                &new_searcher,
                num_candidates,
                movies_offset as i32,
            )
            .await
            .expect("Error while making ANN search")
            .into_inner();

        assert_eq!(new_candidate_movie_ids.len(), n_users * num_candidates);

        // Slice out ANN Retrieval @ top_k (e.g. 20) per user
        let mut new_top_k_movie_ids: Vec<i32> = Vec::with_capacity(n_users * top_k);
        for i in 0..n_users {
            let i0 = i * num_candidates;
            let i1 = i0 + top_k;
            new_top_k_movie_ids.extend_from_slice(&new_candidate_movie_ids[i0..i1]);
        }
        assert_eq!(new_top_k_movie_ids.len(), n_users * top_k);

        // Run Ranker in batches of `ranker_batch_size` and extract top_k per user
        let mut top_new_ranked_movie_ids: Vec<i32> = Vec::with_capacity(n_users * top_k);

        for i0 in (0..n_users).step_by(ranker_batch_size) {
            let i1 = std::cmp::min(i0 + ranker_batch_size, n_users);

            let mut ranked_movies: RankedMovies = harness.orchestrator
                ._make_ranker_request_impl(
                    &user_ids[i0..i1],
                    &timestamps[i0..i1],
                    &user_embeddings[i0 * embed_len..i1 * embed_len],
                    &new_candidate_movie_ids[i0 * num_candidates..i1 * num_candidates],
                    &new_searcher,
                    &user_history,
                    ranker_model_metadata.num_catalog_users,
                    ranker_model_version.clone(),
                )
                .await
                .expect("Error while requesting ranking of cold-start candidates");

            let (batch_top_ids, _batch_top_scores) = get_top_k_desc_scores(
                &mut ranked_movies.movie_ids,
                &mut ranked_movies.scores,
                num_candidates,
                top_k,
            );

            top_new_ranked_movie_ids.extend(batch_top_ids);
        }
        assert_eq!(top_new_ranked_movie_ids.len(), n_users * top_k);

        // Build updated movie_tiers_df and pos_test_df
        let new_catalog_size = new_movies_catalog.len() / embed_len;

        let id_map_lf = df!(
            "old_movie_id" => &old_movie_ids,
            "new_movie_id" => &new_movie_ids,
        )?.lazy();

        let base_movie_tiers_lf = harness.movie_tiers_df.clone()
            .select([
                col("movie_id").cast(DataType::Int32),
                col("movie_tier").cast(DataType::Int32),
            ]);

        let phantom_movie_tiers_lf = id_map_lf.clone()
            .inner_join(base_movie_tiers_lf.clone(), col("old_movie_id"), col("movie_id"))
            .select([
                col("new_movie_id").alias("movie_id"),
                col("movie_tier"),
            ]);

        let new_movie_tiers_df = concat(
            [base_movie_tiers_lf, phantom_movie_tiers_lf],
            UnionArgs::default(),
        )?;

        let new_pos_test_df = harness.pos_test_df.clone()
            .with_columns([col("movie_id").cast(DataType::Int32)])
            .left_join(id_map_lf, col("movie_id"), col("old_movie_id"))
            .with_columns([
                coalesce(&[col("new_movie_id"), col("movie_id")]).alias("movie_id")
            ])
            .drop(by_name(["new_movie_id"], true, false));

        let new_pos_test_tiered = new_pos_test_df.clone()
            .left_join(
                new_movie_tiers_df.clone(),
                col("movie_id"),
                col("movie_id"),
            );

        let user_gt_counts = new_pos_test_tiered
            .group_by([col("user_id")])
            .agg([
                len().alias("total_positives"),
                col("movie_tier").eq(lit(0)).sum().alias("gt_pos_tier_0"),
                col("movie_tier").eq(lit(1)).sum().alias("gt_pos_tier_1"),
                col("movie_tier").eq(lit(2)).sum().alias("gt_pos_tier_2"),
            ]);

        // Evaluate Retrieval @ num_candidates (100), Retrieval @ top_k (20), and Ranker @ top_k (20)
        let tag_retr_100 = format!("retrieval_ann_{}_cold_start_movies", num_candidates);
        let tag_retr_20 = format!("retrieval_ann_{}_cold_start_movies", top_k);
        let tag_ranker_20 = format!("ranker_{}_cold_start_movies", top_k);

        let (res_retr_100, concl_retr_100) = _metrics0(
            num_candidates,
            new_catalog_size,
            &new_pos_test_df,
            &new_movie_tiers_df,
            &harness.user_tiers_df,
            &user_gt_counts,
            user_ids,
            user_embeddings,
            &new_candidate_movie_ids,
            parquet_output_dir,
            summary_output_dir,
            &tag_retr_100,
        ).expect(&format!("{} failed", tag_retr_100));

        let (res_retr_20, concl_retr_20) = _metrics0(
            top_k,
            new_catalog_size,
            &new_pos_test_df,
            &new_movie_tiers_df,
            &harness.user_tiers_df,
            &user_gt_counts,
            user_ids,
            user_embeddings,
            &new_top_k_movie_ids,
            parquet_output_dir,
            summary_output_dir,
            &tag_retr_20,
        ).expect(&format!("{} failed", tag_retr_20));

        let (res_ranker_20, concl_ranker_20) = _metrics0(
            top_k,
            new_catalog_size,
            &new_pos_test_df,
            &new_movie_tiers_df,
            &harness.user_tiers_df,
            &user_gt_counts,
            user_ids,
            user_embeddings,
            &top_new_ranked_movie_ids,
            parquet_output_dir,
            summary_output_dir,
            &tag_ranker_20,
        ).expect(&format!("{} failed", tag_ranker_20));

        let agg_res = serde_json::json!({
            tag_retr_100: {
                "metrics": res_retr_100,
                "automated_conclusions": concl_retr_100
            },
            tag_retr_20: {
                "metrics": res_retr_20,
                "automated_conclusions": concl_retr_20
            },
            tag_ranker_20: {
                "metrics": res_ranker_20,
                "automated_conclusions": concl_ranker_20
            },
        });

        let output_json_file_name = "stratified_metrics_cold_start_movies.json";
        let output_file_path = Path::new(summary_output_dir).join(output_json_file_name);
        let file = File::create(&output_file_path)?;
        serde_json::to_writer_pretty(file, &agg_res)?;

        println!("wrote to {:?}", output_file_path);

        Ok(agg_res)
    }

    ///  simulate cold start movies and return the original ids, new cold start movie ids,
    ///   and the cold start mebeddings.
    ///
    /// # Arguments
    ///
    /// * `harness`:
    /// * `embed_len`:
    ///
    /// returns: Result<(Vec<i32, Global>, Vec<i32, Global>, Vec<f32, Global>), Box<dyn Error+Send+Sync, Global>>
    async fn get_cold_start_movies(harness: &TestHarness,
        candidate_model_version:Option<VersionChoice>, embed_len: usize) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>), Box<dyn std::error::Error + Send + Sync>> {

        let num_per_tier: usize = 167; // 167 * 3 tiers = 501 total

        //  Join test interactions with movie tiers (clone movie_tiers_df to avoid moving out of &TestHarness)
        let pos_test_movie_tiered = harness.pos_test_df.clone()
            .left_join(harness.movie_tiers_df.clone(), col("movie_id"), col("movie_id"));

        // Count test interactions per unique (movie_id, movie_tier), filter >= 5,
        //    shuffle deterministically, and take up to 167 unique movies per tier.
        let candidate_pool_df = pos_test_movie_tiered
            .group_by([
                col("movie_id").cast(DataType::Int32),
                col("movie_tier"),
            ])
            .agg([
                len().alias("test_interaction_count"),
            ])
            .filter(col("test_interaction_count").gt_eq(lit(5u32)))
            // Deterministic pseudo-random order (works without needing the optional `random` feature flag;
            // if `features = ["random"]` is enabled on polars, you can use `col("*").shuffle(Some(42))` instead)
            .with_columns([
                ((col("movie_id").cast(DataType::UInt64) * lit(2654435761u64)) % lit(1_000_000_007u64))
                    .alias("_shuffle_key")
            ])
            .sort(["_shuffle_key"], SortMultipleOptions::default())
            .group_by([col("movie_tier")])
            .head(Some(num_per_tier))
            .collect()?;

        // Extract original_movie_ids as Vec<i32>
        let original_movie_ids: Vec<i32> = candidate_pool_df
            .column("movie_id")?
            .i32()?
            .into_no_null_iter()
            .collect();

        let n_sampled = original_movie_ids.len();

        // Generate contiguous new_movie_ids
        //offset=6040, num_cat_users=6040, movie_id range is [offset+1, offset+3883], so id0 is offset + n_movies + 1
        let id0 = (harness.movies_offset + harness.num_catalog_movies + 1) as i32;
        let new_movie_ids: Vec<i32> = (id0 .. id0 + (n_sampled as i32)).collect();

        // generate the MoviesRequests using the original movie_ids
        //  then replace all the movie_ids in the request with 0s for the OOV
        //  (see recommender_systems class MovieModel movie_emb for details)
        let movie_db : MovieDb = MovieDb::new(harness.movies_path.clone()).unwrap();
        let movies_request_opt : Option<Request<MoviesRequest>> = movie_db.get_request(&original_movie_ids,
            candidate_model_version.clone());

        let request = movies_request_opt.ok_or_else(|| {
            Status::invalid_argument("Error trying to form MoviesRequest")
        })?;

        let mut movies_request = request.into_inner();

        // for the TwoTwoerDNN Candidate model, we need to use 0 for the OOV cold start movie_ids (until that gets refactored)
        movies_request.movie_ids = vec![0; new_movie_ids.len()];

        // create new embeddings with cold start Ids
        let client = CandidateModelClient::new(harness.candidate_uri.clone(), embed_len).await;

        let results : Result<Vec<f32>, Box<dyn Error>> = client.get_movies_embeddings(movies_request,
            candidate_model_version).await;

        let new_embeddings : Vec<f32> = results.expect("Error trying to create movie embeddings with OOV 0 for movie_ids");

        Ok((original_movie_ids, new_movie_ids, new_embeddings))

    }

    ///  simulate cold start users and return the original ids, new cold start user ids,
    ///   and the cold start embeddings.
    ///
    /// # Arguments
    ///
    /// * `harness`:
    /// * `embed_len`:
    ///
    /// returns: Result<(Vec<i32, Global>, Vec<i32, Global>, Vec<f32, Global>), Box<dyn Error+Send+Sync, Global>>
    ///     (original_user_ids, new_user_ids, timestamps, ann.candidate_ids, ann.user_embeddings)
    async fn get_cold_start_users(harness: &TestHarness,
        query_model_version:Option<VersionChoice>, ranker_model_version:Option<VersionChoice>)
        -> Result<(Vec<i32>, Vec<i32>, Vec<i64>, Vec<i32>, Vec<f32>), Box<dyn std::error::Error + Send + Sync>> {

        let num_per_tier: usize = 167; // 167 * 3 tiers = 501 total

        // Join test interactions with user tiers
        let pos_test_user_tiered = harness.pos_test_df.clone()
            .left_join(harness.user_tiers_df.clone(), col("user_id"), col("user_id"));

        // Count test interactions and grab the first (minimum) timestamp per unique (user_id, user_tier),
        // filter >= 5 movies, shuffle deterministically, and take up to 167 unique users per tier.
        let candidate_pool_df = pos_test_user_tiered
            .group_by([
                col("user_id").cast(DataType::Int32),
                col("user_tier"),
            ])
            .agg([
                len().alias("test_interaction_count"),
                col("timestamp").cast(DataType::Int64).min().alias("first_timestamp"),
            ])
            .filter(col("test_interaction_count").gt_eq(lit(5u32)))
            .with_columns([
                ((col("user_id").cast(DataType::UInt64) * lit(2654435761u64)) % lit(1_000_000_007u64))
                    .alias("_shuffle_key")
            ])
            .sort(["_shuffle_key"], SortMultipleOptions::default())
            .group_by([col("user_tier")])
            .head(Some(num_per_tier))
            .collect()?;

        let original_user_ids: Vec<i32> = candidate_pool_df
            .column("user_id")?
            .i32()?
            .into_no_null_iter()
            .collect();

        let first_timestamps: Vec<i64> = candidate_pool_df
            .column("first_timestamp")?
            .i64()?
            .into_no_null_iter()
            .collect();

        let n_sampled = original_user_ids.len();

        //  existing range of user_ids is [1, num_catalog_userss]
        // so start new user ids after last
        let id0 = (harness.num_catalog_users + 1) as i32;
        let new_user_ids: Vec<i32> = (id0 .. id0 + (n_sampled as i32)).collect();

        // generate the UsersRequests using the original user_ids
        //  then replace all the user_ids in the request with 0s for the OOV

        let request: Request<UsersRequest> = harness
            .orchestrator
            .get_users_request(
                &original_user_ids,
                &first_timestamps,
                query_model_version.clone(),
                ranker_model_version.clone(),
            )
            .await.expect("Error trying to form UsersRequest");

        let mut users_request: UsersRequest = request.into_inner();
        users_request.user_ids = vec![0; new_user_ids.len()];

        let tonic_request: Request<UsersRequest> = Request::new(users_request);

        // fetch the ANN, then subtract out the watch history of the original_ids

        let searcher = harness.orchestrator._get_searcher();

        // returns num_candidates for each user
        let res : Result<Response<ApproxNearestNeighborsResponse>, Status> =
            harness.orchestrator._approx_nearest_neighbors_impl(
                tonic_request,
                Some(&original_user_ids),
                &harness.orchestrator._get_user_history(),
                &searcher,
            ).await;

        let ann_res : Response<ApproxNearestNeighborsResponse> = res.expect("Error during request to Wuery and ANN search");
        let ann : ApproxNearestNeighborsResponse = ann_res.into_inner();

        {
            debug_assert_eq!(n_sampled, original_user_ids.len());
            debug_assert_eq!(n_sampled, new_user_ids.len());
            let ranker_metadata = harness.orchestrator.get_or_fetch_ranker_metadata(
                ranker_model_version.clone()
            ).expect("error fetching ranker metadata");
            let num_candidates = ranker_metadata.num_candidates;
            debug_assert_eq!(n_sampled * num_candidates, ann.candidate_ids.len());
            debug_assert_eq!(n_sampled * ranker_metadata.embed_len, ann.user_embeddings.len());
        }

        Ok((original_user_ids, new_user_ids, first_timestamps, ann.candidate_ids, ann.user_embeddings))
    }


    async fn calc_cold_start_user_metrics(
        harness: &TestHarness,
        query_model_version:Option<VersionChoice>,
        ranker_model_version:Option<VersionChoice>,
        ranker_model_metadata: &Arc<RankerModelMetadata>,
        top_k: usize,
        parquet_output_dir: &str,
        summary_output_dir: &str) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {

        let embed_len = ranker_model_metadata.embed_len;
        let num_candidates = ranker_model_metadata.num_candidates;
        let candidate_model_version = query_model_version.clone();
        let movies_offset = ranker_model_metadata.num_catalog_users; // 6041 - 6040 = 1, leaves 0 for OOV

        //select 167 randomly from each (user_tier=0, 1, and 2) of test dataset.
        // create user embeddings using the selected user data but give them ids of 0.
        // make an ANN search of the new cold start user embeddings against the full movie catalog,
        // and subtract out the watched user history of the original id holder.
        let (original_user_ids, new_user_ids, new_timestamps,
            new_candidate_ids, new_user_embeddings) =
            get_cold_start_users(
                harness,
                query_model_version.clone(),
                ranker_model_version.clone()).await.expect("Error while simulating cold start users");

        // concatenate the new user embeddings onto end of the original user_embeddings
        /*let mut new_user_embeddings_catalog: Vec<f32> = Vec::with_capacity(
            user_embeddings.len() + new_user_embeddings.len()
        );
        new_user_embeddings_catalog.extend_from_slice(&user_embeddings);
        new_user_embeddings_catalog.extend_from_slice(&new_user_embeddings);
        */

        let n_cold_users = new_user_ids.len();
        assert_eq!(new_candidate_ids.len(), n_cold_users * num_candidates);
        assert_eq!(new_user_embeddings.len(), n_cold_users * embed_len);

        // Slice out ANN Retrieval @ top_k (e.g. 20) per cold-start user
        let mut new_top_k_movie_ids: Vec<i32> = Vec::with_capacity(n_cold_users * top_k);
        for i in 0..n_cold_users {
            let i0 = i * num_candidates;
            let i1 = i0 + top_k;
            new_top_k_movie_ids.extend_from_slice(&new_candidate_ids[i0..i1]);
        }
        assert_eq!(new_top_k_movie_ids.len(), n_cold_users * top_k);

        // Build UserHistory with empty (padded) entries for `new_user_ids`
        let user_history = harness.orchestrator._get_user_history();
        //CAVEAT: make sure that UserHistory.n_users is not used as new catalog nuum users

        let searcher = harness.orchestrator._get_searcher();

        let new_num_catalog_users = ranker_model_metadata.num_catalog_users + new_user_ids.len();
        let ranker_batch_size = ranker_model_metadata.batch_size;

        let mut top_new_ranked_movie_ids: Vec<i32> = Vec::with_capacity(n_cold_users * top_k);

        for i0 in (0..n_cold_users).step_by(ranker_batch_size) {
            let i1 = std::cmp::min(i0 + ranker_batch_size, n_cold_users);

            let mut ranked_movies: RankedMovies = harness
                .orchestrator
                ._make_ranker_request_impl(
                    &new_user_ids[i0..i1],
                    &new_timestamps[i0..i1],
                    &new_user_embeddings[i0 * embed_len..i1 * embed_len],
                    &new_candidate_ids[i0 * num_candidates..i1 * num_candidates],
                    &searcher,
                    &user_history,
                    ranker_model_metadata.num_catalog_users,
                    ranker_model_version.clone(),
                )
                .await
                .expect("Error while requesting ranking for cold-start users");

            let (batch_top_ids, _batch_top_scores) = get_top_k_desc_scores(
                &mut ranked_movies.movie_ids,
                &mut ranked_movies.scores,
                num_candidates,
                top_k,
            );

            top_new_ranked_movie_ids.extend(batch_top_ids);
        }
        assert_eq!(top_new_ranked_movie_ids.len(), n_cold_users * top_k);

        let user_id_map_lf = df!(
            "old_user_id" => &original_user_ids,
            "new_user_id" => &new_user_ids,
        )?.lazy();

        // Standardize original user_tiers_df to Int32 and append new_user_ids
        // inheriting the exact user_tier of their corresponding old_user_id.
        let base_user_tiers_lf = harness.user_tiers_df.clone()
            .select([
                col("user_id").cast(DataType::Int32),
                col("user_tier").cast(DataType::Int32),
            ]);

        let phantom_user_tiers_lf = user_id_map_lf.clone()
            .inner_join(base_user_tiers_lf.clone(), col("old_user_id"), col("user_id"))
            .select([
                col("new_user_id").alias("user_id"),
                col("user_tier"),
            ]);

        let new_user_tiers_df = concat(
            [base_user_tiers_lf, phantom_user_tiers_lf],
            UnionArgs::default(),
        )?;

        // Swap old_user_id -> new_user_id in pos_test_df so the cold-start user IDs
        // are evaluated against their true held-out positive test interactions.
        let new_pos_test_df = harness.pos_test_df.clone()
            .with_columns([col("user_id").cast(DataType::Int32)])
            .left_join(user_id_map_lf, col("user_id"), col("old_user_id"))
            .with_columns([
                coalesce(&[col("new_user_id"), col("user_id")]).alias("user_id")
            ])
            .drop(by_name(["new_user_id"], true, false));

        // Join with movie_tiers_df to compute user_gt_counts
        let new_pos_test_tiered = new_pos_test_df.clone()
            .left_join(
                harness.movie_tiers_df.clone(),
                col("movie_id"),
                col("movie_id"),
            );

        let user_gt_counts = new_pos_test_tiered
            .group_by([col("user_id")])
            .agg([
                len().alias("total_positives"),
                col("movie_tier").eq(lit(0)).sum().alias("gt_pos_tier_0"),
                col("movie_tier").eq(lit(1)).sum().alias("gt_pos_tier_1"),
                col("movie_tier").eq(lit(2)).sum().alias("gt_pos_tier_2"),
            ]);

        let catalog_size = harness.num_catalog_movies;

        // Evaluate Retrieval @ num_candidates (100), Retrieval @ top_k (20), and Ranker @ top_k (20)
        //    specifically over the `new_user_ids` cohort
        let tag_retr_100 = format!("retrieval_ann_{}_cold_start_users", num_candidates);
        let tag_retr_20 = format!("retrieval_ann_{}_cold_start_users", top_k);
        let tag_ranker_20 = format!("ranker_{}_cold_start_users", top_k);

        let (res_retr_100, concl_retr_100) = _metrics0(
            num_candidates,
            catalog_size,
            &new_pos_test_df,
            &harness.movie_tiers_df,
            &new_user_tiers_df,
            &user_gt_counts,
            &new_user_ids,
            &new_user_embeddings,
            &new_candidate_ids,
            parquet_output_dir,
            summary_output_dir,
            &tag_retr_100,
        ).expect(&format!("{} failed", tag_retr_100));

        let (res_retr_20, concl_retr_20) = _metrics0(
            top_k,
            catalog_size,
            &new_pos_test_df,
            &harness.movie_tiers_df,
            &new_user_tiers_df,
            &user_gt_counts,
            &new_user_ids,
            &new_user_embeddings,
            &new_top_k_movie_ids,
            parquet_output_dir,
            summary_output_dir,
            &tag_retr_20,
        ).expect(&format!("{} failed", tag_retr_20));

        let (res_ranker_20, concl_ranker_20) = _metrics0(
            top_k,
            catalog_size,
            &new_pos_test_df,
            &harness.movie_tiers_df,
            &new_user_tiers_df,
            &user_gt_counts,
            &new_user_ids,
            &new_user_embeddings,
            &top_new_ranked_movie_ids,
            parquet_output_dir,
            summary_output_dir,
            &tag_ranker_20,
        ).expect(&format!("{} failed", tag_ranker_20));

        let agg_res = serde_json::json!({
            tag_retr_100: {
                "metrics": res_retr_100,
                "automated_conclusions": concl_retr_100
            },
            tag_retr_20: {
                "metrics": res_retr_20,
                "automated_conclusions": concl_retr_20
            },
            tag_ranker_20: {
                "metrics": res_ranker_20,
                "automated_conclusions": concl_ranker_20
            },
        });

        let output_json_file_name = "stratified_metrics_cold_start_users.json";
        let output_file_path = Path::new(summary_output_dir).join(output_json_file_name);
        let file = File::create(&output_file_path)?;
        serde_json::to_writer_pretty(file, &agg_res)?;

        println!("wrote to {:?}", output_file_path);

        Ok(agg_res)
    }

    async fn contextual_hubness(
        harness: &TestHarness,
        query_model_version:Option<VersionChoice>,
        ranker_model_version:Option<VersionChoice>,
        ranker_metadata: &Arc<RankerModelMetadata>,
        _num_candidates: usize,
        _top_k: usize,
        user_ids: &[i32],
        timestamps: &[i64],
        parquet_output_dir: &str,
        summary_output_dir: &str) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {

        let n_users = user_ids.len();
        let mut agg_res: HashMap<String, f64> = HashMap::new();
        let mut conclusions: Vec<String> = Vec::new();

        // Execute the Diagnostic gRPC Request
        let tonic_req: Request<UsersRequest> = harness.orchestrator.get_users_request(
            &user_ids,
            &timestamps,
            query_model_version.clone(),
            ranker_model_version.clone()
        ).await?;

        // returns harness.orchestrator.top_k results:
        let results: Result<Response<RankedMoviesAndRepr>, tonic::Status> =
            harness.orchestrator.predict_diagnostic(tonic_req).await;

        let response_inner = results?.into_inner();

        //  `k` (top_k) items per user
        let returned_k = response_inner.k as usize;
        let out_features = ranker_metadata.out_features;
        let tag = format!("ranker_contextual_k_{}", returned_k);

        assert_eq!(response_inner.user_reprs.len(), out_features * n_users);
        assert_eq!(response_inner.cand_reprs.len(), out_features * n_users * returned_k,
           "out_features={}, n_users={}, returned_k={}", out_features, n_users, returned_k);

        // =================================================================
        // CALCULATE CONTEXTUAL GEOMETRIC RISK (Anisotropy / Oversmoothing)
        // Compare the extracted user representation to the actual candidates
        // evaluated in their specific multi-hop sub-graph.
        // =================================================================

        let mut similarities = Vec::with_capacity(n_users * returned_k);

        for u in 0..n_users {
            let u_start = u * out_features;
            // user_reprs shape is (n_users * out_features)
            // cand_reprs shape is (n_users * out_features * top_k)
            let user_vec = &response_inner.user_reprs[u_start .. (u_start + out_features)];

            let i0 = u_start * returned_k;
            //extract each out_features slice starting from i0
            for c in 0..returned_k {
                let c_start = i0 + (out_features * c);
                let cand_vec = &response_inner.cand_reprs[c_start .. (c_start + out_features)];

                let sim = cosine_similarity(user_vec, cand_vec);
                similarities.push(sim as f64);
            }
        }

        let repeated_users: Vec<i32> = user_ids.iter()
            .flat_map(|&u| std::iter::repeat(u).take(returned_k))
            .collect();

        let base_df = df!(
            "user_id" => repeated_users,
            "movie_id" => response_inner.movie_ids,
            "similarity_to_user" => similarities
        )?.lazy();

        // =================================================================
        // ITEM-LEVEL CONTEXTUAL HUBNESS
        // =================================================================

        let item_pop_lf = base_df.clone()
            .group_by([col("movie_id")])
            .agg([
                len().alias("retrieval_count"),
                col("similarity_to_user").mean().alias("avg_sim_to_user") // Contextual Geometric Risk
            ]);

        // Keep `avg_sim_to_user` as Null for unretrieved movies (`retrieval_count == 0`) so
        // `_calculate_hubness_stats` computes `skewness` over all 3,883 catalog items while computing
        // `mean_global_sim` and `pearson_r` strictly over retrieved items without dummy 0.0 dilution.
        let mut full_item_df = harness.movie_tiers_df.clone()
            .select([col("movie_id"), col("movie_tier")])
            .left_join(item_pop_lf, col("movie_id"), col("movie_id"))
            .with_columns([
                col("retrieval_count").fill_null(lit(0u32)),
            ])
            .collect()?;

        // Export Item-Level Contextual Stats to Parquet
        let item_parquet_path = Path::new(parquet_output_dir).join(format!("contextual_hubness_item_stats_k_{}_{}.parquet", returned_k, tag));
        let mut file1 = File::create(&item_parquet_path)?;
        ParquetWriter::new(&mut file1).finish(&mut full_item_df)?;

        let counts_series = full_item_df.column("retrieval_count")?.cast(&DataType::Float64)?;
        let counts_ca = counts_series.f64()?;
        let sims_ca = full_item_df.column("avg_sim_to_user")?.f64()?;

        let (skewness, max_nk, mean_global_sim, pearson_r) =
            _calculate_hubness_stats(counts_ca, sims_ca);

        agg_res.insert(format!("ctx_hubness_skewness_{}", tag), skewness);
        agg_res.insert(format!("ctx_hubness_max_nk_{}", tag), max_nk);
        agg_res.insert(format!("ctx_hubness_mean_sim_{}", tag), mean_global_sim);
        agg_res.insert(format!("ctx_hubness_centroid_corr_{}", tag), pearson_r);

        // =================================================================
        // USER-LEVEL CONTEXTUAL EXPOSURE
        // =================================================================

        let mut user_hub_df = base_df.clone()
            .left_join(full_item_df.clone().lazy(), col("movie_id"), col("movie_id"))
            .group_by([col("user_id")])
            .agg([
                col("retrieval_count").cast(DataType::Float64).mean().alias("avg_slate_nk"),
                col("similarity_to_user").mean().alias("avg_slate_anisotropy")
            ])
            .left_join(harness.user_tiers_df.clone(), col("user_id"), col("user_id"))
            .collect()?;

        // Export User-Level Contextual Stats to Parquet
        let user_parquet_path = Path::new(parquet_output_dir).join(format!("contextual_hubness_user_stats_k_{}_{}.parquet", returned_k, tag));
        let mut file2 = File::create(&user_parquet_path)?;
        ParquetWriter::new(&mut file2).finish(&mut user_hub_df)?;

        let extract_mean = |df: &DataFrame, c: &str| -> f64 {
            df.column(c).ok().and_then(|col| col.cast(&DataType::Float64).ok()).and_then(|col| col.f64().ok().map(|ca| ca.mean().unwrap_or(0.0))).unwrap_or(0.0)
        };

        agg_res.insert(format!("ctx_exposure_mean_nk_{}", tag), extract_mean(&user_hub_df, "avg_slate_nk"));
        agg_res.insert(format!("ctx_exposure_mean_anisotropy_{}", tag), extract_mean(&user_hub_df, "avg_slate_anisotropy"));

        for t in 0..=2 {
            let mask = user_hub_df.column("user_tier")?.as_materialized_series().i32()?.equal(t as i32);
            let tier_df = user_hub_df.filter(&mask)?;

            agg_res.insert(format!("ctx_exposure_mean_nk_user_tier_{}_{}", t, tag), extract_mean(&tier_df, "avg_slate_nk"));
            agg_res.insert(format!("ctx_exposure_mean_anisotropy_user_tier_{}_{}", t, tag), extract_mean(&tier_df, "avg_slate_anisotropy"));
        }

        // =================================================================
        // AUTOMATED CONCLUSIONS (Anisotropy & Oversmoothing specific)
        // =================================================================

        if mean_global_sim > 0.90 {
            conclusions.push(format!("SEVERE OVERSMOOTHING ({}): The global mean cosine similarity between users and their recommended candidates is {:.2}. The GNN message passing has likely suffered from catastrophic anisotropy (the cone effect), collapsing all vectors into near-identical representations.", tag, mean_global_sim));
        } else if mean_global_sim > 0.70 {
            conclusions.push(format!("MODERATE ANISOTROPY ({}): The representations exhibit high contextual similarity ({:.2}), indicating a narrowing of the latent space, but maintaining some angular diversity.", tag, mean_global_sim));
        } else {
            conclusions.push(format!("HEALTHY EXPRESSIVITY ({}): The mean contextual similarity is {:.2}. The GATv2 layers are successfully preserving geometric breadth between users and candidates.", tag, mean_global_sim));
        }

        if pearson_r > 0.5 {
            conclusions.push(format!("CONTEXTUAL HUBNESS ({}): Recommendation frequency strongly correlates with representation collapse (r={:.2}). The model defaults to predicting high scores for items whose embeddings degenerate closest to the user's vector.", tag, pearson_r));
        }

        // Tier 0 = Power users, Tier 2 = Light users
        let power_anisotropy = *agg_res.get(&format!("ctx_exposure_mean_anisotropy_user_tier_0_{}", tag)).unwrap_or(&0.0);
        let light_anisotropy = *agg_res.get(&format!("ctx_exposure_mean_anisotropy_user_tier_2_{}", tag)).unwrap_or(&0.0);

        if power_anisotropy > light_anisotropy + 0.05 {
            conclusions.push(format!("COHORT COLLAPSE WARNING ({}): Power users (Tier 0 Anisotropy: {:.2}) experience significantly higher oversmoothing than Light users (Tier 2: {:.2}). The dense historical graphs of power users are overwhelming the message-passing layers and washing out their embeddings.", tag, power_anisotropy, light_anisotropy));
        }

        // =================================================================
        // JSON EXPORT
        // =================================================================

        let summary_payload = serde_json::json!({
            "metrics": agg_res,
            "automated_conclusions": conclusions
        });

        let summary_path = Path::new(summary_output_dir).join(format!("contextual_hubness_k_{}.json", returned_k));
        let summary_file = File::create(&summary_path)?;
        serde_json::to_writer_pretty(summary_file, &summary_payload)?;

        println!("Wrote contextual hubness metrics to {:?}", summary_path);

        Ok(summary_payload)
    }

    async fn calc_and_write(
        harness: &TestHarness,
        ranker_metadata: &Arc<RankerModelMetadata>,
        num_candidates: usize,
        top_k: usize,
        user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...] for tier being movie_tiers 0, 1, 2
        user_ids: &[i32],            // shape: (n_users)
        user_embeddings: &[f32],     // shape (n_users * embed_len)
        candidate_movie_ids: &[i32],    // shape: (n_users * top_k)   these are the retrieved or ranked movie_ids
        top_k_movie_ids: &[i32],           // shape: (n_users * top_k)   these are the retrieved or ranked movie_ids
        ranker_movie_ids: &[i32],         // shape: (n_users * top_k)
        parquet_output_dir: &str,
        summary_output_dir: &str,
        output_json_file_name: &str,
        keyword: &str,
        metric_func: impl Fn(usize, usize, &TestHarness, &LazyFrame, &[i32], &[f32],
            &[i32],  &str, &str, &str) -> PolarsResult<(HashMap<String, f64>, Vec<String>)>
       ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {

        let tag1 = format!("retrieval_ann_{}", num_candidates);
        let result1 = //tokio::task::spawn_blocking(move || {
            metric_func(
                num_candidates,
                ranker_metadata.num_catalog_movies,
                harness,
                user_gt_counts,
                user_ids,
                user_embeddings,
                candidate_movie_ids,
                parquet_output_dir,
                summary_output_dir,
                &tag1
            );
        //}).await.expect("Spawn blocking panicked")?;

        let (res1, concl1) = result1.expect(
            format!("{} failed for tag1", keyword).as_str());
        println!("have results1");

        let tag2 = format!("retrieval_ann_{}", top_k);
        let result2 = //tokio::task::spawn_blocking(move || {
            metric_func(
                top_k,
                ranker_metadata.num_catalog_movies,
                harness,
                user_gt_counts,
                user_ids,
                user_embeddings,
                top_k_movie_ids,
                parquet_output_dir,
                summary_output_dir,
                &tag2
            );
        //}).await.expect("Spawn blocking panicked")?;

        let (res2, concl2) = result2.expect(
            format!("{} failed for tag2", keyword).as_str());

        println!("have results2");


        // =========== RANKER ===============================

        //let n_users = user_ids.len();


        let tag3 = format!("ranker_{}", top_k);
        let result3 = //tokio::task::spawn_blocking(move || {
            metric_func(
                top_k,
                ranker_metadata.num_catalog_movies,
                harness,
                user_gt_counts,
                user_ids,
                user_embeddings,
                ranker_movie_ids,
                parquet_output_dir,
                summary_output_dir,
                &tag3
            );
        //}).await.expect("Spawn blocking panicked")?;

        println!("have results3");

        let (res3, concl3) = result3.expect(
            format!("{} failed for tag3", keyword).as_str());


        let agg_res = serde_json::json!(
            {
               tag1: {
                    "metrics": res1,
                    "automated_conclusions": concl1
                },
                tag2: {
                    "metrics": res2,
                    "automated_conclusions": concl2
                },
                tag3: {
                    "metrics": res3,
                    "automated_conclusions": concl3
                }
            }
        );

        let output_file_path = Path::new(&summary_output_dir).join(output_json_file_name);
        let file = File::create(output_file_path.clone())?;
        serde_json::to_writer_pretty(file, &agg_res)?;

        println!("{}", format!("wrote to {:?}", output_file_path));

        Ok(agg_res)
    }

    /// calculate retrieval or ranking catalog coverage,
    /// gini, and lorenz curve where
    /// lorenz curve shows the discrete frequency distribution of items appearing in the
    /// generated recommendation user slates across all users.
    /// The lorenz curve X axis is the cumulative percentage of items sorted from least
    /// recommended to most recommended.
    /// The lorenz curve Y axis is the total number of recommendations (or impressions) that that item
    /// received.
    /// if every item in the catalog receives the exact same number of recommendations, then
    /// the lorenz curve is a 45 degree line due to it being a cumulated sum.
    /// That 45 degree line is the line of Perfect Equality (the bottom 50% receive 50% of recommendations).
    /// The gini coefficient is a measure of concentration (essentially the opposite of the
    /// information entropy which measures dispersion).
    /// The gini coefficient is in range [0,1] inclusive where 0 is perfect equality where all
    /// items are recommended the same number of times and a gini of 1 means that 1 item represents
    /// 100% of the recommendations.
    /// The gini coefficient is derived as (the abs value of area between the lorenz 45 degree perfect equality
    /// curve and the real observed lorenz curve) divided by (the area under the perfect equality curve)..
    ///
    ///
    /// # Arguments
    ///
    /// * `top_k`:
    /// * `catalog_size`:
    /// * `pos_test_df`:
    /// * `movie_tiers_df`:
    /// * `user_tiers_df`:
    /// * `user_gt_counts`:
    /// * `user_ids`:
    /// * `neighbors`:
    /// * `parquet_output_dir`:
    /// * `tag`:
    ///
    /// returns: Result<(HashMap<String, f64, RandomState, Global>, Vec<String, Global>), PolarsError>
    pub fn _coverage_and_gini(
        k: usize,                    // either num_candidates or top_k
        catalog_size: usize,         // number of catalog movies
        harness: &TestHarness,
        _user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...] for tier being movie_tiers 0, 1, 2
        user_ids: &[i32],            // shape: (n_users)
        _user_embeddings: &[f32],     // shape (n_users * embed_len)
        neighbors: &[i32],           // shape: (n_users * ks)   these are the retrieved or ranked movie_ids
        parquet_output_dir: &str,
        summary_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {
        let n_users = user_ids.len();
        let mut agg_res: HashMap<String, f64> = HashMap::new();
        let mut conclusions: Vec<String> = Vec::new();

        //  Explode neighbors into a flat dataframe
        let repeated_users: Vec<i32> = user_ids.iter()
            .flat_map(|&u| std::iter::repeat(u).take(k))
            .collect();
        let ranks: Vec<i32> = (0..n_users)
            .flat_map(|_| 1..=k as i32)
            .collect();

        let retrieval_df = df!(
            "user_id" => repeated_users,
            "movie_id" => neighbors,
            "rank" => ranks,
        )?.lazy();

        let mut joined_df = retrieval_df
            .left_join(harness.movie_tiers_df.clone(), col("movie_id"), col("movie_id"))
            .left_join(harness.user_tiers_df.clone(), col("user_id"), col("user_id"))
            .collect()?;

        // Evaluate catalog counts per tier once
        let mt_collected = harness.movie_tiers_df.clone().collect()?;
        let mt_series = mt_collected.column("movie_tier")?.i32()?;
        let mut cat_counts = vec![0.0; 3];
        for t in 0..=2 {
            cat_counts[t as usize] = mt_series.equal(t).sum().unwrap_or(0) as f64;
        }

        // --- COVERAGE CALCS ---
        let total_unique = joined_df.column("movie_id")?.n_unique()? as f64;
        agg_res.insert(format!("coverage_at_{}_full_{}", k, tag), total_unique / (catalog_size as f64));

        for m_tier in 0..=2 {
            let filtered = joined_df.filter(&joined_df.column("movie_tier")?.i32()?.equal(m_tier))?;
            let count = filtered.column("movie_id")?.n_unique()? as f64;
            let cat_count = cat_counts[m_tier as usize];
            let cov = if cat_count > 0.0 { count / cat_count } else { 0.0 };
            agg_res.insert(format!("coverage_at_{}_movie_tier_{}_{}", k, m_tier, tag), cov);
        }

        for u_tier in 0..=2 {
            let u_filtered = joined_df.filter(&joined_df.column("user_tier")?.i32()?.equal(u_tier))?;
            let count = u_filtered.column("movie_id")?.n_unique()? as f64;
            agg_res.insert(format!("coverage_at_{}_user_tier_{}_{}", k, u_tier, tag), count / (catalog_size as f64));

            for m_tier in 0..=2 {
                let um_filtered = u_filtered.filter(&u_filtered.column("movie_tier")?.i32()?.equal(m_tier))?;
                let um_count = um_filtered.column("movie_id")?.n_unique()? as f64;
                let cat_count = cat_counts[m_tier as usize];
                let cov = if cat_count > 0.0 { um_count / cat_count } else { 0.0 };
                agg_res.insert(format!("coverage_at_{}_user_tier_{}_movie_tier_{}_{}", k, u_tier, m_tier, tag), cov);
            }
        }

        // Export user retrievals
        let retrieval_path = Path::new(parquet_output_dir).join(format!("user_coverage_top_{}_{}.parquet", k, tag));
        let mut file = File::create(&retrieval_path)?;
        ParquetWriter::new(&mut file).finish(&mut joined_df)?;

        // --- GINI COEFFICIENT & LORENZ CURVE CALCS ---

        // Create base item frequency frame against the FULL catalog
        let mut freq_df = harness.movie_tiers_df.clone()
            .select([col("movie_id"), col("movie_tier")])
            .left_join(
                joined_df.clone().lazy().group_by([col("movie_id")]).agg([len().alias("retrieval_count")]),
                col("movie_id"),
                col("movie_id")
            )
            .with_columns([col("retrieval_count").fill_null(lit(0u32))])
            .sort(["retrieval_count"], SortMultipleOptions::default().with_order_descending(false))
            .collect()?;

        let freq_path = Path::new(parquet_output_dir).join(format!("item_frequencies_top_{}_{}.parquet", k, tag));
        let mut file = File::create(&freq_path)?;
        ParquetWriter::new(&mut file).finish(&mut freq_df)?;

        // Native Rust Closure for fast Gini Math & Lorenz extraction
        let calc_gini_and_lorenz = |df: &DataFrame, count_col: &str, extract_lorenz: bool| -> PolarsResult<(f64, f64, f64, Vec<f64>)> {
            let n = df.height() as f64;
            if n == 0.0 { return Ok((0.0, 0.0, 0.0, vec![])); }

            // Ensure sorted ascending
            let sorted = df.sort([count_col], SortMultipleOptions::default().with_order_descending(false))?;

            let counts = sorted.column(count_col)?.cast(&DataType::Float64)?;
            let counts_ca = counts.f64()?; // Downcast to ChunkedArray first

            // ChunkedArray::sum() returns an Option<f64>, making unwrap_or valid
            let total_recs = counts_ca.sum().unwrap_or(0.0);

            if total_recs == 0.0 { return Ok((0.0, 0.0, 0.0, vec![])); }

            let mut sum_rank_count = 0.0;
            let mut current_cum_sum = 0.0;
            let mut bottom_80 = 0.0;
            let mut top_10 = 0.0;
            let mut lorenz_curve = vec![0.0; 100];

            for (i, val_opt) in counts_ca.iter().enumerate() {
                let val = val_opt.unwrap_or(0.0);
                let rank = (i + 1) as f64;

                sum_rank_count += rank * val;

                if extract_lorenz {
                    current_cum_sum += val;
                    let cum_recs_pct = current_cum_sum / total_recs;
                    let cum_items_pct = rank / n;

                    if cum_items_pct <= 0.80 { bottom_80 = cum_recs_pct; }
                    if cum_items_pct <= 0.90 { top_10 = 1.0 - cum_recs_pct; } // captures remaining 10%

                    let percentile = (cum_items_pct * 100.0).ceil() as usize;
                    if percentile > 0 && percentile <= 100 {
                        lorenz_curve[percentile - 1] = cum_recs_pct;
                    }
                }
            }

            let gini = (2.0 * sum_rank_count) / (n * total_recs) - ((n + 1.0) / n);

            if extract_lorenz {
                // Fill forward missing percentile buckets
                let mut last_val = 0.0;
                for val in lorenz_curve.iter_mut() {
                    if *val == 0.0 { *val = last_val; } else { last_val = *val; }
                }
            }

            Ok((gini, bottom_80, top_10, lorenz_curve))
        };

        // Calculate Full Catalog Gini & Lorenz
        let (full_gini, bottom_80_share, top_10_share, lorenz_curve)
            = calc_gini_and_lorenz(&freq_df, "retrieval_count", true)?;

        agg_res.insert(format!("gini_at_{}_full_{}", k, tag), full_gini);
        agg_res.insert(format!("lorenz_at_{}_bottom_80_share_{}", k, tag), bottom_80_share);
        agg_res.insert(format!("lorenz_at_{}_top_10_share_{}", k, tag), top_10_share);

        // Write Lorenz curve to JSON for downstream Python plotting
        if !lorenz_curve.is_empty() {
            let lorenz_path = Path::new(summary_output_dir).join(format!("lorenz_curve_top_{}_{}.json", k, tag));
            let lorenz_file = File::create(lorenz_path)?;
            serde_json::to_writer_pretty(lorenz_file, &lorenz_curve).expect("Failed to write Lorenz JSON");
        }

        // Gini by Movie Tier
        for m_tier in 0..=2 {
            let m_tier_freq = freq_df.filter(&freq_df.column("movie_tier")?.i32()?.equal(m_tier))?;
            let (tier_gini, _, _, _) = calc_gini_and_lorenz(&m_tier_freq, "retrieval_count", false)?;
            agg_res.insert(format!("gini_at_{}_movie_tier_{}_{}", k, m_tier, tag), tier_gini);
        }

        // Gini by User Tier
        for u_tier in 0..=2 {
            let tier_retrievals = joined_df.filter(&joined_df.column("user_tier")?.i32()?.equal(u_tier))?;

            let u_tier_freq = harness.movie_tiers_df.clone().select([col("movie_id")])
                .left_join(
                    tier_retrievals.lazy().group_by([col("movie_id")]).agg([len().alias("u_retrieval_count")]),
                    col("movie_id"),
                    col("movie_id")
                )
                .with_columns([col("u_retrieval_count").fill_null(lit(0u32))])
                .collect()?;

            let (u_tier_gini, _, _, _) = calc_gini_and_lorenz(&u_tier_freq, "u_retrieval_count", false)?;
            agg_res.insert(format!("gini_at_{}_user_tier_{}_{}", k, u_tier, tag), u_tier_gini);
        }

        // --- AUTOMATED INSIGHTS & CONCLUSIONS ---

        let total_recs_series = freq_df.column("retrieval_count")?.cast(&DataType::Float64)?;
        let total_recs = total_recs_series.f64()?.sum().unwrap_or(0.0);

        if total_recs > 0.0 {
            if full_gini > 0.90 {
                conclusions.push(format!("SEVERE POPULARITY BIAS at_{} ({}): Gini is {:.2}. The model is acting as a popularity echo chamber, collapsing onto blockbuster items.", k, tag, full_gini));
            } else if full_gini < 0.45 {
                conclusions.push(format!("SUSPICIOUSLY UNIFORM at_{} ({}): Gini is {:.2}. The model may be overly random or popularity suppression is too aggressive.", k, tag, full_gini));
            } else {
                conclusions.push(format!("HEALTHY BIAS at_{} ({}): Gini is {:.2}. The model successfully balances mainstream relevance with catalog exploration.", k, tag, full_gini));
            }

            if bottom_80_share < 0.05 {
                conclusions.push(format!("DEAD TAIL at_{} ({}): The bottom 80% of the catalog receives only {:.1}% of recommendations. Niche items are effectively invisible.", k, tag, bottom_80_share * 100.0));
            } else if bottom_80_share > 0.15 {
                conclusions.push(format!("STRONG TAIL at_{} ({}): The bottom 80% captures {:.1}% of traffic, indicating excellent long-tail surfacing capability.", k, tag, bottom_80_share * 100.0));
            } else {
                conclusions.push(format!("MODERATE TAIL at_{} ({}): The bottom 80% captures {:.1}% of traffic.", k, tag, bottom_80_share * 100.0));
            }

            if top_10_share > 0.75 {
                conclusions.push(format!("HEAD HEAVY at_{} ({}): The top 10% of items consume {:.1}% of all recommendation slots.", k, tag, top_10_share * 100.0));
            } else {
                conclusions.push(format!("DIVERSE HEAD at_{} ({}): The top 10% consume {:.1}% of slots, leaving plenty of room for the torso/tail.", k, tag, top_10_share * 100.0));
            }

            // Tier 0 = Power users, Tier 2 = Light users
            let gini_power = *agg_res.get(&format!("gini_at_{}_user_tier_0_{}", k, tag)).unwrap_or(&1.0);
            let gini_light = *agg_res.get(&format!("gini_at_{}_user_tier_2_{}", k, tag)).unwrap_or(&1.0);

            if gini_power < gini_light - 0.02 {
                conclusions.push(format!("USER PERSONALIZATION at_{} ({}): Power users exhibit lower Gini (more diverse slates) than light users, successfully leveraging rich interaction histories.", k, tag));
            } else if gini_power > gini_light + 0.02 {
                conclusions.push(format!("WARNING (COHORT COLLAPSE) at_{} ({}): Power users have higher concentration (Gini) than light users. The model may be pulling rich histories into dense popularity traps.", k, tag));
            } else {
                conclusions.push(format!("UNIFORM COHORTS at_{} ({}): Light and Power users experience roughly the same level of catalog concentration.", k, tag));
            }
        }

        Ok((agg_res, conclusions))
    }

    pub fn _popularity_bias(
        k: usize,                // either num_candidates or top_k
        _catalog_size: usize,
        harness: &TestHarness,
        _user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...] for tier being movie_tiers 0, 1, 2
        user_ids: &[i32],            // shape: (n_users)
        _user_embeddings: &[f32],     // shape (n_users * embed_len)
        neighbors: &[i32],           // shape: (n_users * k)   these are the retrieved or ranked movie_ids
        parquet_output_dir: &str,
        _summary_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {

        let mut agg_res: HashMap<String, f64> = HashMap::new();
        let mut conclusions: Vec<String> = Vec::new();

        // Calculate Global Item Popularity from Training History
        let item_pop_lf = harness.train_history_df.clone()
            .group_by([col("movie_id")])
            .agg([len().alias("global_pop_count")]);

        // Build Recommendation LazyFrame
        let n_users = user_ids.len();
        let repeated_users: Vec<i32> = user_ids.iter()
            .flat_map(|&u| std::iter::repeat(u).take(k))
            .collect();
        let ranks: Vec<i32> = (0..n_users)
            .flat_map(|_| 1..=k as i32)
            .collect();
        let rec_lf = df!(
            "user_id" => repeated_users,
            "movie_id" => neighbors,
            "rank" => ranks,
        )?.lazy();

        // Calculate Average Popularity of Recommended Slates per User
        let user_rec_pop_lf = rec_lf
            .left_join(item_pop_lf.clone(), col("movie_id"), col("movie_id"))
            .with_columns([col("global_pop_count").fill_null(lit(0u32))])
            .group_by([col("user_id")])
            .agg([col("global_pop_count").mean().alias("rec_mean_pop")]);

        // Calculate Average Popularity of Organic User History
        let user_hist_pop_lf = harness.train_history_df.clone()
            .left_join(item_pop_lf, col("movie_id"), col("movie_id"))
            .group_by([col("user_id")])
            .agg([col("global_pop_count").mean().alias("hist_mean_pop")]);

        //  Combine, Calculate Delta, and Stratify by User Tier
        //  Columns: ["user_id", "rec_mean_pop", "hist_mean_pop", "user_tier", "pop_bias_delta"]
        let mut user_bias_df = user_rec_pop_lf
            .left_join(user_hist_pop_lf, col("user_id"), col("user_id"))
            .left_join(harness.user_tiers_df.clone(), col("user_id"), col("user_id"))
            .with_columns([
                // Positive delta means model amplifies popularity; Negative means model explores niche
                (col("rec_mean_pop") - col("hist_mean_pop")).alias("pop_bias_delta")
            ])
            .collect()?;

        // Write User-Level Metrics to Parquet for Pairwise T-Tests
        let parquet_path = Path::new(parquet_output_dir).join(format!("popularity_bias_k_{}_{}.parquet", k, tag));
        let mut file = File::create(&parquet_path)?;
        ParquetWriter::new(&mut file).finish(&mut user_bias_df)?;

        // Aggregate Global Statistics
        let extract_mean = |df: &DataFrame, c: &str| -> f64 {
            df.column(c).ok()
                .and_then(|col| col.cast(&DataType::Float64).ok())
                .and_then(|col| col.f64().ok().map(|ca| ca.mean().unwrap_or(0.0)))
                .unwrap_or(0.0)
        };

        let global_rec_pop = extract_mean(&user_bias_df, "rec_mean_pop");
        let global_hist_pop = extract_mean(&user_bias_df, "hist_mean_pop");
        let global_delta = extract_mean(&user_bias_df, "pop_bias_delta");

        agg_res.insert(format!("pop_bias_rec_mean_k_{}_{}", k, tag), global_rec_pop);
        agg_res.insert(format!("pop_bias_hist_mean_{}", tag), global_hist_pop);
        agg_res.insert(format!("pop_bias_delta_mean_k_{}_{}", k, tag), global_delta);

        // Aggregate Stratified Statistics by User Tier
        for t in 0..=2 {
            let mask = user_bias_df.column("user_tier")?.as_materialized_series().i32()?.equal(t);
            let tier_df = user_bias_df.filter(&mask)?;

            let t_rec = extract_mean(&tier_df, "rec_mean_pop");
            let t_hist = extract_mean(&tier_df, "hist_mean_pop");
            let t_delta = extract_mean(&tier_df, "pop_bias_delta");

            agg_res.insert(format!("pop_bias_rec_mean_k_{}_user_tier_{}_{}", k, t, tag), t_rec);
            agg_res.insert(format!("pop_bias_hist_mean_user_tier_{}_{}", t, tag), t_hist);
            agg_res.insert(format!("pop_bias_delta_mean_k_{}_user_tier_{}_{}", k, t, tag), t_delta);
        }

        // 9. Automated Conclusions
        let delta_pct = (global_delta / global_hist_pop) * 100.0;

        if delta_pct > 25.0 {
            conclusions.push(format!("SEVERE POPULARITY AMPLIFICATION ({}): The model recommends items that are {:.1}% more popular than users naturally consume. It is acting as a popularity echo chamber.", tag, delta_pct));
        } else if delta_pct > 5.0 {
            conclusions.push(format!("MODERATE POPULARITY BIAS ({}): The model leans toward popular items, inflating slate popularity by {:.1}% over historical behavior.", tag, delta_pct));
        } else if delta_pct < -10.0 {
            conclusions.push(format!("NICHE EXPLORATION ({}): The model actively suppresses popularity, surfacing items {:.1}% less popular than organic consumption.", tag, delta_pct.abs()));
        } else {
            conclusions.push(format!("POPULARITY CALIBRATED ({}): The model perfectly mirrors organic user popularity preferences (Delta: {:.1}%).", tag, delta_pct));
        }

        // Tier 0 = Power users, Tier 2 = Light users
        let power_delta = *agg_res.get(&format!("pop_bias_delta_mean_k_{}_user_tier_0_{}", k, tag)).unwrap_or(&0.0);
        let light_delta = *agg_res.get(&format!("pop_bias_delta_mean_k_{}_user_tier_2_{}", k, tag)).unwrap_or(&0.0);

        if power_delta > light_delta + (global_hist_pop * 0.1) {
            conclusions.push(format!("POWER USER PENALTY ({}): Power users (Tier 0) experience significantly worse popularity bias than Light users (Tier 2). The model ignores their rich histories and defaults to blockbusters.", tag));
        } else if power_delta < light_delta {
            conclusions.push(format!("SUCCESSFUL PERSONALIZATION ({}): The model leverages the rich histories of Power users (Tier 0) to surface less mainstream, personalized items compared to Light users (Tier 2).", tag));
        }

        Ok((agg_res, conclusions))

    }

    /// calculate intra-list diversity and inter-list diversion.
    /// Intra-list diversity (personalization):
    ///      calculates the similarity between User A and User B's
    ///      slates by movie_id.
    /// Inter-list diversity (novelty/ breadth)"
    ///     calculates simularity between Movie X to Movie Y within User A's slate.
    ///     This method requires a representation of the movies such as features or the embeddings.
    ///
    /// # Arguments
    ///
    /// * `k`:
    /// * `catalog_size`:
    /// * `harness`:
    /// * `user_gt_counts`:
    /// * `user_ids`:
    /// * `neighbors`:
    /// * `parquet_output_dir`:
    /// * `summary_output_dir`:
    /// * `tag`:
    ///
    /// returns: Result<(HashMap<String, f64, RandomState, Global>, Vec<String, Global>), PolarsError>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub fn _diversity_metrics(
        k: usize,                    // either num_candidates or top_k
        _catalog_size: usize,
        harness: &TestHarness,
        _user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...] for tier being movie_tiers 0, 1, 2
        user_ids: &[i32],            // shape: (n_users)
        _user_embeddings: &[f32],     // shape (n_users * embed_len)
        neighbors: &[i32],           // shape: (n_users * k)   these are the retrieved or ranked movie_ids
        parquet_output_dir: &str,
        _summary_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {

        let mut agg_res: HashMap<String, f64> = HashMap::new();
        let mut conclusions: Vec<String> = Vec::new();
        let n_users = user_ids.len();

        let mut slates: Vec<Vec<i32>> = Vec::with_capacity(n_users);
        for chunk in neighbors.chunks_exact(k) {
            let mut slate = chunk.to_vec();
            slate.sort_unstable();
            slates.push(slate);
        }

        let ids_df = df!("user_id" => user_ids)?.lazy();
        let aligned_tiers_df = ids_df
            .left_join(harness.user_tiers_df.clone(), col("user_id"), col("user_id"))
            .with_columns([col("user_tier").fill_null(lit(0i32))])
            .collect()?;

        let user_tiers: Vec<i32> = aligned_tiers_df
            .column("user_tier")?
            .i32()?
            .iter()
            .map(|opt| opt.unwrap_or(0))
            .collect();

        // =================================================================
        // INTER-LIST DIVERSITY (Personalization)
        // =================================================================
        println!("Calculating Inter-list diversity (Statistically Sampled)...");

        // OPTIMIZATION: Cap the inner loop to ~300 comparisons to prevent O(N^2) lockup.
        //    in test positives there are 48_285 ratings for 5096 unique users.
        let max_samples = 1_000;//300;
        let step = (n_users / max_samples).max(1);
        // inter-list is calculated for each user, but the comparison is to only max_samples now instead o

        let interlist_diversity_scores: Vec<f64> = slates.par_iter().map(|slate_a| {
            let mut total_jaccard = 0.0;
            let mut valid_comparisons = 0.0;

            for slate_b in slates.iter().step_by(step) {
                let mut i = 0;
                let mut j = 0;
                let mut intersection = 0;

                while i < k && j < k {
                    if slate_a[i] == slate_b[j] {
                        intersection += 1;
                        i += 1;
                        j += 1;
                    } else if slate_a[i] < slate_b[j] {
                        i += 1;
                    } else {
                        j += 1;
                    }
                }

                // Skip identical slates (self-comparisons)
                if intersection == k { continue; }

                let union = (2 * k) - intersection;
                total_jaccard += (intersection as f64) / (union as f64);
                valid_comparisons += 1.0;
            }

            if valid_comparisons > 0.0 {
                1.0 - (total_jaccard / valid_comparisons)
            } else {
                0.0
            }
        }).collect();

        // =================================================================
        // INTRA-LIST DIVERSITY (Breadth)
        // =================================================================
        println!("Calculating Intra-list diversity...");

        let movie_embeddings: Vec<f32> = harness.orchestrator._get_movies_embedding_catalog();
        let embed_len = harness.embed_len;
        let movies_offset = harness.movies_offset;

        let intralist_diversity_scores: Vec<f64> = slates.par_iter().map(|slate| {
            let mut total_distance = 0.0;
            let num_pairs = (k * (k - 1)) / 2;

            for i in 0..k {
                for j in (i+1)..k {
                    let raw_a = slate[i] as usize;
                    let raw_b = slate[j] as usize;

                    // OPTIMIZATION: Safely remove the global ID offset so we don't index out of bounds
                    let idx_a = if raw_a >= movies_offset { raw_a - movies_offset } else { raw_a };
                    let idx_b = if raw_b >= movies_offset { raw_b - movies_offset } else { raw_b };

                    let start_a = idx_a * embed_len;
                    let start_b = idx_b * embed_len;

                    let emb_a = &movie_embeddings[start_a..(start_a + embed_len)];
                    let emb_b = &movie_embeddings[start_b..(start_b + embed_len)];

                    let sim = cosine_similarity(emb_a, emb_b);
                    total_distance += 1.0 - (sim as f64);
                }
            }

            if num_pairs > 0 {
                total_distance / (num_pairs as f64)
            } else {
                0.0
            }
        }).collect();

        // =================================================================
        // METRICS AGGREGATION & EXPORT
        // =================================================================
        let mut metrics_df = df!(
            "user_id" => user_ids,
            "user_tier" => user_tiers,
            "interlist_diversity" => &interlist_diversity_scores,
            "intralist_diversity" => &intralist_diversity_scores
        )?;

        let parquet_path = Path::new(parquet_output_dir).join(format!("diversity_metrics_k_{}_{}.parquet", k, tag));
        let mut file = File::create(&parquet_path)?;
        ParquetWriter::new(&mut file).finish(&mut metrics_df)?;

        let extract_mean = |df: &DataFrame, c: &str| -> f64 {
            df.column(c).ok()
                .and_then(|col| col.cast(&DataType::Float64).ok())
                .and_then(|col| col.f64().ok().map(|ca| ca.mean().unwrap_or(0.0)))
                .unwrap_or(0.0)
        };

        let global_interlist = extract_mean(&metrics_df, "interlist_diversity");
        let global_intralist = extract_mean(&metrics_df, "intralist_diversity");

        agg_res.insert(format!("interlist_diversity_mean_k_{}_{}", k, tag), global_interlist);
        agg_res.insert(format!("intralist_diversity_mean_k_{}_{}", k, tag), global_intralist);

        for t in 0..=2 {
            let mask = metrics_df.column("user_tier")?.as_materialized_series().i32()?.equal(t as i32);
            let tier_df = metrics_df.filter(&mask)?;

            let t_inter = extract_mean(&tier_df, "interlist_diversity");
            let t_intra = extract_mean(&tier_df, "intralist_diversity");

            agg_res.insert(format!("interlist_diversity_mean_k_{}_user_tier_{}_{}", k, t, tag), t_inter);
            agg_res.insert(format!("intralist_diversity_mean_k_{}_user_tier_{}_{}", k, t, tag), t_intra);
        }

        if global_interlist > 0.90 {
            conclusions.push(format!("HIGH PERSONALIZATION ({}): Global Inter-list Diversity is {:.2}%. The model successfully delivers highly unique slates customized to individual users.", tag, global_interlist * 100.0));
        } else if global_interlist > 0.60 {
            conclusions.push(format!("MODERATE PERSONALIZATION ({}): Inter-list Diversity is {:.2}%. Users receive a mix of personalized items and global blockbusters.", tag, global_interlist * 100.0));
        } else {
            conclusions.push(format!("LOW PERSONALIZATION WARNING ({}): Inter-list Diversity is only {:.2}%. The model is serving almost identical slates to everyone, indicating severe popularity bias or catastrophic forgetting.", tag, global_interlist * 100.0));
        }

        if global_intralist < 0.10 {
            conclusions.push(format!("MONOTONOUS SLATES ({}): Global Intra-list Diversity is very low ({:.2}). The model is recommending walls of virtually identical items (e.g., 20 Marvel movies in a row).", tag, global_intralist));
        } else if global_intralist > 0.35 {
            conclusions.push(format!("HIGH BREADTH ({}): Intra-list Diversity is high ({:.2}), indicating slates feature a wide variety of semantic genres/topics.", tag, global_intralist));
        }

        // Tier 0 = Power users, Tier 2 = Light users
        let power_inter = *agg_res.get(&format!("interlist_diversity_mean_k_{}_user_tier_0_{}", k, tag)).unwrap_or(&0.0);
        let light_inter = *agg_res.get(&format!("interlist_diversity_mean_k_{}_user_tier_2_{}", k, tag)).unwrap_or(&0.0);

        if power_inter > light_inter + 0.05 {
            conclusions.push(format!("TIERED PERSONALIZATION ({}): Power users (Tier 0: {:.2}%) receive significantly more personalized (diverse) slates than Light users (Tier 2: {:.2}%), effectively leveraging their rich interaction histories.", tag, power_inter * 100.0, light_inter * 100.0));
        }

        Ok((agg_res, conclusions))
    }

    pub fn _embedding_hubness(
        k: usize,                    // either num_candidates or top_k
        _catalog_size: usize,
        harness: &TestHarness,
        _user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...]
        user_ids: &[i32],            // shape: (n_users)
        user_embeddings: &[f32],     // shape (n_users * embed_len)
        neighbors: &[i32],           // shape: (n_users * k)
        parquet_output_dir: &str,
        #[allow(dead_code)]
        _summary_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {

        /*
        TwoTower embeddings:
           - relevant pairs should have good alignment, measured by coine sim or dot product
           - as a whole, all embeddings should be well distributed over the latent space.
             if the embeddings are normalized, the latent space is a hypersphere and the
             embeddings are on the surface of it.  negative pairs should have good separation
             and positive pairs should proximally close.
           - stratified partitions of the embeddings for movies by popularity or user by popularity
             should all be co-spatial too, that is, they share the same natural semantic clusters.
             The differences in spatial mappings on the hyper-sphere are from the latent
             meaning of the features (e.g. genre, occupation, city).
             - an error in popularity stratification can be seen as popular items being centrally
               clusters on the hypersphere with medium popular items a little further away and
               least popular items even further away from the central popular items.
               - to correct for such an error:
                  - log Q correction
                  - hard negative sampling
                  - temperature scaling
        Hubness:
           - an error in filling the embedding space where one sees that a small number of
             vector embeddings (hubs) are the nearest neighbors to the rest of the embeddings.
             This cone-like appearance should be avoided:
                - apply more regularization to avoid popularity bias, etc.
         */

        let mut agg_res: HashMap<String, f64> = HashMap::new();
        let mut conclusions: Vec<String> = Vec::new();
        let n_users = user_ids.len();

        let movie_embeddings: Vec<f32> = harness.orchestrator._get_movies_embedding_catalog();
        let embed_len = harness.embed_len;
        let movies_offset = harness.movies_offset;

        // =================================================================
        // CALCULATE GLOBAL USER CENTROID
        // =================================================================
        let mut user_centroid = vec![0.0f32; embed_len];
        for chunk in user_embeddings.chunks_exact(embed_len) {
            for (i, &val) in chunk.iter().enumerate() {
                user_centroid[i] += val;
            }
        }
        for val in user_centroid.iter_mut() {
            *val /= n_users as f32;
        }

        // =================================================================
        //ITEM-LEVEL HUBNESS (N_k & Centroid Proximity)
        // =================================================================

        let item_pop_lf = df!("movie_id" => neighbors)?.lazy()
            .group_by([col("movie_id")])
            .agg([len().alias("retrieval_count")]);

        let mut full_item_df = harness.movie_tiers_df.clone()
            .select([col("movie_id").cast(DataType::Int32), col("movie_tier")])
            .left_join(item_pop_lf, col("movie_id"), col("movie_id"))
            .with_columns([col("retrieval_count").fill_null(lit(0u32))])
            .collect()?;

        let movie_ids_series = full_item_df.column("movie_id")?.cast(&DataType::Int32)?;
        let movie_ids_ca = movie_ids_series.i32()?;
        let mut similarities = Vec::with_capacity(movie_ids_ca.len());

        for opt_id in movie_ids_ca.iter() {
            let sim = if let Some(id) = opt_id {
                let raw_id = id as usize;
                let idx = if raw_id >= movies_offset { raw_id - movies_offset } else { raw_id };
                let start = idx * embed_len;

                if start + embed_len <= movie_embeddings.len() {
                    cosine_similarity(&movie_embeddings[start..(start + embed_len)], &user_centroid)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            similarities.push(sim as f64);
        }

        full_item_df.with_column(Column::from(Series::new("sim_to_user_centroid".into(), similarities)))?;

        // Export Item-Level stats to Parquet
        let item_parquet_path = Path::new(parquet_output_dir).join(format!("hubness_item_stats_k_{}_{}.parquet", k, tag));
        let mut file = File::create(&item_parquet_path)?;
        ParquetWriter::new(&mut file).finish(&mut full_item_df)?;

        // Delegate statistical math to the helper
        let counts_series = full_item_df.column("retrieval_count")?.cast(&DataType::Float64)?;
        let counts_ca = counts_series.f64()?;
        let sims_ca = full_item_df.column("sim_to_user_centroid")?.f64()?;

        let (skewness, max_nk, _mean_sim, pearson_r) = _calculate_hubness_stats(counts_ca, sims_ca);

        agg_res.insert(format!("hubness_skewness_k_{}_{}", k, tag), skewness);
        agg_res.insert(format!("hubness_max_nk_k_{}_{}", k, tag), max_nk);
        agg_res.insert(format!("hubness_centroid_corr_k_{}_{}", k, tag), pearson_r);

        // =================================================================
        // USER-LEVEL HUB EXPOSURE
        // =================================================================

        let repeated_users: Vec<i32> = user_ids.iter().flat_map(|&u| std::iter::repeat(u).take(k)).collect();
        let retrieval_df = df!(
            "user_id" => repeated_users,
            "movie_id" => neighbors
        )?.lazy();

        let mut user_hub_df = retrieval_df
            .left_join(full_item_df.clone().lazy(), col("movie_id"), col("movie_id"))
            .group_by([col("user_id")])
            .agg([
                col("retrieval_count").cast(DataType::Float64).mean().alias("avg_slate_nk"),
                col("sim_to_user_centroid").mean().alias("avg_slate_sim_to_centroid")
            ])
            .left_join(harness.user_tiers_df.clone(), col("user_id"), col("user_id"))
            .collect()?;

        // Export User-Level stats to Parquet
        let user_parquet_path = Path::new(parquet_output_dir).join(format!("hubness_user_stats_k_{}_{}.parquet", k, tag));
        let mut file2 = File::create(&user_parquet_path)?;
        ParquetWriter::new(&mut file2).finish(&mut user_hub_df)?;

        // Stratify User-Level Metrics
        let extract_mean = |df: &DataFrame, c: &str| -> f64 {
            df.column(c).ok().and_then(|col| col.cast(&DataType::Float64).ok()).and_then(|col| col.f64().ok().map(|ca| ca.mean().unwrap_or(0.0))).unwrap_or(0.0)
        };

        agg_res.insert(format!("hub_exposure_mean_nk_k_{}_{}", k, tag), extract_mean(&user_hub_df, "avg_slate_nk"));
        agg_res.insert(format!("hub_exposure_mean_sim_k_{}_{}", k, tag), extract_mean(&user_hub_df, "avg_slate_sim_to_centroid"));

        for t in 0..=2 {
            let mask = user_hub_df.column("user_tier")?.as_materialized_series().i32()?.equal(t as i32);
            let tier_df = user_hub_df.filter(&mask)?;
            agg_res.insert(format!("hub_exposure_mean_nk_k_{}_user_tier_{}_{}", k, t, tag), extract_mean(&tier_df, "avg_slate_nk"));
        }

        // =================================================================
        // AUTOMATED CONCLUSIONS
        // =================================================================

        let max_nk_pct = (max_nk / n_users as f64) * 100.0;

        if skewness > 5.0 && pearson_r > 0.5 {
            conclusions.push(format!("SEVERE GEOMETRIC HUBNESS ({}): The retrieval distribution is highly skewed (Skewness: {:.1}), and retrieval frequency heavily correlates (r={:.2}) with proximity to the global user centroid. The space has collapsed into a few universal attractors.", tag, skewness, pearson_r));
        } else if skewness > 3.0 {
            conclusions.push(format!("MODERATE HUBNESS ({}): The retrieval distribution is moderately right-skewed (Skewness: {:.1}). A handful of items are dominating slates, but they are not strictly tied to the global user centroid (r={:.2}).", tag, skewness, pearson_r));
        } else {
            conclusions.push(format!("HEALTHY LATENT SPACE ({}): Skewness is low ({:.1}), indicating recommendation volume is naturally distributed without geometric black holes.", tag, skewness));
        }

        if max_nk_pct > 50.0 {
            conclusions.push(format!("CRITICAL HUB ALERT ({}): A single item was recommended to {:.1}% of all users. Review item embeddings for missing normalization or excessive popularity biases.", tag, max_nk_pct));
        }

        // Tier 0 = Power users, Tier 2 = Light users
        let power_exposure = *agg_res.get(&format!("hub_exposure_mean_nk_k_{}_user_tier_0_{}", k, tag)).unwrap_or(&0.0);
        let light_exposure = *agg_res.get(&format!("hub_exposure_mean_nk_k_{}_user_tier_2_{}", k, tag)).unwrap_or(&0.0);

        if power_exposure > light_exposure * 1.5 {
            conclusions.push(format!("TIERED HUB VULNERABILITY ({}): Power users (Tier 0 Avg N_k: {:.0}) are being aggressively funneled into hub items far more than Light users (Tier 2 Avg N_k: {:.0}). The dense history of Power users is pulling them into the center of the latent space.", tag, power_exposure, light_exposure));
        }

        Ok((agg_res, conclusions))
    }

    /// Calculates the Hubness statistics for a given distribution of retrieval counts
    /// and their associated geometric risk (e.g., distance to centroid or anisotropy score).
    ///
    /// Returns: (skewness, max_nk, mean_geometric_risk, pearson_correlation)
    pub fn _calculate_hubness_stats(
        counts_ca: &Float64Chunked,
        risk_ca: &Float64Chunked,
    ) -> (f64, f64, f64, f64) {

        let mean_nk = counts_ca.mean().unwrap_or(0.0);
        let std_nk = counts_ca.std(1).unwrap_or(0.0);
        let max_nk = counts_ca.max().unwrap_or(0.0);

        // 1. Compute 3rd standardized moment (Skewness) across the FULL catalog (including 0-count items)
        let mut skewness = 0.0;
        let mut n_counts = 0.0;
        if std_nk > 0.0 {
            for opt_nk in counts_ca.iter() {
                if let Some(nk) = opt_nk {
                    let z = (nk - mean_nk) / std_nk;
                    skewness += z * z * z;
                    n_counts += 1.0;
                }
            }
        }
        let final_skewness = if n_counts > 0.0 { skewness / n_counts } else { 0.0 };

        // 2. Compute mean_risk and Pearson correlation strictly over items with non-null risk
        //    (so unretrieved items in contextual_hubness do not inject dummy 0.0s)
        let valid_pairs: Vec<(f64, f64)> = counts_ca
            .iter()
            .zip(risk_ca.iter())
            .filter_map(|(opt_nk, opt_risk)| match (opt_nk, opt_risk) {
                (Some(nk), Some(risk)) => Some((nk, risk)),
                _ => None,
            })
            .collect();

        let n_valid = valid_pairs.len() as f64;
        if n_valid <= 1.0 {
            return (final_skewness, max_nk, 0.0, 0.0);
        }

        let mean_nk_valid = valid_pairs.iter().map(|(nk, _)| *nk).sum::<f64>() / n_valid;
        let mean_risk = valid_pairs.iter().map(|(_, r)| *r).sum::<f64>() / n_valid;

        let mut var_nk = 0.0;
        let mut var_risk = 0.0;
        let mut cov = 0.0;
        for &(nk, r) in &valid_pairs {
            let d_nk = nk - mean_nk_valid;
            let d_r = r - mean_risk;
            var_nk += d_nk * d_nk;
            var_risk += d_r * d_r;
            cov += d_nk * d_r;
        }
        let denom = (var_nk * var_risk).sqrt();
        let pearson_r = if denom > 1e-12 { cov / denom } else { 0.0 };

        (final_skewness, max_nk, mean_risk, pearson_r)
    }

    fn _metrics(
        k: usize,                    // either num_candidates or top_k
        catalog_size: usize,
        harness: &TestHarness,
        user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...]
        user_ids: &[i32],            // shape: (n_users)
        user_embeddings: &[f32],     // shape (n_users * embed_len)
        neighbors: &[i32],           // shape: (n_users * k)
        parquet_output_dir: &str,
        summary_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {

        _metrics0(k, catalog_size,
            &harness.pos_test_df, &harness.movie_tiers_df, &harness.user_tiers_df,
            user_gt_counts,
            user_ids, user_embeddings, neighbors, parquet_output_dir, summary_output_dir, tag)

    }


    /// Evaluates Retrieval or Ranker output and returns a tuple of (Metrics Dictionary, Conclusions List)
    /// Evaluates Retrieval or Ranker output and returns a tuple of (Metrics Dictionary, Conclusions List)
    fn _metrics0(
        k: usize,                    // either num_candidates or top_k
        catalog_size: usize,
        pos_test_df: &LazyFrame,
        movie_tiers_df: &LazyFrame,
        user_tiers_df: &LazyFrame,
        user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...]
        user_ids: &[i32],            // shape: (n_users)
        _user_embeddings: &[f32],    // shape (n_users * embed_len)
        neighbors: &[i32],           // shape: (n_users * k)
        parquet_output_dir: &str,
        _summary_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {

        let n_users = user_ids.len();

        // Build Base Metrics Data Structures (Expected baselines & IDCG Cache)
        let expected_random_recall = (k as f64) / (catalog_size as f64);
        let idcg_cache = build_idcg_cache(k);
        let expected_random_dcg_part1 = idcg_cache[k] / (catalog_size as f64);

        // Efficiently construct the flat retrieval DataFrame from vectors
        let repeated_users: Vec<i32> = user_ids.iter()
            .flat_map(|&u| std::iter::repeat(u).take(k))
            .collect();

        let ranks: Vec<i32> = (0..n_users)
            .flat_map(|_| 1..=k as i32)
            .collect();

        let retrieval_df = df!(
            "user_id" => repeated_users,
            "movie_id" => neighbors,
            "rank" => ranks,
        )?.lazy();

        // Standardize join key types to Int32 to prevent Polars schema mismatch errors
        let pos_test_clean = pos_test_df.clone().with_columns([
            col("user_id").cast(DataType::Int32),
            col("movie_id").cast(DataType::Int32),
        ]);
        let movie_tiers_clean = movie_tiers_df.clone().with_columns([
            col("movie_id").cast(DataType::Int32),
            col("movie_tier").cast(DataType::Int32),
        ]);

        // Join with truth datasets
        let joined_lf = retrieval_df
            .join_builder()
            .with(pos_test_clean)
            .left_on([col("user_id"), col("movie_id")])
            .right_on([col("user_id"), col("movie_id")])
            .how(JoinType::Left)
            .finish()
            .left_join(movie_tiers_clean, col("movie_id"), col("movie_id"));

        // Construct Aggregation Expressions
        // Note: Use `.max()` instead of `.first()` for Reciprocal Rank (`1.0 / rank`)
        // because hash joins do not preserve row order by `rank`, and `max(1/rank) == 1/min(rank)`.
        let is_hit = col("rating").is_not_null();
        let dcg_expr = lit(1.0) / (col("rank").cast(DataType::Float64) + lit(1.0)).log(lit(2.0));
        let rr_expr = lit(1.0) / col("rank").cast(DataType::Float64);

        let mut agg_exprs = vec![
            is_hit.clone().sum().alias("hits_global"),
            dcg_expr.clone().filter(is_hit.clone()).sum().alias("dcg_global"),
            rr_expr.clone().filter(is_hit.clone()).max().fill_null(lit(0.0)).alias("rr_global"),
        ];

        for t in 0..=2 {
            let tier_hit = is_hit.clone().and(col("movie_tier").eq(lit(t)));
            agg_exprs.push(tier_hit.clone().sum().alias(format!("hits_tier_{}", t)));
            agg_exprs.push(dcg_expr.clone().filter(tier_hit.clone()).sum().alias(format!("dcg_tier_{}", t)));
            agg_exprs.push(rr_expr.clone().filter(tier_hit).max().alias(format!("rr_tier_raw_{}", t)));
        }

        // Group by User ID and Apply Metrics
        let metrics_lf = joined_lf
            .group_by([col("user_id")])
            .agg(agg_exprs)
            .left_join(user_tiers_df.clone(), col("user_id"), col("user_id"))
            .inner_join(user_gt_counts.clone(), col("user_id"), col("user_id"));

        // Extract IDCG cache logic into a map closure
        let idcg_cache_clone = idcg_cache.clone();
        let apply_idcg = move |col: Column| -> PolarsResult<Column> {
            let s = col.as_materialized_series().cast(&DataType::UInt32)?;
            let ca = s.u32()?;
            let out: Float64Chunked = ca.iter().map(|opt_val| {
                let count = opt_val.unwrap_or(0) as usize;
                Some(idcg_cache_clone[count.min(k)])
            }).collect();
            Ok(Column::from(out.into_series()))
        };

        let mut col_exprs = vec![
            // Global base metrics
            col("total_positives")
                .map(
                    apply_idcg.clone(),
                    |_, f| Ok(Field::new(f.name().clone(), DataType::Float64))
                )
                .alias("idcg_global"),
            (col("hits_global").cast(DataType::Float64) / col("total_positives").cast(DataType::Float64)).alias("recall_global"),
            (col("hits_global").cast(DataType::Float64) / lit(k as f64)).alias("precision_global"),

            // Random baselines
            lit(expected_random_recall).alias("expected_random_recall"),
            (col("total_positives").cast(DataType::Float64) / lit(catalog_size as f64)).alias("expected_random_precision"),
        ];

        // Create a second vector for expressions that depend on the columns created above
        let mut ndcg_exprs = vec![
            (col("dcg_global") / col("idcg_global")).fill_nan(lit(0.0)).alias("ndcg_global"),
            (col("total_positives").cast(DataType::Float64) * lit(expected_random_dcg_part1) / col("idcg_global"))
                .alias("expected_random_ndcg"),
        ];

        // Add Movie Tier specific derivations (masked to users who actually have gt_pos_tier_{t} > 0)
        for t in 0..=2 {
            let gt_col = format!("gt_pos_tier_{}", t);
            let idcg_col = format!("idcg_tier_{}", t);

            // PHASE 1: Add to col_exprs
            col_exprs.push(
                when(col(&gt_col).gt(lit(0)))
                    .then(col(&gt_col).map(
                        apply_idcg.clone(),
                        |_, f| Ok(Field::new(f.name().clone(), DataType::Float64))
                    ))
                    .otherwise(lit(Null{}))
                    .alias(&idcg_col)
            );

            col_exprs.push(
                when(col(&gt_col).gt(lit(0)))
                    .then(col(&format!("hits_tier_{}", t)).cast(DataType::Float64) / col(&gt_col).cast(DataType::Float64))
                    .otherwise(lit(Null{}))
                    .alias(format!("recall_tier_{}", t))
            );

            col_exprs.push(
                when(col(&gt_col).gt(lit(0)))
                    .then(col(&format!("hits_tier_{}", t)).cast(DataType::Float64) / lit(k as f64))
                    .otherwise(lit(Null{}))
                    .alias(format!("precision_tier_{}", t))
            );

            col_exprs.push(
                when(col(&gt_col).gt(lit(0)))
                    .then(col(&format!("rr_tier_raw_{}", t)).fill_null(lit(0.0)))
                    .otherwise(lit(Null{}))
                    .alias(format!("rr_tier_{}", t))
            );

            // PHASE 2: Add dependent calculations to ndcg_exprs
            ndcg_exprs.push(
                when(col(&idcg_col).is_not_null())
                    .then(col(&format!("dcg_tier_{}", t)) / col(&idcg_col))
                    .otherwise(lit(Null{}))
                    .fill_nan(lit(0.0))
                    .alias(format!("ndcg_tier_{}", t))
            );
        }

        // Apply calculations in two sequential steps using `ndcg_exprs`
        let mut metrics_df = metrics_lf
            .with_columns(col_exprs)
            .with_columns(ndcg_exprs)
            .collect()?;

        // Write to Parquet
        let parquet_path = Path::new(parquet_output_dir).join(format!("stratified_metrics_top_{}_{}.parquet", k, tag));
        let mut file = File::create(&parquet_path)?;
        ParquetWriter::new(&mut file).finish(&mut metrics_df)?;

        // Extract Aggregated Results into HashMap
        let mut agg_res = HashMap::new();
        agg_res.insert("n_samples_global".to_string(), metrics_df.height() as f64);

        agg_res.insert(format!("recall_at_{}_mean", k), get_col_mean(&metrics_df, "recall_global"));
        agg_res.insert(format!("recall_at_{}_std", k), get_col_std(&metrics_df, "recall_global"));
        agg_res.insert(format!("precision_at_{}_mean", k), get_col_mean(&metrics_df, "precision_global"));
        agg_res.insert(format!("precision_at_{}_std", k), get_col_std(&metrics_df, "precision_global"));
        agg_res.insert(format!("ndcg_at_{}_mean", k), get_col_mean(&metrics_df, "ndcg_global"));
        agg_res.insert(format!("ndcg_at_{}_std", k), get_col_std(&metrics_df, "ndcg_global"));
        agg_res.insert(format!("mrr_at_{}_mean", k), get_col_mean(&metrics_df, "rr_global"));
        agg_res.insert(format!("mrr_at_{}_std", k), get_col_std(&metrics_df, "rr_global"));

        agg_res.insert(format!("recall_at_{}_mean_random", k), get_col_mean(&metrics_df, "expected_random_recall"));
        agg_res.insert(format!("precision_at_{}_mean_random", k), get_col_mean(&metrics_df, "expected_random_precision"));
        agg_res.insert(format!("ndcg_at_{}_mean_random", k), get_col_mean(&metrics_df, "expected_random_ndcg"));

        // User Tier & Movie Tier Extractions
        for t in 0..=2 {
            let mask = metrics_df
                .column("user_tier")?
                .as_materialized_series()
                .i32()?
                .equal(t as i32);

            let filtered_df = metrics_df.filter(&mask)?;

            agg_res.insert(format!("n_samples_user_tier_{}", t), filtered_df.height() as f64);
            agg_res.insert(format!("recall_at_{}_mean_user_tier_{}", k, t), get_col_mean(&filtered_df, "recall_global"));
            agg_res.insert(format!("recall_at_{}_std_user_tier_{}", k, t), get_col_std(&filtered_df, "recall_global"));
            agg_res.insert(format!("precision_at_{}_mean_user_tier_{}", k, t), get_col_mean(&filtered_df, "precision_global"));
            agg_res.insert(format!("precision_at_{}_std_user_tier_{}", k, t), get_col_std(&filtered_df, "precision_global"));
            agg_res.insert(format!("ndcg_at_{}_mean_user_tier_{}", k, t), get_col_mean(&filtered_df, "ndcg_global"));
            agg_res.insert(format!("ndcg_at_{}_std_user_tier_{}", k, t), get_col_std(&filtered_df, "ndcg_global"));
            agg_res.insert(format!("mrr_at_{}_mean_user_tier_{}", k, t), get_col_mean(&filtered_df, "rr_global"));
            agg_res.insert(format!("mrr_at_{}_std_user_tier_{}", k, t), get_col_std(&filtered_df, "rr_global"));

            // Movie Tier Extractions (automatically ignores Null rows where gt_pos_tier_{t} == 0)
            let recall_col = format!("recall_tier_{}", t);
            let n_movie_samples = metrics_df.column(&recall_col)?.is_not_null().sum().unwrap_or(0);

            agg_res.insert(format!("n_samples_movie_tier_{}", t), n_movie_samples as f64);
            agg_res.insert(format!("recall_at_{}_mean_movie_tier_{}", k, t), get_col_mean(&metrics_df, &recall_col));
            agg_res.insert(format!("precision_at_{}_mean_movie_tier_{}", k, t), get_col_mean(&metrics_df, &format!("precision_tier_{}", t)));
            agg_res.insert(format!("ndcg_at_{}_mean_movie_tier_{}", k, t), get_col_mean(&metrics_df, &format!("ndcg_tier_{}", t)));
            agg_res.insert(format!("mrr_at_{}_mean_movie_tier_{}", k, t), get_col_mean(&metrics_df, &format!("rr_tier_{}", t)));
        }

        //  Generate Automated Conclusions
        let mut conclusions = Vec::new();

        let global_recall = *agg_res.get(&format!("recall_at_{}_mean", k)).unwrap_or(&0.0);
        let random_recall = *agg_res.get(&format!("recall_at_{}_mean_random", k)).unwrap_or(&0.0);
        let global_precision = *agg_res.get(&format!("precision_at_{}_mean", k)).unwrap_or(&0.0);
        let random_precision = *agg_res.get(&format!("precision_at_{}_mean_random", k)).unwrap_or(&0.0);

        if global_recall > random_recall * 3.0 {
            conclusions.push(format!("STRONG RECALL LIFT: Global Recall@{} ({:.1}%) heavily outperforms the random expected baseline ({:.1}%). The candidate generator is extracting meaningful semantic signal.", k, global_recall * 100.0, random_recall * 100.0));
        } else if global_recall > random_recall {
            conclusions.push(format!("MODERATE RECALL LIFT: Global Recall@{} ({:.1}%) beats the random baseline ({:.1}%), but there is room for improvement.", k, global_recall * 100.0, random_recall * 100.0));
        } else {
            conclusions.push("RECALL FAILURE: The model fails to outperform a random candidate generator on Recall.".to_string());
        }

        if global_precision > random_precision * 3.0 {
            conclusions.push(format!("STRONG PRECISION LIFT: Global Precision@{} ({:.2}%) heavily outperforms the random expected precision ({:.2}%), indicating high density of relevant items in the retrieved slates.", k, global_precision * 100.0, random_precision * 100.0));
        } else if global_precision > random_precision {
            conclusions.push(format!("MODERATE PRECISION LIFT: Global Precision@{} ({:.2}%) is better than random ({:.2}%), suggesting basic relevance filtering is functioning.", k, global_precision * 100.0, random_precision * 100.0));
        } else {
            conclusions.push("PRECISION FAILURE: The model retrieves slates with the same or worse precision than a completely random draw.".to_string());
        }

        let global_mrr = *agg_res.get(&format!("mrr_at_{}_mean", k)).unwrap_or(&0.0);
        if global_mrr > 0.50 {
            conclusions.push(format!("EXCELLENT SURFACING SPEED: Global MRR@{} is {:.2}, meaning the first relevant recommendation appears, on average, within the top 2 slots.", k, global_mrr));
        } else if global_mrr > 0.20 {
            conclusions.push(format!("MODERATE SURFACING SPEED: Global MRR@{} is {:.2}, meaning the first relevant recommendation typically appears within the top 5 slots.", k, global_mrr));
        } else {
            conclusions.push(format!("SLOW SURFACING WARNING: Global MRR@{} is heavily degraded ({:.2}), indicating users must scroll significantly to find the first relevant item.", k, global_mrr));
        }

        // User Tiers: Tier 0 = Power Users (top 20%), Tier 2 = Light Users (bottom 20%)
        let recall_power = *agg_res.get(&format!("recall_at_{}_mean_user_tier_0", k)).unwrap_or(&0.0);
        let recall_light = *agg_res.get(&format!("recall_at_{}_mean_user_tier_2", k)).unwrap_or(&0.0);

        if recall_power > recall_light * 1.5 {
            conclusions.push(format!("EXCELLENT HISTORY UTILIZATION: Power users (Tier 0 Recall: {:.1}%) perform massively better than light users (Tier 2 Recall: {:.1}%). The model successfully translates rich interaction histories into highly relevant slates.", recall_power * 100.0, recall_light * 100.0));
        } else if recall_power > recall_light {
            conclusions.push(format!("MODERATE HISTORY UTILIZATION: Power users (Tier 0: {:.1}%) see better recall than light users (Tier 2: {:.1}%).", recall_power * 100.0, recall_light * 100.0));
        } else {
            conclusions.push(format!("STRONG LIGHT-USER GENERALIZATION: Light/niche users (Tier 2 Recall: {:.1}%) achieve recall comparable to or exceeding power users (Tier 0 Recall: {:.1}%).", recall_light * 100.0, recall_power * 100.0));
        }

        let ndcg_power = *agg_res.get(&format!("ndcg_at_{}_mean_user_tier_0", k)).unwrap_or(&0.0);
        let ndcg_light = *agg_res.get(&format!("ndcg_at_{}_mean_user_tier_2", k)).unwrap_or(&0.0);

        if ndcg_power > ndcg_light {
            conclusions.push(format!("HEALTHY RANKING STRATIFICATION: NDCG is higher for power users (Tier 0: {:.2}% vs Tier 2: {:.2}%). The model not only retrieves the right items for power users but places them higher in the slate.", ndcg_power * 100.0, ndcg_light * 100.0));
        } else {
            conclusions.push(format!("STRONG NICHE USER RANKING: Light/niche users achieve strong top-K ranking quality (Tier 2 NDCG: {:.2}% vs Tier 0: {:.2}%).", ndcg_light * 100.0, ndcg_power * 100.0));
        }

        // Movie Tiers: Tier 0 = Head/Popular (top 20%), Tier 2 = Tail/Niche (bottom 20%)
        let ndcg_head = *agg_res.get(&format!("ndcg_at_{}_mean_movie_tier_0", k)).unwrap_or(&0.0);
        let ndcg_tail = *agg_res.get(&format!("ndcg_at_{}_mean_movie_tier_2", k)).unwrap_or(&0.0);

        if ndcg_head > ndcg_tail * 1.5 {
            conclusions.push(format!("POPULARITY BIAS DETECTED: Ranking quality heavily favors head/popular movies (Tier 0 NDCG: {:.2}%) compared to long-tail/niche items (Tier 2 NDCG: {:.2}%).", ndcg_head * 100.0, ndcg_tail * 100.0));
        } else if ndcg_tail > ndcg_head {
            conclusions.push(format!("STRONG LONG-TAIL PERFORMANCE: Model successfully surfaces and ranks niche/long-tail items (Tier 2 NDCG: {:.2}%) effectively relative to popular head items (Tier 0 NDCG: {:.2}%).", ndcg_tail * 100.0, ndcg_head * 100.0));
        } else {
            conclusions.push(format!("BALANCED ITEM TIER RANKING: NDCG performance remains consistent across movie tiers (Head Tier 0: {:.2}%, Tail Tier 2: {:.2}%), indicating stable retrieval across popular and long-tail catalogs.", ndcg_head * 100.0, ndcg_tail * 100.0));
        }

        Ok((agg_res, conclusions))
    }

    /// Helper function to pre-calculate IDCG values up to K.
    /// idcg_cache[n] = sum(1/log2(r+1)) for r=1..=n
    fn build_idcg_cache(top_k: usize) -> Vec<f64> {
        let mut cache = vec![0.0; top_k + 1];
        let mut current_idcg = 0.0;
        for r in 1..=top_k {
            current_idcg += 1.0 / ((r as f64) + 1.0).log2();
            cache[r] = current_idcg;
        }
        cache
    }

    /// Helper to safely extract a scalar mean from a DataFrame column
    fn get_col_mean(df: &DataFrame, col_name: &str) -> f64 {
        df.column(col_name)
            .and_then(|c| c.cast(&DataType::Float64))
            .and_then(|c| c.f64().map(|ca| ca.mean().unwrap_or(0.0)))
            .unwrap_or(0.0)
    }

    /// Helper to safely extract a scalar std deviation from a DataFrame column
    fn get_col_std(df: &DataFrame, col_name: &str) -> f64 {
        df.column(col_name)
            .and_then(|c| c.cast(&DataType::Float64))
            // std(1) calculates sample standard deviation (N-1) natively
            .and_then(|c| c.f64().map(|ca| ca.std(1).unwrap_or(0.0)))
            .unwrap_or(0.0)
    }

    /// Calculates the cosine similarity between two dense vectors.
    /// Returns a value between -1.0 and 1.0 (or 0.0 if either vector is completely empty/zero).
    #[inline]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vectors must be of the same length");

        let mut dot_product = 0.0;
        let mut norm_a_sq = 0.0;
        let mut norm_b_sq = 0.0;

        // Using .iter().zip() allows the Rust compiler (LLVM) to auto-vectorize
        // this loop into SIMD instructions for massive performance gains.
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot_product += x * y;
            norm_a_sq += x * x;
            norm_b_sq += y * y;
        }

        if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
            0.0
        } else {
            dot_product / (norm_a_sq.sqrt() * norm_b_sq.sqrt())
        }
    }

    async fn cross_metric_analysis(
        base_metrics: &serde_json::Value,
        coverage: &serde_json::Value,
        _pop_bias: &serde_json::Value,
        diversity: &serde_json::Value,
        emb_hubness: &serde_json::Value,
        ctx_hubness: &serde_json::Value,
        rank_agreement: &serde_json::Value,
        cold_start_movies: &serde_json::Value,
        cold_start_users: &serde_json::Value,
        num_candidates: usize,
        top_k: usize,
        summary_output_dir: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {

        let tag1 = format!("retrieval_ann_{}", num_candidates);
        let tag2 = format!("retrieval_ann_{}", top_k);
        let tag3 = format!("ranker_{}", top_k);

        let tag_cs_mov_ret_100 = format!("retrieval_ann_{}_cold_start_movies", num_candidates);
        let tag_cs_mov_ret_20 = format!("retrieval_ann_{}_cold_start_movies", top_k);
        let tag_cs_mov_rnk_20 = format!("ranker_{}_cold_start_movies", top_k);

        let tag_cs_usr_ret_100 = format!("retrieval_ann_{}_cold_start_users", num_candidates);
        let tag_cs_usr_ret_20 = format!("retrieval_ann_{}_cold_start_users", top_k);
        let tag_cs_usr_rnk_20 = format!("ranker_{}_cold_start_users", top_k);

        // Helper to safely extract a f64 from the nested JSON tree
        let extract = |json: &serde_json::Value, tag: &str, metric: &str| -> f64 {
            json.get(tag)
                .and_then(|t| t.get("metrics"))
                .and_then(|m| m.get(metric))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        };

        let mut conclusions = Vec::new();

        // =================================================================
        // 1. FUNNEL ANALYSIS (Retriever @ 100 vs Retriever @ 20)
        // =================================================================
        let recall_100 = extract(base_metrics, &tag1, &format!("recall_at_{}_mean", num_candidates));
        let recall_20 = extract(base_metrics, &tag2, &format!("recall_at_{}_mean", top_k));

        if recall_20 > 0.0 && recall_100 > 0.0 {
            let retention = (recall_20 / recall_100) * 100.0;
            conclusions.push(format!(
                "FUNNEL RETENTION: Shrinking the candidate pool from {} to {} retains {:.1}% of the recall (Recall drops from {:.2}% to {:.2}%).",
                num_candidates, top_k, retention, recall_100 * 100.0, recall_20 * 100.0
            ));
        }

        // =================================================================
        // 2. RANKER VS RETRIEVER LIFT (Ranker @ 20 vs Retriever @ 20)
        // =================================================================
        let mrr_ret_20 = extract(base_metrics, &tag2, &format!("mrr_at_{}_mean", top_k));
        let mrr_rnk_20 = extract(base_metrics, &tag3, &format!("mrr_at_{}_mean", top_k));

        if mrr_ret_20 > 0.0 {
            let mrr_lift = ((mrr_rnk_20 - mrr_ret_20) / mrr_ret_20) * 100.0;

            if mrr_lift > 5.0 {
                conclusions.push(format!(
                    "SURFACING SPEED LIFT: The Ranker improved MRR@{} by {:.1}% (from {:.2} to {:.2}). The Cross-Encoder successfully pushes the first relevant item much closer to rank 1.",
                    top_k, mrr_lift, mrr_ret_20, mrr_rnk_20
                ));
            } else if mrr_lift < -2.0 {
                conclusions.push(format!(
                    "SURFACING SPEED DEGRADATION: The Ranker degraded MRR@{} by {:.1}% (from {:.2} to {:.2}). It forces the user to scroll further to find their first hit compared to raw retrieval.",
                    top_k, mrr_lift.abs(), mrr_ret_20, mrr_rnk_20
                ));
            } else {
                conclusions.push(format!(
                    "SURFACING SPEED NEUTRAL: The Ranker's MRR@{} ({:.2}) is practically identical to the Retriever's ({:.2}).",
                    top_k, mrr_rnk_20, mrr_ret_20
                ));
            }
        }

        let ndcg_ret_20 = extract(base_metrics, &tag2, &format!("ndcg_at_{}_mean", top_k));
        let ndcg_rnk_20 = extract(base_metrics, &tag3, &format!("ndcg_at_{}_mean", top_k));

        let mut ndcg_lift = 0.0;
        if ndcg_ret_20 > 0.0 {
            ndcg_lift = ((ndcg_rnk_20 - ndcg_ret_20) / ndcg_ret_20) * 100.0;
            if ndcg_lift > 5.0 {
                conclusions.push(format!("RANKER LIFT: The Ranker improved NDCG@{} by {:.1}% over pure retrieval. The Cross-Encoder is successfully refining the candidate slates.", top_k, ndcg_lift));
            } else if ndcg_lift < -2.0 {
                conclusions.push(format!("RANKER DEGRADATION: The Ranker degraded NDCG@{} by {:.1}%. The Cross-Encoder is miscalibrated and is burying the true positives retrieved by the ANN.", top_k, ndcg_lift.abs()));
            } else {
                conclusions.push(format!("RANKER NEUTRAL: The Ranker provided negligible NDCG lift ({:.1}%) over pure retrieval.", ndcg_lift));
            }
        }

        // =================================================================
        // 3. RANKER CHURN CORRELATION
        // =================================================================
        let overlap = extract(rank_agreement, "metrics", &format!("mean_overlap_at_{}", top_k));

        if overlap > 0.0 && ndcg_lift < 0.0 {
            conclusions.push(format!(
                "CHURN VS LIFT DEGRADATION: The Ranker is aggressively replacing items (only retaining {:.1}% of the Retriever's slate), but doing so hurts overall performance (NDCG lift: {:.1}%). The ranking model is introducing noise rather than signal.",
                overlap * 100.0, ndcg_lift
            ));
        } else if overlap < 0.40 && ndcg_lift > 5.0 {
            conclusions.push(format!(
                "SUCCESSFUL FILTERING: The Ranker aggressively filtered out {:.1}% of the Retriever's items, resulting in a positive NDCG lift of {:.1}%. It successfully separated the signal from the ANN's noise.",
                (1.0 - overlap) * 100.0, ndcg_lift
            ));
        }

        let cov_ret_20 = extract(coverage, &tag2, &format!("coverage_at_{}_full_{}", top_k, tag2));
        let cov_rnk_20 = extract(coverage, &tag3, &format!("coverage_at_{}_full_{}", top_k, tag3));

        if cov_ret_20 > 0.0 {
            let cov_cost = ((cov_rnk_20 - cov_ret_20) / cov_ret_20) * 100.0;
            if cov_cost < -15.0 {
                conclusions.push(format!("RANKER COVERAGE COLLAPSE: To achieve its ranking, the Ranker sacrificed {:.1}% of the Retriever's catalog coverage. It is behaving much more conservatively than the ANN.", cov_cost.abs()));
            }
        }

        // =================================================================
        // 4. ROOT CAUSE CORRELATIONS (Gini vs. Hubness)
        // =================================================================
        let gini_rnk_20 = extract(coverage, &tag3, &format!("gini_at_{}_full_{}", top_k, tag3));
        let div_rnk_20 = extract(diversity, &tag3, &format!("interlist_diversity_mean_k_{}_{}", top_k, tag3));

        // Contextual hubness JSON is not wrapped in tag blocks, so we extract directly from "metrics"
        let ctx_hub_corr = ctx_hubness.get("metrics")
            .and_then(|m| m.get(format!("ctx_hubness_centroid_corr_ranker_contextual_k_{}", top_k)))
            .and_then(|v| v.as_f64()).unwrap_or(0.0);

        let emb_hub_corr = extract(emb_hubness, &tag1, &format!("hubness_centroid_corr_k_{}_{}", num_candidates, tag1));

        if gini_rnk_20 > 0.80 {
            if ctx_hub_corr > 0.40 {
                conclusions.push(format!("RANKER DIAGNOSIS (STRUCTURAL BIAS): The Ranker's severe popularity bias (Gini: {:.2}) is structurally driven by contextual hubness (r={:.2}). The GATv2 layers are oversmoothing candidate embeddings.", gini_rnk_20, ctx_hub_corr));
            } else {
                conclusions.push(format!("RANKER DIAGNOSIS (ORGANIC BIAS): The Ranker exhibits high popularity bias (Gini: {:.2}), but it is NOT driven by geometric hubness (r={:.2}). The model is intentionally favoring popular items based on interaction data.", gini_rnk_20, ctx_hub_corr));
            }
        }

        if div_rnk_20 < 0.50 && emb_hub_corr > 0.40 {
            conclusions.push(format!("RETRIEVER DIAGNOSIS (POISONED POOL): The Ranker's poor inter-list diversity ({:.1}%) is likely inherited from the Retriever. The Retriever's latent space is dominated by universal geometric hubs (r={:.2}), poisoning the candidate pool before ranking even occurs.", div_rnk_20 * 100.0, emb_hub_corr));
        }

        // =================================================================
        // 5. ITEM COLD-START ANALYSIS (OOV=0 Phantom Movies vs. Warm Catalog)
        // =================================================================
        let cs_mov_recall_100 = extract(cold_start_movies, &tag_cs_mov_ret_100, &format!("recall_at_{}_mean", num_candidates));
        let cs_mov_ndcg_ret_20 = extract(cold_start_movies, &tag_cs_mov_ret_20, &format!("ndcg_at_{}_mean", top_k));
        let cs_mov_ndcg_rnk_20 = extract(cold_start_movies, &tag_cs_mov_rnk_20, &format!("ndcg_at_{}_mean", top_k));
        let cs_mov_mrr_ret_20 = extract(cold_start_movies, &tag_cs_mov_ret_20, &format!("mrr_at_{}_mean", top_k));
        let cs_mov_mrr_rnk_20 = extract(cold_start_movies, &tag_cs_mov_rnk_20, &format!("mrr_at_{}_mean", top_k));

        if recall_100 > 0.0 && cs_mov_recall_100 > 0.0 {
            let mov_ret_retention = (cs_mov_recall_100 / recall_100) * 100.0;
            if mov_ret_retention >= 85.0 {
                conclusions.push(format!(
                    "ITEM COLD-START RESILIENCE (RETRIEVER): When 501 test movies are replaced with unmemorized OOV=0 phantom items, Recall@{} retains {:.1}% of warm performance ({:.2}% vs {:.2}%). The Candidate Tower's content features (genres/metadata) generalize strongly into the embedding space.",
                    num_candidates, mov_ret_retention, cs_mov_recall_100 * 100.0, recall_100 * 100.0
                ));
            } else if mov_ret_retention >= 60.0 {
                conclusions.push(format!(
                    "MODERATE ITEM COLD-START DROP (RETRIEVER): Replacing 501 test movies with OOV=0 phantom items retains {:.1}% of warm Recall@{} ({:.2}% vs {:.2}%). The Candidate Tower relies partially on memorized movie ID embeddings.",
                    mov_ret_retention, num_candidates, cs_mov_recall_100 * 100.0, recall_100 * 100.0
                ));
            } else {
                conclusions.push(format!(
                    "SEVERE ITEM ID MEMORIZATION (RETRIEVER): Recall@{} collapses to {:.1}% of warm baseline ({:.2}% vs {:.2}%) when 501 test movies use OOV=0 IDs. The Candidate Tower is over-reliant on memorized item IDs and under-utilizing movie side-features.",
                    num_candidates, mov_ret_retention, cs_mov_recall_100 * 100.0, recall_100 * 100.0
                ));
            }
        }

        if cs_mov_ndcg_ret_20 > 0.0 {
            let cs_mov_rnk_lift = ((cs_mov_ndcg_rnk_20 - cs_mov_ndcg_ret_20) / cs_mov_ndcg_ret_20) * 100.0;
            let cs_mov_mrr_lift = if cs_mov_mrr_ret_20 > 0.0 {
                ((cs_mov_mrr_rnk_20 - cs_mov_mrr_ret_20) / cs_mov_mrr_ret_20) * 100.0
            } else {
                0.0
            };

            if cs_mov_rnk_lift > 5.0 {
                conclusions.push(format!(
                    "ITEM COLD-START RANKER LIFT: Even with OOV=0 phantom movies in the candidate pool, the Ranker improves NDCG@{} by {:.1}% (and MRR@{} by {:.1}%) over pure retrieval. The GATv2 graph attention effectively scores cold-start item embeddings against user history.",
                    top_k, cs_mov_rnk_lift, top_k, cs_mov_mrr_lift
                ));
            } else if cs_mov_rnk_lift < -2.0 {
                conclusions.push(format!(
                    "ITEM COLD-START RANKER PENALTY: The Ranker degrades NDCG@{} by {:.1}% when scoring cold-start movies ({:.2}% vs Retriever {:.2}%). The Cross-Encoder is biased toward memorized warm movie embeddings and buries new OOV items.",
                    top_k, cs_mov_rnk_lift.abs(), cs_mov_ndcg_rnk_20 * 100.0, cs_mov_ndcg_ret_20 * 100.0
                ));
            }
        }

        // Compare Blockbuster (Tier 0 = Head) vs Niche (Tier 2 = Tail) drop under Item Cold-Start
        let warm_mov_head_ndcg = extract(base_metrics, &tag3, &format!("ndcg_at_{}_mean_movie_tier_0", top_k));
        let cs_mov_head_ndcg = extract(cold_start_movies, &tag_cs_mov_rnk_20, &format!("ndcg_at_{}_mean_movie_tier_0", top_k));
        let warm_mov_tail_ndcg = extract(base_metrics, &tag3, &format!("ndcg_at_{}_mean_movie_tier_2", top_k));
        let cs_mov_tail_ndcg = extract(cold_start_movies, &tag_cs_mov_rnk_20, &format!("ndcg_at_{}_mean_movie_tier_2", top_k));

        if warm_mov_head_ndcg > 0.0 && warm_mov_tail_ndcg > 0.0 {
            let head_drop_pct = ((warm_mov_head_ndcg - cs_mov_head_ndcg) / warm_mov_head_ndcg) * 100.0;
            let tail_drop_pct = ((warm_mov_tail_ndcg - cs_mov_tail_ndcg) / warm_mov_tail_ndcg) * 100.0;

            if head_drop_pct > tail_drop_pct + 10.0 {
                conclusions.push(format!(
                    "HEAD ITEM ID DEPENDENCY: Under cold-start movie simulation, Tier 0 (head) NDCG@{} drops by {:.1}% ({:.2}% -> {:.2}%), compared to a {:.1}% change for Tier 2 (long-tail) items. Popular items rely heavily on memorized collaborative ID weights rather than metadata.",
                    top_k, head_drop_pct, warm_mov_head_ndcg * 100.0, cs_mov_head_ndcg * 100.0, tail_drop_pct
                ));
            }
        }

        // =================================================================
        // 6. USER COLD-START ANALYSIS (OOV=0 Users & Empty History vs. Warm Users)
        // =================================================================
        let cs_usr_recall_100 = extract(cold_start_users, &tag_cs_usr_ret_100, &format!("recall_at_{}_mean", num_candidates));
        let cs_usr_rand_recall_100 = extract(cold_start_users, &tag_cs_usr_ret_100, &format!("recall_at_{}_mean_random", num_candidates));
        let cs_usr_ndcg_ret_20 = extract(cold_start_users, &tag_cs_usr_ret_20, &format!("ndcg_at_{}_mean", top_k));
        let cs_usr_ndcg_rnk_20 = extract(cold_start_users, &tag_cs_usr_rnk_20, &format!("ndcg_at_{}_mean", top_k));
        let cs_usr_mrr_ret_20 = extract(cold_start_users, &tag_cs_usr_ret_20, &format!("mrr_at_{}_mean", top_k));
        let cs_usr_mrr_rnk_20 = extract(cold_start_users, &tag_cs_usr_rnk_20, &format!("mrr_at_{}_mean", top_k));

        if recall_100 > 0.0 && cs_usr_recall_100 > 0.0 {
            let usr_ret_retention = (cs_usr_recall_100 / recall_100) * 100.0;
            let rand_mult = if cs_usr_rand_recall_100 > 0.0 {
                cs_usr_recall_100 / cs_usr_rand_recall_100
            } else {
                0.0
            };

            if usr_ret_retention >= 65.0 {
                conclusions.push(format!(
                    "STRONG DEMOGRAPHIC COLD-START (USER RETRIEVER): With user_id=0 (OOV), demographic features alone (age, gender, occupation) retain {:.1}% of warm Recall@{} ({:.2}% vs {:.2}%, {:.1}x random baseline).",
                    usr_ret_retention, num_candidates, cs_usr_recall_100 * 100.0, recall_100 * 100.0, rand_mult
                ));
            } else if cs_usr_recall_100 > cs_usr_rand_recall_100 * 2.0 {
                conclusions.push(format!(
                    "MODERATE DEMOGRAPHIC COLD-START (USER RETRIEVER): Cold-start users retain {:.1}% of warm Recall@{} ({:.2}% vs {:.2}%), beating the random baseline by {:.1}x, though collaborative ID/history carries the majority of personalization signal.",
                    usr_ret_retention, num_candidates, cs_usr_recall_100 * 100.0, recall_100 * 100.0, rand_mult
                ));
            } else {
                conclusions.push(format!(
                    "WEAK USER COLD-START SIGNAL: Cold-start User Recall@{} is {:.2}% (only {:.1}% of warm Recall and {:.1}x random). User demographic features alone provide minimal geometric separation in the Query Tower.",
                    num_candidates, cs_usr_recall_100 * 100.0, usr_ret_retention, rand_mult
                ));
            }
        }

        if cs_usr_ndcg_ret_20 > 0.0 {
            let cs_usr_rnk_lift = ((cs_usr_ndcg_rnk_20 - cs_usr_ndcg_ret_20) / cs_usr_ndcg_ret_20) * 100.0;
            let cs_usr_mrr_lift = if cs_usr_mrr_ret_20 > 0.0 {
                ((cs_usr_mrr_rnk_20 - cs_usr_mrr_ret_20) / cs_usr_mrr_ret_20) * 100.0
            } else {
                0.0
            };

            if cs_usr_rnk_lift > 3.0 {
                conclusions.push(format!(
                    "ZERO-HISTORY GRAPH RANKER LIFT: Even when cold-start users have an empty watch history in the GATv2 subgraph, the Ranker improves NDCG@{} by {:.1}% (and MRR@{} by {:.1}%) over pure retrieval.",
                    top_k, cs_usr_rnk_lift, top_k, cs_usr_mrr_lift
                ));
            } else if cs_usr_rnk_lift < -3.0 {
                conclusions.push(format!(
                    "ZERO-HISTORY GRAPH DEGRADATION: When cold-start users have an empty watch history, the GATv2 Ranker degrades NDCG@{} by {:.1}% ({:.2}% vs Retriever {:.2}%) and MRR@{} by {:.1}%. Without historical graph edges, message-passing injects noise; consider bypassing the GNN ranker or falling back to ANN scores when user history is empty.",
                    top_k, cs_usr_rnk_lift.abs(), cs_usr_ndcg_rnk_20 * 100.0, cs_usr_ndcg_ret_20 * 100.0, top_k, cs_usr_mrr_lift.abs()
                ));
            } else {
                conclusions.push(format!(
                    "ZERO-HISTORY GRAPH STABILITY: For cold-start users with empty watch histories, the GATv2 Ranker preserves the Retriever's slate quality (NDCG@{} delta: {:.1}%).",
                    top_k, cs_usr_rnk_lift
                ));
            }
        }

        // Power User (Tier 0) History Ablation
        let warm_usr_power_recall_20 = extract(base_metrics, &tag3, &format!("recall_at_{}_mean_user_tier_0", top_k));
        let cs_usr_power_recall_20 = extract(cold_start_users, &tag_cs_usr_rnk_20, &format!("recall_at_{}_mean_user_tier_0", top_k));
        if warm_usr_power_recall_20 > 0.0 && cs_usr_power_recall_20 > 0.0 {
            let history_value_lift = ((warm_usr_power_recall_20 - cs_usr_power_recall_20) / cs_usr_power_recall_20) * 100.0;
            conclusions.push(format!(
                "POWER USER HISTORY ABLATION: Stripping Power Users (Tier 0) of their interaction history and user ID reduces Ranker Recall@{} from {:.2}% to {:.2}% (confirming that learned history + ID provides a {:+.1}% relative lift over raw demographics).",
                top_k, warm_usr_power_recall_20 * 100.0, cs_usr_power_recall_20 * 100.0, history_value_lift
            ));
        }

        // =================================================================
        // SYSTEMIC COLD-START ASYMMETRY (Users vs. Movies)
        // =================================================================
        if ndcg_rnk_20 > 0.0 && cs_mov_ndcg_rnk_20 > 0.0 && cs_usr_ndcg_rnk_20 > 0.0 {
            let mov_rnk_retention = (cs_mov_ndcg_rnk_20 / ndcg_rnk_20) * 100.0;
            let usr_rnk_retention = (cs_usr_ndcg_rnk_20 / ndcg_rnk_20) * 100.0;

            if usr_rnk_retention + 15.0 < mov_rnk_retention {
                conclusions.push(format!(
                    "COLD-START BOTTLENECK (USER SIDE): End-to-end Ranker NDCG@{} retention is significantly lower for Cold-Start Users ({:.1}% of warm) than Cold-Start Movies ({:.1}% of warm). Onboarding new users (e.g., via interactive preference elicitation or richer user features) is a higher-leverage priority than item metadata.",
                    top_k, usr_rnk_retention, mov_rnk_retention
                ));
            } else if mov_rnk_retention + 15.0 < usr_rnk_retention {
                conclusions.push(format!(
                    "COLD-START BOTTLENECK (ITEM SIDE): End-to-end Ranker NDCG@{} retention is significantly lower for Cold-Start Movies ({:.1}% of warm) than Cold-Start Users ({:.1}% of warm). Enriching item content features (e.g., plot/synopsis embeddings, cast, tags) is the highest-leverage priority.",
                    top_k, mov_rnk_retention, usr_rnk_retention
                ));
            } else {
                conclusions.push(format!(
                    "BALANCED COLD-START RESILIENCE: User Cold-Start ({:.1}% NDCG@{} retention) and Item Cold-Start ({:.1}% NDCG@{} retention) exhibit comparable end-to-end resilience.",
                    usr_rnk_retention, top_k, mov_rnk_retention, top_k
                ));
            }
        }

        // =================================================================
        // EXPORT
        // =================================================================
        let payload = serde_json::json!({
            "cross_metric_conclusions": conclusions
        });

        let out_path = std::path::Path::new(summary_output_dir).join("cross_metric_analysis.json");
        let file = std::fs::File::create(&out_path)?;
        serde_json::to_writer_pretty(file, &payload)?;
        println!("Wrote cross-metric analysis to {:?}", out_path);

        Ok(payload)
    }

    fn write_master_summary_and_conclusions(
        base_metrics: &serde_json::Value,
        coverage: &serde_json::Value,
        pop_bias: &serde_json::Value,
        diversity: &serde_json::Value,
        emb_hubness: &serde_json::Value,
        ctx_hubness: &serde_json::Value,
        rank_agreement: &serde_json::Value,
        cold_start_movies: &serde_json::Value,
        cold_start_users: &serde_json::Value,
        embedding_diagnostics: &serde_json::Value,
        cross_metrics_concl: &serde_json::Value,
        num_candidates: usize,
        top_k: usize,
        summary_output_dir: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        let tag_ret_100 = format!("retrieval_ann_{}", num_candidates);
        let tag_ret_20 = format!("retrieval_ann_{}", top_k);
        let tag_rnk_20 = format!("ranker_{}", top_k);

        let extract_metric = |json: &serde_json::Value, tag: &str, key: &str| -> f64 {
            json.get(tag)
                .and_then(|t| t.get("metrics"))
                .and_then(|m| m.get(key))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        };

        let extract_conclusions = |json: &serde_json::Value, tag: &str| -> Vec<String> {
            json.get(tag)
                .and_then(|t| t.get("automated_conclusions"))
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default()
        };

        let extract_direct_conclusions = |json: &serde_json::Value, key: &str| -> Vec<String> {
            json.get(key)
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default()
        };

        // 1. Compute Executive Two-Stage Funnel & GATv2 Geometry Verdicts
        let mut executive_summary: Vec<String> = Vec::new();

        let rec_100 = extract_metric(base_metrics, &tag_ret_100, &format!("recall_at_{}_mean", num_candidates));
        let rec_ret_20 = extract_metric(base_metrics, &tag_ret_20, &format!("recall_at_{}_mean", top_k));
        let rec_rnk_20 = extract_metric(base_metrics, &tag_rnk_20, &format!("recall_at_{}_mean", top_k));

        let ndcg_ret_20 = extract_metric(base_metrics, &tag_ret_20, &format!("ndcg_at_{}_mean", top_k));
        let ndcg_rnk_20 = extract_metric(base_metrics, &tag_rnk_20, &format!("ndcg_at_{}_mean", top_k));

        let mrr_ret_20 = extract_metric(base_metrics, &tag_ret_20, &format!("mrr_at_{}_mean", top_k));
        let mrr_rnk_20 = extract_metric(base_metrics, &tag_rnk_20, &format!("mrr_at_{}_mean", top_k));

        if rec_100 > 0.0 && rec_ret_20 > 0.0 {
            let ret_capture_pct = (rec_ret_20 / rec_100) * 100.0;
            let rnk_capture_pct = (rec_rnk_20 / rec_100) * 100.0;
            let recall_lift_pct = ((rec_rnk_20 - rec_ret_20) / rec_ret_20) * 100.0;
            let ndcg_lift_pct = if ndcg_ret_20 > 0.0 {
                ((ndcg_rnk_20 - ndcg_ret_20) / ndcg_ret_20) * 100.0
            } else {
                0.0
            };
            let mrr_lift_pct = if mrr_ret_20 > 0.0 {
                ((mrr_rnk_20 - mrr_ret_20) / mrr_ret_20) * 100.0
            } else {
                0.0
            };

            executive_summary.push(format!(
                "TWO-STAGE FUNNEL SUMMARY (@{} -> @{}): Stage-1 ANN Recall@{} is {:.2}%. Unranked ANN Recall@{} retains {:.1}% of that pool ({:.2}%), while Stage-2 GATv2 Ranker@{} captures {:.1}% of the pool ({:.2}%), delivering {:+.1}% Recall@{}, {:+.1}% NDCG@{} ({:.4} -> {:.4}), and {:+.1}% MRR@{} ({:.4} -> {:.4}) lift over ANN@{}.",
                num_candidates, top_k, num_candidates, rec_100 * 100.0,
                top_k, ret_capture_pct, rec_ret_20 * 100.0,
                top_k, rnk_capture_pct, rec_rnk_20 * 100.0,
                recall_lift_pct, top_k, ndcg_lift_pct, top_k, ndcg_ret_20, ndcg_rnk_20,
                mrr_lift_pct, top_k, mrr_ret_20, mrr_rnk_20, top_k
            ));
        }

        // Composite Tier-Balanced NDCG@20 (0.33 * Head + 0.33 * Torso + 0.33 * Tail)
        let ret_t0_ndcg = extract_metric(base_metrics, &tag_ret_20, &format!("ndcg_at_{}_mean_movie_tier_0", top_k));
        let ret_t1_ndcg = extract_metric(base_metrics, &tag_ret_20, &format!("ndcg_at_{}_mean_movie_tier_1", top_k));
        let ret_t2_ndcg = extract_metric(base_metrics, &tag_ret_20, &format!("ndcg_at_{}_mean_movie_tier_2", top_k));
        let rnk_t0_ndcg = extract_metric(base_metrics, &tag_rnk_20, &format!("ndcg_at_{}_mean_movie_tier_0", top_k));
        let rnk_t1_ndcg = extract_metric(base_metrics, &tag_rnk_20, &format!("ndcg_at_{}_mean_movie_tier_1", top_k));
        let rnk_t2_ndcg = extract_metric(base_metrics, &tag_rnk_20, &format!("ndcg_at_{}_mean_movie_tier_2", top_k));

        let comp_ndcg_ret = 0.33 * (ret_t0_ndcg + ret_t1_ndcg + ret_t2_ndcg);
        let comp_ndcg_rnk = 0.33 * (rnk_t0_ndcg + rnk_t1_ndcg + rnk_t2_ndcg);
        executive_summary.push(format!(
            "TIER-STRATIFIED RANKING BREAKDOWN (@{}): Composite NDCG@{} shifted from {:.4} (Retriever) to {:.4} (Ranker) | Head (Tier 0): {:.4} -> {:.4} | Torso (Tier 1): {:.4} -> {:.4} | Tail (Tier 2): {:.4} -> {:.4}.",
            top_k, top_k, comp_ndcg_ret, comp_ndcg_rnk,
            ret_t0_ndcg, rnk_t0_ndcg, ret_t1_ndcg, rnk_t1_ndcg, ret_t2_ndcg, rnk_t2_ndcg
        ));

        // GATv2 Spherical Fidelity Conclusions
        let mut gatv2_fidelity_conclusions: Vec<String> = Vec::new();
        for (entity_label, json_key) in [
            ("Movie Candidate", "GATv2_spherical_fidelity_movie"),
            ("User Query", "GATv2_spherical_fidelity_user"),
        ] {
            if let Some(fid) = embedding_diagnostics.get(json_key) {
                let r_bar = fid.get("mean_resultant_length_avg").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let r_p10 = fid.get("mean_resultant_length_p10").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let kappa = fid.get("vmf_kappa_median").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let mean_ang = fid.get("mean_angle_deg_avg").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let p90_ang = fid.get("p90_angle_deg_avg").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let mad_ang = fid.get("robust_mad_angle_deg_avg").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let pct_hi = fid.get("pct_high_fidelity_r_ge_090").and_then(|v| v.as_f64()).unwrap_or(1.0) * 100.0;
                let pct_poly = fid.get("pct_context_polysemous_r_lt_070").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0;

                let status = if r_bar >= 0.90 {
                    "HIGH SPHERICAL STABILITY"
                } else if r_bar >= 0.75 {
                    "MODERATE CONTEXTUAL ADAPTATION"
                } else {
                    "HIGH CONTEXTUAL POLYSEMY"
                };

                let msg = format!(
                    "{} ({}): Mean Resultant Length R_bar = {:.4} (P10 = {:.4}, Median vMF kappa = {:.1}), Mean Geodesic Deviation = {:.1} deg (P90 = {:.1} deg, Robust MAD = {:.1} deg). {:.1}% of entities have tight centroids (R_bar >= 0.90), while {:.1}% exhibit strong context shift (R_bar < 0.70).",
                    status, entity_label, r_bar, r_p10, kappa, mean_ang, p90_ang, mad_ang, pct_hi, pct_poly
                );
                gatv2_fidelity_conclusions.push(msg.clone());
                executive_summary.push(msg);
            }
        }

        // 2. Assemble the Master `summary_conclusions.json` Payload
        let tag_cs_mov_ret_100 = format!("retrieval_ann_{}_cold_start_movies", num_candidates);
        let tag_cs_mov_ret_20 = format!("retrieval_ann_{}_cold_start_movies", top_k);
        let tag_cs_mov_rnk_20 = format!("ranker_{}_cold_start_movies", top_k);

        let tag_cs_usr_ret_100 = format!("retrieval_ann_{}_cold_start_users", num_candidates);
        let tag_cs_usr_ret_20 = format!("retrieval_ann_{}_cold_start_users", top_k);
        let tag_cs_usr_rnk_20 = format!("ranker_{}_cold_start_users", top_k);

        let summary_conclusions_payload = serde_json::json!({
            "executive_summary": executive_summary,
            "cross_metric_analysis": extract_direct_conclusions(cross_metrics_concl, "cross_metric_conclusions"),
            "stratified_metrics": {
                &tag_ret_100: extract_conclusions(base_metrics, &tag_ret_100),
                &tag_ret_20: extract_conclusions(base_metrics, &tag_ret_20),
                &tag_rnk_20: extract_conclusions(base_metrics, &tag_rnk_20),
            },
            "coverage_and_gini": {
                &tag_ret_100: extract_conclusions(coverage, &tag_ret_100),
                &tag_ret_20: extract_conclusions(coverage, &tag_ret_20),
                &tag_rnk_20: extract_conclusions(coverage, &tag_rnk_20),
            },
            "popularity_bias": {
                &tag_ret_100: extract_conclusions(pop_bias, &tag_ret_100),
                &tag_ret_20: extract_conclusions(pop_bias, &tag_ret_20),
                &tag_rnk_20: extract_conclusions(pop_bias, &tag_rnk_20),
            },
            "diversity_metrics": {
                &tag_ret_100: extract_conclusions(diversity, &tag_ret_100),
                &tag_ret_20: extract_conclusions(diversity, &tag_ret_20),
                &tag_rnk_20: extract_conclusions(diversity, &tag_rnk_20),
            },
            "embedding_hubness": {
                &tag_ret_100: extract_conclusions(emb_hubness, &tag_ret_100),
                &tag_ret_20: extract_conclusions(emb_hubness, &tag_ret_20),
                &tag_rnk_20: extract_conclusions(emb_hubness, &tag_rnk_20),
            },
            "contextual_hubness": extract_direct_conclusions(ctx_hubness, "automated_conclusions"),
            "rank_agreement_and_churn": extract_direct_conclusions(rank_agreement, "automated_conclusions"),
            "gatv2_spherical_fidelity": gatv2_fidelity_conclusions,
            "cold_start_movies": {
                &tag_cs_mov_ret_100: extract_conclusions(cold_start_movies, &tag_cs_mov_ret_100),
                &tag_cs_mov_ret_20: extract_conclusions(cold_start_movies, &tag_cs_mov_ret_20),
                &tag_cs_mov_rnk_20: extract_conclusions(cold_start_movies, &tag_cs_mov_rnk_20),
            },
            "cold_start_users": {
                &tag_cs_usr_ret_100: extract_conclusions(cold_start_users, &tag_cs_usr_ret_100),
                &tag_cs_usr_ret_20: extract_conclusions(cold_start_users, &tag_cs_usr_ret_20),
                &tag_cs_usr_rnk_20: extract_conclusions(cold_start_users, &tag_cs_usr_rnk_20),
            }
        });

        let conclusions_path = Path::new(summary_output_dir).join("summary_conclusions.json");
        let conclusions_file = File::create(&conclusions_path)?;
        serde_json::to_writer_pretty(conclusions_file, &summary_conclusions_payload)?;
        println!("Wrote master conclusions to {:?}", conclusions_path);

        // 3. Also export consolidated `summary_metrics.json` for easy model-to-model comparison
        let summary_metrics_payload = serde_json::json!({
            "stratified_metrics": base_metrics,
            "coverage": coverage,
            "popularity_bias": pop_bias,
            "diversity_metrics": diversity,
            "embedding_hubness": emb_hubness,
            "contextual_hubness": ctx_hubness.get("metrics"),
            "rank_agreement": rank_agreement.get("metrics"),
            "gatv2_spherical_fidelity": embedding_diagnostics,
            "cold_start_movies": cold_start_movies,
            "cold_start_users": cold_start_users,
        });

        let metrics_path = Path::new(summary_output_dir).join("summary_metrics.json");
        let metrics_file = File::create(&metrics_path)?;
        serde_json::to_writer_pretty(metrics_file, &summary_metrics_payload)?;
        println!("Wrote master metrics to {:?}", metrics_path);

        Ok(summary_conclusions_payload)
    }

    async fn rank_agreement_and_churn(
        top_k: usize,
        user_ids: &[i32],
        retriever_movie_ids: &[i32], // shape: n_users * top_k
        ranker_movie_ids: &[i32],    // shape: n_users * top_k
        parquet_output_dir: &str,
        summary_output_dir: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {

        let n_users = user_ids.len();

        let mut total_overlap = 0.0;
        let mut total_spearman = 0.0;
        let mut valid_spearman_users = 0.0;

        let mut overlaps = Vec::with_capacity(n_users);
        let mut spearmans = Vec::with_capacity(n_users);

        for u in 0..n_users {
            let start = u * top_k;
            let end = start + top_k;
            let ret_slate = &retriever_movie_ids[start..end];
            let rnk_slate = &ranker_movie_ids[start..end];

            // Map the retriever's slate items to their original rank
            let mut ret_map = std::collections::HashMap::new();
            for (i, &item) in ret_slate.iter().enumerate() {
                ret_map.insert(item, i);
            }

            // Find the intersection (items that survived the Ranker)
            let mut intersection = Vec::new();
            for (j, &item) in rnk_slate.iter().enumerate() {
                if let Some(&i) = ret_map.get(&item) {
                    intersection.push((item, i, j)); // (movie_id, ret_rank_raw, rnk_rank_raw)
                }
            }

            // Calculate Overlap
            let overlap_ratio = intersection.len() as f64 / top_k as f64;
            total_overlap += overlap_ratio;
            overlaps.push(overlap_ratio);

            // Calculate Spearman's Rank Correlation on the intersection
            let n = intersection.len() as f64;
            let spearman = if n > 1.0 {

                // Sort by ret_rank_raw to assign relative ranks 1..N
                intersection.sort_by_key(|k| k.1);
                let mut ret_rel_ranks = std::collections::HashMap::new();
                for (rel_rank, tuple) in intersection.iter().enumerate() {
                    ret_rel_ranks.insert(tuple.0, rel_rank as f64 + 1.0);
                }

                // Sort by rnk_rank_raw to assign relative ranks 1..N
                intersection.sort_by_key(|k| k.2);
                let mut d_sq_sum = 0.0;
                for (rel_rank, tuple) in intersection.iter().enumerate() {
                    let rnk_rel_rank = rel_rank as f64 + 1.0;
                    let ret_rel_rank = *ret_rel_ranks.get(&tuple.0).unwrap();
                    let d = ret_rel_rank - rnk_rel_rank;
                    d_sq_sum += d * d;
                }

                let rho = 1.0 - ((6.0 * d_sq_sum) / (n * (n * n - 1.0)));
                valid_spearman_users += 1.0;
                total_spearman += rho;
                rho
            } else {
                // If 0 or 1 items overlap, correlation is effectively 0
                0.0
            };
            spearmans.push(spearman);
        }

        let mean_overlap = total_overlap / n_users as f64;
        let mean_spearman = if valid_spearman_users > 0.0 {
            total_spearman / valid_spearman_users
        } else {
            0.0
        };

        // Export metrics to Parquet
        let mut df = df!(
            "user_id" => user_ids,
            "overlap_at_k" => overlaps,
            "spearman_rho" => spearmans
        )?;
        let parquet_path = Path::new(parquet_output_dir).join(format!("rank_agreement_k_{}.parquet", top_k));
        let mut file = File::create(&parquet_path)?;
        ParquetWriter::new(&mut file).finish(&mut df)?;

        // Aggregate Results
        let mut agg_res = HashMap::new();
        agg_res.insert(format!("mean_overlap_at_{}", top_k), mean_overlap);
        agg_res.insert(format!("mean_spearman_rho_at_{}", top_k), mean_spearman);

        // Automated Conclusions
        let mut conclusions = Vec::new();
        if mean_overlap < 0.20 {
            conclusions.push(format!("MASSIVE CHURN: The Ranker completely discards and replaces over {:.0}% of the Retriever's top items (Overlap: {:.1}%). It is acting as an extremely aggressive filter.", (1.0 - mean_overlap)*100.0, mean_overlap*100.0));
        } else if mean_overlap > 0.80 {
            conclusions.push(format!("LOW CHURN: The Ranker retains {:.1}% of the Retriever's top items. It primarily functions to fine-tune the existing slate rather than filter it.", mean_overlap*100.0));
        } else {
            conclusions.push(format!("MODERATE CHURN: The Ranker retains {:.1}% of the Retriever's top items.", mean_overlap*100.0));
        }

        if mean_spearman > 0.80 {
            conclusions.push(format!("HIGH RANK AGREEMENT: For items that survive into the final slate, the Ranker heavily agrees with the Retriever's ordering (Spearman ρ = {:.2}).", mean_spearman));
        } else if mean_spearman < 0.30 {
            conclusions.push(format!("AGGRESSIVE RE-RANKING: The Ranker drastically scrambles the relative ordering of retrieved items (Spearman ρ = {:.2}), proving it scores candidate items very differently than the ANN space.", mean_spearman));
        } else {
            conclusions.push(format!("MODERATE RE-RANKING: The Ranker adjusts the relative ordering of retrieved items moderately (Spearman ρ = {:.2}).", mean_spearman));
        }

        let summary_payload = serde_json::json!({
            "metrics": agg_res,
            "automated_conclusions": conclusions
        });

        let summary_path = Path::new(summary_output_dir).join(format!("rank_agreement_k_{}.json", top_k));
        let summary_file = File::create(&summary_path)?;
        serde_json::to_writer_pretty(summary_file, &summary_payload)?;
        println!("Wrote rank agreement metrics to {:?}", summary_path);

        Ok(summary_payload)
    }

    async fn calc_embedding_diagnostics(
        harness: &TestHarness,
        query_model_version:Option<VersionChoice>,
        ranker_model_version:Option<VersionChoice>,
        ranker_model_metadata: &Arc<RankerModelMetadata>,
        summary_output_dir: &str)
        -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {

        // Deduplicate (user_id, timestamp) pairs before calling predict_diagnostic so multi-item
        // rating sessions at the exact same timestamp are not queried redundantly (which would
        // artificially inflate user spherical fidelity R_bar toward 1.0).
        let pos_test_tiered = harness.pos_test_df.clone()
            .select([col("user_id"), col("timestamp")])
            .unique(None, UniqueKeepStrategy::First)
            .sort(["user_id", "timestamp"], SortMultipleOptions::default())
            .collect()?;

        let user_ids: Vec<i32> = pos_test_tiered
            .column("user_id")?
            .cast(&DataType::Int32)? // Safe guard in case type inferred differently
            .i32()?
            .rechunk()
            .cont_slice()
            .expect("user_id column contained nulls; cannot slice contiguously")
            .to_vec();

        let timestamps: Vec<i64> = pos_test_tiered
            .column("timestamp")?
            .cast(&DataType::Int64)?
            .i64()?
            .rechunk()
            .cont_slice()
            .expect("timestamp column contained nulls; cannot slice contiguously")
            .to_vec();

        let tonic_req: Request<UsersRequest> = harness.orchestrator.get_users_request(
            &user_ids,
            &timestamps,
            query_model_version.clone(),
            ranker_model_version.clone()
        ).await.expect("error while forming UsersRequest");

        // returns harness.orchestrator.top_k results.  handles batching.
        let results: Result<Response<RankedMoviesAndRepr>, tonic::Status> =
            harness.orchestrator.predict_diagnostic(tonic_req).await;

        let response_inner = results
            .expect("Error while getting diagnostic predictions fromranker").into_inner();

        //  `k` (top_k) items per user
        let returned_k = response_inner.k as usize;
        let out_features = ranker_model_metadata.out_features;
        let tag = format!("ranker_contextual_k_{}", returned_k);

        assert_eq!(response_inner.user_reprs.len(), out_features * user_ids.len());
        assert_eq!(response_inner.cand_reprs.len(), out_features * user_ids.len() * returned_k,
            "out_features={}, n_users={}, returned_k={}", out_features, user_ids.len(), returned_k);
        assert_eq!(response_inner.movie_ids.len(), user_ids.len() * returned_k);

        // make movie_id, movie_tier LazyFrame
        let movie_df = harness.movie_tiers_df.clone().collect()?;
        let movie_ids_ca = movie_df.column("movie_id")?.cast(&DataType::Int32)?;
        let _movie_ids = movie_ids_ca.i32()?;
        let movie_tiers_ca = movie_df.column("movie_tier")?.cast(&DataType::Int32)?;
        let movie_tiers = movie_tiers_ca.i32()?;
        // Pre-allocate the FxHashMap with the exact number of rows for zero reallocations
        let mut movie_tier_map: FxHashMap<i32, i32> = FxHashMap::with_capacity_and_hasher(
            movie_df.height(),
            Default::default(),
        );
        for (mid_opt, tier_opt) in _movie_ids.iter().zip(movie_tiers.iter()) {
            // Standard pattern to safely ignore any Polars Null values
            if let (Some(mid), Some(tier)) = (mid_opt, tier_opt) {
                movie_tier_map.insert(mid, tier);
            }
        }

        // make user_id, user_tier LazyFrame
        let user_df = harness.user_tiers_df.clone().collect()?;
        let user_ids_ca = user_df.column("user_id")?.cast(&DataType::Int32)?;
        let _user_ids = user_ids_ca.i32()?;
        let user_tiers_ca = user_df.column("user_tier")?.cast(&DataType::Int32)?;
        let user_tiers = user_tiers_ca.i32()?;
        // Pre-allocate the FxHashMap with the exact number of rows for zero reallocations
        let mut user_tier_map: FxHashMap<i32, i32> = FxHashMap::with_capacity_and_hasher(
            user_df.height(),
            Default::default(),
        );
        for (id_opt, tier_opt) in _user_ids.iter().zip(user_tiers.iter()) {
            // Standard pattern to safely ignore any Polars Null values
            if let (Some(id), Some(tier)) = (id_opt, tier_opt) {
                user_tier_map.insert(id, tier);
            }
        }

        let (user_repr_df, user_fidelity)
            = deduplicate_and_eval_spherical_reprs(
                &user_ids,
                &response_inner.user_reprs,
                out_features,
                &user_tier_map,
                "user_id",
                "user_tier",
        ).expect("Error while de-duplicating user_ids and embeddings");

        let (movie_repr_df, movie_fidelity)
            = deduplicate_and_eval_spherical_reprs(
            &response_inner.movie_ids,
            &response_inner.cand_reprs,
            out_features,
            &movie_tier_map,
            "movie_id",
            "movie_tier",
        ).expect("Error while de-duplicating movie_ids and embeddings");

        println!("GATv2 Movie Spherical Fidelity: {:#?}", movie_fidelity);
        println!("GATv2 User Spherical Fidelity: {:#?}", user_fidelity);

        let output_embed_dir = Path::new(summary_output_dir).join("embedding_plots");
        //  Generate the 6 UMAP & t-SNE PNGs (User, Movie, and Joint Shared Hypersphere)
        let saved_pngs = _create_umap_tsne_plots(
            &user_repr_df,
            &movie_repr_df,
            &*output_embed_dir,
            "gatv2_emb",
        ).expect("error while creating tsne and umap plots");

        let agg_res = serde_json::json!({
            "GATv2_spherical_fidelity_movie": movie_fidelity,
            "GATv2_spherical_fidelity_user": user_fidelity,
        });

        let output_json_file_name = "gatv2_spherical_fidelity.json";
        let output_file_path = Path::new(summary_output_dir).join(output_json_file_name);
        let file = File::create(&output_file_path)?;
        serde_json::to_writer_pretty(file, &agg_res)?;

        println!("wrote to {:?}", output_file_path);

        Ok(agg_res)
    }

    /// Deduplicates multiple unit-normalized GATv2 representations per entity (`movie_id` or `user_id`)
    /// into a single L2-normalized centroid vector, returning:
    ///   1. A Polars `DataFrame` with columns `[id_col, "embedding", tier_col, "n_occurrences",
    ///      "mean_resultant_length", "spherical_variance", "vmf_kappa", "mean_angle_deg",
    ///      "p90_angle_deg", "robust_mad_angle_deg"]` ready for `_create_umap_tsne_plots`.
    ///   2. A `SphericalFidelitySummary` struct with global directional fidelity metrics.
    fn deduplicate_and_eval_spherical_reprs(
        entity_ids: &[i32],
        raw_embeddings_flat: &[f32],
        embed_dim: usize,
        tier_lookup: &FxHashMap<i32, i32>,
        id_col_name: &str,
        tier_col_name: &str,
    ) -> Result<(DataFrame, SphericalFidelitySummary), Box<dyn Error>> {
        assert_eq!(
            entity_ids.len() * embed_dim,
            raw_embeddings_flat.len(),
            "raw_embeddings_flat length must equal entity_ids.len() * embed_dim"
        );

        // Group row indices by entity_id
        let mut groups: FxHashMap<i32, Vec<usize>> = FxHashMap::default();
        for (row_idx, &eid) in entity_ids.iter().enumerate() {
            groups.entry(eid).or_default().push(row_idx);
        }

        let mut sorted_eids: Vec<i32> = groups.keys().copied().collect();
        sorted_eids.sort_unstable();
        let n_unique = sorted_eids.len();

        let mut out_ids: Vec<i32> = Vec::with_capacity(n_unique);
        let mut out_tiers: Vec<i32> = Vec::with_capacity(n_unique);
        let mut out_embs: Vec<Series> = Vec::with_capacity(n_unique);
        let mut out_counts: Vec<u32> = Vec::with_capacity(n_unique);
        let mut out_r_bars: Vec<f64> = Vec::with_capacity(n_unique);
        let mut out_sph_vars: Vec<f64> = Vec::with_capacity(n_unique);
        let mut out_kappas: Vec<f64> = Vec::with_capacity(n_unique);
        let mut out_mean_angles: Vec<f64> = Vec::with_capacity(n_unique);
        let mut out_p90_angles: Vec<f64> = Vec::with_capacity(n_unique);
        let mut out_mad_angles: Vec<f64> = Vec::with_capacity(n_unique);

        // Accumulators for multi-occurrence summary stats
        let mut multi_r_bars: Vec<f64> = Vec::new();
        let mut multi_kappas: Vec<f64> = Vec::new();
        let mut multi_mean_angles: Vec<f64> = Vec::new();
        let mut multi_p90_angles: Vec<f64> = Vec::new();
        let mut multi_mad_angles: Vec<f64> = Vec::new();

        let d_f64 = embed_dim as f64;

        for eid in sorted_eids {
            let row_indices = &groups[&eid];
            let count = row_indices.len();
            let count_f64 = count as f64;

            // 1. Accumulate unit-normalized vectors for this entity
            let mut sum_vec = vec![0.0_f64; embed_dim];
            let mut unit_vecs: Vec<Vec<f64>> = Vec::with_capacity(count);

            for &r_idx in row_indices {
                let start = r_idx * embed_dim;
                let slice = &raw_embeddings_flat[start..start + embed_dim];
                let norm_sq: f64 = slice.iter().map(|&x| (x as f64) * (x as f64)).sum();
                let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
                let mut u_vec = Vec::with_capacity(embed_dim);
                for (d, &val) in slice.iter().enumerate() {
                    let u_val = (val as f64) * inv_norm;
                    sum_vec[d] += u_val;
                    u_vec.push(u_val);
                }
                unit_vecs.push(u_vec);
            }

            // Mean vector v_bar and Mean Resultant Length R_bar = ||v_bar||_2
            let mean_vec: Vec<f64> = sum_vec.into_iter().map(|s| s / count_f64).collect();
            let r_bar_raw: f64 = mean_vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let r_bar = r_bar_raw.clamp(0.0, 1.0);
            let sph_var = 1.0 - r_bar;

            // L2-normalized centroid mu = mean_vec / ||mean_vec||_2
            let inv_r = 1.0 / r_bar_raw.max(1e-12);
            let centroid_f64: Vec<f64> = mean_vec.iter().map(|&x| x * inv_r).collect();
            let centroid_f32: Vec<f32> = centroid_f64.iter().map(|&x| x as f32).collect();

            let (kappa, mean_ang, p90_ang, mad_ang) = if count >= 2 {
                // Banerjee et al. (2005) von Mises-Fisher concentration approximation
                let r_safe = r_bar.min(0.9999);
                let k_val = (r_safe * (d_f64 - r_safe * r_safe)) / (1.0 - r_safe * r_safe).max(1e-9);

                // Geodesic angles theta_k = arccos(<v_k, mu>) in degrees
                let mut angles_deg: Vec<f64> = unit_vecs
                    .iter()
                    .map(|u| {
                        let dot: f64 = u
                            .iter()
                            .zip(centroid_f64.iter())
                            .map(|(&a, &b)| a * b)
                            .sum();
                        dot.clamp(-1.0, 1.0).acos().to_degrees()
                    })
                    .collect();

                angles_deg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let m_ang = angles_deg.iter().sum::<f64>() / count_f64;
                let p90_idx = ((0.90 * ((count - 1) as f64)).round() as usize).min(count - 1);
                let p90 = angles_deg[p90_idx];
                let med_ang = percentile_sorted(&angles_deg, 0.50);

                let mut abs_devs: Vec<f64> = angles_deg.iter().map(|&a| (a - med_ang).abs()).collect();
                abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let mad = 1.4826 * percentile_sorted(&abs_devs, 0.50);

                multi_r_bars.push(r_bar);
                multi_kappas.push(k_val);
                multi_mean_angles.push(m_ang);
                multi_p90_angles.push(p90);
                multi_mad_angles.push(mad);

                (k_val, m_ang, p90, mad)
            } else {
                (f64::NAN, 0.0, 0.0, 0.0)
            };

            out_ids.push(eid);
            out_tiers.push(*tier_lookup.get(&eid).unwrap_or(&2));
            out_embs.push(Series::new(PlSmallStr::EMPTY, centroid_f32));
            out_counts.push(count as u32);
            out_r_bars.push(r_bar);
            out_sph_vars.push(sph_var);
            out_kappas.push(kappa);
            out_mean_angles.push(mean_ang);
            out_p90_angles.push(p90_ang);
            out_mad_angles.push(mad_ang);
        }

        let df = DataFrame::new(
            n_unique,
            vec![
                Column::new(PlSmallStr::from_str(id_col_name), out_ids),
                Column::new(PlSmallStr::from_static("embedding"), out_embs),
                Column::new(PlSmallStr::from_str(tier_col_name), out_tiers),
                Column::new(PlSmallStr::from_static("n_occurrences"), out_counts),
                Column::new(PlSmallStr::from_static("mean_resultant_length"), out_r_bars),
                Column::new(PlSmallStr::from_static("spherical_variance"), out_sph_vars),
                Column::new(PlSmallStr::from_static("vmf_kappa"), out_kappas),
                Column::new(PlSmallStr::from_static("mean_angle_deg"), out_mean_angles),
                Column::new(PlSmallStr::from_static("p90_angle_deg"), out_p90_angles),
                Column::new(PlSmallStr::from_static("robust_mad_angle_deg"), out_mad_angles),
            ],
        )?;

        let n_multi = multi_r_bars.len();
        let mut sorted_r = multi_r_bars.clone();
        sorted_r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mut sorted_k = multi_kappas.clone();
        sorted_k.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let r_avg = if n_multi > 0 {
            multi_r_bars.iter().sum::<f64>() / (n_multi as f64)
        } else {
            1.0
        };
        let summary = SphericalFidelitySummary {
            n_unique_entities: n_unique,
            n_multi_occurrence_entities: n_multi,
            mean_resultant_length_avg: r_avg,
            mean_resultant_length_p10: if n_multi > 0 {
                percentile_sorted(&sorted_r, 0.10)
            } else {
                1.0
            },
            spherical_variance_avg: 1.0 - r_avg,
            vmf_kappa_median: if n_multi > 0 {
                percentile_sorted(&sorted_k, 0.50)
            } else {
                0.0
            },
            mean_angle_deg_avg: if n_multi > 0 {
                multi_mean_angles.iter().sum::<f64>() / (n_multi as f64)
            } else {
                0.0
            },
            p90_angle_deg_avg: if n_multi > 0 {
                multi_p90_angles.iter().sum::<f64>() / (n_multi as f64)
            } else {
                0.0
            },
            robust_mad_angle_deg_avg: if n_multi > 0 {
                multi_mad_angles.iter().sum::<f64>() / (n_multi as f64)
            } else {
                0.0
            },
            pct_high_fidelity_r_ge_090: if n_multi > 0 {
                (multi_r_bars.iter().filter(|&&r| r >= 0.90).count() as f64) / (n_multi as f64)
            } else {
                1.0
            },
            pct_context_polysemous_r_lt_070: if n_multi > 0 {
                (multi_r_bars.iter().filter(|&&r| r < 0.70).count() as f64) / (n_multi as f64)
            } else {
                0.0
            },
        };

        Ok((df, summary))
    }


    /// Generates 6 high-resolution PNG projection plots (3 UMAP + 3 t-SNE) in pure Rust:
    ///   1. User embeddings stratified by `user_tier` (t-SNE & UMAP)
    ///   2. Movie embeddings stratified by `movie_tier` (t-SNE & UMAP)
    ///   3. Joint User + Movie embeddings co-projected in the SAME shared hypersphere
    ///      (t-SNE & UMAP using geodesic cosine distance), stratified by entity & tier.
    ///
    /// # Arguments
    /// * `user_repr_df` - Polars `DataFrame` with columns `["user_id", "embedding", "user_tier"]`
    /// * `movie_repr_df` - Polars `DataFrame` with columns `["movie_id", "embedding", "movie_tier"]`
    /// * `output_dir` - Directory path where PNG files will be written
    /// * `output_filename_root` - Prefix/root string for saved PNG filenames
    pub fn _create_umap_tsne_plots(
        user_repr_df: &DataFrame,
        movie_repr_df: &DataFrame,
        output_dir: &Path,
        output_filename_root: &str,
    ) -> Result<Vec<PathBuf>, Box<dyn Error>> {

        ensure_plotters_font_registered();

        std::fs::create_dir_all(output_dir)?;
        let mut saved_files: Vec<PathBuf> = Vec::with_capacity(6);

        let (x_users, y_users) =
            extract_unit_embeddings_and_tiers(user_repr_df, "embedding", "user_tier")?;
        let (x_movies, y_movies) =
            extract_unit_embeddings_and_tiers(movie_repr_df, "embedding", "movie_tier")?;

        // 1. User Embeddings (tier-stratified t-SNE & UMAP)
        let user_tsne_coords = run_cosine_tsne(&x_users, 30.0, 500, 42);
        let user_tsne_path = output_dir.join(format!(
            "{}_tsne_users_user_tier.png",
            output_filename_root
        ));
        render_tier_scatter_png(
            &user_tsne_coords,
            &y_users,
            &format!(
                "{}: User Embedding t-SNE by user_tier (N={})",
                output_filename_root,
                x_users.len()
            ),
            "t-SNE 1",
            "t-SNE 2",
            "user_tier",
            &user_tsne_path,
        )?;
        saved_files.push(user_tsne_path);

        let user_umap_coords = run_cosine_umap(&x_users, 15, 0.1, 200, 42);
        let user_umap_path = output_dir.join(format!(
            "{}_umap_users_user_tier.png",
            output_filename_root
        ));
        render_tier_scatter_png(
            &user_umap_coords,
            &y_users,
            &format!(
                "{}: User Embedding UMAP by user_tier (N={})",
                output_filename_root,
                x_users.len()
            ),
            "UMAP 1",
            "UMAP 2",
            "user_tier",
            &user_umap_path,
        )?;
        saved_files.push(user_umap_path);

        // 2. Movie Embeddings (tier-stratified t-SNE & UMAP)
        let movie_tsne_coords = run_cosine_tsne(&x_movies, 30.0, 500, 42);
        let movie_tsne_path = output_dir.join(format!(
            "{}_tsne_movies_movie_tier.png",
            output_filename_root
        ));
        render_tier_scatter_png(
            &movie_tsne_coords,
            &y_movies,
            &format!(
                "{}: Movie Embedding t-SNE by movie_tier (N={})",
                output_filename_root,
                x_movies.len()
            ),
            "t-SNE 1",
            "t-SNE 2",
            "movie_tier",
            &movie_tsne_path,
        )?;
        saved_files.push(movie_tsne_path);

        let movie_umap_coords = run_cosine_umap(&x_movies, 15, 0.1, 200, 42);
        let movie_umap_path = output_dir.join(format!(
            "{}_umap_movies_movie_tier.png",
            output_filename_root
        ));
        render_tier_scatter_png(
            &movie_umap_coords,
            &y_movies,
            &format!(
                "{}: Movie Embedding UMAP by movie_tier (N={})",
                output_filename_root,
                x_movies.len()
            ),
            "UMAP 1",
            "UMAP 2",
            "movie_tier",
            &movie_umap_path,
        )?;
        saved_files.push(movie_umap_path);

        // 3a. Bipartite Catalog-Anchored Joint UMAP:
        //     Project each User directly onto the Movie UMAP coordinates as a softmax-weighted
        //     barycenter of its top-15 nearest Movies in cosine similarity!
        //     This uses strictly User -> Movie cross-edges and eliminates the User-User k-NN ring artifact.
        let user_on_movie_umap_coords =
            project_users_onto_movie_manifold(&x_users, &x_movies, &movie_umap_coords, 15);
        let bipartite_umap_path = output_dir.join(format!(
            "{}_umap_catalog_anchored_joint_users_movies.png",
            output_filename_root
        ));
        render_joint_scatter_png(
            &user_on_movie_umap_coords,
            &y_users,
            &movie_umap_coords,
            &y_movies,
            &format!(
                "{}: Catalog-Anchored Bipartite UMAP (Users Projected via Top-15 Movie Cross-Edges)",
                output_filename_root
            ),
            "Movie UMAP 1",
            "Movie UMAP 2",
            &bipartite_umap_path,
        )?;
        saved_files.push(bipartite_umap_path);

        // 3b. Modality-Centered Joint User + Movie Embeddings in the Shared Hypersphere
        let n_u = x_users.len();
        let x_users_centered = center_and_renormalize(&x_users);
        let x_movies_centered = center_and_renormalize(&x_movies);
        let mut x_joint: Vec<Vec<f32>> = Vec::with_capacity(n_u + x_movies.len());
        x_joint.extend_from_slice(&x_users_centered);
        x_joint.extend_from_slice(&x_movies_centered);

        let joint_tsne_coords = run_cosine_tsne(&x_joint, 30.0, 500, 42);
        let joint_tsne_path = output_dir.join(format!(
            "{}_tsne_joint_users_movies.png",
            output_filename_root
        ));
        render_joint_scatter_png(
            &joint_tsne_coords[..n_u],
            &y_users,
            &joint_tsne_coords[n_u..],
            &y_movies,
            &format!(
                "{}: Joint User & Movie Shared Hypersphere t-SNE (Users={}, Movies={})",
                output_filename_root,
                n_u,
                x_movies.len()
            ),
            "t-SNE 1",
            "t-SNE 2",
            &joint_tsne_path,
        )?;
        saved_files.push(joint_tsne_path);

        let joint_umap_coords = run_cosine_umap(&x_joint, 15, 0.1, 200, 42);
        let joint_umap_path = output_dir.join(format!(
            "{}_umap_joint_users_movies.png",
            output_filename_root
        ));
        render_joint_scatter_png(
            &joint_umap_coords[..n_u],
            &y_users,
            &joint_umap_coords[n_u..],
            &y_movies,
            &format!(
                "{}: Joint User & Movie Shared Hypersphere UMAP (Users={}, Movies={})",
                output_filename_root,
                n_u,
                x_movies.len()
            ),
            "UMAP 1",
            "UMAP 2",
            &joint_umap_path,
        )?;
        saved_files.push(joint_umap_path);

        Ok(saved_files)
    }

    // ============================================================================
    // Internal Helpers: Polars Extraction, Cosine UMAP, Cosine t-SNE, Plotters PNG
    // ============================================================================

    fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let weight = pos - (lo as f64);
            sorted[lo] * (1.0 - weight) + sorted[hi] * weight
        }
    }

    /// Extracts unit-normalized embedding vectors and integer tier labels from a Polars `DataFrame`.
    /// Supports both `DataType::List(Float32/Float64)` and `DataType::Array(Float32/Float64, _)`.
    fn extract_unit_embeddings_and_tiers(
        df: &DataFrame,
        emb_col: &str,
        tier_col: &str,
    ) -> Result<(Vec<Vec<f32>>, Vec<i32>), Box<dyn Error>> {
        let n_rows = df.height();
        let tier_series = df.column(tier_col)?.cast(&DataType::Int32)?;
        let tier_ca = tier_series.i32()?;
        let tiers: Vec<i32> = (0..n_rows).map(|i| tier_ca.get(i).unwrap_or(0)).collect();

        let emb_series = df.column(emb_col)?;
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n_rows);

        if let Ok(list_ca) = emb_series.list() {
            for i in 0..n_rows {
                let sub = list_ca
                    .get_as_series(i)
                    .ok_or("Missing embedding row in List column")?;
                let sub_f32 = sub.cast(&DataType::Float32)?;
                let ca = sub_f32.f32()?;
                let mut vec_i: Vec<f32> = (0..ca.len()).map(|j| ca.get(j).unwrap_or(0.0)).collect();
                l2_normalize_in_place(&mut vec_i);
                embeddings.push(vec_i);
            }
        } else {
            // Fallback if stored as FixedSizeList / Array
            let list_series = emb_series.cast(&DataType::List(Box::new(DataType::Float32)))?;
            let list_ca = list_series.list()?;
            for i in 0..n_rows {
                let sub = list_ca
                    .get_as_series(i)
                    .ok_or("Missing embedding row in Array column")?;
                let ca = sub.f32()?;
                let mut vec_i: Vec<f32> = (0..ca.len()).map(|j| ca.get(j).unwrap_or(0.0)).collect();
                l2_normalize_in_place(&mut vec_i);
                embeddings.push(vec_i);
            }
        }

        Ok((embeddings, tiers))
    }

    #[inline]
    fn l2_normalize_in_place(v: &mut [f32]) {
        let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
        let inv = 1.0 / norm_sq.sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x *= inv;
        }
    }

    #[inline]
    fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        (1.0 - dot).max(0.0)
    }

    /// Computes exact top-k nearest neighbors under cosine distance for each row in `x`.
    fn compute_cosine_knn(x: &[Vec<f32>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let n = x.len();
        let k_actual = k.min(n.saturating_sub(1));
        let mut knn_indices = vec![Vec::with_capacity(k_actual); n];
        let mut knn_dists = vec![Vec::with_capacity(k_actual); n];

        for i in 0..n {
            let mut row_dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_dist(&x[i], &x[j])))
                .collect();

            if k_actual > 0 && row_dists.len() > k_actual {
                row_dists.select_nth_unstable_by(k_actual, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
                });
                row_dists.truncate(k_actual);
            }
            row_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            knn_indices[i] = row_dists.iter().map(|&(idx, _)| idx).collect();
            knn_dists[i] = row_dists.iter().map(|&(_, d)| d).collect();
        }

        (knn_indices, knn_dists)
    }

    /// Deterministic 2D power-iteration PCA initialization so UMAP and t-SNE start from a
    /// globally aligned coordinate frame.
    fn pca_2d_init(x: &[Vec<f32>], scale: f32) -> Vec<[f32; 2]> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let d = x[0].len();
        let mut mean = vec![0.0_f64; d];
        for row in x {
            for (j, &v) in row.iter().enumerate() {
                mean[j] += v as f64;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }

        // Covariance matrix C = X_centered^T X_centered (d x d, where d = 16)
        let mut cov = vec![vec![0.0_f64; d]; d];
        for row in x {
            for r in 0..d {
                let xr = (row[r] as f64) - mean[r];
                for c in r..d {
                    cov[r][c] += xr * ((row[c] as f64) - mean[c]);
                }
            }
        }
        for r in 0..d {
            for c in 0..r {
                cov[r][c] = cov[c][r];
            }
        }

        // Power iteration for top 2 eigenvectors
        let mut v1 = (0..d).map(|j| ((j + 1) as f64).sin()).collect::<Vec<f64>>();
        for _ in 0..40 {
            let mut next = vec![0.0_f64; d];
            for r in 0..d {
                for c in 0..d {
                    next[r] += cov[r][c] * v1[c];
                }
            }
            let norm = next.iter().map(|&z| z * z).sum::<f64>().sqrt().max(1e-12);
            for z in next.iter_mut() {
                *z /= norm;
            }
            v1 = next;
        }

        // Deflate cov by v1
        let mut lambda1 = 0.0_f64;
        for r in 0..d {
            for c in 0..d {
                lambda1 += v1[r] * cov[r][c] * v1[c];
            }
        }
        for r in 0..d {
            for c in 0..d {
                cov[r][c] -= lambda1 * v1[r] * v1[c];
            }
        }

        let mut v2 = (0..d).map(|j| ((j + 2) as f64).cos()).collect::<Vec<f64>>();
        for _ in 0..40 {
            let mut next = vec![0.0_f64; d];
            for r in 0..d {
                for c in 0..d {
                    next[r] += cov[r][c] * v2[c];
                }
            }
            let norm = next.iter().map(|&z| z * z).sum::<f64>().sqrt().max(1e-12);
            for z in next.iter_mut() {
                *z /= norm;
            }
            v2 = next;
        }

        let mut coords = Vec::with_capacity(n);
        let mut max_abs = 1e-6_f32;
        for row in x {
            let mut c0 = 0.0_f64;
            let mut c1 = 0.0_f64;
            for j in 0..d {
                let centered = (row[j] as f64) - mean[j];
                c0 += centered * v1[j];
                c1 += centered * v2[j];
            }
            let p0 = c0 as f32;
            let p1 = c1 as f32;
            max_abs = max_abs.max(p0.abs()).max(p1.abs());
            coords.push([p0, p1]);
        }

        for p in coords.iter_mut() {
            p[0] = (p[0] / max_abs) * scale;
            p[1] = (p[1] / max_abs) * scale;
        }
        coords
    }

    /// Self-contained Cosine UMAP implementation in pure Rust (fuzzy simplicial set + SGD layout).
    pub fn run_cosine_umap(
        x: &[Vec<f32>],
        n_neighbors: usize,
        min_dist: f32,
        n_epochs: usize,
        seed: u64,
    ) -> Vec<[f32; 2]> {
        let n = x.len();
        if n <= 2 {
            return vec![[0.0, 0.0]; n];
        }

        let (knn_idx, knn_dists) = compute_cosine_knn(x, n_neighbors);
        let target_log2 = (n_neighbors as f32).ln() / std::f32::consts::LN_2;

        // 1. Smooth kNN distances: find rho_i (distance to nearest neighbor) and sigma_i via binary search
        let mut directed_edges: FxHashMap<(usize, usize), f32> =
            FxHashMap::with_capacity_and_hasher(n * n_neighbors * 2, Default::default());

        for i in 0..n {
            let dists = &knn_dists[i];
            if dists.is_empty() {
                continue;
            }
            let rho = dists[0].max(0.0);
            let mut lo = 1e-5_f32;
            let mut hi = 10.0_f32;
            let mut sigma = 1.0_f32;

            for _ in 0..32 {
                sigma = 0.5 * (lo + hi);
                let sum_p: f32 = dists
                    .iter()
                    .map(|&d| (-(d - rho).max(0.0) / sigma).exp())
                    .sum();
                if sum_p > target_log2 {
                    hi = sigma;
                } else {
                    lo = sigma;
                }
            }

            for (&j, &d) in knn_idx[i].iter().zip(dists.iter()) {
                let w = (-(d - rho).max(0.0) / sigma).exp();
                directed_edges.insert((i, j), w);
            }
        }

        // 2. Fuzzy set union: w_sym(i, j) = w_ij + w_ji - w_ij * w_ji
        let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(directed_edges.len());
        for (&(i, j), &w_ij) in directed_edges.iter() {
            let w_ji = *directed_edges.get(&(j, i)).unwrap_or(&0.0);
            if i < j || w_ji == 0.0 {
                let w_sym = w_ij + w_ji - w_ij * w_ji;
                if w_sym > 1e-3 {
                    edges.push((i, j, w_sym));
                    edges.push((j, i, w_sym));
                }
            }
        }

        // UMAP curve parameters (a, b) for min_dist = 0.1, spread = 1.0
        let (a, b) = if (min_dist - 0.1).abs() < 0.05 {
            (1.576943_f32, 0.8950609_f32)
        } else {
            (1.929_f32, 0.7915_f32)
        };

        let mut coords = pca_2d_init(x, 6.0);
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        let neg_sample_rate = 5usize;

        for epoch in 0..n_epochs {
            let alpha = 1.0_f32 * (1.0 - (epoch as f32) / (n_epochs as f32));
            for &(i, j, w) in &edges {
                // Sample edge with probability w
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let u_rand = ((rng_state >> 40) as f32) / ((1u64 << 24) as f32);
                if u_rand > w {
                    continue;
                }

                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dist_sq = (dx * dx + dy * dy).max(1e-4);
                let dist_pow_b = dist_sq.powf(b);

                // Attractive gradient
                let grad_coeff = (-2.0 * a * b * dist_pow_b / dist_sq) / (1.0 + a * dist_pow_b);
                let gx = (grad_coeff * dx).clamp(-4.0, 4.0) * alpha;
                let gy = (grad_coeff * dy).clamp(-4.0, 4.0) * alpha;
                coords[i][0] += gx;
                coords[i][1] += gy;
                coords[j][0] -= gx;
                coords[j][1] -= gy;

                // Negative sampling repulsive gradient
                for _ in 0..neg_sample_rate {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let k = (rng_state as usize) % n;
                    if k == i {
                        continue;
                    }
                    let ndx = coords[i][0] - coords[k][0];
                    let ndy = coords[i][1] - coords[k][1];
                    let ndist_sq = (ndx * ndx + ndy * ndy).max(1e-3);
                    let ndist_pow_b = ndist_sq.powf(b);
                    let rep_coeff =
                        (2.0 * b) / ((0.001 + ndist_sq) * (1.0 + a * ndist_pow_b));
                    let rgx = (rep_coeff * ndx).clamp(-4.0, 4.0) * alpha;
                    let rgy = (rep_coeff * ndy).clamp(-4.0, 4.0) * alpha;
                    coords[i][0] += rgx;
                    coords[i][1] += rgy;
                }
            }
        }

        coords
    }

    /// Self-contained Cosine t-SNE in pure Rust (sparse k-NN Gaussian affinities + Student-t KL descent).
    pub fn run_cosine_tsne(
        x: &[Vec<f32>],
        perplexity: f32,
        n_iter: usize,
        seed: u64,
    ) -> Vec<[f32; 2]> {
        let n = x.len();
        if n <= 2 {
            return vec![[0.0, 0.0]; n];
        }

        let k_neighbors = ((3.0 * perplexity) as usize).clamp(15, 90).min(n - 1);
        let (knn_idx, knn_dists) = compute_cosine_knn(x, k_neighbors);
        let target_entropy = perplexity.ln();

        // 1. Compute conditional probabilities p_{j|i} via binary search for beta_i = 1 / (2 * sigma_i^2)
        let mut p_directed: FxHashMap<(usize, usize), f32> =
            FxHashMap::with_capacity_and_hasher(n * k_neighbors * 2, Default::default());

        for i in 0..n {
            let dists = &knn_dists[i];
            let mut beta = 1.0_f32;
            let mut beta_min = 0.0_f32;
            let mut beta_max = f32::INFINITY;
            let mut probs = vec![0.0_f32; dists.len()];

            for _ in 0..40 {
                let mut sum_p = 0.0_f32;
                for (m, &d) in dists.iter().enumerate() {
                    let p = (-beta * d * d).exp();
                    probs[m] = p;
                    sum_p += p;
                }
                sum_p = sum_p.max(1e-12);
                let mut h = 0.0_f32;
                for p in probs.iter_mut() {
                    *p /= sum_p;
                    if *p > 1e-12 {
                        h -= *p * p.ln();
                    }
                }
                let diff = h - target_entropy;
                if diff.abs() < 1e-4 {
                    break;
                }
                if diff > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_infinite() {
                        beta * 2.0
                    } else {
                        0.5 * (beta + beta_max)
                    };
                } else {
                    beta_max = beta;
                    beta = 0.5 * (beta + beta_min);
                }
            }

            for (&j, &p) in knn_idx[i].iter().zip(probs.iter()) {
                p_directed.insert((i, j), p);
            }
        }

        // Symmetrize P_{ij} = (p_{j|i} + p_{i|j}) / (2N)
        let mut sym_edges: Vec<(usize, usize, f32)> = Vec::with_capacity(p_directed.len());
        let denom = 2.0 * (n as f32);
        for (&(i, j), &p_ij) in p_directed.iter() {
            let p_ji = *p_directed.get(&(j, i)).unwrap_or(&0.0);
            if i < j || p_ji == 0.0 {
                let p_sym = ((p_ij + p_ji) / denom).max(1e-12);
                sym_edges.push((i, j, p_sym));
            }
        }

        let mut y = pca_2d_init(x, 1.0);
        let mut vel = vec![[0.0_f32; 2]; n];
        let mut gains = vec![[1.0_f32; 2]; n];
        let mut grad = vec![[0.0_f32; 2]; n];
        let mut rng_state = seed ^ 0x517CC1B727220A95;
        let rep_samples = 64usize.min(n - 1);
        let lr = 200.0_f32;

        for iter in 0..n_iter {
            let exaggeration = if iter < 125 { 4.0_f32 } else { 1.0_f32 };
            let momentum = if iter < 125 { 0.5_f32 } else { 0.8_f32 };

            for g in grad.iter_mut() {
                g[0] = 0.0;
                g[1] = 0.0;
            }

            // Estimate normalization Z and repulsive forces via Monte Carlo sampling over all N nodes
            let mut z_sum = 0.0_f64;
            let sample_scale = ((n - 1) as f32) / (rep_samples as f32);

            for i in 0..n {
                for _ in 0..rep_samples {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let j = (rng_state as usize) % n;
                    if j == i {
                        continue;
                    }
                    let dx = y[i][0] - y[j][0];
                    let dy = y[i][1] - y[j][1];
                    let q_num = 1.0 / (1.0 + dx * dx + dy * dy);
                    z_sum += (q_num * sample_scale) as f64;
                }
            }
            let z_inv = (1.0 / z_sum.max(1e-9)) as f32;

            // Attractive forces along sparse k-NN edges: 4 * (exaggeration * P_ij) * q_num * (y_i - y_j)
            for &(i, j, p_ij) in &sym_edges {
                let dx = y[i][0] - y[j][0];
                let dy = y[i][1] - y[j][1];
                let q_num = 1.0 / (1.0 + dx * dx + dy * dy);
                let f_attr = 4.0 * (p_ij * exaggeration) * q_num;
                let fx = f_attr * dx;
                let fy = f_attr * dy;
                grad[i][0] += fx;
                grad[i][1] += fy;
                grad[j][0] -= fx;
                grad[j][1] -= fy;
            }

            // Repulsive forces: -4 * (q_num / Z) * q_num * (y_i - y_j)
            for i in 0..n {
                for _ in 0..rep_samples {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let j = (rng_state as usize) % n;
                    if j == i {
                        continue;
                    }
                    let dx = y[i][0] - y[j][0];
                    let dy = y[i][1] - y[j][1];
                    let q_num = 1.0 / (1.0 + dx * dx + dy * dy);
                    let f_rep = 4.0 * (q_num * z_inv) * q_num * sample_scale;
                    grad[i][0] -= f_rep * dx;
                    grad[i][1] -= f_rep * dy;
                }
            }

            // Update with adaptive gains and momentum
            let mut mean_x = 0.0_f32;
            let mut mean_y = 0.0_f32;
            for i in 0..n {
                for d in 0..2 {
                    let same_sign = (grad[i][d] > 0.0) == (vel[i][d] > 0.0);
                    gains[i][d] = if same_sign {
                        (gains[i][d] * 0.8).max(0.01)
                    } else {
                        gains[i][d] + 0.2
                    };
                    vel[i][d] = momentum * vel[i][d] - lr * gains[i][d] * grad[i][d];
                    y[i][d] += vel[i][d];
                }
                mean_x += y[i][0];
                mean_y += y[i][1];
            }
            mean_x /= n as f32;
            mean_y /= n as f32;
            for p in y.iter_mut() {
                p[0] -= mean_x;
                p[1] -= mean_y;
            }
        }

        y
    }

    fn compute_axis_bounds(coords: &[[f32; 2]]) -> (f32, f32, f32, f32) {
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for &[x, y] in coords {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        let pad_x = ((max_x - min_x) * 0.07).max(0.5);
        let pad_y = ((max_y - min_y) * 0.07).max(0.5);
        (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)
    }

    /// Renders a single-modality tier-stratified scatter plot to PNG using `plotters`.
    fn render_tier_scatter_png(
        coords: &[[f32; 2]],
        tiers: &[i32],
        title: &str,
        x_label: &str,
        y_label: &str,
        legend_title: &str,
        out_path: &Path,
    ) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(out_path, (1200, 960)).into_drawing_area();
        root.fill(&WHITE)?;

        let (min_x, max_x, min_y, max_y) = compute_axis_bounds(coords);

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 24).into_font())
            .margin(25)
            .x_label_area_size(45)
            .y_label_area_size(55)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .axis_desc_style(("sans-serif", 18))
            .label_style(("sans-serif", 14))
            .light_line_style(RGBColor(235, 235, 235))
            .draw()?;

        // Tab10 palette matching Python seaborn ("tab10": 0=Blue, 1=Orange, 2=Green)
        let tier_colors: [(i32, RGBColor); 3] = [
            (0, RGBColor(31, 119, 180)),
            (1, RGBColor(255, 127, 14)),
            (2, RGBColor(44, 160, 44)),
        ];

        for (tier_val, color) in tier_colors {
            let pts: Vec<(f32, f32)> = coords
                .iter()
                .zip(tiers.iter())
                .filter(|&(_, &t)| t == tier_val)
                .map(|(&[x, y], _)| (x, y))
                .collect();

            let label = format!("{} {}", legend_title, tier_val);
            chart
                .draw_series(
                    pts.into_iter()
                        .map(|(x, y)| Circle::new((x, y), 3, color.mix(0.80).filled())),
                )?
                .label(label)
                .legend(move |(x, y)| Circle::new((x + 8, y), 5, color.filled()));
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.9))
            .border_style(BLACK.mix(0.4))
            .label_font(("sans-serif", 15))
            .draw()?;

        root.present()?;
        Ok(())
    }

    /// Renders the Joint User + Movie Shared-Hypersphere scatter plot to PNG using `plotters`.
    fn render_joint_scatter_png(
        user_coords: &[[f32; 2]],
        user_tiers: &[i32],
        movie_coords: &[[f32; 2]],
        movie_tiers: &[i32],
        title: &str,
        x_label: &str,
        y_label: &str,
        out_path: &Path,
    ) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(out_path, (1300, 980)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut all_coords: Vec<[f32; 2]> = Vec::with_capacity(user_coords.len() + movie_coords.len());
        all_coords.extend_from_slice(user_coords);
        all_coords.extend_from_slice(movie_coords);
        let (min_x, max_x, min_y, max_y) = compute_axis_bounds(&all_coords);

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 23).into_font())
            .margin(25)
            .x_label_area_size(45)
            .y_label_area_size(55)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .axis_desc_style(("sans-serif", 18))
            .label_style(("sans-serif", 14))
            .light_line_style(RGBColor(235, 235, 235))
            .draw()?;

        // 1. Draw Users first (Circles in cool blue shades)
        let user_styles: [(i32, &str, RGBColor); 3] = [
            (0, "User Tier 0 (Power)", RGBColor(31, 119, 180)),
            (1, "User Tier 1 (Mid)", RGBColor(107, 174, 214)),
            (2, "User Tier 2 (Light)", RGBColor(158, 202, 225)),
        ];

        for (t_val, label_str, color) in user_styles {
            let pts: Vec<(f32, f32)> = user_coords
                .iter()
                .zip(user_tiers.iter())
                .filter(|&(_, &t)| t == t_val)
                .map(|(&[x, y], _)| (x, y))
                .collect();

            chart
                .draw_series(
                    pts.into_iter()
                        .map(|(x, y)| Circle::new((x, y), 3, color.mix(0.45).filled())),
                )?
                .label(label_str)
                .legend(move |(x, y)| Circle::new((x + 8, y), 5, color.filled()));
        }

        // 2. Draw Movies on top (Triangles in high-contrast Red / Orange / Green)
        let movie_styles: [(i32, &str, RGBColor); 3] = [
            (1, "Movie Tier 1 (Torso)", RGBColor(255, 127, 14)),
            (2, "Movie Tier 2 (Tail)", RGBColor(44, 160, 44)),
            (0, "Movie Tier 0 (Head)", RGBColor(214, 39, 40)),
        ];

        for (t_val, label_str, color) in movie_styles {
            let pts: Vec<(f32, f32)> = movie_coords
                .iter()
                .zip(movie_tiers.iter())
                .filter(|&(_, &t)| t == t_val)
                .map(|(&[x, y], _)| (x, y))
                .collect();

            chart
                .draw_series(
                    pts.into_iter()
                        .map(|(x, y)| TriangleMarker::new((x, y), 5, color.mix(0.85).filled())),
                )?
                .label(label_str)
                .legend(move |(x, y)| TriangleMarker::new((x + 8, y), 6, color.filled()));
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.92))
            .border_style(BLACK.mix(0.4))
            .label_font(("sans-serif", 15))
            .draw()?;

        root.present()?;
        Ok(())
    }


    /// Centers a modality's unit vectors by subtracting their mean direction and re-normalizing to S^(d-1).
    /// Removes any constant modality-cone translation offset (Liang et al., NeurIPS 2022).
    fn center_and_renormalize(x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let d = x[0].len();
        let mut mean = vec![0.0_f32; d];
        for row in x {
            for (j, &v) in row.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f32;
        }
        let mut out = Vec::with_capacity(n);
        for row in x {
            let mut v: Vec<f32> = row.iter().zip(mean.iter()).map(|(&a, &m)| a - m).collect();
            l2_normalize_in_place(&mut v);
            out.push(v);
        }
        out
    }

    /// Projects each user directly into the 2D Movie UMAP coordinate frame using strictly
    /// bipartite User -> Movie cosine k-NN edges (matching `reducer_movie.transform(X_users)`).
    fn project_users_onto_movie_manifold(
        x_users: &[Vec<f32>],
        x_movies: &[Vec<f32>],
        movie_coords_2d: &[[f32; 2]],
        k: usize,
    ) -> Vec<[f32; 2]> {
        let n_u = x_users.len();
        let n_m = x_movies.len();
        if n_u == 0 || n_m == 0 {
            return vec![[0.0, 0.0]; n_u];
        }
        let k_actual = k.min(n_m).max(1);
        let mut user_coords = Vec::with_capacity(n_u);

        for u_vec in x_users {
            let mut dists: Vec<(usize, f32)> = (0..n_m)
                .map(|m_idx| (m_idx, cosine_dist(u_vec, &x_movies[m_idx])))
                .collect();

            if dists.len() > k_actual {
                dists.select_nth_unstable_by(k_actual, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
                });
                dists.truncate(k_actual);
            }
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let rho = dists[0].1.max(0.0);
            let sigma = (dists[k_actual - 1].1 - rho).max(1e-3);
            let mut w_sum = 0.0_f32;
            let mut cx = 0.0_f32;
            let mut cy = 0.0_f32;

            for &(m_idx, d) in &dists {
                let w = (-(d - rho).max(0.0) / sigma).exp();
                cx += w * movie_coords_2d[m_idx][0];
                cy += w * movie_coords_2d[m_idx][1];
                w_sum += w;
            }
            w_sum = w_sum.max(1e-9);
            user_coords.push([cx / w_sum, cy / w_sum]);
        }

        user_coords
    }

}