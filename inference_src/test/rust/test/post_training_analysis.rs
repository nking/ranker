use std::{fs, io};
use std::path::PathBuf;use polars::prelude::{LazyFrame, PolarsResult};

// Assuming your UserRequest is accessible here
pub mod helper {
    // Tell Rust to literally include the code from helper.rs here
    include!("helper.rs");
    include!("helper_users.rs");
}
use inference_engine::app_config::AppConfig;
use inference_engine::orchestrator::Orchestrator;
use helper::{get_train_val_test_liked_uris, DataSize};
use inference_engine::model_client::tf_serving::model_spec::VersionChoice;
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
    #[allow(dead_code)]
    num_catalog_movies: usize,
    movie_tiers_df: LazyFrame,
    user_tiers_df: LazyFrame,
}

impl TestHarness {
    async fn new() -> Self {
        println!("\n[SETUP]: Initializing test resource...");
        let config_path = get_config_json_uri().to_string();
        let config = AppConfig::load_from_file(&config_path)
            .expect("Could not load tiny config json file");

        let query_uri = config.query_uri.clone();
        let ranker_uri = config.ranker_uri.clone();
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
            user_db_path
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
        let movies_offset = ranker_metadata.num_catalog_users + 1;
        let num_catalog_movies = ranker_metadata.num_catalog_movies;

        Self {
            orchestrator: orchestrator, test_uris: test_ratings_uris, ratings_uris: ratings_uris,
            movies_df: movies_df,
            movies_offset: movies_offset,
            num_catalog_movies: num_catalog_movies,
            train_history_df: train_history_df,
            pos_test_df: pos_test_df,
            movie_tiers_df: movie_tiers_df,
            user_tiers_df: user_tiers_df
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
    use std::fs::File;
    use std::path::Path;
    use polars::df;
    use polars::prelude::*;
    use tonic::Request;
    // Bring everything from the outer scope (TestHarness, helper functions, etc.) into the test module
    use super::*;

    use inference_engine::model_client::tf_serving::model_spec::VersionChoice;
    use inference_engine::pb::{ApproxNearestNeighborsResponse, UsersRequest};
    use inference_engine::ranker_model_metadata::RankerModelMetadata;
    use inference_engine::util::{get_top_k_desc_scores};
    use crate::helper::get_unique_user_and_first_timestamp;

    #[tokio::test(flavor = "multi_thread")]
    pub async fn test_analysis() ->  Result<(), Box<dyn std::error::Error + Send + Sync>>{

        let harness = TestHarness::new().await;

        let query_model_version = Some(VersionChoice::Version(1));

        // change these for the model choices and output directory to write stats to
        let ranker_model_version = Some(VersionChoice::Version(1));
        let output_base_dir = get_bin_dir().unwrap().to_string_lossy().into_owned();
        let summary_output_dir = format!("{}/post_training_analysis", output_base_dir.trim_end_matches('/'));
        let _ = recreate_directory(summary_output_dir.as_str()).unwrap();
        let parquet_output_dir = format!("{}/parquet_metrics",  output_base_dir.trim_end_matches('/'));
        let _ = recreate_directory(parquet_output_dir.as_str()).unwrap();


        // calculate @k metrics for retrieval and ranking for given model versions
        let _ = calc_metrics_at_k(&harness, query_model_version.clone(), ranker_model_version.clone(),
            &summary_output_dir, &parquet_output_dir).await;

        Ok(())
        // harness destructs when goes out of scope when method frame is done
    }

    async fn calc_metrics_at_k(harness: &TestHarness, query_model_version:Option<VersionChoice>,
        ranker_model_version:Option<VersionChoice>,
        summary_output_dir: &str, parquet_output_dir: &str) ->  Result<(), Box<dyn std::error::Error + Send + Sync>> {

        /*let ranker_version_num: i64 = match ranker_model_version {
            Some(VersionChoice::Version(v)) => v,
            _ => 1, // Default fallback version if None
        };*/

        let ranker_metadata = harness.orchestrator.get_or_fetch_ranker_metadata(
            ranker_model_version.clone()
        ).expect("error fetching ranker metadata");
        let ranker_batch_size = ranker_metadata.batch_size;
        let num_candidates = ranker_metadata.num_candidates;

        // calculate top_k=20 for retrieval then ranker

        let ranker_metadata =
            harness.orchestrator.get_or_fetch_ranker_metadata(ranker_model_version.clone())?;
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

        //polars is already spinning up threads to use all cores available, so run these
        //   sequentially rather than in parallel.

        let _ = calc_and_write(
            &harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "stratified_metrics.json".as_ref(),
            "metrics".as_ref(),
            _metrics
        ).await;

        let _ = calc_and_write(
            &harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "coverage.json".as_ref(),
            "coverage".as_ref(),
            _coverage_and_gini
        ).await;

        let _ = calc_and_write(
            &harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "popularity_bias.json".as_ref(),
            "popularity_bias".as_ref(),
            _popularity_bias
        ).await;

        let _ = calc_and_write(
            &harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "interlist_diversity.json".as_ref(),
            "interlist_diversity".as_ref(),
            _interlist_diversity
        ).await;

        let _ = calc_and_write(
            &harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "intralist_diversity.json".as_ref(),
            "intralist_diversity".as_ref(),
            _intralist_diversity
        ).await;

        /*
        // only relevant for retrieval, but might be nice to have it here too along with umpa and tsne plots
        let _ = calc_and_write(
            &harness,
            &ranker_metadata,
            num_candidates,
            top_k,
            &user_gt_counts,
            &user_ids,
            &candidate_movie_ids, //num_candidates
            &top_k_movie_ids, //top_k
            &ranker_movie_ids,
            &parquet_output_dir,
            &summary_output_dir,
            "embedding_hubness.json".as_ref(),
            "embedding_hubness".as_ref(),
            _embedding_hubness
        ).await;
        */

        //TODO:  analyze the funnel from num_candidates to top_k
        //TODO:  analyze whether the ranker ranking improves upon the retrieval for same top_k

        Ok(())
    }

    async fn calc_and_write(
        harness: &TestHarness,
        ranker_metadata: &Arc<RankerModelMetadata>,
        num_candidates: usize,
        top_k: usize,
        user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...] for tier being movie_tiers 0, 1, 2
        user_ids: &[i32],            // shape: (n_users)
        candidate_movie_ids: &[i32],           // shape: (n_users * top_k)   these are the retrieved or ranked movie_ids
        top_k_movie_ids: &[i32],           // shape: (n_users * top_k)   these are the retrieved or ranked movie_ids
        ranker_movie_ids: &[i32],         // shape: (n_users * top_k)
        parquet_output_dir: &str,
        summary_output_dir: &str,
        output_json_file_name: &str,
        keyword: &str,
        metric_func: impl Fn(usize, usize, &TestHarness, &LazyFrame, &[i32],
            &[i32],  &str, &str, &str) -> PolarsResult<(HashMap<String, f64>, Vec<String>)>
       ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

        let tag1 = format!("retrieval_ann_{}", num_candidates);
        let result1 = //tokio::task::spawn_blocking(move || {
            metric_func(
                num_candidates,
                ranker_metadata.num_catalog_movies,
                harness,
                &user_gt_counts,
                &user_ids,
                &candidate_movie_ids,
                &parquet_output_dir,
                &summary_output_dir,
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
                &user_gt_counts,
                &user_ids,
                &top_k_movie_ids,
                &parquet_output_dir,
                &summary_output_dir,
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
                &user_gt_counts,
                &user_ids,
                &ranker_movie_ids,
                &parquet_output_dir,
                &summary_output_dir,
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

        Ok(())
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

            let gini_power = *agg_res.get(&format!("gini_at_{}_user_tier_2_{}", k, tag)).unwrap_or(&1.0);
            let gini_light = *agg_res.get(&format!("gini_at_{}_user_tier_0_{}", k, tag)).unwrap_or(&1.0);

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
        catalog_size: usize,
        harness: &TestHarness,
        user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...] for tier being movie_tiers 0, 1, 2
        user_ids: &[i32],            // shape: (n_users)
        neighbors: &[i32],           // shape: (n_users * k)   these are the retrieved or ranked movie_ids
        parquet_output_dir: &str,
        summary_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {
        /*

         */
        Ok((HashMap::new(), Vec::new()))
    }

    pub fn _intralist_diversity(
        k: usize,                    // either num_candidates or top_k
        catalog_size: usize,
        harness: &TestHarness,
        user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...] for tier being movie_tiers 0, 1, 2
        user_ids: &[i32],            // shape: (n_users)
        neighbors: &[i32],           // shape: (n_users * k)   these are the retrieved or ranked movie_ids
        parquet_output_dir: &str,
        summary_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {
        Ok((HashMap::new(), Vec::new()))
    }

    pub fn _interlist_diversity(
        k: usize,                     // either num_candidates or top_k
        catalog_size: usize,
        harness: &TestHarness,
        user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...] for tier being movie_tiers 0, 1, 2
        user_ids: &[i32],            // shape: (n_users)
        neighbors: &[i32],           // shape: (n_users * k)   these are the retrieved or ranked movie_ids
        parquet_output_dir: &str,
        summary_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {
        Ok((HashMap::new(), Vec::new()))
    }

    /// Evaluates Retrieval or Ranker output and returns a tuple of (Metrics Dictionary, Conclusions List)
    fn _metrics(
        k: usize,                    // either num_candidates or top_k
        catalog_size: usize,
        harness: &TestHarness,
        user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...]
        user_ids: &[i32],            // shape: (n_users)
        neighbors: &[i32],           // shape: (n_users * k)
        parquet_output_dir: &str,
        #[allow(dead_code)]
        summary_output_dir: &str,
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

        // Join with truth datasets
        let joined_lf = retrieval_df
            .join_builder()
            .with(harness.pos_test_df.clone())
            .left_on([col("user_id"), col("movie_id")])
            .right_on([col("user_id"), col("movie_id")])
            .how(JoinType::Left)
            .finish()
            .left_join(harness.movie_tiers_df.clone(), col("movie_id"), col("movie_id"));

        // Construct Aggregation Expressions
        let is_hit = col("rating").is_not_null();
        let dcg_expr = lit(1.0) / (col("rank").cast(DataType::Float64) + lit(1.0)).log(lit(2.0));

        let mut agg_exprs = vec![
            is_hit.clone().sum().alias("hits_global"),
            dcg_expr.clone().filter(is_hit.clone()).sum().alias("dcg_global"),
        ];

        for t in 0..=2 {
            let tier_hit = is_hit.clone().and(col("movie_tier").eq(lit(t)));
            agg_exprs.push(tier_hit.clone().sum().alias(&format!("hits_tier_{}", t)));
            agg_exprs.push(dcg_expr.clone().filter(tier_hit).sum().alias(&format!("dcg_tier_{}", t)));
        }

        // Group by User ID and Apply Metrics
        let metrics_lf = joined_lf
            .group_by([col("user_id")])
            .agg(agg_exprs)
            .left_join(harness.user_tiers_df.clone(), col("user_id"), col("user_id"))
            .inner_join(user_gt_counts.clone(), col("user_id"), col("user_id"));

        // Extract IDCG cache logic into a map_elements closure
        let idcg_cache_clone = idcg_cache.clone();
        let apply_idcg = move |col: Column| -> PolarsResult<Column> {
            // Extract the Series from the Column
            let s = col.as_materialized_series();
            // Downcast to u32
            let ca = s.u32()?;
            let out: Float64Chunked = ca.iter().map(|opt_val| {
                // u32 can't be negative, so just cast directly to usize
                let count = opt_val.unwrap_or(0) as usize;
                // Wrap the f64 in Some() so collect() can build the ChunkedArray
                Some(idcg_cache_clone[count.min(k)])
            }).collect();
            // Return a Column
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
                .alias("expected_random_ndcg")
        ];

        // Add Movie Tier specific derivations
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
                    .alias(&format!("recall_tier_{}", t))
            );

            col_exprs.push(
                (col(&format!("hits_tier_{}", t)).cast(DataType::Float64) / lit(k as f64))
                    .alias(&format!("precision_tier_{}", t))
            );

            // PHASE 2: Add dependent calculations to ndcg_exprs
            ndcg_exprs.push(
                when(col(&idcg_col).is_not_null())
                    .then(col(&format!("dcg_tier_{}", t)) / col(&idcg_col))
                    .otherwise(lit(Null{}))
                    .fill_nan(lit(0.0))
                    .alias(&format!("ndcg_tier_{}", t))
            );
        }

        // Apply calculations in two sequential steps
        let mut metrics_df = metrics_lf
            .with_columns(col_exprs)
            .with_columns([
                (col("dcg_global") / col("idcg_global")).fill_nan(lit(0.0)).alias("ndcg_global"),
                (col("total_positives").cast(DataType::Float64) * lit(expected_random_dcg_part1) / col("idcg_global"))
                    .alias("expected_random_ndcg")
            ])
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

        agg_res.insert(format!("recall_at_{}_mean_random", k), get_col_mean(&metrics_df, "expected_random_recall"));
        agg_res.insert(format!("precision_at_{}_mean_random", k), get_col_mean(&metrics_df, "expected_random_precision"));
        agg_res.insert(format!("ndcg_at_{}_mean_random", k), get_col_mean(&metrics_df, "expected_random_ndcg"));

        // User Tier Extractions
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

            // Movie Tier Extractions (using non-null valid data rows)
            let recall_col = format!("recall_tier_{}", t);
            let n_movie_samples = metrics_df.column(&recall_col)?.is_not_null().sum().unwrap_or(0);

            agg_res.insert(format!("n_samples_movie_tier_{}", t), n_movie_samples as f64);
            agg_res.insert(format!("recall_at_{}_mean_movie_tier_{}", k, t), get_col_mean(&metrics_df, &recall_col));
            agg_res.insert(format!("precision_at_{}_mean_movie_tier_{}", k, t), get_col_mean(&metrics_df, &format!("precision_tier_{}", t)));
            agg_res.insert(format!("ndcg_at_{}_mean_movie_tier_{}", k, t), get_col_mean(&metrics_df, &format!("ndcg_tier_{}", t)));
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

        let recall_t0 = *agg_res.get(&format!("recall_at_{}_mean_user_tier_0", k)).unwrap_or(&0.0);
        let recall_t2 = *agg_res.get(&format!("recall_at_{}_mean_user_tier_2", k)).unwrap_or(&0.0);

        if recall_t2 > recall_t0 * 1.5 {
            conclusions.push(format!("EXCELLENT HISTORY UTILIZATION: Power users (Tier 2 Recall: {:.1}%) perform massively better than light users (Tier 0 Recall: {:.1}%). The model successfully translates rich interaction histories into highly relevant slates.", recall_t2 * 100.0, recall_t0 * 100.0));
        } else if recall_t2 > recall_t0 {
            conclusions.push("MODERATE HISTORY UTILIZATION: Power users see slightly better recall than light users.".to_string());
        } else {
            conclusions.push("HISTORY NEGLECT WARNING: Power users perform worse or equal to light users, indicating the model struggles to parse dense interaction histories.".to_string());
        }

        let ndcg_t0 = *agg_res.get(&format!("ndcg_at_{}_mean_user_tier_0", k)).unwrap_or(&0.0);
        let ndcg_t2 = *agg_res.get(&format!("ndcg_at_{}_mean_user_tier_2", k)).unwrap_or(&0.0);

        if ndcg_t2 > ndcg_t0 {
            conclusions.push(format!("HEALTHY RANKING STRATIFICATION: NDCG scales positively with user tier (Tier 2: {:.2}% vs Tier 0: {:.2}%). The model not only retrieves the right items for power users but places them higher in the slate.", ndcg_t2 * 100.0, ndcg_t0 * 100.0));
        } else {
            conclusions.push("POOR RANKING STRATIFICATION: NDCG does not improve for power users, suggesting the model retrieves relevant items but places them randomly within the top K.".to_string());
        }

        let ndcg_mt0 = *agg_res.get(&format!("ndcg_at_{}_mean_movie_tier_0", k)).unwrap_or(&0.0);
        let ndcg_mt2 = *agg_res.get(&format!("ndcg_at_{}_mean_movie_tier_2", k)).unwrap_or(&0.0);

        if ndcg_mt2 > ndcg_mt0 * 1.5 {
            conclusions.push(format!("POPULARITY BIAS DETECTED: Ranking quality heavily favors head/popular movies (Tier 2 NDCG: {:.2}%) compared to long-tail/niche items (Tier 0 NDCG: {:.2}%).", ndcg_mt2 * 100.0, ndcg_mt0 * 100.0));
        } else if ndcg_mt0 > ndcg_mt2 {
            conclusions.push(format!("STRONG LONG-TAIL PERFORMANCE: Model successfully surfaces and ranks niche/long-tail items (Tier 0 NDCG: {:.2}%) effectively relative to popular items (Tier 2 NDCG: {:.2}%).", ndcg_mt0 * 100.0, ndcg_mt2 * 100.0));
        } else {
            conclusions.push(format!("BALANCED ITEM TIER RANKING: NDCG performance remains consistent across movie tiers (Tier 0: {:.2}%, Tier 2: {:.2}%), indicating stable retrieval across popular and long-tail catalogs.", ndcg_mt0 * 100.0, ndcg_mt2 * 100.0));
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
}