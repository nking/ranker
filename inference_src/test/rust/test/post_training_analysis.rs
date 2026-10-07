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
    summary_output_dir: String,
    parquet_output_dir: String,
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

        // change this to another directory if wanted
        let output_base_dir = get_bin_dir().unwrap().to_string_lossy().into_owned();
        let summary_output_dir = format!("{}/post_training_analysis",
            output_base_dir.trim_end_matches('/'));
        let _ = recreate_directory(summary_output_dir.as_str()).unwrap();
        let parquet_output_dir = format!("{}/parquet_metrics",
            output_base_dir.trim_end_matches('/'));
        let _ = recreate_directory(parquet_output_dir.as_str()).unwrap();

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
            summary_output_dir: summary_output_dir,
            parquet_output_dir: parquet_output_dir,
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
    use inference_engine::pb::{ApproxNearestNeighborsResponse, RankedMovies, UsersRankOnlyRequest, UsersRequest};
    use inference_engine::pb::recommender_service_server::RecommenderService;
    use crate::helper::get_unique_user_and_first_timestamp;

    #[tokio::test(flavor = "multi_thread")]
    pub async fn test_analysis() ->  Result<(), Box<dyn std::error::Error + Send + Sync>>{

        // change these for the model choices and output directory to write stats to
        let query_model_version = Some(VersionChoice::Version(1));
        let ranker_model_version = Some(VersionChoice::Version(1));

        let harness = TestHarness::new().await;

        let _ = calc_metrics_at_k(&harness, query_model_version.clone(), ranker_model_version.clone()).await;


        Ok(())
        // harness destructs when goes out of scope when method frame is done
    }

    async fn calc_metrics_at_k(harness: &TestHarness, query_model_version:Option<VersionChoice>,
        ranker_model_version:Option<VersionChoice>) ->  Result<(), Box<dyn std::error::Error + Send + Sync>> {

        // calculate top_k=20 for retrieval then ranker

        let ranker_metatadata =
            harness.orchestrator.get_or_fetch_ranker_metadata(ranker_model_version.clone())?;
        let top_k = harness.orchestrator.top_k;

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
        )?;

        // =========== RETRIEVAL ===============================

        edit to retrieva all and keep top 20 for metrics but use all 100 for rank_only request

        // no batch_size constraints for the query model or ANN requests
        let tonic_request: Request<UsersRequest>  =
            harness.orchestrator.get_users_request(&user_ids, &timestamps,
                query_model_version.clone(), ranker_model_version.clone()).await?;
        let mut users_request : UsersRequest = tonic_request.into_inner();
        users_request.k = Some(top_k as u32);
        let ann_reqs = Request::new(users_request.clone());
        let ann_res: ApproxNearestNeighborsResponse = harness.orchestrator
            ._approx_nearest_neighbors(ann_reqs).await?.into_inner();
        // length: n_users * k
        let ann_movie_ids: Vec<i32> = ann_res.candidate_ids;

        let tag1 = "retrieval_ann";

        let result1 = //tokio::task::spawn_blocking(move || {
            metrics(
                top_k,
                ranker_metatadata.num_catalog_movies,
                &harness.pos_test_df,
                &harness.movie_tiers_df,
                &harness.user_tiers_df,
                &user_gt_counts,
                &user_ids,
                &ann_movie_ids,
                &harness.parquet_output_dir,
                tag1
            );
        //}).await.expect("Spawn blocking panicked")?;

        let (res1, concl1) = result1.expect("metrics failed for tag1");

        println!("have results1");

        // =========== RANKER ===============================

        let ranker_version_num: i64 = match ranker_model_version {
            Some(VersionChoice::Version(v)) => v,
            _ => 1, // Default fallback version if None
        };
        let query_version_num: i64 = match query_model_version {
            Some(VersionChoice::Version(v)) => v,
            _ => 1, // Default fallback version if None
        };

        let n_users = user_ids.len();
        let ranker_batch_size =
            harness.orchestrator.get_ranker_batch_size(ranker_model_version.clone()).expect("cannot get ranker batch_size");

        let mut ranker_movie_ids: Vec<i32> = Vec::with_capacity(ann_movie_ids.len());

        // put through ranker
        // a request should be batch_size
        for i0 in (0..n_users).step_by(ranker_batch_size) {

            let i1 = std::cmp::min(i0 + ranker_batch_size, n_users);

            let j0 = i0 * top_k;
            let j1 = i1 * top_k;

            let rank_req = Request::new(UsersRankOnlyRequest {
                user_ids: user_ids[i0..i1].to_vec(),
                timestamps: timestamps[i0..i1].to_vec(),
                candidate_ids: ann_movie_ids[j0..j1].to_vec(),
                query_model_version: query_version_num,
                ranker_model_version: ranker_version_num
            });

            println!("about to rank movie_ids for users: {}-{}", i0, i1);

            let ranked_movies: RankedMovies = match harness.orchestrator.ranks_only(rank_req).await {
                Ok(response) => response.into_inner(),
                Err(e) => {
                    eprintln!("\n[ERROR] Ranker request failed for user batch indices {} to {}", i0, i1);
                    eprintln!("   - Number of users in batch: {}", i1 - i0);
                    eprintln!("   - Number of candidates sent: {}", j1 - j0);
                    eprintln!("   - Error Details: {:#?}", e);

                    // Bubble the error up to the function's return type
                    return Err(e.into());
                }
            };

            ranker_movie_ids.extend(ranked_movies.movie_ids);
        }

        let tag2 = "ranker";
        let result2 = //tokio::task::spawn_blocking(move || {
            metrics(
                top_k,
                ranker_metatadata.num_catalog_movies,
                &harness.pos_test_df,
                &harness.movie_tiers_df,
                &harness.user_tiers_df,
                &user_gt_counts,
                &user_ids,
                &ranker_movie_ids,
                &harness.parquet_output_dir,
                tag2
            );
        //}).await.expect("Spawn blocking panicked")?;

        println!("have results2");

        let (res2, concl2) = result2.expect("metrics failed for tag2");

        let agg_res = serde_json::json!(
            {
                format!("{}_at_{}", tag1, top_k): {
                    "metrics": res1,
                    "automated_conclusions": concl1
                },
                 format!("{}_at_{}", tag2, top_k): {
                    "metrics": res2,
                    "automated_conclusions": concl2
                }
            }
        );

        let output_file_path = Path::new(&harness.summary_output_dir).join(format!("stratified_metrics_top_{}.json", top_k));
        let file = File::create(output_file_path.clone())?;
        serde_json::to_writer_pretty(file, &agg_res)?;

        println!("{}", format!("wrote to {:?}", output_file_path));

        Ok(())
    }

    /// Evaluates Retrieval or Ranker output and returns a tuple of (Metrics Dictionary, Conclusions List)
    pub fn metrics(
        top_k: usize,
        catalog_size: usize,
        pos_test_df: &LazyFrame,     // [user_id, movie_id, rating, timestamp]
        movie_tiers_df: &LazyFrame,  // [movie_id, movie_tier]
        user_tiers_df: &LazyFrame,   // [user_id, user_tier]
        user_gt_counts: &LazyFrame,  // [user_id, total_positives, gt_pos_tier_0, ...]
        user_ids: &[i32],            // shape: (n_users)
        neighbors: &[i32],           // shape: (n_users * top_k)
        parquet_output_dir: &str,
        tag: &str,
    ) -> PolarsResult<(HashMap<String, f64>, Vec<String>)> {

        let n_users = user_ids.len();

        // Build Base Metrics Data Structures (Expected baselines & IDCG Cache)
        let expected_random_recall = (top_k as f64) / (catalog_size as f64);
        let idcg_cache = build_idcg_cache(top_k);
        let expected_random_dcg_part1 = idcg_cache[top_k] / (catalog_size as f64);

        // Efficiently construct the flat retrieval DataFrame from vectors
        let repeated_users: Vec<i32> = user_ids.iter()
            .flat_map(|&u| std::iter::repeat(u).take(top_k))
            .collect();

        let ranks: Vec<i32> = (0..n_users)
            .flat_map(|_| 1..=top_k as i32)
            .collect();

        let retrieval_df = df!(
            "user_id" => repeated_users,
            "movie_id" => neighbors,
            "rank" => ranks,
        )?.lazy();

        // Join with truth datasets
        let joined_lf = retrieval_df
            .join_builder()
            .with(pos_test_df.clone())
            .left_on([col("user_id"), col("movie_id")])
            .right_on([col("user_id"), col("movie_id")])
            .how(JoinType::Left)
            .finish()
            .left_join(movie_tiers_df.clone(), col("movie_id"), col("movie_id"));

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
            .left_join(user_tiers_df.clone(), col("user_id"), col("user_id"))
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
                Some(idcg_cache_clone[count.min(top_k)])
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
            (col("hits_global").cast(DataType::Float64) / lit(top_k as f64)).alias("precision_global"),

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
                (col(&format!("hits_tier_{}", t)).cast(DataType::Float64) / lit(top_k as f64))
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
        let parquet_path = Path::new(parquet_output_dir).join(format!("stratified_metrics_top_{}_{}.parquet", top_k, tag));
        let mut file = File::create(&parquet_path)?;
        ParquetWriter::new(&mut file).finish(&mut metrics_df)?;

        // Extract Aggregated Results into HashMap
        let mut agg_res = HashMap::new();
        agg_res.insert("n_samples_global".to_string(), metrics_df.height() as f64);

        agg_res.insert(format!("recall_at_{}_mean", top_k), get_col_mean(&metrics_df, "recall_global"));
        agg_res.insert(format!("recall_at_{}_std", top_k), get_col_std(&metrics_df, "recall_global"));
        agg_res.insert(format!("precision_at_{}_mean", top_k), get_col_mean(&metrics_df, "precision_global"));
        agg_res.insert(format!("precision_at_{}_std", top_k), get_col_std(&metrics_df, "precision_global"));
        agg_res.insert(format!("ndcg_at_{}_mean", top_k), get_col_mean(&metrics_df, "ndcg_global"));
        agg_res.insert(format!("ndcg_at_{}_std", top_k), get_col_std(&metrics_df, "ndcg_global"));

        agg_res.insert(format!("recall_at_{}_mean_random", top_k), get_col_mean(&metrics_df, "expected_random_recall"));
        agg_res.insert(format!("precision_at_{}_mean_random", top_k), get_col_mean(&metrics_df, "expected_random_precision"));
        agg_res.insert(format!("ndcg_at_{}_mean_random", top_k), get_col_mean(&metrics_df, "expected_random_ndcg"));

        // User Tier Extractions
        for t in 0..=2 {
            let mask = metrics_df
                .column("user_tier")?
                .as_materialized_series()
                .i32()?
                .equal(t as i32); // Notice we also remove the `?` at the end here!

            let filtered_df = metrics_df.filter(&mask)?;

            agg_res.insert(format!("n_samples_user_tier_{}", t), filtered_df.height() as f64);
            agg_res.insert(format!("recall_at_{}_mean_user_tier_{}", top_k, t), get_col_mean(&filtered_df, "recall_global"));
            agg_res.insert(format!("recall_at_{}_std_user_tier_{}", top_k, t), get_col_std(&filtered_df, "recall_global"));
            agg_res.insert(format!("precision_at_{}_mean_user_tier_{}", top_k, t), get_col_mean(&filtered_df, "precision_global"));
            agg_res.insert(format!("precision_at_{}_std_user_tier_{}", top_k, t), get_col_std(&filtered_df, "precision_global"));
            agg_res.insert(format!("ndcg_at_{}_mean_user_tier_{}", top_k, t), get_col_mean(&filtered_df, "ndcg_global"));
            agg_res.insert(format!("ndcg_at_{}_std_user_tier_{}", top_k, t), get_col_std(&filtered_df, "ndcg_global"));

            // Movie Tier Extractions (using non-null valid data rows)
            let recall_col = format!("recall_tier_{}", t);
            let n_movie_samples = metrics_df.column(&recall_col)?.is_not_null().sum().unwrap_or(0);

            agg_res.insert(format!("n_samples_movie_tier_{}", t), n_movie_samples as f64);
            agg_res.insert(format!("recall_at_{}_mean_movie_tier_{}", top_k, t), get_col_mean(&metrics_df, &recall_col));
            agg_res.insert(format!("precision_at_{}_mean_movie_tier_{}", top_k, t), get_col_mean(&metrics_df, &format!("precision_tier_{}", t)));
            agg_res.insert(format!("ndcg_at_{}_mean_movie_tier_{}", top_k, t), get_col_mean(&metrics_df, &format!("ndcg_tier_{}", t)));
        }

        //  Generate Automated Conclusions
        let mut conclusions = Vec::new();

        let global_recall = *agg_res.get(&format!("recall_at_{}_mean", top_k)).unwrap_or(&0.0);
        let random_recall = *agg_res.get(&format!("recall_at_{}_mean_random", top_k)).unwrap_or(&0.0);
        let global_precision = *agg_res.get(&format!("precision_at_{}_mean", top_k)).unwrap_or(&0.0);
        let random_precision = *agg_res.get(&format!("precision_at_{}_mean_random", top_k)).unwrap_or(&0.0);

        if global_recall > random_recall * 3.0 {
            conclusions.push(format!("STRONG RECALL LIFT: Global Recall@{} ({:.1}%) heavily outperforms the random expected baseline ({:.1}%). The candidate generator is extracting meaningful semantic signal.", top_k, global_recall * 100.0, random_recall * 100.0));
        } else if global_recall > random_recall {
            conclusions.push(format!("MODERATE RECALL LIFT: Global Recall@{} ({:.1}%) beats the random baseline ({:.1}%), but there is room for improvement.", top_k, global_recall * 100.0, random_recall * 100.0));
        } else {
            conclusions.push("RECALL FAILURE: The model fails to outperform a random candidate generator on Recall.".to_string());
        }

        if global_precision > random_precision * 3.0 {
            conclusions.push(format!("STRONG PRECISION LIFT: Global Precision@{} ({:.2}%) heavily outperforms the random expected precision ({:.2}%), indicating high density of relevant items in the retrieved slates.", top_k, global_precision * 100.0, random_precision * 100.0));
        } else if global_precision > random_precision {
            conclusions.push(format!("MODERATE PRECISION LIFT: Global Precision@{} ({:.2}%) is better than random ({:.2}%), suggesting basic relevance filtering is functioning.", top_k, global_precision * 100.0, random_precision * 100.0));
        } else {
            conclusions.push("PRECISION FAILURE: The model retrieves slates with the same or worse precision than a completely random draw.".to_string());
        }

        let recall_t0 = *agg_res.get(&format!("recall_at_{}_mean_user_tier_0", top_k)).unwrap_or(&0.0);
        let recall_t2 = *agg_res.get(&format!("recall_at_{}_mean_user_tier_2", top_k)).unwrap_or(&0.0);

        if recall_t2 > recall_t0 * 1.5 {
            conclusions.push(format!("EXCELLENT HISTORY UTILIZATION: Power users (Tier 2 Recall: {:.1}%) perform massively better than light users (Tier 0 Recall: {:.1}%). The model successfully translates rich interaction histories into highly relevant slates.", recall_t2 * 100.0, recall_t0 * 100.0));
        } else if recall_t2 > recall_t0 {
            conclusions.push("MODERATE HISTORY UTILIZATION: Power users see slightly better recall than light users.".to_string());
        } else {
            conclusions.push("HISTORY NEGLECT WARNING: Power users perform worse or equal to light users, indicating the model struggles to parse dense interaction histories.".to_string());
        }

        let ndcg_t0 = *agg_res.get(&format!("ndcg_at_{}_mean_user_tier_0", top_k)).unwrap_or(&0.0);
        let ndcg_t2 = *agg_res.get(&format!("ndcg_at_{}_mean_user_tier_2", top_k)).unwrap_or(&0.0);

        if ndcg_t2 > ndcg_t0 {
            conclusions.push(format!("HEALTHY RANKING STRATIFICATION: NDCG scales positively with user tier (Tier 2: {:.2}% vs Tier 0: {:.2}%). The model not only retrieves the right items for power users but places them higher in the slate.", ndcg_t2 * 100.0, ndcg_t0 * 100.0));
        } else {
            conclusions.push("POOR RANKING STRATIFICATION: NDCG does not improve for power users, suggesting the model retrieves relevant items but places them randomly within the top K.".to_string());
        }

        let ndcg_mt0 = *agg_res.get(&format!("ndcg_at_{}_mean_movie_tier_0", top_k)).unwrap_or(&0.0);
        let ndcg_mt2 = *agg_res.get(&format!("ndcg_at_{}_mean_movie_tier_2", top_k)).unwrap_or(&0.0);

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