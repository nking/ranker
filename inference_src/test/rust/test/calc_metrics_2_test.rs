#[cfg(test)]
mod calc_metrics_2_tests {
    use tokio::sync::oneshot;
    use tokio::task::JoinHandle;

    struct TestServerGuard {
        tx_shutdown: Option<oneshot::Sender<()>>,
        server_handle: Option<JoinHandle<()>>,
    }

    impl Drop for TestServerGuard {
        fn drop(&mut self) {
            // Trigger the shutdown signal
            if let Some(tx) = self.tx_shutdown.take() {
                let _ = tx.send(());
            }

            // Safely block and join the background server thread using block_in_place
            if let Some(handle) = self.server_handle.take() {
                let _ = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(handle)
                });
            }
        }
    }

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
        include!("helper_users.rs");
    }

    use helper::{get_project_dir, load_and_concat_parquet, get_unique_user_and_first_timestamp,
        get_user_movie_tier_map};
    use std::path::PathBuf;
    use std::collections::{HashMap, HashSet};
    use polars::chunked_array::ops::IsLastDistinct;
    use polars::error::PolarsResult;
    use polars::prelude::{ChunkCompareEq, DataFrame, LazyFrame};
    use tonic::{Request, Response};
    use inference_engine::app_config::AppConfig;
    use inference_engine::app_runner::AppRunner;
    use inference_engine::model_client::RankerModelClient;
    use inference_engine::movie_tiers::load_from_file;
    use inference_engine::pb::recommender_service_client::RecommenderServiceClient;
    use inference_engine::pb::{RankedMovies, UsersRequest};
    use inference_engine::ranker_model_metadata::RankerModelMetadata;
    use inference_engine::user_db::UserDb;
    use crate::calc_metrics_2_tests::helper::get_config_json_uri;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    pub async fn test_calc_test_metrics() {

        // ======= setup server ======================
        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        // the default confg is for the batch ranker model, so change to the single inference model:
        let ranker_metadat_uri = config.ranker_metadata_uri.clone();
        let single_uri = ranker_metadat_uri.replace("batch", "single");

        let ranker_metadata = RankerModelMetadata::load_from_file(&single_uri).unwrap();

        let client = RankerModelClient::new(config.ranker_uri.clone(), ranker_metadata.clone()).await;

        let _top_k = config.top_k;
        let _user_db_path: PathBuf = config.user_db_path.clone();
        let _persisted_index_path : PathBuf = config.persisted_index_path.clone();

        let _max_history = ranker_metadata.max_history;
        let _num_candidates = ranker_metadata.num_candidates;
        let _num_catalog_users = ranker_metadata.num_catalog_users;

        let runner = AppRunner::new(config.to_owned());

        let (tx_shutdown, rx_shutdown) = oneshot::channel::<()>();
        let (tx_addr, rx_addr) = oneshot::channel::<std::net::SocketAddr>();
        // Spawn the server in a background Tokio task
        let server_handle = tokio::spawn(async move {
            let shutdown_future = async {
                rx_shutdown.await.ok();
            };
            runner.run(shutdown_future, Some(tx_addr)).await.expect("Server crashed");
        });
        let addr = rx_addr.await.expect("Failed to receive server address");
        // Initialize the RAII Guard.
        // It will automatically trigger shutdown and join if the test finishes or panics.
        let _server_guard = TestServerGuard {
            tx_shutdown: Some(tx_shutdown),
            server_handle: Some(server_handle),
        };

        let endpoint = format!("http://{}", addr);

        // ===== run tests ==========

        if true {
            calc_tier_stratified_metrics(config.clone(), endpoint).await;
        }


        // server is shutdown by the guard when this method is out of scope

    }

    async fn calc_tier_stratified_metrics(config: AppConfig,  endpoint: String) -> Result<(), Box<dyn std::error::Error>> {

        // get the movie_tiers file
        let movie_tiers : HashMap<i32, i32> = load_from_file(&config.movie_tiers_path)?;

        // get the ground truth
        let proj_dir : String = get_project_dir().unwrap().to_string_lossy().into_owned();
        let test_liked = vec![
            format!("{}/src/test/resources/data/ratings_test_liked.parquet", proj_dir)];

        let (movie_tier_df_map, user_ids, timestamps) : (HashMap<i32, DataFrame>,Vec<i32>, Vec<i64> )
               = get_user_datastructures(&[&test_liked[0]], &movie_tiers);

        let user_db : UserDb = UserDb::new(&config.user_db_path).expect("Failed to initialize UserDb from binary path");

        let ranker_metadata = RankerModelMetadata::load_from_file(&config.ranker_metadata_uri).unwrap();

        let ranker_batch_size = ranker_metadata.batch_size;

        let client = RecommenderServiceClient::connect(endpoint.clone()).await?;

        let mut ndcg_tiers : Vec<f32> = vec![0.; 3];
        let mut recall_tiers : Vec<f32> = vec![0.; 3];
        let mut count_tiers : Vec<i32> = vec![0; 3];

        for (chunk_users, chunk_timestamps) in user_ids.chunks(ranker_batch_size).zip(timestamps.chunks(ranker_batch_size)) {
            // chunks are &[i32] slices

            let option_tonic_request :  Option<Request<UsersRequest>> = user_db.get_request(chunk_users, chunk_timestamps);
            let tonic_req = option_tonic_request.ok_or("request not found for chunk")?;
            //let users_req = tonic_req.into_inner();

            let mut active_client = client.clone();

            let results: Result<Response<RankedMovies>, tonic::Status> =
                active_client.predict(tonic_req).await;
            let ranked_movies = results.unwrap().into_inner();

            // calc metrics by tier and add to ndcg_tiers and recal_tiers
            sum_metrics(&movie_tier_df_map, &ranked_movies,
                &mut ndcg_tiers, &mut recall_tiers, &mut count_tiers);

        }


        Ok(())
    }

    pub fn sum_metrics(
        movie_tier_df_map_ref: &HashMap<i32, DataFrame>,
        ranked_movies_ref: &RankedMovies,
        ndcg_sum_tiers_ref: &mut [f32],
        recall_sum_tiers_ref: &mut [f32],
        count_tiers_ref: &mut [i32],
    ) -> Result<(), Box<dyn std::error::Error>> {

        // ====================================================================
        // PRE-COMPUTATION (O(N) Time)
        // Build a map of user_id -> HashSet<movie_id> for each tier ONCE.
        // ====================================================================
        let mut tier_user_gt_maps: Vec<HashMap<i32, HashSet<i32>>> = vec![HashMap::new(); 3];

        for tier in 0i32..3i32 {
            let Some(df_tier) = movie_tier_df_map_ref.get(&tier) else {
                continue;
            };

            let user_ca = df_tier.column("user_id")?.i32()?;
            let movie_ca = df_tier.column("movie_id")?.i32()?; // Ensure this matches your DF (movie_id vs movie_ids)

            for (u, m) in user_ca.into_no_null_iter().zip(movie_ca.into_no_null_iter()) {
                tier_user_gt_maps[tier as usize]
                    .entry(u)
                    .or_default()
                    .insert(m);
            }
        }

        // ====================================================================
        // 2. METRIC AGGREGATION
        // ====================================================================
        let k = ranked_movies_ref.num_candidates as usize;

        for (&user_id, predicted_movies) in ranked_movies_ref
            .user_ids
            .iter()
            .zip(ranked_movies_ref.movie_ids.chunks_exact(k))
        {
            for tier in 0..3 {
                // O(1) lookup to get the pre-computed set of movies for this user/tier combo
                let Some(gt_set) = tier_user_gt_maps[tier].get(&user_id) else {
                    continue; // User has no target movies in this tier
                };

                if gt_set.is_empty() {
                    continue;
                }

                // --- Recall ---
                let hits = predicted_movies.iter().filter(|id| gt_set.contains(id)).count();
                let max_possible_hits = std::cmp::min(gt_set.len(), predicted_movies.len());
                let recall = hits as f32 / max_possible_hits as f32;

                // --- NDCG ---
                let mut dcg: f32 = 0.0;
                for (rank_idx, movie_id) in predicted_movies.iter().enumerate() {
                    if gt_set.contains(movie_id) {
                        // FIX: + 2.0 prevents log2(1.0) == 0.0 division by zero panic
                        dcg += 1.0 / (rank_idx as f32 + 2.0).log2();
                    }
                }

                let mut idcg: f32 = 0.0;
                for rank_idx in 0..max_possible_hits {
                    idcg += 1.0 / (rank_idx as f32 + 2.0).log2();
                }

                let ndcg: f32 = if idcg == 0.0 { 0.0 } else { dcg / idcg };

                // --- Accumulate ---
                recall_sum_tiers_ref[tier] += recall;
                ndcg_sum_tiers_ref[tier] += ndcg;
                count_tiers_ref[tier] += 1;
            }
        }

        Ok(())
    }

    pub fn get_user_datastructures(
        ratings_uris: &[&str],
        movie_tiers: &HashMap<i32, i32>
    ) -> Result<(Vec<HashMap<i32, HashSet<i32>>>, Vec<i32>, Vec<i64>), Box<dyn std::error::Error>> {

        let df_gt = load_and_concat_parquet(ratings_uris)?;

        let movie_tier_maps_vec = get_user_movie_tier_map(df_gt.clone(), movie_tiers)?;

        let (user_ids, timestamps) = get_unique_user_and_first_timestamp(df_gt)?;

        Ok((movie_tier_maps_vec, user_ids, timestamps))
    }
}