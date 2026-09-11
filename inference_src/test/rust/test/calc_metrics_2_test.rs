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
    use std::fs::File;
    use std::io::Write;
    use serde_json::{Value};
    use tonic::{Request, Response};
    use inference_engine::app_config::AppConfig;
    use inference_engine::app_runner::AppRunner;
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
            let proj_dir : String = get_project_dir().unwrap().to_string_lossy().into_owned();
            let test_ratings_paths = vec![
                format!("{}/src/test/resources/data/ratings_test_liked.parquet", proj_dir)];
            let output_path = format!("{}/bin/test_metrics.json", proj_dir);

            calc_tier_stratified_metrics(config.clone(), endpoint, test_ratings_paths, output_path).await.expect("Error: while calculating metrics");

        }


        // server is shutdown by the guard when this method is out of scope

    }

    async fn calc_tier_stratified_metrics(config: AppConfig, endpoint: String,
        test_ratings_paths: Vec<String>, output_path: String) -> Result<(), Box<dyn std::error::Error>> {

        // get the movie_tiers file.  key=movie_id, value=movie_tier
        let movie_tiers : HashMap<i32, i32> = load_from_file(&config.movie_tiers_path)?;

        let num_catalog_movies = movie_tiers.len();

        //movie_tier_vec_map: tier -> user_id -> movies
        let (movie_tier_vec_map, user_ids, timestamps) : (Vec<HashMap<i32, HashSet<i32>>>,Vec<i32>, Vec<i64> )
               = get_user_datastructures(&[&test_ratings_paths[0]], &movie_tiers)?;

        let user_db : UserDb = UserDb::new(&config.user_db_path).expect("Failed to initialize UserDb from binary path");

        let ranker_metadata = RankerModelMetadata::load_from_file(&config.ranker_metadata_uri).unwrap();

        let ranker_batch_size = ranker_metadata.batch_size;

        let client = RecommenderServiceClient::connect(endpoint.clone()).await?;

        let mut ndcg_tiers : Vec<f64> = vec![0.; 3];
        let mut recall_tiers : Vec<f64> = vec![0.; 3];
        let mut rand_ndcg_tiers : Vec<f64> = vec![0.; 3];
        let mut rand_recall_tiers : Vec<f64> = vec![0.; 3];
        let mut count_metric_tiers: Vec<i32> = vec![0; 3];
        let mut count_predicted_tiers : Vec<f64> = vec![0.; 3];

        let mut recommended_set : HashSet<i32> = HashSet::new();

        for (chunk_users, chunk_timestamps) in user_ids.chunks(ranker_batch_size).zip(timestamps.chunks(ranker_batch_size)) {
            // chunks are &[i32] slices

            let option_tonic_request :  Option<Request<UsersRequest>> = user_db.get_request(chunk_users, chunk_timestamps);
            let tonic_req = option_tonic_request.ok_or("request not found for chunk")?;
            //let users_req = tonic_req.into_inner();

            let mut active_client = client.clone();

            let results: Result<Response<RankedMovies>, tonic::Status> =
                active_client.predict(tonic_req).await;
            let ranked_movies = results.unwrap().into_inner();

            // calc metrics by tier and add to ndcg_tiers and recall_tiers
            sum_metrics(&movie_tier_vec_map, &movie_tiers, &ranked_movies,
                &mut ndcg_tiers, &mut recall_tiers, &mut rand_ndcg_tiers,
                &mut rand_recall_tiers, &mut count_metric_tiers,
                &mut count_predicted_tiers,
                &mut recommended_set
            )?;

        }

        // count the tier movies in the test dataset
        let test_tier_counts : Vec<i32> = count_tier_movies_in_test(movie_tier_vec_map);
        let movie_catalog_tier_counts : Vec<i32> = count_tier_movies_in_catalog(movie_tiers);

        // write to outpath : String as json file and print pretty here
        let tier_names = ["head", "torso", "tail"];
        let top_k = config.top_k;
        let mut results_map = serde_json::Map::new();
        for tier in 0..3 {
            let count = count_metric_tiers[tier];
            let mean_ndcg = if count == 0 {0.} else {ndcg_tiers[tier] / count as f64};
            let mean_recall = if count == 0 {0.} else {recall_tiers[tier] / count as f64};
            let mean_rand_ndcg = if count == 0 {0.} else {rand_ndcg_tiers[tier] / count as f64};
            let mean_rand_recall = if count == 0 {0.} else {rand_recall_tiers[tier] / count as f64};
            let mean_predicted_tiers = if count == 0 {0.} else {count_predicted_tiers[tier] / count as f64};
            let tier_name = tier_names[tier];

            results_map.insert(format!("ndcg_{}_{}", tier_name, top_k), Value::from(mean_ndcg));
            results_map.insert(format!("recall_{}_{}", tier_name, top_k), Value::from(mean_recall));
            results_map.insert(format!("random_ndcg_{}_{}", tier_name, top_k), Value::from(mean_rand_ndcg));
            results_map.insert(format!("random_recall_{}_{}", tier_name, top_k), Value::from(mean_rand_recall));
            results_map.insert(format!("count_users_{}_{}", tier_name, top_k), Value::from(count));

            results_map.insert(format!("count_test_{}", tier_name), Value::from(test_tier_counts[tier]));
            results_map.insert(format!("count_movie_catalog_{}", tier_name), Value::from(movie_catalog_tier_counts[tier]));

            results_map.insert(format!("frac_pred_tiers_{}_{}", tier_name, top_k), Value::from(mean_predicted_tiers));
        }

        let cat_cov = (recommended_set.len() as f32)/(num_catalog_movies as f32);
        results_map.insert(format!("catalog coverage"), Value::from(cat_cov));

        // Format as pretty-printed JSON string
        let pretty_json = serde_json::to_string_pretty(&results_map)?;

        // Print to stdout
        println!("{}", pretty_json);

        // Write to file
        let mut file = File::create(&output_path)?;
        file.write_all(pretty_json.as_bytes())?;

        Ok(())
    }

    fn count_tier_movies_in_catalog(movie_tiers : HashMap<i32, i32>) -> Vec<i32> {
        let mut counts : Vec<i32> = vec![0; 3];
        for (_u, t) in movie_tiers {
            counts[t as usize] += 1;
        }
        counts
    }

    fn count_tier_movies_in_test(movie_tier_vec_map: Vec<HashMap<i32, HashSet<i32>>>) -> Vec<i32> {

        let mut counts : Vec<i32> = vec![0; 3];
        let mut tier_set = std::collections::HashSet::new();
        for tier in 0..3 {
            for inner_set in movie_tier_vec_map[tier].values() {
                tier_set.extend(inner_set.iter().copied());
            }
            counts[tier] = tier_set.len() as i32;
            tier_set.clear();
        }
        counts
    }

    pub fn sum_metrics(
        movie_tier_vec_map_ref: &Vec<HashMap<i32, HashSet<i32>>>,
        movie_tiers_ref : &HashMap<i32, i32>,
        ranked_movies_ref: &RankedMovies,
        ndcg_sum_tiers_ref: &mut [f64],
        recall_sum_tiers_ref: &mut [f64],
        rand_ndcg_sum_tiers_ref: &mut [f64],
        rand_recall_sum_tiers_ref: &mut [f64],
        count_metric_tiers_ref: &mut [i32],
        count_predicted_tiers_ref: &mut [f64],
        recommended_set_ref: &mut HashSet<i32>,
    ) -> Result<(), Box<dyn std::error::Error>> {

        let k = ranked_movies_ref.num_candidates as usize;
        let catalog_size = movie_tiers_ref.len() as f64;

        // make 1 hashset for each movie_tier.
        let movie_tier_sets: Vec<HashSet<i32>> = (0..3)
            .map(|target_tier| {
                movie_tiers_ref
                    .iter()
                    .filter(|&(_, &tier_val)| tier_val == target_tier)
                    .map(|(&movie_id, _)| movie_id)
                    .collect()
            })
            .collect();

        // Pre-calculate the maximum possible DCG for K items (used for analytical random NDCG)
        // This is sum(1 / log2(rank + 1)) for all K ranks
        let max_k_dcg: f64 = (0..k).map(|r| 1.0 / (r as f64 + 2.0).log2()).sum();

        for (&user_id, predicted_movies) in ranked_movies_ref.user_ids.iter()
            .zip(ranked_movies_ref.movie_ids.chunks_exact(k))
        {
            for tier in 0..3 {
                // O(1) lookup to get the pre-computed set of movies for this user/tier combo
                let Some(gt_set) = movie_tier_vec_map_ref[tier].get(&user_id) else {
                    continue; // User has no target movies in this tier
                };

                if gt_set.is_empty() {
                    continue;
                }

                let gt_len = gt_set.len() as f64;
                let max_possible_hits = std::cmp::min(gt_set.len(), predicted_movies.len());

                // --- Recall ---
                let hits = predicted_movies.iter().filter(|id| gt_set.contains(id)).count();
                let recall = hits as f64 / max_possible_hits as f64;

                // --- NDCG ---
                let mut dcg: f64 = 0.0;
                for (rank_idx, movie_id) in predicted_movies.iter().enumerate() {
                    if gt_set.contains(movie_id) {
                        dcg += 1.0 / (rank_idx as f64 + 2.0).log2();
                    }
                }
                let mut idcg: f64 = 0.0;
                for rank_idx in 0..max_possible_hits {
                    idcg += 1.0 / (rank_idx as f64 + 2.0).log2();
                }
                let ndcg: f64 = if idcg == 0.0 { 0.0 } else { dcg / idcg };

                // the number of the tiers predicted
                let intersection_count = predicted_movies
                    .iter()
                    .filter(|&movie_id|  movie_tier_sets[tier].contains(movie_id))
                    .count();
                count_predicted_tiers_ref[tier] += intersection_count as f64 / predicted_movies.len() as f64;

                // baseline metrics, calculate for random ordering and selection,
                // that is,
                // a uniform random ranker
                let expected_rand_recall = gt_len.max(k as f64) / catalog_size;

                // Expected Random DCG = (|GT| / N) * max_k_dcg
                let expected_rand_dcg = (gt_len / catalog_size) * max_k_dcg;
                let expected_rand_ndcg = if idcg == 0.0 { 0.0 } else { expected_rand_dcg / idcg };

                for &movie_id in predicted_movies {
                    recommended_set_ref.insert(movie_id);
                }

                // --- Accumulate ---
                recall_sum_tiers_ref[tier] += recall;
                ndcg_sum_tiers_ref[tier] += ndcg;
                rand_recall_sum_tiers_ref[tier] += expected_rand_recall;
                rand_ndcg_sum_tiers_ref[tier] += expected_rand_ndcg;
                count_metric_tiers_ref[tier] += 1;
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