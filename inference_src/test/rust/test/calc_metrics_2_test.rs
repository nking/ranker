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
        get_user_movie_tier_map, create_user_movie_map};
    use std::path::PathBuf;
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::io::Write;
    use rustc_hash::FxHashMap;
    use serde_json::{Value};
    use tonic::{Request, Response};
    use inference_engine::app_config::AppConfig;
    use inference_engine::app_runner::AppRunner;
    use inference_engine::movie_tiers::load_from_file;
    use inference_engine::pb::recommender_service_client::RecommenderServiceClient;
    use inference_engine::pb::{RankedMovies, UsersRequest};
    use inference_engine::ranker_model_metadata::RankerModelMetadata;
    use inference_engine::user_db::UserDb;
    use inference_engine::user_history::{build_map_async, UserMapEntry};
    use crate::calc_metrics_2_tests::helper::{get_config_json_uri, get_train_val_test_liked_uris, DataSize, calc_normalized_emd_3, mean_and_std};

    use rand::seq::SliceRandom;
    use rand::thread_rng;

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

        let proj_dir : String = get_project_dir().unwrap().to_string_lossy().into_owned();
        let test_ratings_paths = vec![
            format!("{}/src/test/resources/data/ratings_test_liked.parquet", proj_dir)];

        // ===== movie_tier stats =======
        let output_path = format!("{}/bin/test_movie_tier_metrics.json", proj_dir);

        let movie_catalog_tier_fractions : Vec<f64> = calc_movie_tier_stratified_metrics(config.clone(), endpoint.clone(),
            test_ratings_paths.clone(), output_path).await.expect("Error: while calculating movie metrics");

        let output_path = format!("{}/bin/test_user_tier_metrics.json", proj_dir);

        let _res = calc_user_tier_stratified_metrics(config.clone(),
            test_ratings_paths.clone(), endpoint.clone(), output_path).await.expect("Error: while calculating user metrics");

        let output_path = format!("{}/bin/test_user_movie_tier_intersection_metrics.json", proj_dir);

        let _res = calc_user_movie_tier_intersection_stratified_metrics(config.clone(),
            test_ratings_paths, endpoint, &movie_catalog_tier_fractions, output_path).await.expect("Error: while calculating user metrics");

        // server is shutdown by the guard when this method is out of scope

    }

    async fn calc_user_movie_tier_intersection_stratified_metrics(config: AppConfig,  test_ratings_paths: Vec<String>,
        endpoint: String, movie_catalog_tier_fractions: &Vec<f64>, output_path: String) -> Result<(), Box<dyn std::error::Error>> {

        // for each user_tier=2 (users with fewest number of ratings)
        //   for history: count movie_tiers => get fractions
        //   for recommended: count movie_tiers => get fractions
        //   calc normalized EMD
        //   calc random EMD

        // key=movie_id, val=movie_tier
        let movie_tiers : FxHashMap<i32, i32> = load_from_file(&config.movie_tiers_path)?;

        // user_tiers: key=user_id, val=user_tier
        // user_movie_tier_history_frac_map: key=user_id, val=fraction of movie_tiers
        let (user_tiers, user_movie_tier_hist_frac_map, user_ids, timestamps) :
            (FxHashMap<i32, i32>, FxHashMap<i32, Vec<f64>>, Vec<i32>, Vec<i64>)
            = get_user_datastructures_3(&[&test_ratings_paths[0]], &movie_tiers,
            //Some(2)
            None
        ).await?;

        let user_db : UserDb = UserDb::new(&config.user_db_path).expect("Failed to initialize UserDb from binary path");

        let ranker_metadata = RankerModelMetadata::load_from_file(&config.ranker_metadata_uri).unwrap();

        let ranker_batch_size = ranker_metadata.batch_size;

        let client = RecommenderServiceClient::connect(endpoint.clone()).await?;

        // for each user in user_movie_tier_hist_frac_map, create user_movie_tier_recomend_frac_map
        let mut user_movie_tier_recommend_frac_map : FxHashMap<i32, Vec<f64>> = FxHashMap::default();

        let num_candidates: usize = ranker_metadata.num_candidates;

        for (chunk_users, chunk_timestamps) in user_ids.chunks(ranker_batch_size).zip(timestamps.chunks(ranker_batch_size)) {
            let option_tonic_request: Option<Request<UsersRequest>> = user_db.get_request(chunk_users, chunk_timestamps);
            let tonic_req = option_tonic_request.ok_or("request not found for chunk")?;
            //let users_req = tonic_req.into_inner();

            let mut active_client = client.clone();

            let results: Result<Response<RankedMovies>, tonic::Status> =
                active_client.predict(tonic_req).await;
            let ranked_movies = results?.into_inner();

            for (&user_id, predicted_movies) in ranked_movies.user_ids.iter()
                .zip(ranked_movies.movie_ids.chunks_exact(num_candidates)){

                // Stack-allocated array for counting tiers 0, 1, and 2
                let mut counts = [0_usize; 3];

                // there should always be num_candidates recommendations

                for &movie_id in predicted_movies {
                    if let Some(&tier) = movie_tiers.get(&movie_id) {
                        if (0..3).contains(&tier) {
                            counts[tier as usize] += 1;
                        }
                    }
                }

                // Convert counts to fractions
                let fractions: Vec<f64> = counts
                    .iter()
                    .map(|&c| c as f64 / num_candidates as f64)
                    .collect();

                user_movie_tier_recommend_frac_map.insert(user_id, fractions);

            }
        }

        // now we have the historic user_movie_tier_hist_frac_map
        //     and the recommended  user_movie_tier_recommend_frac_map
        // and because they are ordinal, that is, order matters, we will use Earth-Mover's Distance,
        // a.k.a. Wasserstein distances.

        // calculate normalized EMDs for user_tier=2 because it's harder to model.
        // compare to what the EMD would be if the recommended distribution just followed the catalog proportionally

        let mut real_emds: FxHashMap<i32, Vec<f64>> = FxHashMap::from_iter([
            (0, Vec::new()),
            (1, Vec::new()),
            (2, Vec::new()),
        ]);

        let mut catalog_emds: FxHashMap<i32, Vec<f64>> = FxHashMap::from_iter([
            (0, Vec::new()),
            (1, Vec::new()),
            (2, Vec::new()),
        ]);

        // Vectors to hold distributions so we can zip and shuffle them later
        let mut all_hist_distrs = Vec::new();
        let mut all_rec_distrs = Vec::new();

        let mut tier_1_hist_distrs = Vec::new();
        let mut tier_1_rec_distrs = Vec::new();

        let mut tier_2_hist_distrs = Vec::new();
        let mut tier_2_rec_distrs = Vec::new();

        for (&user_id, &tier) in user_tiers.iter() {
            if let (Some(hist), Some(rec)) = (
                user_movie_tier_hist_frac_map.get(&user_id),
                user_movie_tier_recommend_frac_map.get(&user_id),
            ) {
                let emd = calc_normalized_emd_3(hist, rec);
                real_emds.entry(tier).or_default().push(emd);

                catalog_emds.entry(tier).or_default().push(
                    calc_normalized_emd_3(hist, movie_catalog_tier_fractions)
                );

                all_hist_distrs.push(hist.clone());
                all_rec_distrs.push(rec.clone());

                if tier == 2 {
                    tier_2_hist_distrs.push(hist.clone());
                    tier_2_rec_distrs.push(rec.clone());
                } else if tier == 1{
                    tier_1_hist_distrs.push(hist.clone());
                    tier_1_rec_distrs.push(rec.clone());
                }
            }
        }

        let mut results_map = serde_json::Map::new();


        let mut real_mean = Vec::with_capacity(3);
        let mut stdv = Vec::with_capacity(3);
        for tier in 0..3 {
            if let Some(emds) = real_emds.get(&tier) {
                let (mean, stdev) = mean_and_std(emds);
                real_mean.push(mean);
                stdv.push(stdev);
                //println!("Real Mean EMD (tier={}):   {:.4} (std: {:.4})", tier, mean, stdev);
                results_map.insert(format!("emd_real_mean_for_usertier_{}", tier), Value::from(round_to_4_decimal_places(mean)));
                results_map.insert(format!("emd_real_stdev_for_usertier_{}", tier), Value::from(round_to_4_decimal_places(stdev)));
            }
        }

        let mut cat_mean = Vec::with_capacity(3);
        let mut cat_stdv = Vec::with_capacity(3);
        for tier in 0..3 {
            if let Some(emds) = catalog_emds.get(&tier) {
                let (mean, stdev) = mean_and_std(emds);
                cat_mean.push(mean);
                cat_stdv.push(stdev);
                //println!("EMD  (tier={}) to catalog distr:   {:.4} (std: {:.4})", tier, mean, stdev);
                results_map.insert(format!("emd_to_catalog_mean_for_usertier_{}", tier), Value::from(round_to_4_decimal_places(mean)));
                results_map.insert(format!("emd_to_catalog_stdev_for_usertier_{}", tier), Value::from(round_to_4_decimal_places(stdev)));
            }
        }

        //movie_tier_fractions
        // Calculate comparison
        let mut rng = thread_rng();
        // Shuffle the historical distributions so they no longer match the users
        all_hist_distrs.shuffle(&mut rng);
        let mut all_random_emds = Vec::with_capacity(all_rec_distrs.len());
        // Pair the shuffled histories against the original recommendations
        for (shuffled_hist, rec) in all_hist_distrs.iter().zip(all_rec_distrs.iter()) {
            let emd = calc_normalized_emd_3(shuffled_hist, rec);
            all_random_emds.push(emd);
        }
        let (all_rand_mean, all_rand_std) = mean_and_std(&all_random_emds);
        results_map.insert("emd_random_shuffle_all_mean".to_string(), Value::from(round_to_4_decimal_places(all_rand_mean)));
        results_map.insert("emd_random_shuffle_all_stdev".to_string(), Value::from(round_to_4_decimal_places(all_rand_std)));


        tier_2_hist_distrs.shuffle(&mut rng);
        let mut tier_2_random_emds = Vec::with_capacity(tier_2_rec_distrs.len());
        // Pair the shuffled histories against the original recommendations
        for (shuffled_hist, rec) in tier_2_hist_distrs.iter().zip(tier_2_rec_distrs.iter()) {
            let emd = calc_normalized_emd_3(shuffled_hist, rec);
            tier_2_random_emds.push(emd);
        }
        let (tier_2_rand_mean, tier_2_rand_std) = mean_and_std(&tier_2_random_emds);
        results_map.insert("emd_random_shuffle_usertier_2_mean".to_string(), Value::from(round_to_4_decimal_places(tier_2_rand_mean)));
        results_map.insert("emd_random_shuffle_usertier_2_stdev".to_string(), Value::from(round_to_4_decimal_places(tier_2_rand_std)));


        tier_1_hist_distrs.shuffle(&mut rng);
        let mut tier_1_random_emds = Vec::with_capacity(tier_1_rec_distrs.len());
        // Pair the shuffled histories against the original recommendations
        for (shuffled_hist, rec) in tier_1_hist_distrs.iter().zip(tier_1_rec_distrs.iter()) {
            let emd = calc_normalized_emd_3(shuffled_hist, rec);
            tier_1_random_emds.push(emd);
        }
        let (tier_1_rand_mean, tier_1_rand_std) = mean_and_std(&tier_1_random_emds);
        results_map.insert("emd_random_shuffle_usertier_1_mean".to_string(), Value::from(round_to_4_decimal_places(tier_1_rand_mean)));
        results_map.insert("emd_random_shuffle_usertier_1_stdev".to_string(), Value::from(round_to_4_decimal_places(tier_1_rand_std)));


        // Output the results
        //println!("Real Mean EMD:   {:.4} (std: {:.4})", real_mean, real_std);
        //println!("Random Mean EMD from shuffle all user history to recommendation data: {:.4} (std: {:.4})", all_rand_mean, all_rand_std);
        //println!("Random Mean EMD from shuffle user_tier=2 history to recommendation data: {:.4} (std: {:.4})", tier_2_rand_mean, tier_2_rand_std);

        let pretty_json = serde_json::to_string_pretty(&results_map)?;
        println!("{}", pretty_json);
        let mut file = File::create(&output_path)?;
        file.write_all(pretty_json.as_bytes())?;

        println!("=== Tier 2 Users EMD Alignment ===");

        Ok(())
    }

    fn round_to_4_decimal_places(x : f64) -> f64 {
        (x * 10000.0).round() / 10000.0
    }

    async fn calc_user_tier_stratified_metrics(config: AppConfig,  test_ratings_paths: Vec<String>,
        endpoint: String, output_path: String) -> Result<(), Box<dyn std::error::Error>> {

        // calc user stratified metrics
        let user_tiers : FxHashMap<i32, i32> = get_user_tiers().await?;

        let (test_user_movie_map, user_ids, timestamps) : (HashMap<i32, HashSet<i32>>,Vec<i32>, Vec<i64>)
            = get_user_datastructures_2(&[&test_ratings_paths[0]])?;

        let user_db : UserDb = UserDb::new(&config.user_db_path).expect("Failed to initialize UserDb from binary path");

        let ranker_metadata = RankerModelMetadata::load_from_file(&config.ranker_metadata_uri).unwrap();

        let ranker_batch_size = ranker_metadata.batch_size;

        let client = RecommenderServiceClient::connect(endpoint.clone()).await?;

        let num_catalog_movies = ranker_metadata.num_catalog_movies;

        let mut ndcg_tiers : Vec<f64> = vec![0.; 3];
        let mut recall_tiers : Vec<f64> = vec![0.; 3];
        let mut rand_ndcg_tiers : Vec<f64> = vec![0.; 3];
        let mut rand_recall_tiers : Vec<f64> = vec![0.; 3];
        let mut count_metric_tiers: Vec<i32> = vec![0; 3];

        // from test dataset, need the user_ids and timestamps
        // see get_user_datastructures

        for (chunk_users, chunk_timestamps) in user_ids.chunks(ranker_batch_size).zip(timestamps.chunks(ranker_batch_size)) {

            let option_tonic_request :  Option<Request<UsersRequest>> = user_db.get_request(chunk_users, chunk_timestamps);
            let tonic_req = option_tonic_request.ok_or("request not found for chunk")?;
            //let users_req = tonic_req.into_inner();

            let mut active_client = client.clone();

            let results: Result<Response<RankedMovies>, tonic::Status> =
                active_client.predict(tonic_req).await;
            let ranked_movies = results?.into_inner();

            sum_user_tier_metrics(
                &test_user_movie_map,
                &user_tiers,
                &ranked_movies,
                &mut ndcg_tiers,
                &mut recall_tiers,
                &mut rand_ndcg_tiers,
                &mut rand_recall_tiers,
                &mut count_metric_tiers,
                num_catalog_movies)?;
        }

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
            let tier_name = tier_names[tier];

            results_map.insert(format!("ndcg_{}_{}", tier_name, top_k), Value::from(mean_ndcg));
            results_map.insert(format!("recall_{}_{}", tier_name, top_k), Value::from(mean_recall));
            results_map.insert(format!("random_ndcg_{}_{}", tier_name, top_k), Value::from(mean_rand_ndcg));
            results_map.insert(format!("random_recall_{}_{}", tier_name, top_k), Value::from(mean_rand_recall));
            results_map.insert(format!("count_users_{}_{}", tier_name, top_k), Value::from(count));
        }

        println!("\n==========================================================");
        println!("  User Tier Metrics  (based on freq of user ratings length) ");
        println!("============================================================");

        // Format as pretty-printed JSON string
        let pretty_json = serde_json::to_string_pretty(&results_map)?;

        // Print to stdout
        println!("{}", pretty_json);

        // Write to file
        let mut file = File::create(&output_path)?;
        file.write_all(pretty_json.as_bytes())?;

        Ok(())
    }

    async fn get_user_tiers() -> Result<FxHashMap<i32, i32>, Box<dyn std::error::Error>> {

        let ratings_map = get_train_val_test_liked_uris(DataSize::Full, false);
        let ratings_history_uris: Vec<String> = vec![
            ratings_map.get("train_liked").unwrap().clone(),
            ratings_map.get("train_3").unwrap().clone(),
            ratings_map.get("train_disliked").unwrap().clone(),
            ratings_map.get("val_liked").unwrap().clone(),
            ratings_map.get("val_3").unwrap().clone(),
            ratings_map.get("val_disliked").unwrap().clone(),
        ];
        let ratings_history_uris: Vec<&str> = ratings_history_uris
            .iter()
            .map(|s| s.as_str())
            .collect();
        // make user history, then for the length of each users' histories, make a histogram
        // and extract the top 20% as head, next 60% as torso and bottom 20% as tail
        // these are then the user_tiers
        let (user_hash, _max_history_len) : (FxHashMap<i32, UserMapEntry>, usize)
            = build_map_async(&ratings_history_uris).await;

        let mut user_rating_lengths: Vec<(i32, usize)> = user_hash
            .iter()
            .map(|(&user_id, entry)| (user_id, entry.movie_ids.len()))
            .collect();

        // Sort ascending by activity length
        user_rating_lengths.sort_unstable_by_key(|&(_, len)| len);

        let n = user_rating_lengths.len();
        let tail_cutoff = (n as f64 * 0.20).round() as usize;
        let head_cutoff = (n as f64 * 0.80).round() as usize;

        //  Assign tiers (0 = head, 1 = torso, 2 = tail)
        let mut user_tiers: FxHashMap<i32, i32> = FxHashMap::default();
        user_tiers.reserve(n);

        for (i, &(user_id, _)) in user_rating_lengths.iter().enumerate() {
            let tier = if i < tail_cutoff {
                2 // Bottom 20% (tail)
            } else if i >= head_cutoff {
                0 // Top 20% (head)
            } else {
                1 // Middle 60% (torso)
            };
            user_tiers.insert(user_id, tier);
        }

        Ok(user_tiers)
    }

    async fn calc_movie_tier_stratified_metrics(config: AppConfig, endpoint: String,
        test_ratings_paths: Vec<String>, output_path: String) -> Result<Vec<f64>, Box<dyn std::error::Error>> {

        // get the movie_tiers file.  key=movie_id, value=movie_tier
        let movie_tiers : FxHashMap<i32, i32> = load_from_file(&config.movie_tiers_path)?;

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
            let ranked_movies = results?.into_inner();

            // calc metrics by tier and add to ndcg_tiers and recall_tiers
            sum_movie_tier_metrics(&movie_tier_vec_map, &movie_tiers, &ranked_movies,
                &mut ndcg_tiers, &mut recall_tiers, &mut rand_ndcg_tiers,
                &mut rand_recall_tiers, &mut count_metric_tiers,
                &mut count_predicted_tiers,
                &mut recommended_set
            )?;

        }

        // count the tier movies in the test dataset
        let test_tier_counts : Vec<i32> = count_tier_unique_movies_in_test(movie_tier_vec_map);
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
        results_map.insert("catalog coverage".to_string(), Value::from(cat_cov));

        // Format as pretty-printed JSON string
        let pretty_json = serde_json::to_string_pretty(&results_map)?;

        println!("\n============================================================");
        println!("  Movie Tier Metrics  (based on freq of movie ratings length) ");
        println!("==============================================================");

        // Print to stdout
        println!("{}", pretty_json);

        // Write to file
        let mut file = File::create(&output_path)?;
        file.write_all(pretty_json.as_bytes())?;

        let total: i32 = movie_catalog_tier_counts.iter().sum();
        let movie_catalog_tier_frac: Vec<f64> = if total == 0 {
            vec![0.0; movie_catalog_tier_counts.len()]
        } else {
            movie_catalog_tier_counts.iter()
                .map(|&c| c as f64 / total as f64)
                .collect()
        };

        Ok(movie_catalog_tier_frac)
    }

    fn count_tier_movies_in_catalog(movie_tiers : FxHashMap<i32, i32>) -> Vec<i32> {
        let mut counts : Vec<i32> = vec![0; 3];
        for (_u, t) in movie_tiers {
            counts[t as usize] += 1;
        }
        counts
    }

    fn count_tier_unique_movies_in_test(movie_tier_vec_map: Vec<HashMap<i32, HashSet<i32>>>) -> Vec<i32> {

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

    pub fn sum_movie_tier_metrics(
        movie_tier_vec_map_ref: &Vec<HashMap<i32, HashSet<i32>>>,
        movie_tiers_ref : &FxHashMap<i32, i32>,
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

    pub fn sum_user_tier_metrics(
        test_user_movie_map_ref: &HashMap<i32, HashSet<i32>>,
        user_tiers_ref : &FxHashMap<i32, i32>,
        ranked_movies_ref: &RankedMovies,
        ndcg_sum_tiers_ref: &mut [f64],
        recall_sum_tiers_ref: &mut [f64],
        rand_ndcg_sum_tiers_ref: &mut [f64],
        rand_recall_sum_tiers_ref: &mut [f64],
        count_metric_tiers_ref: &mut [i32],
        catalog_size : usize
    ) -> Result<(), Box<dyn std::error::Error>> {

        let k = ranked_movies_ref.num_candidates as usize;

        // make 1 hashset for each user_tier.
       let user_tier_sets: Vec<HashSet<i32>> = (0..3)
            .map(|target_tier| {
                user_tiers_ref
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
                if !user_tier_sets[tier].contains(&user_id) {
                    continue;
                };

                let gt_set = test_user_movie_map_ref.get(&user_id).ok_or("user not found")?;

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

                // baseline metrics, calculate for random ordering and selection,
                // that is,
                // a uniform random ranker
                let expected_rand_recall = gt_len.max(k as f64) / catalog_size as f64;

                // Expected Random DCG = (|GT| / N) * max_k_dcg
                let expected_rand_dcg = (gt_len / catalog_size as f64) * max_k_dcg;
                let expected_rand_ndcg = if idcg == 0.0 { 0.0 } else { expected_rand_dcg / idcg };

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
        movie_tiers: &FxHashMap<i32, i32>
    ) -> Result<(Vec<HashMap<i32, HashSet<i32>>>, Vec<i32>, Vec<i64>), Box<dyn std::error::Error>> {

        let df_gt = load_and_concat_parquet(ratings_uris)?;

        let movie_tier_maps_vec = get_user_movie_tier_map(df_gt.clone(), movie_tiers)?;

        let (user_ids, timestamps) = get_unique_user_and_first_timestamp(df_gt)?;

        Ok((movie_tier_maps_vec, user_ids, timestamps))
    }

    pub fn get_user_datastructures_2(ratings_uris: &[&str]) -> Result<(HashMap<i32, HashSet<i32>>, Vec<i32>, Vec<i64>), Box<dyn std::error::Error>> {

        let df_gt = load_and_concat_parquet(ratings_uris)?;

        let (user_ids, timestamps) = get_unique_user_and_first_timestamp(df_gt.clone())?;

        // key=user_id, value=hashset(movie_id)
        let test_user_movie_map = create_user_movie_map(df_gt)?;

        Ok((test_user_movie_map, user_ids, timestamps))
    }

    async fn get_user_datastructures_3(ratings_uris: &[&str], movie_tier_map: &FxHashMap<i32, i32>,
        filter_for_user_tier: Option<i32>)
        -> Result<(FxHashMap<i32, i32>, FxHashMap<i32, Vec<f64>>, Vec<i32>, Vec<i64>), Box<dyn std::error::Error>> {

        let df_gt = load_and_concat_parquet(ratings_uris)?;

        let (mut user_ids, mut timestamps) : (Vec<i32>, Vec<i64>) = get_unique_user_and_first_timestamp(df_gt.clone())?;

        let (user_tiers, mut user_movie_tier_history_frac_map) : (FxHashMap<i32, i32>, FxHashMap<i32, Vec<f64>>)
            = _get_datasets_3(movie_tier_map).await?;


        if let Some(target_tier) = filter_for_user_tier {
            // movie up the tier=2 elements and truncate vector
            let mut write_idx = 0;
            for read_idx in 0..user_ids.len() {
                let user_id = user_ids[read_idx];
                if user_tiers.get(&user_id) == Some(&target_tier) {
                    user_ids[write_idx] = user_id;
                    timestamps[write_idx] = timestamps[read_idx];
                    write_idx += 1;
                }
            }
            user_ids.truncate(write_idx);
            timestamps.truncate(write_idx);

            // modify in-place
            user_movie_tier_history_frac_map.retain(|user_id, _| {
                user_tiers.get(user_id) == Some(&target_tier)
            });
        }

        Ok((user_tiers, user_movie_tier_history_frac_map, user_ids, timestamps))
    }

    async fn _get_datasets_3(movie_tiers: &FxHashMap<i32, i32>)
        -> Result<(FxHashMap<i32, i32>, FxHashMap<i32, Vec<f64>>), Box<dyn std::error::Error>> {

        let ratings_map = get_train_val_test_liked_uris(DataSize::Full, false);
        let ratings_history_uris: Vec<String> = vec![
            ratings_map.get("train_liked").unwrap().clone(),
            ratings_map.get("train_3").unwrap().clone(),
            ratings_map.get("train_disliked").unwrap().clone(),
            ratings_map.get("val_liked").unwrap().clone(),
            ratings_map.get("val_3").unwrap().clone(),
            ratings_map.get("val_disliked").unwrap().clone(),
        ];
        let ratings_history_uris: Vec<&str> = ratings_history_uris
            .iter()
            .map(|s| s.as_str())
            .collect();

        // make user history, then for the length of each users' histories, make a histogram
        // and extract the top 20% as head, next 60% as torso and bottom 20% as tail
        // these are then the user_tiers
        let (user_hash, _max_history_len) : (FxHashMap<i32, UserMapEntry>, usize)
            = build_map_async(&ratings_history_uris).await;

        let mut user_rating_lengths: Vec<(i32, usize)> = user_hash
            .iter()
            .map(|(&user_id, entry)| (user_id, entry.movie_ids.len()))
            .collect();

        // Sort ascending by activity length
        user_rating_lengths.sort_unstable_by_key(|&(_, len)| len);

        let n = user_rating_lengths.len();
        let tail_cutoff = (n as f64 * 0.20).round() as usize;
        let head_cutoff = (n as f64 * 0.80).round() as usize;

        //  Assign tiers (0 = head, 1 = torso, 2 = tail)
        let mut user_tiers: FxHashMap<i32, i32> = FxHashMap::default();
        user_tiers.reserve(n);

        for (i, &(user_id, _)) in user_rating_lengths.iter().enumerate() {
            let tier = if i < tail_cutoff {
                2 // Bottom 20% (tail)
            } else if i >= head_cutoff {
                0 // Top 20% (head)
            } else {
                1 // Middle 60% (torso)
            };
            user_tiers.insert(user_id, tier);
        }

        // given user_hash: FxHashMap<i32, UserMapEntry> with key=user_id, and val = UserMapEntry which has movie_ids
        // given movie_tiers: &FxHashMap<i32, i32>)
        // create FxHashMap<i32, Vec<i32>> with key=user_id, val=fraction of movie_tier for each user
        // by  counting the movie_tiers for each user's UserMapEntry and divide by total to get fractions of each as a Vector
        let mut user_movie_tier_fractions: FxHashMap<i32, Vec<f64>> = FxHashMap::default();
        user_movie_tier_fractions.reserve(user_hash.len());

        for (&user_id, entry) in &user_hash {
            let total_movies = entry.movie_ids.len();

            if total_movies == 0 {
                user_movie_tier_fractions.insert(user_id, vec![0.0; 3]);
                continue;
            }

            // Stack-allocated array for counting tiers 0, 1, and 2
            let mut counts = [0_usize; 3];

            for &movie_id in &entry.movie_ids {
                if let Some(&tier) = movie_tiers.get(&movie_id) {
                    if (0..3).contains(&tier) {
                        counts[tier as usize] += 1;
                    }
                }
            }

            // Convert counts to fractions
            let fractions: Vec<f64> = counts
                .iter()
                .map(|&c| c as f64 / total_movies as f64)
                .collect();

            user_movie_tier_fractions.insert(user_id, fractions);
        }

        Ok((user_tiers, user_movie_tier_fractions))
    }

}