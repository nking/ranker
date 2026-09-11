///
/// a look at user specificity with respect to the target ratings (see test_Y().
/// and with respect to the input latent space (see test_X)
///
#[cfg(test)]
mod tail_user_specificity_tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::BufReader;
    use rustc_hash::FxHashMap;
    use serde_json::Value;
    use tonic::Status;
    use inference_engine::app_config::AppConfig;
    use inference_engine::bayesian::{load_and_count_movies, Movie, CatalogStats, build_bayesian_catalog};
    use inference_engine::pb::recommender_service_client::RecommenderServiceClient;
    use inference_engine::user_db::UserDb;
    use inference_engine::user_history::{build_map_async, UserMapEntry};
    use crate::tail_user_specificity_tests::helper::{get_embeddings_uris, get_model_param_json_uri};

    use tokio::task::JoinHandle;
    use inference_engine::app_runner::AppRunner;
    use inference_engine::embeddings_util::read_user_embeddings;

    use serial_test::serial;
    use inference_engine::movie_tiers::load_from_file;

    //use super::*;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    struct TestServerGuard {
        tx_shutdown: Option<tokio::sync::oneshot::Sender<()>>,
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

    /// calculate the movie global rating distribution and note the top 100 movies and the tail 80%.
    /// users who  rated the tail 20% highly are the tail of the "behavioral" distribution and are
    /// tested for specificity of recommendations here:
    ///
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial]
    pub async fn test_Y() {

        // files for use in tests:
        let config_path = "./config/default.json";
        let config = AppConfig::load_from_file(config_path).unwrap();

        let movies_map : HashMap<i32, Movie> = load_and_count_movies(&config);

        let catalog_stats : CatalogStats = build_bayesian_catalog(&movies_map);

        let movie_tiers : HashMap<i32, i32> = load_from_file(&config.movie_tiers_path).unwrap();


        //let movie_tiers = load_movie_tiers(&config.movie_tiers_path);

        let ratings_uris = &config.ratings_uris;
        let tmp: Vec<&str> = ratings_uris.iter().map(|s| s.as_str()).collect();
        let slice: &[&str] = tmp.as_slice();

        let (user_ratings_map, longest_history)  : (FxHashMap<i32, UserMapEntry>, usize) = build_map_async(slice).await;

        println!("user_map len={}, longest_history={}", user_ratings_map.len(), longest_history);

        let mut scored_users: Vec<(i32, f32)> = Vec::with_capacity(user_ratings_map.len());

        for (&user_id, entry) in user_ratings_map.iter() {
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            let history_iter = entry.movie_ids.iter().zip(entry.ratings.iter());

            for (&movie_id, &rating) in history_iter {
                let rating_f32 = rating as f32;
                // Only score items that exist in our Bayesian cache
                if let Some(&s_i) = catalog_stats.bayesian_scores.get(&movie_id) {
                    numerator += rating_f32 * s_i;
                    denominator += rating_f32;
                } else {
                    println!("WARNING:  shouldn't be missing any movies: {}", &movie_id);
                }
            }

            // Only include users who had at least one valid rated item in the catalog
            if denominator > 0.0 {
                let m_u = numerator / denominator;
                scored_users.push((user_id, m_u));
            }
        }

        // Sort users by Mainstreamness (Lowest to Highest)
        scored_users.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Identify the Tail (Bottom 20%)
        let tail_cutoff = (scored_users.len() as f32 * 0.20).floor() as usize;

        // Extract just the user IDs for the tail distribution
        let tail_user_ids: Vec<i32> = scored_users.iter()
            .take(tail_cutoff)
            .map(|(u_id, _)| *u_id)
            .collect();

        println!(
            "Identified {} Tail Users out of {} total unique users.",
            tail_user_ids.len(),
            scored_users.len()
        );

        // --- VISUALIZE DISTRIBUTION (ASCII HISTOGRAM) ---
        let min_score = scored_users.first().unwrap().1;
        let max_score = scored_users.last().unwrap().1;

        // The exact mainstreamness score at the 20% threshold
        let cutoff_score = scored_users[tail_cutoff].1;

        print_histogram(&scored_users, min_score, max_score, cutoff_score,
            "USER MAINSTREAMNESS (M_u) DISTRIBUTION".parse().unwrap(), true);


        run_baysian_shrinkage_stats(config, catalog_stats, scored_users, tail_user_ids).await;

    }

    /// calculate the centroid of the latent space embeddings and find the users who are furthest
    /// from the centroid at as the tail users. and test for specificity here
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial]
    pub async fn test_X() {

        let (user_embeddings_uri, _) = get_embeddings_uris();
        let config_path = "./config/default.json";
        let config = AppConfig::load_from_file(config_path).unwrap();

        let (user_embeddings, num_embeddings, embed_len) : (Vec<f32>, usize, usize)
            = read_user_embeddings(user_embeddings_uri.as_str());

        let user_db : UserDb = UserDb::new(&config.user_db_path).expect("Failed to initialize UserDb from binary path");

        let mut centroid: Vec<f32> = Vec::new();

        println!("Fetching user embeddings to calculate latent centroid...");

        //  Gather all embeddings and sum them up
        for i in 0..num_embeddings {
            //let user_id = i + 1;
            let i00 = i * embed_len;
            let i01 = i00 + embed_len;
            let embedding = &user_embeddings[i00..i01];

            // Initialize centroid vector on the first pass
            if centroid.is_empty() {
                centroid = vec![0.0; embedding.len()];
            }

            // Accumulate for the centroid average
            for (i, &val) in embedding.iter().enumerate() {
                centroid[i] += val;
            }
        }

        for val in &mut centroid {
            *val /= num_embeddings as f32;
        }

        // 3. Calculate Euclidean distance from the centroid for each user
        let mut user_distances: Vec<(i32, f32)> = Vec::with_capacity(num_embeddings);
        for i in 0..num_embeddings {
            let user_id = i + 1;
            let i00 = i * embed_len;
            let i01 = i00 + embed_len;
            let emb = &user_embeddings[i00..i01];

            let mut dist_sq = 0.0;
            for (i, &val) in emb.iter().enumerate() {
                let diff = val - centroid[i];
                dist_sq += diff * diff;
            }
            user_distances.push((user_id as i32, dist_sq));
        }

        // 4. Sort by distance descending (most distant users at the top)
        user_distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 5. Extract the new distance-based Tail Cohort (e.g., top 20% most distant)
        let tail_cutoff = (user_distances.len() as f32 * 0.20) as usize;

        let latent_tail_user_ids: Vec<i32> = user_distances.iter()
            .take(tail_cutoff)
            .map(|(id, _)| *id)
            .collect();

        println!("Centroid calculation complete.");
        println!("Identified {} latent tail users.", latent_tail_user_ids.len());

        let min_score = user_distances.last().unwrap().1;
        let max_score = user_distances.first().unwrap().1;

        // The exact mainstreamness score at the 20% threshold
        let cutoff_score = user_distances[tail_cutoff].1;

        print_histogram(&user_distances, min_score, max_score, cutoff_score,
            "LATENT TAIL COHORT DISTRIBUTION (Squared Distance)".parse().unwrap(), false);


        let movies_map : HashMap<i32, Movie> = load_and_count_movies(&config);

        let catalog_stats : CatalogStats = build_bayesian_catalog(&movies_map);

        run_baysian_shrinkage_stats(config, catalog_stats, user_distances, latent_tail_user_ids).await;

    }

    async fn run_baysian_shrinkage_stats(config: AppConfig, catalog_stats: CatalogStats,
        scored_users: Vec<(i32, f32)>, tail_user_ids: Vec<i32>) {

        // turn on server, with guard to shutdown when method goes out of scope
        let runner = AppRunner::new(config.to_owned());

        let (tx_shutdown, rx_shutdown) = tokio::sync::oneshot::channel::<()>();
        let (tx_addr, rx_addr) = tokio::sync::oneshot::channel::<std::net::SocketAddr>();
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

        //first in MovieLens dataset is Tuesday, April 25, 2000, at 11:05:32 PM UTC == 956703932
        //last is 1,046,454,590 which is February 28, 2003, at 17:23:10 UTC
        //The Unix timestamp 964152495 corresponds to July 21, 2000, at 04:08:15 UTC and its in the training data time range.
        const TIMESTAMP : i64 = 964152495;

        // ===== E2E Purity loop on `tail_users` to see if performance drops. =====
        let mut e2e_total_rec_score: f32 = 0.0;
        let mut e2e_eval_count: usize = 0;

        let mut tail_ann_score = 0.0;
        let mut tail_ann_count = 0;

        let user_db: UserDb = UserDb::new(config.user_db_path).expect("Failed to initialize UserDb from binary path");
        let client = RecommenderServiceClient::connect(endpoint.clone()).await.unwrap();

        // The proto documentation recommends not exceeding 256 users per request[cite: 3].
        let batch_size = 256;

        let timestamps = vec![TIMESTAMP; tail_user_ids.len()];
        for (user_batch, time_batch) in tail_user_ids.chunks(batch_size).zip(timestamps.chunks(batch_size)) {

            // Fetch request from DB for the whole batch[cite: 2]
            let user_req_opt = user_db.get_request(user_batch, time_batch);
            assert!(user_req_opt.is_some(), "Batch chunk should exist in database");

            let users_request_msg = user_req_opt.unwrap().into_inner();

            // =====================================================================
            // Predict (Graph Ranker) Evaluation
            // =====================================================================

            let mut active_client = client.clone();
            let predict_req = tonic::Request::new(users_request_msg.clone());

            let response_response = active_client
                .predict(predict_req)
                .await
                .map_err(|err| Status::internal(format!("ranking request failed: {}", err)))
                .unwrap();

            let response = response_response.into_inner();

            // Group parallel arrays by user_id to maintain per-user bias checking[cite: 3]
            let mut user_stats: HashMap<i32, (f32, usize)> = HashMap::new();

            for (&uid, &movie_id) in response.user_ids.iter().zip(response.movie_ids.iter()) {
                if let Some(&s_i) = catalog_stats.bayesian_scores.get(&movie_id) {
                    let stat = user_stats.entry(uid).or_insert((0.0, 0));
                    stat.0 += s_i;  // Sum of scores
                    stat.1 += 1;    // Count of scored items
                }
            }

            // Calculate per-user averages and check for popularity bias
            for (&uid, &(rec_score_sum, scored_items)) in &user_stats {
                if scored_items > 0 {
                    let user_avg_s_i = rec_score_sum / scored_items as f32;
                    e2e_total_rec_score += user_avg_s_i;
                    e2e_eval_count += 1;

                    // if the model aggressively defaults to highly mainstream items
                    // adjust this threshold based on the catalog's global mean S_i
                    if user_avg_s_i > 4.5 {
                        println!(
                            "\n[Popularity Bias Warning] User {}: Avg Rec S_i is {:.2}. Model may be falling back to global popularity.",
                            uid, user_avg_s_i
                        );
                    }
                }
            }

            // =====================================================================
            // 2. ANN Evaluation
            // =====================================================================

            let mut active_client = client.clone();
            let ann_req = tonic::Request::new(users_request_msg); // Consume the final copy

            if let Ok(response_response) = active_client.approx_nearest_neighbors(ann_req).await {
                let response = response_response.into_inner();

                // ApproxNearestNeighborsResponse returns a flat list of candidate_ids[cite: 3].
                for &movie_id in &response.candidate_ids {
                    if let Some(&s_i) = catalog_stats.bayesian_scores.get(&movie_id) {
                        tail_ann_score += s_i;
                        tail_ann_count += 1;
                    }
                }
            }
        }

        let tail_avg_ann_retrieval_s_i = if tail_ann_count > 0 {
            tail_ann_score / tail_ann_count as f32
        } else {
            0.0
        };

        let tail_eval_metric = if e2e_eval_count > 0 {
            e2e_total_rec_score / e2e_eval_count as f32
        } else {
            0.0
        };

        // ==== new we compare the results to a random sample of all users
        // if the random S_i is significantly larger than tail_eval_metric, then the model is
        // specializing for the tail distribution users,
        // else if the random S_ is near tail_eval_metric, the model is showing popularity bias.
        // --- GLOBAL BASELINE EVALUATION ---

        // Extract all user IDs from the previously scored map
        let global_user_ids: Vec<i32> = scored_users.iter().map(|(u_id, _)| *u_id).collect();

        let mut global_total_rec_score: f32 = 0.0;
        let mut global_eval_count: usize = 0;

        let mut global_ann_score = 0.0;
        let mut global_ann_count = 0;

        println!("\nStarting Global Batched Baseline Evaluation...");

        // The proto documentation recommends not exceeding 256 users per request.
        let batch_size = 256;

        let timestamps = vec![TIMESTAMP; global_user_ids.len()];
        // Zip through chunks of users and their parallel timestamps
        for (user_chunk, time_chunk) in global_user_ids.chunks(batch_size).zip(timestamps.chunks(batch_size)) {

            // Fetch once from DB for the whole batch
            let user_req_opt = user_db.get_request(user_chunk, time_chunk);

            if user_req_opt.is_none() {
                continue;
            }

            // Extract the inner message so we can clone it for both Predict and ANN calls
            let users_request_msg = user_req_opt.unwrap().into_inner();

            // =====================================================================
            // Predict (Graph Ranker) Evaluation
            // =====================================================================

            let mut active_client = client.clone();
            let predict_req = tonic::Request::new(users_request_msg.clone());

            if let Ok(response_response) = active_client.predict(predict_req).await {
                let response = response_response.into_inner();

                // RankedMovies returns parallel arrays. Group them by user_id to maintain
                // the original macro-average math (average per user, then global sum).
                let mut user_stats: HashMap<i32, (f32, usize)> = HashMap::new();

                for (&uid, &movie_id) in response.user_ids.iter().zip(response.movie_ids.iter()) {
                    if let Some(&s_i) = catalog_stats.bayesian_scores.get(&movie_id) {
                        let stat = user_stats.entry(uid).or_insert((0.0, 0));
                        stat.0 += s_i;  // Sum of scores
                        stat.1 += 1;    // Count of scored items
                    }
                }

                // Calculate the user average and add to global scores
                for (_, (rec_score_sum, scored_items)) in user_stats {
                    if scored_items > 0 {
                        let user_avg_s_i = rec_score_sum / scored_items as f32;
                        global_total_rec_score += user_avg_s_i;
                        global_eval_count += 1;
                    }
                }
            }

            // =====================================================================
            // ANN Evaluation
            // =====================================================================

            let mut active_client = client.clone();
            let ann_req = tonic::Request::new(users_request_msg); // Consume the final copy

            if let Ok(response_response) = active_client.approx_nearest_neighbors(ann_req).await {
                let response = response_response.into_inner();

                // ApproxNearestNeighborsResponse returns a flat list of candidate_ids.
                // We can just iterate through and sum them exactly like the original code.
                for &movie_id in &response.candidate_ids {
                    if let Some(&s_i) = catalog_stats.bayesian_scores.get(&movie_id) {
                        global_ann_score += s_i;
                        global_ann_count += 1;
                    }
                }
            }
        }

        let global_avg_ann_retrieval_s_i = if global_ann_count > 0 {
            global_ann_score / global_ann_count as f32
        } else {
            0.0
        };

        let global_eval_metric = if global_eval_count > 0 {
            global_total_rec_score / global_eval_count as f32
        } else {
            0.0
        };

        // --- FINAL ANALYSIS ---

        println!("\n\n===========================================");
        println!("        POPULARITY BIAS ANALYSIS           ");
        println!("===========================================");

        println!("Global ANN Retrieval Pool Avg S_i: {:.4}", global_avg_ann_retrieval_s_i);
        println!("Tail Cohort ANN Retrieval Pool Avg S_i: {:.4}", tail_avg_ann_retrieval_s_i);
        let diff = global_avg_ann_retrieval_s_i - tail_avg_ann_retrieval_s_i;
        let diff_sigma = diff / catalog_stats.std_dev;
        println!("Delta (Global - Tail):    {:.4} = {:.3} σ", diff, diff_sigma);
        if diff_sigma >= 0.6 {
            println!("✅ SUCCESS: The retrieval successfully specializes for user item preference! \
            It recommends significantly more niche items to Tail users than to the Global population.");
        } else if diff_sigma >= 0.5 {
            println!("MODERATE: The retrieval shows a moderate ability to recommend niche items to users \
            who prefer them. Popularity bias may still be heavily influencing the rerieval.");
        } else if diff_sigma >= 0.2 {
            println!("⚠️ SMALL: The retrieval shows low behavioral differentiation for tail users.\
            The effect is statistically detectable but it is an operationally marginal difference \
            between global and tail user consumption patterns");
        } else {
            println!("❌ FAILURE: The retrieval suffers from strong popularity bias. \
            Tail users are receiving the exact same mainstream recommendations as the rest of the population.");
        }
        println!("===========================================");
        println!("Global Average Retrieval+Ranking Top-K S_i: {:.4}", global_eval_metric);
        println!("Tail Average Retrieval+Ranking Top-K S_i:   {:.4}", tail_eval_metric);
        let diff = global_eval_metric - tail_eval_metric;
        let diff_sigma = diff / catalog_stats.std_dev;
        println!("Delta (Global - Tail): {:.4}  = {:.3} σ", diff, diff_sigma);
        if diff_sigma >= 0.6 {
            println!("✅ SUCCESS: The model successfully specializes for user item preference! \
            It recommends significantly more niche items to Tail users than to the Global population.");
        } else if diff_sigma >= 0.5 {
            println!("MODERATE: The model shows a moderate ability to recommend niche items to users \
            who prefer them. Popularity bias may still be heavily influencing the ranker.");
        } else if diff_sigma >= 0.2 {
            println!("⚠️ SMALL: The model shows low behavioral differentiation for tail users.\
            The effect is statistically detectable but it is an operationally marginal difference \
            between global and tail user consumption patterns");
        } else {
            println!("❌ FAILURE: The model suffers from strong popularity bias. Tail users are receiving the exact same mainstream recommendations as the rest of the population.");
        }
    }

    #[test]
    #[serial]
    pub fn test_load_movies() {

        let config_path = "./config/default.json";
        let config = AppConfig::load_from_file(config_path).unwrap();

        let movies_map : HashMap<i32, Movie> = load_and_count_movies(&config);

        // model params:
        let params_json_uri = get_model_param_json_uri();
        let file = File::open(params_json_uri).unwrap();
        let reader = BufReader::new(file);
        let dict: HashMap<String, Value> = serde_json::from_reader(reader).unwrap();
        let num_catalog_movies = dict.get("num_catalog_movies")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        assert_eq!(num_catalog_movies, movies_map.len());

        for (movie_id, movie) in &movies_map {
            assert_eq!(movie_id, &movie.movie_id);
            assert!(movie.title.len() > 0);
            assert!(movie.genres.len() > 0);
            assert!(movie.rating_counts.len() > 0);
        }
    }


    pub fn print_histogram(scored_users : &Vec<(i32, f32)> , min_score: f32, max_score: f32,
        cutoff_score: f32, title: String, ascending_scores: bool) {
        let num_buckets = 30;
        let bucket_width = (max_score - min_score) / num_buckets as f32;

        let mut buckets = vec![0; num_buckets];

        for &(_, score) in scored_users {
            let mut bucket_idx = ((score - min_score) / bucket_width).floor() as usize;
            if bucket_idx >= num_buckets {
                bucket_idx = num_buckets - 1; // Catch edge case for the absolute max value
            }
            buckets[bucket_idx] += 1;
        }

        let max_count = *buckets.iter().max().unwrap_or(&1);
        let max_bar_length = 50; // Maximum terminal characters for the longest bar

        println!("\n=======================================================");
        println!("       {}          ", title);
        println!("=======================================================");
        let direction :String = if ascending_scores {"(Bottom 20%) <=".to_string() } else {"(Top 20%) >=".to_string()};
        println!("Total Users: {} | Tail Cutoff {}  {:.4}", scored_users.len(),
            direction, cutoff_score);
        println!("-------------------------------------------------------");

        for i in 0..num_buckets {
            let bucket_min = min_score + (i as f32 * bucket_width);
            let bucket_max = bucket_min + bucket_width;
            let count = buckets[i];

            // Scale bar length to fit terminal
            let bar_length = ((count as f32 / max_count as f32) * max_bar_length as f32).round() as usize;

            // Use a solid block for the Tail (Bottom 20%), and a shaded block for the rest
            let bar_char = if ascending_scores {
                if bucket_min < cutoff_score { "█" } else { "▒" }
            } else {
                // For test_X, shade the buckets that are greater than or equal to the cutoff
                if bucket_max >= cutoff_score { "█" } else { "▒" }
            };
            let bar: String = std::iter::repeat(bar_char).take(bar_length).collect();

            // Print the bucket range, count, and visual bar
            println!("{:.4} - {:.4} | {:>4} | {}", bucket_min, bucket_max, count, bar);
        }
        println!("=======================================================\n");
    }


}