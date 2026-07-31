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
    pub async fn test_Y() {

        // files for use in tests:
        let config_path = "./config/default.json";
        let config = AppConfig::load_from_file(config_path).unwrap();

        let movies_map : HashMap<i32, Movie> = load_and_count_movies(&config);

        let catalog_stats : CatalogStats = build_bayesian_catalog(&movies_map);

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


        // ===== E2E Purity loop  on `tail_users` to see if performance drops. =====
        let mut e2e_total_rec_score: f32 = 0.0;
        let mut e2e_eval_count: usize = 0;

        let mut tail_ann_score = 0.0;
        let mut tail_ann_count = 0;

        let user_db : UserDb = UserDb::new(config.user_db_path).expect("Failed to initialize UserDb from binary path");
        let client = RecommenderServiceClient::connect(endpoint.clone()).await.unwrap();

        for (_i, &user_id) in tail_user_ids.iter().enumerate() {

            let user_req_opt = user_db.get_request(user_id as i64);
            assert!(user_req_opt.is_some(), "User ID {} should exist in database", user_id);
            let tonic_req = user_req_opt.unwrap();

            let mut active_client = client.clone();
            let response_response = active_client
                .predict(tonic_req)
                .await
                .map_err(|err| Status::internal(format!("ranking request failed: {}", err)))
                .unwrap();

            let response = response_response.into_inner();

            let retrieved_ids = response.movie_ids;
            let actual_k = retrieved_ids.len();

            if actual_k > 0 {
                let mut rec_score_sum = 0.0;
                let mut scored_items = 0;

                for &movie_id in &retrieved_ids {
                    if let Some(&s_i) = catalog_stats.bayesian_scores.get(&movie_id) {
                        rec_score_sum += s_i;
                        scored_items += 1;
                    }
                }

                if scored_items > 0 {
                    let user_avg_s_i = rec_score_sum / scored_items as f32;
                    e2e_total_rec_score += user_avg_s_i;
                    e2e_eval_count += 1;

                    // if the model aggressively defaults to highly mainstream items
                    // adjust this threshold based on the catalog's global mean S_i
                    if user_avg_s_i > 4.5 {
                        println!(
                            "\n[Popularity Bias Warning] User {}: Avg Rec S_i is {:.2}. Model may be falling back to global popularity.",
                            user_id, user_avg_s_i
                        );
                    }
                }
            }

            // look for popularity affinity in the ANN of embeddings from bi-encoder trained models ====

            let user_req_opt = user_db.get_request(user_id as i64);
            assert!(user_req_opt.is_some(), "User ID {} should exist in database", user_id);
            let tonic_req = user_req_opt.unwrap();

            let mut active_client = client.clone();
            if let Ok(response_response)
                = active_client.approx_nearest_neighbors(tonic_req).await {
                let response = response_response.into_inner();
                let retrieved_ids = response.candidate_ids;
                let _actual_k = retrieved_ids.len();
                for &movie_id in &retrieved_ids {
                    if let Some(&s_i) = catalog_stats.bayesian_scores.get(&movie_id) {
                        tail_ann_score += s_i;
                        tail_ann_count += 1;
                    }
                }
            }

            //use std::io::{self, Write};
            //print!("\rProgress: {}/{} users tested", i + 1, tail_user_ids.len());
            //io::stdout().flush().unwrap();
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

        println!("\nStarting Global Baseline Evaluation...");

        for (_i, &user_id) in global_user_ids.iter().enumerate() {

            let user_req_opt = user_db.get_request(user_id as i64);

            // Safe unwrap/continue in case some users are missing from the request DB
            if user_req_opt.is_none() {
                continue;
            }
            let tonic_req = user_req_opt.unwrap();

            let mut active_client = client.clone();

            // We use Ok() to gracefully skip errors if the server drops a request
            // under high load, rather than panicking the entire test loop.
            if let Ok(response_response) = active_client.predict(tonic_req).await {
                let response = response_response.into_inner();
                let retrieved_ids = response.movie_ids;
                let actual_k = retrieved_ids.len();

                if actual_k > 0 {
                    let mut rec_score_sum = 0.0;
                    let mut scored_items = 0;

                    for &movie_id in &retrieved_ids {
                        if let Some(&s_i) = catalog_stats.bayesian_scores.get(&movie_id) {
                            rec_score_sum += s_i;
                            scored_items += 1;
                        }
                    }

                    if scored_items > 0 {
                        let user_avg_s_i = rec_score_sum / scored_items as f32;
                        global_total_rec_score += user_avg_s_i;
                        global_eval_count += 1;
                    }
                }
            }

            // look for popularity affinity in the ANN of embeddings from bi-encoder trained models ====

            let user_req_opt = user_db.get_request(user_id as i64);
            assert!(user_req_opt.is_some(), "User ID {} should exist in database", user_id);
            let tonic_req = user_req_opt.unwrap();

            let mut active_client = client.clone();
            if let Ok(response_response)
                = active_client.approx_nearest_neighbors(tonic_req).await {
                let response = response_response.into_inner();
                let retrieved_ids = response.candidate_ids;
                let _actual_k = retrieved_ids.len();
                for &movie_id in &retrieved_ids {
                    if let Some(&s_i) = catalog_stats.bayesian_scores.get(&movie_id) {
                        global_ann_score += s_i;
                        global_ann_count += 1;
                    }
                }
            }

            //use std::io::{self, Write};
            //print!("\rProgress: {}/{} global users tested", i + 1, global_user_ids.len());
            //io::stdout().flush().unwrap();
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