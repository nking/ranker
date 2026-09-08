///
/// in this class we are testing the end-to-end inference results by choosing test users who have a
///    strong positive signal for a single movie genre, and then sampling their negative candidates
///    proportionally from their actual negative distribution.  We choose a positvee candidate
///    as an unwatched movie from the same genre as their dominant genre.
///
///  Note that this test represents very few users, but it does show the ability of the model to
///  order items.
///
#[cfg(test)]
mod pairwise_pref_tests {
    use tokio::sync::oneshot;
    use tokio::task::JoinHandle;
    use std::error::Error;

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
    }

    use helper::{get_model_param_json_uri};
    use std::fs::File;
    use std::collections::{HashMap, HashSet};
    use std::io::BufReader;
    use serde_json::Value;
    use inference_engine::app_config::AppConfig;
    use inference_engine::app_runner::AppRunner;

    use polars::prelude::*;
    use std::io::{self, Write};
    use std::path::Path;
    use tonic::Status;
    // use recommender_grpc::ranker_client::RankerClient;
    // use recommender_grpc::RankRequest;
    use tonic::transport::Channel;
    use inference_engine::pb::RankOnlyRequest;
    use inference_engine::pb::recommender_service_client::RecommenderServiceClient;
    //use inference_engine::util::timestamp_now;
    use crate::pairwise_pref_tests::helper::{get_project_dir};
    use rand::Rng;
    use inference_engine::user_db::UserDb;

    //TODO: this could be imrpoved now that the default request is a batch request
    
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    pub async fn test_pairwise_pref() -> Result<(), Box<dyn Error>> {

        // ======= setup server ======================

        let params_json_uri = get_model_param_json_uri();
        let file = File::open(params_json_uri).unwrap();
        let reader = BufReader::new(file);
        let dict: HashMap<String, Value> = serde_json::from_reader(reader).unwrap();
        let _max_history = dict.get("max_history")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_candidates = dict.get("num_candidates")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let _num_catalog_users = dict.get("num_catalog_users")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let config_path = "./config/default.json";
        let config = AppConfig::load_from_file(config_path).unwrap();

        let top_k = config.top_k;

        let user_db_path = &config.user_db_path;

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


        // ===== run tests =============================================================

        // originally Option<PathBuf>
        let proj_dir : String = get_project_dir().unwrap().to_string_lossy().into_owned();

        let train_val_liked = vec![
            format!("{}/src/test/resources/data/ratings_train_liked.parquet",  proj_dir),
            format!("{}/src/test/resources/data/ratings_val_liked.parquet", proj_dir)];
        let train_val_liked: &[&str] = &[&train_val_liked[0], &train_val_liked[1]];

        let train_val_disliked = vec![
            format!("{}/src/test/resources/data/ratings_train_disliked.parquet",  proj_dir),
            format!("{}/src/test/resources/data/ratings_val_disliked.parquet", proj_dir)];
        let train_val_disliked: &[&str] = &[&train_val_disliked[0], &train_val_disliked[1]];

        let test_liked = vec![
            format!("{}/src/test/resources/data/ratings_test_liked.parquet", proj_dir)];
        let test_liked: &[&str] = &[&test_liked[0]];


        let movies_path = config.movies_path;

        let context = generate_eval_context_on_the_fly(
            &train_val_liked,
            &train_val_disliked,
            &test_liked,
            &movies_path,
            0.75, // Dominance threshold
            20,   // Min history
            num_candidates
        )?;

        println!("Context Built Successfully!");
        println!("Identified {} highly polarized users.", context.polarized_users.len());

        let mut client = RecommenderServiceClient::connect(endpoint.clone()).await?;

        let win_rate = run_pairwise_preference_test(&mut client, &context, top_k, user_db_path).await?;

        assert!(win_rate >= 0.55, "Ranker failed the pairwise preference test!");

        Ok(())


        // server is shutdown by the guard when this method is out of scope

    }


    // ============================================================================
    // 1. DATA STRUCTURES
    // ============================================================================

    #[derive(Debug, Clone)]
    pub struct PolarizedUser {
        pub user_id: i32,
        pub pos_dominant_genre: String,
        pub neg_genre_dist: Vec<(String, u32)>, // The histogram of disliked genres
    }

    #[derive(Debug, Clone)]
    pub struct MovieStats {
        pub movie_id: i32,
        pub genres: Vec<String>,
        pub global_rating_count: u32,
    }

    pub struct EvalContext {
        pub polarized_users: Vec<PolarizedUser>,
        pub movies: HashMap<i32, MovieStats>,
        pub user_histories: HashMap<i32, HashSet<i32>>,
        pub num_candidates: usize,
    }

    // ============================================================================
    // 2. DATA PIPELINE (POLARS)
    // ============================================================================

    /// Helper to concatenate multiple parquet files into a single LazyFrame
    fn load_and_concat_parquet(paths: &[&str]) -> PolarsResult<LazyFrame> {
        let frames: Result<Vec<LazyFrame>, _> = paths
            .iter()
            .map(|&path| {
                let pl_path = PlRefPath::from(path);
                LazyFrame::scan_parquet(pl_path, ScanArgsParquet::default())
            })
            .collect();
        concat(frames?, UnionArgs::default())
    }

    /// Core function to calculate the dominant genre for a given ratings dataset
    fn extract_dominant_profiles(
        ratings_lf: LazyFrame,
        exploded_movies_lf: LazyFrame,
        prefix: &str,
    ) -> LazyFrame {
        let user_totals = ratings_lf.clone()
            .group_by([col("user_id")])
            .agg([len().alias("total_ratings")]);

        // 2. Join ratings with exploded movies to map ratings to individual genres
        let joined = ratings_lf.inner_join(exploded_movies_lf, col("movie_id"), col("movie_id"));

        // 3. Count instances of each genre per user
        let genre_counts = joined
            .group_by([col("user_id"), col("genre")])
            .agg([len().alias("genre_count")]);

        // 4. Find the max genre per user
        let dominant_genres = genre_counts
            .group_by([col("user_id")])
            .agg([
                col("genre")
                    .sort_by([col("genre_count")], SortMultipleOptions::default().with_order_descending(true))
                    .first()
                    .alias(format!("{}_dominant_genre", prefix)),

                col("genre_count")
                    .max()
                    .alias(format!("{}_dominant_count", prefix)),
            ]);

        // 5. Calculate ratio using the true movie rating count as the denominator
        dominant_genres
            .inner_join(user_totals, col("user_id"), col("user_id"))
            .with_column(
                (col(format!("{}_dominant_count", prefix)).cast(DataType::Float64)
                    / col("total_ratings").cast(DataType::Float64))
                    .alias(format!("{}_max_ratio", prefix)),
            )
            .rename(["total_ratings"], [format!("{}_total_ratings", prefix)], true)
    }

    pub fn generate_eval_context_on_the_fly(
        ratings_train_val_liked_paths: &[&str],
        ratings_train_val_disliked_paths: &[&str],
        ratings_test_liked_paths: &[&str],
        movies_path: &str,
        threshold: f64,
        min_history: u32,
        num_candidates: usize
    ) -> PolarsResult<EvalContext> {
        let movies_path_ref = PlRefPath::from(movies_path);
        let movies_lf = LazyFrame::scan_parquet(movies_path_ref, ScanArgsParquet::default())?
            .with_column(col("movie_id").cast(DataType::Int32));
        let train_val_liked = load_and_concat_parquet(ratings_train_val_liked_paths)?;
        let train_val_disliked = load_and_concat_parquet(ratings_train_val_disliked_paths)?;
        let test_liked = load_and_concat_parquet(ratings_test_liked_paths)?;

        // Explode genres string (e.g. "Action|Sci-Fi" -> two rows)
        let exploded_movies = movies_lf.clone()
            // 1. Split the string into a List (row count stays 3883)
            .with_column(
                col("genres").str().split(lit("|")).alias("genre")
            )
            // 2. Explode the LazyFrame using the cols() selector
            .explode(
                cols(["genre"]),
                ExplodeOptions { empty_as_null: false, keep_nulls: false }
            );

        // Process Positive Profiles
        let pos_profiles = extract_dominant_profiles(train_val_liked.clone(), exploded_movies.clone(), "pos");
        let pos_cohort = pos_profiles.filter(
            col("pos_total_ratings").gt_eq(lit(min_history))
                .and(col("pos_max_ratio").gt_eq(lit(threshold)))
        );

        // 2. Ensure ground truth in test set
        let test_users = test_liked.select([col("user_id")]).unique(None, UniqueKeepStrategy::First);

        let pos_df = pos_cohort.collect()?;
        let test_df = test_users.collect()?;

        println!("--- DEBUG COUNTS ---");
        println!("pos_df row count: {}", pos_df.height());
        println!("test_df row count: {}", test_df.height());

        // Join positive cohort and test users to establish the target user pool
        let polarized_df = pos_df.inner_join(&test_df, ["user_id"], ["user_id"])?;

        println!("polarized_df row count (after test join): {}", polarized_df.height());
        println!("--------------------");

        // 3. Extract raw disliked (user_id, genre) pairs for histogram building in Rust
        let neg_pairs_df = train_val_disliked.clone()
            .inner_join(exploded_movies.clone(), col("movie_id"), col("movie_id"))
            .select([col("user_id"), col("genre")])
            .collect()?;

        // Build a native Rust histogram mapping: user_id -> { genre -> count }
        let mut user_neg_counts: HashMap<i32, HashMap<String, u32>> = HashMap::new();
        let neg_uids = neg_pairs_df.column("user_id")?.i32()?;
        let neg_genres = neg_pairs_df.column("genre")?.str()?;

        for i in 0..neg_pairs_df.height() {
            if let (Some(uid), Some(g)) = (neg_uids.get(i), neg_genres.get(i)) {
                *user_neg_counts.entry(uid).or_default().entry(g.to_string()).or_insert(0) += 1;
            }
        }

        // Global Movie Popularity Generation
        let all_train_val = concat(
            vec![train_val_liked.clone(), train_val_disliked.clone()],
            UnionArgs::default()
        )?;

        let movie_popularity = all_train_val.clone()
            .group_by([col("movie_id")])
            .agg([col("user_id").count().alias("global_rating_count")]);

        // Filter popularity to only include movies present in movies_lf using an inner join
        let filtered_popularity = movie_popularity
            .inner_join(movies_lf.clone(), col("movie_id"), col("movie_id"))
            .select([col("movie_id"), col("global_rating_count")]);

        let movies_enriched_df = movies_lf
            .left_join(filtered_popularity, col("movie_id"), col("movie_id"))
            .with_column(col("global_rating_count").fill_null(lit(0u32)))
            .collect()?;

        // Build the History HashSet
        let all_history_df = all_train_val.clone()
            .group_by([col("user_id")])
            .agg([col("movie_id").alias("history")])
            .collect()?;

        // ========================================================================
        // EXPORT TO RUST STRUCTS
        // ========================================================================

        let mut eval_context = EvalContext {
            polarized_users: Vec::new(),
            movies: HashMap::new(),
            user_histories: HashMap::new(),
            num_candidates: num_candidates
        };

        let user_ids = polarized_df.column("user_id")?.i32()?;
        let pos_genres = polarized_df.column("pos_dominant_genre")?.str()?;

        for i in 0..polarized_df.height() {
            if let (Some(uid), Some(pos_g)) = (user_ids.get(i), pos_genres.get(i)) {
                if let Some(neg_map) = user_neg_counts.remove(&uid) {
                    let neg_dist: Vec<(String, u32)> = neg_map.into_iter().collect();
                    if !neg_dist.is_empty() {
                        eval_context.polarized_users.push(PolarizedUser {
                            user_id: uid,
                            pos_dominant_genre: pos_g.to_string(),
                            neg_genre_dist: neg_dist,
                        });
                    }
                }
            }
        }

        // 2. Populate MovieStats
        let m_ids = movies_enriched_df.column("movie_id")?.i32()?;
        let m_genres = movies_enriched_df.column("genres")?.str()?;
        let m_counts = movies_enriched_df.column("global_rating_count")?.u32()?;

        for i in 0..movies_enriched_df.height() {
            if let (Some(mid), Some(g_str), Some(count)) = (m_ids.get(i), m_genres.get(i), m_counts.get(i)) {
                let genres_vec = g_str.split('|').map(|s| s.to_string()).collect();
                eval_context.movies.insert(mid, MovieStats {
                    movie_id: mid,
                    genres: genres_vec,
                    global_rating_count: count,
                });
            }
        }

        // 3. Populate User Histories
        let _h_uids = all_history_df.column("user_id")?.i32()?;
        let _h_lists = all_history_df.column("history")?.list()?;

        let all_ratings_df = all_train_val.collect()?;
        let h_uids = all_ratings_df.column("user_id")?.i32()?;
        let h_mids = all_ratings_df.column("movie_id")?.i32()?;

        let mut user_histories: HashMap<i32, HashSet<i32>> = HashMap::new();
        for i in 0..all_ratings_df.height() {
            if let (Some(uid), Some(mid)) = (h_uids.get(i), h_mids.get(i)) {
                user_histories.entry(uid).or_default().insert(mid);
            }
        }

        Ok(eval_context)
    }


    // ============================================================================
    // Multi-PAIRWISE PREFERENCE TEST LOGIC
    // ============================================================================

    fn find_popularity_matched_pairs(
        pos_genre: &str,
        neg_genre_dist: &[(String, u32)],
        history: &HashSet<i32>,
        movies: &HashMap<i32, MovieStats>,
        tolerance_pct: f64,
        num_pairs: usize,
    ) -> Option<Vec<(i32, i32)>> {

        // 1. Collect and sort all available positive candidates for this genre
        let mut pos_candidates: Vec<&MovieStats> = movies.values()
            .filter(|m| !history.contains(&m.movie_id) && m.genres.contains(&pos_genre.to_string()))
            .collect();
        pos_candidates.sort_unstable_by_key(|m| m.global_rating_count);

        // 2. Sample negative candidates proportionally using the user's negative genre histogram
        let mut neg_candidates = Vec::new();
        let mut seen_neg_ids = HashSet::new();
        let mut attempts = 0;

        while neg_candidates.len() < num_pairs && attempts < num_pairs * 25 {
            attempts += 1;

            if let Some(sampled_neg_genre) = sample_negative_genre(neg_genre_dist) {
                // Guard A: Ignore if the sampled negative genre matches the positive dominant genre
                if sampled_neg_genre == pos_genre {
                    continue;
                }

                if let Some(m) = movies.values().find(|m| {
                    !history.contains(&m.movie_id)
                        && !seen_neg_ids.contains(&m.movie_id)
                        && m.genres.contains(&sampled_neg_genre)
                        && !m.genres.contains(&pos_genre.to_string()) // Guard B: Prevent multi-genre cross-contamination
                }) {
                    seen_neg_ids.insert(m.movie_id);
                    neg_candidates.push(m);
                }
            }
        }
        neg_candidates.sort_unstable_by_key(|m| m.global_rating_count);

        // Ensure we have enough candidates to form the requested number of pairs
        if pos_candidates.len() < num_pairs || neg_candidates.len() < num_pairs {
            return None;
        }

        // 3. Match positive and negative candidates by popularity
        let mut matched_pairs = Vec::with_capacity(num_pairs);
        let mut pos_idx = 0;
        let mut neg_idx = 0;

        while pos_idx < pos_candidates.len() && neg_idx < neg_candidates.len() && matched_pairs.len() < num_pairs {
            let pos_m = pos_candidates[pos_idx];
            let neg_m = neg_candidates[neg_idx];

            let diff = pos_m.global_rating_count.abs_diff(neg_m.global_rating_count) as f64;
            let max_diff = (pos_m.global_rating_count as f64 * tolerance_pct).max(10.0);

            if diff <= max_diff {
                matched_pairs.push((pos_m.movie_id, neg_m.movie_id));
                pos_idx += 1;
                neg_idx += 1;
            } else if pos_m.global_rating_count < neg_m.global_rating_count {
                pos_idx += 1;
            } else {
                neg_idx += 1;
            }
        }

        if matched_pairs.len() == num_pairs {
            Some(matched_pairs)
        } else {
            None
        }
    }

    pub async fn run_pairwise_preference_test(
        client: &mut RecommenderServiceClient<Channel>,
        context: &EvalContext,
        top_k: usize,
        user_db_path : impl AsRef<Path>
    ) -> Result<f64, tonic::Status> {

        //first in MovieLens dataset is Tuesday, April 25, 2000, at 11:05:32 PM UTC == 956703932
        //last is 1,046,454,590 which is February 28, 2003, at 17:23:10 UTC
        //The Unix timestamp 964152495 corresponds to July 21, 2000, at 04:08:15 UTC and its in the training data time range.
        const TIMESTAMP : i64 = 964152495;

        let timestamps = vec![TIMESTAMP];

        // variables for the parity metrics:
        let mut wins = 0;
        let mut valid_tests = 0;
        let popularity_tolerance = 0.15;
        let num_pairs = context.num_candidates / 2;

        println!("Starting Pairwise Preference Evaluation (Batch Size: {})...", context.num_candidates);

        // also we are calculating ranking purity on the synthetic candidate list.
        // the candidates are a pure 50/50 split of positives/negatives.
        // we count how many of the top_k sorted are positives

        // variables for the purity metrics for the randomly chosen pos and neg ids:
        let mut purity_total: f32 = 0.0;
        let mut purity_eval_count: usize = 0;

        // variables for the purity metrics for the recommended movies
        let mut e2e_total_purity: f32 = 0.0;
        let mut e2e_purity_eval_count: usize = 0;

        let user_db : UserDb = UserDb::new(user_db_path).expect("Failed to initialize UserDb from binary path");

        for (i, user) in context.polarized_users.iter().enumerate() {
            let history = context.user_histories.get(&user.user_id).cloned().unwrap_or_default();

            // Find multiple popularity-matched pairs using the negative distribution
            if let Some(matched_pairs) = find_popularity_matched_pairs(
                &user.pos_dominant_genre,
                &user.neg_genre_dist,
                &history,
                &context.movies,
                popularity_tolerance,
                num_pairs,
            ) {
                // Flatten pairs into a single candidate ID vector matching num_candidates length
                let mut candidate_ids = Vec::with_capacity(context.num_candidates);
                for (pos_id, neg_id) in &matched_pairs {
                    candidate_ids.push(*pos_id);
                    candidate_ids.push(*neg_id);
                }

                //println!("CANDIDATE_IDS {:?}", candidate_ids.clone());

                let req = RankOnlyRequest {
                    user_id: user.user_id,
                    timestamp: TIMESTAMP,
                    candidate_ids,
                };

                let mut active_client = client.clone();
                let response_response = active_client
                    .rank_only_return_all(tonic::Request::new(req))
                    .await
                    .map_err(|err| Status::internal(format!("ranking request failed: {}", err)))?;

                let response = response_response.into_inner();

                let mut score_map = HashMap::with_capacity(response.movie_ids.len());
                for (&id, &score) in response.movie_ids.iter().zip(response.scores.iter()) {
                    score_map.insert(id, score);
                }

                // Evaluate every pair within the batch response
                for (pos_id, neg_id) in matched_pairs {
                    // 2. Assert that all matched_pair IDs are present in the response hashmap
                    if !score_map.contains_key(&pos_id) {
                        println!("\n[Error] Positive movie ID {} missing from recommender response for user {}", pos_id, user.user_id);
                        println!("Full response score map ({0} items): {:?} {:?}", score_map.len(), score_map);
                        panic!("Positive movie ID missing from response");
                    }
                    if !score_map.contains_key(&neg_id) {
                        println!("\n[Error] Negative movie ID {} missing from recommender response for user {}", neg_id, user.user_id);
                        println!("Full response score map ({0} items): {:?} {:?}", score_map.len(), score_map);
                        panic!("Negative movie ID missing from response");
                    }

                    // 3. Look up pos and neg scores from the response hashmap
                    let pos_score = score_map[&pos_id];
                    let neg_score = score_map[&neg_id];

                    if pos_score > neg_score {
                        wins += 1;
                    } else {
                        // Detailed diagnostic print for failures
                        let pos_movie = context.movies.get(&pos_id);
                        let neg_movie = context.movies.get(&neg_id);

                        let pos_genres = pos_movie.map(|m| m.genres.join("|")).unwrap_or_default();
                        let neg_genres = neg_movie.map(|m| m.genres.join("|")).unwrap_or_default();
                        let pos_pop = pos_movie.map(|m| m.global_rating_count).unwrap_or(0);
                        let neg_pop = neg_movie.map(|m| m.global_rating_count).unwrap_or(0);

                        println!(
                            "\n[Failure] User {}:\n  - POS [ID: {} | Target Genre: {}]: Genres [{}] | Pop: {} | Score: {:.4}\n  - NEG [ID {}]: Genres [{}] | Pop: {} | Score: {:.4}",
                            user.user_id, pos_id, user.pos_dominant_genre, pos_genres, pos_pop, pos_score,
                            neg_id, neg_genres, neg_pop, neg_score
                        );
                    }
                    valid_tests += 1;
                }

                // ==========================================
                // Top-K Genre Purity Evaluation
                // ==========================================

                // 1. Zip the IDs and scores together so we can sort them listwise
                let mut ranked_items: Vec<(i32, f32)> = response.movie_ids.iter().cloned()
                    .zip(response.scores.iter().cloned())
                    .collect();

                //Sort descending by score
                ranked_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take the Top K (handling cases where the batch might be smaller than K)
                let top_k_items: Vec<(i32, f32)> = ranked_items.into_iter().take(top_k).collect();
                let actual_k = top_k_items.len();

                //  Calculate Purity: How many of the Top K belong to the user's dominant genre?
                let mut hits = 0;
                for (movie_id, _score) in &top_k_items {
                    if let Some(movie) = context.movies.get(movie_id) {
                        // Assuming genres is a Vec<String>
                        if movie.genres.contains(&user.pos_dominant_genre) {
                            hits += 1;
                        }
                    }
                }

                if actual_k > 0 {
                    let user_purity = hits as f32 / actual_k as f32;
                    purity_total += user_purity;
                    purity_eval_count += 1;

                    // Optional diagnostic: Flag if the model completely buried the target genre
                    if user_purity == 0.0 {
                        println!(
                            "[Purity Warning] User {}: 0 hits in Top {} for genre '{}'",
                            user.user_id, actual_k, user.pos_dominant_genre
                        );
                    }
                }
            }

            // ============ test for genre purity of recommended  movies =======================
            let user_ids = vec![user.user_id];
            let user_req_opt = user_db.get_request(&user_ids, &timestamps);
            assert!(user_req_opt.is_some(), "User ID {} should exist in database", user.user_id);
            let tonic_req = user_req_opt.unwrap();

            let mut active_client = client.clone();
            let response_response = active_client
                //.predict(tonic::Request::new(tonic_req.into_inner()))
                .predict(tonic_req)
                .await
                .map_err(|err| Status::internal(format!("ranking request failed: {}", err)))?;

            let response = response_response.into_inner();

            // Already truncated to top_k by the service
            let retrieved_ids = response.movie_ids;
            let actual_k = retrieved_ids.len();

            if actual_k > 0 {
                let mut hits = 0;

                for &movie_id in &retrieved_ids {
                    if let Some(movie) = context.movies.get(&movie_id) {
                        if movie.genres.contains(&user.pos_dominant_genre) {
                            hits += 1;
                        }
                    }
                }

                let user_purity = hits as f32 / actual_k as f32;
                e2e_total_purity += user_purity;
                e2e_purity_eval_count += 1;

                // Optional: Log cases where the end-to-end system completely missed the mark
                if user_purity == 0.0 {
                    println!(
                        "[E2E Purity Warning] User {}: 0 hits in Top {} for genre '{}'. ANN might be drifting.",
                        user.user_id, actual_k, user.pos_dominant_genre
                    );
                }
            }

            print!("\rProgress: {}/{} users tested", i + 1, context.polarized_users.len());
            io::stdout().flush().unwrap();
        }

        println!("\n\n=== Pairwise Preference Results ===");
        println!("Total Valid Pair Comparisons: {}", valid_tests);

        if valid_tests == 0 {
            println!("No valid matched pairs found.");
            return Ok(0.0);
        }

        let win_rate = wins as f64 / valid_tests as f64;
        println!("Pairwise Positive Genre Win Rate (random pos & neg pairs, ranker-only):  {:.2}% ({}/ {})", win_rate * 100.0, wins, valid_tests);

        println!("\n=== Listwise Ranking Results (random pos & neg 50%/50%, ranker-only) ===");
        if purity_eval_count > 0 {
            let average_purity = (purity_total / purity_eval_count as f32) * 100.0;
            println!("Top-{} Genre Purity: {:.2}%", top_k, average_purity);
        }

        println!("\n=== End-to-End Pipeline Results (ANN + Ranker) ===");
        if e2e_purity_eval_count > 0 {
            let average_purity = (e2e_total_purity / e2e_purity_eval_count as f32) * 100.0;
            println!("Top-K Dominant Genre Purity: {:.2}%", average_purity);
        }

        println!("\n=>Results of end-to-end hit rate ratio > purity ratios shows that the ANN search
                is usefully providing a pool to choose from.");

        println!("\n=>Results near 50% show either that: \ngenres is not an important feature for \
              the models,\nor that the random choice of an unwatched movies from the dominant genre \
              is not a good recommendation for the user\n  (similarly for negatives not being a good anti-recommendation), \
              \nor the model is not specializing enough for polarized users.\
              ");

        Ok(win_rate)
    }


    fn sample_negative_genre(dist: &[(String, u32)]) -> Option<String> {
        if dist.is_empty() { return None; }

        let total_weight: u32 = dist.iter().map(|(_, count)| count).sum();
        let mut rng = rand::thread_rng();

        // Generate a random threshold between 0 and total_weight
        let threshold = rng.gen_range(0..total_weight);

        let mut cumulative = 0;
        for (genre, count) in dist {
            cumulative += count;
            if threshold < cumulative {
                return Some(genre.clone());
            }
        }

        // Fallback in case of floating point/rounding weirdness (rare with ints)
        dist.last().map(|(g, _)| g.clone())
    }

}