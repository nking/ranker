use std::path::PathBuf;

// Assuming your UserRequest is accessible here
pub mod helper {
    // Tell Rust to literally include the code from helper.rs here
    include!("helper.rs");
    include!("helper_users.rs");
}
use inference_engine::app_config::AppConfig;
use inference_engine::orchestrator::Orchestrator;
use helper::{get_train_val_test_liked_uris, DataSize, get_most_frequent_users};
use crate::helper::get_tiny_config_json_uri;

struct TestHarness {
    orchestrator: Orchestrator,
    test_uris: Vec<String>,
    ratings_uris: Vec<String>,
}

impl TestHarness {
    async fn new() -> Self {
        println!("\n[SETUP]: Initializing test resource...");
        let config_path = get_tiny_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let query_uri = config.query_uri.clone();
        let ranker_uri = config.ranker_uri.clone();
        let ranker_n_local_devices = config.ranker_n_local_devices;
        let top_k = config.top_k;
        let user_db_path: PathBuf = config.user_db_path.clone();
        let persisted_index_path : PathBuf = config.persisted_index_path.clone();
        let _params_json_uri : String = config.params_json_path;

        let movie_embeddings_uri : String = config.movie_embeddings_path;

        // we want to be able to test against this recommender, so don't include the test uris
        let ratings_map = get_train_val_test_liked_uris(DataSize::Tiny3, false);
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
            config.query_metadata_uri,
            config.ranker_metadata_uri,
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

        Self {
            orchestrator: orchestrator, test_uris: test_ratings_uris, ratings_uris: ratings_uris
        }
    }
}

// --- TEARDOWN LOGIC ---
impl Drop for TestHarness {
    fn drop(&mut self) {
        println!("[TEARDOWN]");
    }
}

#[cfg(test)]
mod orchestrator_tests {
    use rustc_hash::FxHashMap;
    // Bring everything from the outer scope (TestHarness, helper functions, etc.) into the test module
    use super::*;

    //bring the gRPC trait into scope so its methods (.predict) are visible
    use inference_engine::pb::recommender_service_server::RecommenderService;
    use inference_engine::pb::{RankedMovies, UsersRequest};
    use tonic::Response;
    use inference_engine::user_history::{_testable_build_map_async, UserMapEntry};

    #[tokio::test(flavor = "multi_thread")]
    async fn test_orchestrator() -> Result<(), Box<dyn std::error::Error>>{
        // Setup runs here
        let harness = TestHarness::new().await;

        // ===================================
        //   get a couple of test users
        // ==================================
        let n_users : usize = 2;
        let temp_vec: Vec<&str> = harness.test_uris.iter().map(|s| s.as_str()).collect();
        let slice: &[&str] = &temp_vec;
        let user_ids = get_most_frequent_users(slice, n_users).unwrap();
        let tmp_timestamps : Vec<i64> = vec![2524608000; n_users]; // for year 2050
        let users_req: UsersRequest = harness.orchestrator.get_users_request(&user_ids, &tmp_timestamps)
            .await?.into_inner();

        let temp_vec: Vec<&str> = harness.ratings_uris.iter().map(|s| s.as_str()).collect();
        let slice: &[&str] = &temp_vec;
        let user_history_map : (FxHashMap<i32, UserMapEntry>, usize) =_testable_build_map_async(slice).await;
        let mut user_histories = Vec::with_capacity(user_ids.len());
        user_histories.extend(
            user_ids.iter()
                .filter_map(|id| user_history_map.0.get(id))
                .cloned() // Omit .cloned() if you want a Vec<&UserMapEntry> instead
        );

        let mid_points: Vec<usize> = user_histories.iter()
            .map(|mapentry| mapentry.timestamps.len()/2).collect();

        let mock_single_request = UsersRequest {
            user_ids: vec![user_ids[0]],
            genders: vec![users_req.genders[0].to_string()],
            occupations: vec![users_req.occupations[0]],
            ages: vec![users_req.ages[0]],
            timestamps: vec![user_histories[0].timestamps[mid_points[0]]],
            n_users: 1
        };

        let tonic_req = tonic::Request::new(mock_single_request);

        // Call .predict() directly on _harness.orchestrator (DO NOT move it out with `let orchestrator = ...`)
        let results: Result<Response<RankedMovies>, tonic::Status> =
            harness.orchestrator.predict(tonic_req).await;

        assert!(results.is_ok(), "Prediction failed: {:?}", results.err());
        let ranked_movies = results.unwrap().into_inner();

        assert_eq!(harness.orchestrator.top_k as u32, ranked_movies.num_candidates);
        assert_eq!(ranked_movies.user_ids.len(), 1);
        assert_eq!(ranked_movies.scores.len(), 1 * harness.orchestrator.top_k);
        assert_eq!(ranked_movies.movie_ids.len(), 1 * harness.orchestrator.top_k);

        assert_eq!(user_ids[0], ranked_movies.user_ids[0]);


        // ================================
        let mock_batch_request = UsersRequest {
            user_ids: user_ids.clone(),
            genders: vec![users_req.genders[0].to_string(), users_req.genders[1].to_string()],
            occupations: vec![users_req.occupations[0], users_req.occupations[1]],
            ages: vec![users_req.ages[0], users_req.ages[1]],
            timestamps: vec![user_histories[0].timestamps[mid_points[0]], user_histories[1].timestamps[mid_points[1]]],
            n_users: 2
        };

        let tonic_req = tonic::Request::new(mock_batch_request);

        // Call .predict() directly on _harness.orchestrator (DO NOT move it out with `let orchestrator = ...`)
        let results: Result<Response<RankedMovies>, tonic::Status> =
            harness.orchestrator.predict(tonic_req).await;

        assert!(results.is_ok(), "Prediction failed: {:?}", results.err());
        let ranked_movies = results.unwrap().into_inner();

        assert_eq!(harness.orchestrator.top_k as u32, ranked_movies.num_candidates);
        assert_eq!(ranked_movies.user_ids.len(), user_ids.len());
        assert_eq!(ranked_movies.scores.len(), user_ids.len() * harness.orchestrator.top_k);
        assert_eq!(ranked_movies.movie_ids.len(), user_ids.len() * harness.orchestrator.top_k);

        println!("Got {} recommendations!", ranked_movies.movie_ids.len());
        for i in 0..ranked_movies.movie_ids.len() {
            println!("{} {}", ranked_movies.movie_ids[i], ranked_movies.scores[i]);
        }

        assert_eq!(user_ids[0], ranked_movies.user_ids[0]);
        assert_eq!(user_ids[1], ranked_movies.user_ids[1]);

        Ok(())

        // Teardown automatically runs here when `_harness` goes out of scope at the end of the test function
    }
}