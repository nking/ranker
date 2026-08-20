use std::path::PathBuf;

// Assuming your UserRequest is accessible here
pub mod helper {
    // Tell Rust to literally include the code from helper.rs here
    include!("helper.rs");
}
use inference_engine::app_config::AppConfig;
use inference_engine::orchestrator::Orchestrator;
use helper::{get_config_json_uri, get_train_val_test_liked_uris, DataSize};

struct TestHarness {
    orchestrator: Orchestrator,
}

impl TestHarness {
    async fn new() -> Self {
        println!("\n[SETUP]: Initializing test resource...");
        let config_path = get_config_json_uri();
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
        let ratings_map = get_train_val_test_liked_uris(DataSize::Tiny, false);
        let ratings_uris: Vec<&str> = vec![
            ratings_map.get("train_liked").unwrap(),
            ratings_map.get("train_3").unwrap(),
            ratings_map.get("train_disliked").unwrap(),
            ratings_map.get("val_liked").unwrap(),
            ratings_map.get("val_3").unwrap(),
            ratings_map.get("val_disliked").unwrap(),
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
        Self {
            orchestrator: orchestrator,
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
    // Bring everything from the outer scope (TestHarness, helper functions, etc.) into the test module
    use super::*;

    //bring the gRPC trait into scope so its methods (.predict) are visible
    use inference_engine::pb::recommender_service_server::RecommenderService;
    use inference_engine::pb::{RankedMovies, UsersRequest};
    use tonic::Response;

    #[tokio::test]
    async fn test_orchestrator() {
        // Setup runs here
        let _harness = TestHarness::new().await;

        let mock_request = UsersRequest {
            user_ids: vec![42],
            genders: vec!["M".to_string()],
            occupations: vec![10],
            ages: vec![25],
            timestamps: vec![1620000000],
            n_users: 1
        };

        let tonic_req = tonic::Request::new(mock_request);

        // Call .predict() directly on _harness.orchestrator (DO NOT move it out with `let orchestrator = ...`)
        let results: Result<Response<RankedMovies>, tonic::Status> =
            _harness.orchestrator.predict(tonic_req).await;

        assert!(results.is_ok(), "Prediction failed: {:?}", results.err());
        let response = results.unwrap().into_inner();

        println!("Got {} recommendations!", response.movie_ids.len());
        for i in 0..response.movie_ids.len() {
            println!("{} {}", response.movie_ids[i], response.scores[i]);
        }

        assert_eq!(42, response.user_ids[0]);

        // Teardown automatically runs here when `_harness` goes out of scope at the end of the test function
    }
}