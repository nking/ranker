#[cfg(test)]
mod orchestrator_tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::PathBuf;
    use serde_json::Value;
    use tonic::{Response};

    // Assuming your UserRequest is accessible here
    use inference_engine::pb::{RankedMovies, UserRequest};
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }
    use inference_engine::app_config::AppConfig;
    use inference_engine::orchestrator::Orchestrator;
    use inference_engine::pb::recommender_service_server::RecommenderService;
    use crate::orchestrator_tests::helper::{get_config_json_uri, get_train_val_test_liked_uris, DataSize};

    #[tokio::test]
    async fn test_orchestrator() {

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let query_uri = config.query_uri.clone();
        let ranker_uri = config.ranker_uri.clone();
        let ranker_n_local_devices = config.ranker_n_local_devices;
        let top_k = config.top_k;
        let user_db_path: PathBuf = config.user_db_path.clone();
        let persisted_index_path : PathBuf = config.persisted_index_path.clone();
        let params_json_uri : String = config.params_json_path;

        let movie_embeddings_uri : String = config.movie_embeddings_path;

        let file = File::open(params_json_uri).unwrap();
        let reader = BufReader::new(file);
        let dict: HashMap<String, Value> = serde_json::from_reader(reader)
            .expect("reading json file of model params into a dictionary");

        let max_history = dict.get("max_history").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let num_candidates = dict.get("num_candidates").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let num_catalog_users = dict.get("num_catalog_users").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

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
            &movie_embeddings_uri,
            ratings_uris,
            max_history,
            num_candidates,
            num_catalog_users,
            ranker_n_local_devices,
            top_k,
            persisted_index_path,
            user_db_path
        ).await.unwrap();

        let mock_request = UserRequest {
            user_id: 42,
            gender: "M".to_string(),
            occupation: 10,
            age: 25,
            timestamp: 1620000000,
        };

        let tonic_req = tonic::Request::new(mock_request);

        let results : Result<Response<RankedMovies>, tonic::Status>
            = orchestrator.predict(tonic_req).await;

        assert!(results.is_ok(), "Prediction failed: {:?}", results.err());
        let response = results.unwrap().into_inner();
        println!("Got {} recommendations!", response.movie_ids.len());
        for i in 0..response.movie_ids.len() {
            println!("{} {}", response.movie_ids[i], response.scores[i]);
        }
        assert_eq!(42, response.user_id)

    }

}