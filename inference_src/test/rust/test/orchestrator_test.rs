#[cfg(test)]
mod orchestrator_tests {
    use std::collections::HashMap;
    use std::error::Error;
    use std::fs::File;
    use std::io::BufReader;
    use serde_json::Value;
    use tonic::{Request, Response};
    use inference_engine::client::{QueryModelClient, RankerModelClient};
    use inference_engine::graph_builder::{create_fake_padded_super_batch, JraphGraph};
    use super::*;
    // Assuming your UserRequest is accessible here
    use inference_engine::pb::{RankedMovies, UserRequest};
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }
    use helper::{get_param_json_uri, get_embeddings_uris};
    use inference_engine::orchestrator::Orchestrator;
    use inference_engine::pb::recommender_service_server::RecommenderService;
    use crate::orchestrator_tests::helper::{get_movies_uri, get_train_val_test_liked_uris, DataSize};

    #[tokio::test]
    async fn test_orchestrator() {
        let query_uri = "http://172.17.0.1:8500";
        let ranker_uri = "http://172.17.0.1:8510";

        let (user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris();

        let params_json_uri = get_param_json_uri();
        let file = File::open(params_json_uri).unwrap();
        let reader = BufReader::new(file);
        // 3. Deserialize JSON directly into a HashMap
        let dict: HashMap<String, Value> = serde_json::from_reader(reader).unwrap();
        let max_history = dict.get("max_history")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_candidates = dict.get("num_candidates")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let embed_len = dict.get("embed_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_catalog_users = dict.get("num_catalog_users")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

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
            &query_uri,
            &ranker_uri,
            &movie_embeddings_uri,
            ratings_uris,
            max_history,
            num_candidates,
            num_catalog_users
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
        
    }

}