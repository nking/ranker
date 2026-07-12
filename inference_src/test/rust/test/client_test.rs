#[cfg(test)]
mod client_tests {
    use std::error::Error;
    use inference_engine::client::{QueryModelClient, RankerModelClient};
    use inference_engine::graph_builder::{create_fake_padded_super_batch, JraphGraph};
    use super::*;
    // Assuming your UserRequest is accessible here
    use inference_engine::pb::UserRequest;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }
    use crate::client_tests::helper::{get_embeddings_uris};

    #[tokio::test]
    async fn test_query_model_connection() {
        let uri = "http://172.17.0.1:8500";

        let client = QueryModelClient::new(uri).await;

        let mock_request = UserRequest {
            user_id: 42,
            gender: "M".to_string(),
            occupation: 10,
            age: 25,
            timestamp: 1620000000,
        };

        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<(Vec<f32>), Box<dyn Error>> = client.get_user_embedding(mock_request).await;

        assert!(result.is_ok(), "Failed to get embedding: {:?}", result.err());

        let embedding = result.unwrap();
        println!("Received embedding of length: {}", embedding.len());

        // Assert the expected dimension length (e.g., 128)
        assert!(!embedding.is_empty(), "Embedding vector is empty!");
    }

    #[tokio::test]
    async fn test_ranker_model_connection() {
        let uri = "http://172.17.0.1:8510";

        let client = RankerModelClient::new(uri).await;

        let batch_size = 3;
        let max_history = 4;
        let num_candidates = 5;
        let user_id_range = (1, 10);
        let movie_id_range = (1, 10);
        let n_local_devices = 1;

        let (user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris();

        let padded_super_graph : JraphGraph  = create_fake_padded_super_batch(batch_size,
            max_history, num_candidates, user_id_range,
            movie_id_range, n_local_devices,
            &user_embeddings_uri, &movie_embeddings_uri
        );

        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<(Vec<f32>), Box<dyn Error>> = client.get_candidate_ranks(
            padded_super_graph, 16).await;

        assert!(result.is_ok(), "Failed to get ranks: {:?}", result.err());

        let ranks = result.unwrap();
        println!("Received ranks: {}", ranks.len());

        // Assert the expected dimension length (e.g., 128)
        assert!(!ranks.is_empty(), "ranks vector is empty!");
    }
}