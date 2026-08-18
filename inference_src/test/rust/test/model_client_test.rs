#[cfg(test)]
mod client_tests {
    use std::collections::HashMap;
    use std::error::Error;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::{Path, PathBuf};
    use serde_json::Value;
    use inference_engine::app_config::AppConfig;
    use inference_engine::model_client::{QueryModelClient, RankerModelClient};
    use inference_engine::graph_builder::{create_fake_padded_super_batch, JraphGraph};
    //use super::*;
    // Assuming your UserRequest is accessible here
    use inference_engine::pb::UserRequest;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }
    use crate::client_tests::helper::{get_config_json_uri, get_embeddings_metadata_uris, get_embeddings_uris};

    #[tokio::test]
    async fn test_query_model_connection() {
        let uri = String::from("http://172.17.0.1:8500");

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
        let result : Result<Vec<f32>, Box<dyn Error>> = client.get_user_embedding(&mock_request).await;

        assert!(result.is_ok(), "Failed to get embedding: {:?}", result.err());

        let embedding = result.unwrap();
        println!("Received embedding of length: {}", embedding.len());

        // Assert the expected dimension length (e.g., 128)
        assert!(!embedding.is_empty(), "Embedding vector is empty!");
    }

    #[tokio::test]
    async fn test_ranker_model_connection() -> Result<(), Box<dyn std::error::Error>>{
        let uri = String::from("http://172.17.0.1:8510");

        let client = RankerModelClient::new(uri).await;let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();


        let top_k = config.top_k;
        let user_db_path: PathBuf = config.user_db_path.clone();
        let persisted_index_path : PathBuf = config.persisted_index_path.clone();
        let params_json_uri : String = config.params_json_path;

        let file = File::open(params_json_uri).unwrap();
        let reader = BufReader::new(file);
        let dict: HashMap<String, Value> = serde_json::from_reader(reader)
            .expect("reading json file of model params into a dictionary");

        let max_history = dict.get("max_history").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let num_candidates = dict.get("num_candidates").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let num_catalog_users = dict.get("num_catalog_users").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        // the placeholder mode used:
        //"signature_name": "serving_batch", "batch_size": 256, "max_history": 60, "num_candidates": 60, "max_nodes": 31040, "max_edges": 30784, "max_graphs": 258, "embed_len": 16
        //    the range of movie_id_range must be >= (num_history + num_candidates + 1)
        let batch_size = 1;
        let user_id_range = (1, 10);
        let movie_id_range = (6041, 6041 + (max_history + num_candidates + 2));
        let n_local_devices = 1;

        let (user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris();

        let padded_super_graph : JraphGraph  = create_fake_padded_super_batch(batch_size,
            max_history, num_candidates, user_id_range,
            movie_id_range, n_local_devices,
            &user_embeddings_uri, &movie_embeddings_uri
        );

        let (user_emb_metadata_uri, movie_emb_metadata_uri) : (String, String)
            = get_embeddings_metadata_uris();

        let json_content = tokio::fs::read_to_string(&user_emb_metadata_uri).await?;
        let dict: HashMap<String, Value> = serde_json::from_str(&json_content)?;
        let embed_len = dict.get("embed_dim").and_then(|v| v.as_u64()).unwrap();

        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<Vec<f32>, Box<dyn Error>> = client.get_candidate_ranks(
            padded_super_graph, embed_len as usize).await;

        assert!(result.is_ok(), "Failed to get ranks: {:?}", result.err());

        let ranks = result.unwrap();
        println!("Received ranks: {}", ranks.len());

        // Assert the expected dimension length (e.g., 128)
        assert!(!ranks.is_empty(), "ranks vector is empty!");

        Ok(())

    }
}