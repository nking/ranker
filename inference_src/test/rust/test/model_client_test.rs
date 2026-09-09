#[cfg(test)]
mod client_tests {
    use std::collections::HashMap;
    use std::error::Error;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::PathBuf;
    use serde_json::Value;
    use inference_engine::app_config::AppConfig;
    use inference_engine::model_client::{QueryModelClient, RankerModelClient};
    use inference_engine::graph_builder::{create_fake_padded_super_batch, JraphGraph};
    //use super::*;
    // Assuming your UserRequest is accessible here
    use inference_engine::pb::UsersRequest;
    use inference_engine::query_model_metadata::QueryModelMetadata;
    use inference_engine::ranker_model_metadata::RankerModelMetadata;

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }
    use crate::client_tests::helper::{get_config_json_uri, get_embeddings_metadata_uris, get_embeddings_uris};

    #[tokio::test]
    async fn test_query_model_connection() {

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let query_metadata = QueryModelMetadata::load_from_file(&config.query_metadata_uri).unwrap();

        let client = QueryModelClient::new(config.query_uri, query_metadata.embed_len).await;

        let mock_request = UsersRequest {
            user_ids: vec![42],
            genders: vec!["M".to_string()],
            occupations: vec![10],
            ages: vec![25],
            timestamps: vec![1620000000],
            n_users: 1,
        };

        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<Vec<f32>, Box<dyn Error>> = client.get_users_embeddings(mock_request).await;

        assert!(result.is_ok(), "Failed to get embedding: {:?}", result.err());

        let embedding = result.unwrap();
        assert_eq!(client.embed_len, embedding.len());

        // Assert the expected dimension length (e.g., 128)
        assert!(!embedding.is_empty(), "Embedding vector is empty!");
    }

    #[tokio::test]
    async fn test_ranker_model_connection() -> Result<(), Box<dyn std::error::Error>>{

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        // the default confg is for the batch ranker model, so change to the single inference model:
        let ranker_metadat_uri = config.ranker_metadata_uri;
        let single_uri = ranker_metadat_uri.replace("batch", "single");

        let ranker_metadata = RankerModelMetadata::load_from_file(&single_uri).unwrap();

        let client = RankerModelClient::new(config.ranker_uri, ranker_metadata.clone()).await;

        let _top_k = config.top_k;
        let _user_db_path: PathBuf = config.user_db_path.clone();
        let _persisted_index_path : PathBuf = config.persisted_index_path.clone();
        let params_json_uri : String = config.params_json_path;

        let max_history = ranker_metadata.max_history;
        let num_candidates = ranker_metadata.num_candidates;
        let _num_catalog_users = ranker_metadata.num_catalog_users;

        // the placeholder mode used:
        //"signature_name": "serving_batch", "batch_size": 256, "max_history": 60, "num_candidates": 60, "max_nodes": 31040, "max_edges": 30784, "max_graphs": 258, "embed_len": 16
        //    the range of movie_id_range must be >= (num_history + num_candidates + 1)
        let batch_size = 1;
        let user_id_range = (1, 10);
        let movie_id_range = (6041, 6041 + (max_history + num_candidates + 2));
        let n_local_devices = 1;

        let (user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris();

        let ranker_batch_size : usize = client.metadata.batch_size;

        let padded_super_graph : JraphGraph  = create_fake_padded_super_batch(
            batch_size,
            ranker_batch_size,
            max_history, num_candidates, user_id_range,
            movie_id_range, n_local_devices,
            &user_embeddings_uri, &movie_embeddings_uri
        );

        let (user_emb_metadata_uri, _movie_emb_metadata_uri) : (String, String)
            = get_embeddings_metadata_uris();

        let json_content = tokio::fs::read_to_string(&user_emb_metadata_uri).await?;
        let dict: HashMap<String, Value> = serde_json::from_str(&json_content)?;
        let embed_len = dict.get("embed_dim").and_then(|v| v.as_u64()).unwrap();

        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<Vec<f32>, Box<dyn Error>> = client.get_candidate_ranks(
            padded_super_graph, embed_len as usize).await;

        assert!(result.is_ok(), "Failed to get ranks: {:?}", result.err());

        let ranks: Vec<f32> = result.unwrap();
        println!("Received ranks: {}", ranks.len());

        // the batch_size * num_candidates are the scored values, all else are dummy scores
        let ranks : &[f32] = &ranks[0..batch_size * num_candidates];

        Ok(())

    }
}