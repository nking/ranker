#[cfg(test)]
mod client_tests {
    use std::collections::HashMap;
    use std::error::Error;
    use std::path::PathBuf;
    use serde_json::Value;
    use inference_engine::app_config::AppConfig;
    use inference_engine::model_client::{CandidateModelClient, QueryModelClient, RankerModelClient, RerankerModelClient};
    use inference_engine::ranker_graph_builder::{create_fake_padded_super_batch, JraphGraph};
    use inference_engine::tensorflow::serving::model_spec::VersionChoice;
    //use super::*;
    // Assuming your UserRequest is accessible here
    use inference_engine::pb::{UsersRequest, MoviesRequest};
    use inference_engine::query_model_metadata::QueryModelMetadata;
    use inference_engine::ranker_model_metadata::RankerModelMetadata;

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }
    use crate::client_tests::helper::{get_config_json_uri, get_embeddings_metadata_uris, get_embeddings_uris, get_project_dir};

    #[tokio::test]
    async fn test_query_model_connection() {

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let q_ver = 1;
        let r_ver = 1;

        let query_metadata_uri = format!(
            "{}/{}/assets.extra/{}",
            config.query_models_path.trim_end_matches('/'), q_ver, "hyperparameters.json");

        let query_metadata = QueryModelMetadata::load_from_file(&query_metadata_uri).unwrap();

        let client = QueryModelClient::new(config.query_uri.clone(), config.query_models_path.clone()).await;

        let mock_request = UsersRequest {
            user_ids: vec![42],
            genders: vec!["M".to_string()],
            occupations: vec![10],
            ages: vec![25],
            timestamps: vec![1620000000],
            n_users: 1,
            k : None,
            query_model_version:Some(q_ver),
            ranker_model_version:Some(r_ver)
        };

        let query_model_version_choice = Some(VersionChoice::Version(1));

        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>
            = client.get_users_embeddings(mock_request, query_model_version_choice).await;

        assert!(result.is_ok(), "Failed to get user embedding: {:?}", result.err());

        let embedding = result.unwrap();
        assert_eq!(query_metadata.embed_len, embedding.len());

        // Assert the expected dimension length (e.g., 128)
        assert!(!embedding.is_empty(), "User Embedding vector is empty!");

        let latest_version = client.get_latest_model_version().await.expect("Couldn't get latest model version");

        let expected_latest_version: i64 = std::fs::read_dir(&config.query_models_path)
            .expect("Failed to read model repository directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                // Ignore files; only keep directories
                if !entry.file_type().ok()?.is_dir() {
                    return None;
                }
                // Parse directory name into i64 (ignores non-numeric directory names)
                entry.file_name().to_str()?.parse::<i64>().ok()
            })
            .max()
            .expect("No valid numeric version subdirectories found");

        assert_eq!(latest_version, expected_latest_version);

        let q_version_choice = Some(VersionChoice::Version(q_ver));
        let metadata = client.get_or_fetch_metadata(q_version_choice).await;
        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();

        let r = client.update_latest_model_version().await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_candidate_model_connection() {

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let c_ver = 1;
        let model_version_choice = Some(VersionChoice::Version(c_ver));

        let client = CandidateModelClient::new(config.candidate_uri,
            config.candidate_models_path).await;

        let metadata = client.get_or_fetch_metadata(model_version_choice.clone()).await;
        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();

        let mock_request = MoviesRequest {
            movie_ids: vec![6041],
            genres: vec!["Action|Sci-Fi".to_string()],
            n_movies : 1,
            candidate_model_version:Some(1),
        };


        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>
            = client.get_movies_embeddings(mock_request, model_version_choice.clone()).await;

        assert!(result.is_ok(), "Failed to get movie embedding: {:?}", result.err());

        let embedding = result.unwrap();
        assert_eq!(metadata.embed_len, embedding.len());

        // Assert the expected dimension length (e.g., 128)
        assert!(!embedding.is_empty(), "Movie Embedding vector is empty!");

        let latest_version = client.get_latest_model_version().await.expect("Couldn't get latest model version");

        let expected_latest_version: i64 = std::fs::read_dir(client.get_models_path())
            .expect("Failed to read model repository directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                // Ignore files; only keep directories
                if !entry.file_type().ok()?.is_dir() {
                    return None;
                }
                // Parse directory name into i64 (ignores non-numeric directory names)
                entry.file_name().to_str()?.parse::<i64>().ok()
            })
            .max()
            .expect("No valid numeric version subdirectories found");

        assert_eq!(latest_version, expected_latest_version);

        let r = client.update_latest_model_version().await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_ranker_model_connection() -> Result<(), Box<dyn std::error::Error>>{

        let ver = 1;
        let model_version_choice = Some(VersionChoice::Version(ver));

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let filename = if config.ranker_serving_is_batched {
            "metadata_batch.json"
        } else {
            "metadata_single.json"
        };
        let ranker_metadata_uri = format!(
            "{}/{}/assets.extra/{}",
            config.ranker_models_path.trim_end_matches('/'), ver, filename);

        // the default config is for the batch ranker model, so change to the single inference model:
        let single_uri = ranker_metadata_uri.replace("batch", "single");

        let ranker_metadata = RankerModelMetadata::load_from_file(&single_uri).unwrap();

        // instantiated for not using the batch configuration:
        let client = RankerModelClient::new(config.ranker_uri.clone(),
            config.ranker_models_path.clone(), false).await;

        let _top_k = config.ranker_top_k;
        let _user_db_path: PathBuf = config.user_db_path.clone();
        let _persisted_index_path : PathBuf = config.persisted_index_path.clone();

        let max_history = ranker_metadata.max_history;
        let num_candidates = ranker_metadata.num_candidates;
        let _num_catalog_users = ranker_metadata.num_catalog_users;

        // the placeholder mode used:
        //"signature_name": "serving_batch", "batch_size": 256, "max_history": 60, "num_candidates": 60, "max_nodes": 31040, "max_edges": 30784, "max_graphs": 258, "embed_len": 16
        //    the range of movie_id_range must be >= (num_history + num_candidates + 1)
        let batch_size : usize = ranker_metadata.batch_size;
        let n_users = 1;
        assert_eq!(batch_size, 1);
        let user_id_range = (1, 10);
        let movie_id_range = (6041, 6040 + (max_history + num_candidates + 2));
        let n_local_devices = 1;


        let (user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris(ver as i32);

        let padded_super_graph : JraphGraph  = create_fake_padded_super_batch(
            n_users,
            batch_size,
            max_history, num_candidates, user_id_range,
            movie_id_range, n_local_devices,
            &user_embeddings_uri, &movie_embeddings_uri
        );

        let (user_emb_metadata_uri, _movie_emb_metadata_uri) : (String, String)
            = get_embeddings_metadata_uris(ver as i32);

        let json_content = tokio::fs::read_to_string(&user_emb_metadata_uri).await?;
        let dict: HashMap<String, Value> = serde_json::from_str(&json_content)?;
        let embed_len = dict.get("embed_dim").and_then(|v| v.as_u64()).unwrap();

        let query_model_version = Some(VersionChoice::Version(1));

        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>
            = client.get_candidate_ranks(padded_super_graph, embed_len as usize, query_model_version.clone()).await;

        assert!(result.is_ok(), "Failed to get ranks: {:?}", result.err());

        let ranks: Vec<f32> = result.unwrap();
        // the batch_size * num_candidates are the scored values, all else are dummy scores, piled up at end
        let _ranks : &[f32] = &ranks[0..batch_size * num_candidates];
        assert_eq!(_ranks.len(), batch_size * num_candidates);

        // ============================================================================================
        // switch to a batch client to test diagnostics

        let ranker_metadata = RankerModelMetadata::load_from_file(&ranker_metadata_uri).unwrap();
        // instantiated for not using the batch configuration:
        let client = RankerModelClient::new(config.ranker_uri.clone(),
            config.ranker_models_path.clone(), true).await;

        let ranker_batch_size : usize = ranker_metadata.batch_size;
        let out_features = ranker_metadata.out_features;

        let padded_super_graph : JraphGraph  = create_fake_padded_super_batch(
            n_users,
            ranker_batch_size,
            max_history, num_candidates, user_id_range,
            movie_id_range, n_local_devices,
            &user_embeddings_uri, &movie_embeddings_uri
        );

        let result = client.get_candidate_diagnostics(
            padded_super_graph, embed_len as usize, query_model_version.clone()).await;
        assert!(result.is_ok(), "Failed to get ranks: {:?}", result.err());
        let (mut outputs, mut user_reprs, mut cand_reprs) = result
            .expect("error with request or response for ranker diagnostics");

        outputs.truncate(batch_size * num_candidates);
        user_reprs.truncate(batch_size * out_features);
        cand_reprs.truncate(batch_size * num_candidates * out_features);

        let latest_version = client.get_latest_model_version().await.expect("Couldn't get latest model version");

        let model_dir = get_project_dir()
            .expect("Failed to get project directory")
            .join("src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/");

        let expected_latest_version: i64 = std::fs::read_dir(&model_dir)
            .expect("Failed to read model repository directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                // Ignore files; only keep directories
                if !entry.file_type().ok()?.is_dir() {
                    return None;
                }
                // Parse directory name into i64 (ignores non-numeric directory names)
                entry.file_name().to_str()?.parse::<i64>().ok()
            })
            .max()
            .expect("No valid numeric version subdirectories found");

        assert_eq!(latest_version, expected_latest_version);

        let metadata = client.get_or_fetch_metadata(model_version_choice.clone()).await;
        assert!(metadata.is_ok());

        let r = client.update_latest_model_version().await;
        assert!(r.is_ok());

        Ok(())

    }


    #[tokio::test]
    async fn test_reranker_model_connection() -> Result<(), Box<dyn std::error::Error>>{

        let ver = 1;
        let version_option = Some(ver);

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let client = RerankerModelClient::new(config.reranker_uri.clone(),
            config.reranker_models_path.clone()).await;

        let latest_version = client.get_latest_model_version().await.expect("Couldn't get latest model version");

        let expected_latest_version: i64 = std::fs::read_dir(&config.reranker_models_path)
            .expect("Failed to read model repository directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                // Ignore files; only keep directories
                if !entry.file_type().ok()?.is_dir() {
                    return None;
                }
                // Parse directory name into i64 (ignores non-numeric directory names)
                entry.file_name().to_str()?.parse::<i64>().ok()
            })
            .max()
            .expect("No valid numeric version subdirectories found");

        assert_eq!(latest_version, expected_latest_version);

        let metadata = client.get_or_fetch_metadata(
            version_option.clone()
        ).await;
        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();

        let r = client.update_latest_model_version().await;
        assert!(r.is_ok());

        let metadata = client.get_or_fetch_metadata(version_option.clone()).await;
        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();

        let models_path = client.get_models_path();
        assert_eq!(models_path, config.reranker_models_path);

        let r = client.is_model_ready(version_option.clone()).await);\\;
        assert!(r.is_ok());

        //TODO: get_reranked_scores

        Ok(())

    }
}