#[cfg(test)]
mod app_config_tests {
    //use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile; // Requires adding `tempfile = "3"` to Cargo.toml [dev-dependencies]
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }
    use inference_engine::app_config::AppConfig;
    use crate::app_config_tests::helper::get_config_json_uri;

    #[test]
    fn test_config_deserialization() {
        let json_data = r#"{
              "server_addr": "0.0.0.0:50051",
              "query_uri": "http://172.17.0.1:8500",
              "ranker_uri": "http://172.17.0.1:8510",
              "params_json_path": "../../../src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/1/assets.extra/metadata_batch.json",
              "movie_embeddings_path": "../../../src/test/resources/data/tower_versions/1/movie_emb-00000-of-00001.parquet",
              "movie_tiers_path": "../../../src/test/resources/data/movie_tiers.json",
              "ratings_uris": [
                "../../../src/test/resources/data/tiny3/ratings_train_liked.parquet",
                "../../../src/test/resources/data/tiny3/ratings_train_3.parquet",
                "../../../src/test/resources/data/tiny3/ratings_train_disliked.parquet",
                "../../../src/test/resources/data/tiny3/ratings_val_liked.parquet",
                "../../../src/test/resources/data/tiny3/ratings_val_3.parquet",
                "../../../src/test/resources/data/tiny3/ratings_val_disliked.parquet"
              ],
              "user_db_path":"../../../src/test/resources/data/users.bin",
              "movies_path": "../../../src/test/resources/data/movies.parquet",
              "ranker_n_local_devices": 1,
              "top_k": 20,
              "persisted_index_path": "./target/movie_embeddings_indexer",
              "query_saved_models_uri": "../../../src/test/resources/model_repositories/saved_model_formats/bi-encoder/query/",
              "ranker_saved_models_uri": "../../../src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/",
              "ranker_serving_is_batched" : "true"
        }"#;

        // Create a temporary file to test the load function
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", json_data).unwrap();

        let config = AppConfig::load_from_file(temp_file.path().to_str().unwrap()).unwrap();

        assert_eq!(config.top_k, 50);
        assert_eq!(config.ranker_n_local_devices, 2);
        assert_eq!(config.ratings_uris.len(), 2);
        assert_eq!(config.server_addr.port(), 50051);
    }

    #[test]
    pub fn test_default_config() {
        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();
        assert_eq!(config.top_k, 20);
        assert_eq!(config.ranker_n_local_devices, 1);
        assert_eq!(config.ratings_uris.len(), 6);
        assert_eq!(config.server_addr.port(), 50051);
    }
}