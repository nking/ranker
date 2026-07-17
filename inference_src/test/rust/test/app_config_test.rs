#[cfg(test)]
mod app_config_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile; // Requires adding `tempfile = "3"` to Cargo.toml [dev-dependencies]
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }
    use crate::app_config_tests::helper::{get_config_json_uri};

    use inference_engine::app_config::AppConfig;

    #[test]
    fn test_config_deserialization() {
        let json_data = r#"{
            "server_addr": "127.0.0.1:50051",
            "query_uri": "http://localhost:8500",
            "ranker_uri": "http://localhost:8510",
            "params_json_path": "./params.json",
            "movie_embeddings_path": "./movies.bin",
            "ratings_uris": ["file1.csv", "file2.csv"],
            "ranker_n_local_devices": 2,
            "top_k": 50,
            "persisted_index_path": "./index_dir"
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