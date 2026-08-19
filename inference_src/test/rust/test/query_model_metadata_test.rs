#[cfg(test)]
mod query_model_metadata_tests {
    //use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile; // Requires adding `tempfile = "3"` to Cargo.toml [dev-dependencies]
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }
    use crate::query_model_metadata_tests::helper::{get_config_json_uri};

    use inference_engine::app_config::AppConfig;
    use inference_engine::query_model_metadata::QueryModelMetadata;
    use crate::query_model_metadata_tests::helper::{get_query_metadata_uri};

    #[test]
    pub fn test_default_config() {
        let config_path = get_query_metadata_uri();
        let metadata = QueryModelMetadata::load_from_file(&config_path).unwrap();
        
        assert!(metadata.embed_len > 0, "embed_len should be populated");
        assert!(metadata.num_catalog_movies > 0, "num_catalog_movies should be populated");
        assert!(metadata.num_catalog_users > 0, "num_catalog_users should be populated");
        assert!(!metadata.git_commit_hash.is_empty(), "git_commit_hash should not be empty");
    }

}