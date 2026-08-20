#[cfg(test)]
mod ranker_model_metadata_tests {
    //use super::*;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    use inference_engine::ranker_model_metadata::RankerModelMetadata;
    use crate::ranker_model_metadata_tests::helper::{get_ranker_metadata_single_uri, get_ranker_metadata_batch_uri};

    #[test]
    pub fn test_default_config() {
        let config_path = get_ranker_metadata_single_uri();
        let metadata = RankerModelMetadata::load_from_file(&config_path).unwrap();
        
        assert!(metadata.num_candidates > 0, "num_candidates should be populated");
        assert!(metadata.embed_len > 0, "embed_len should be populated");
        assert!(metadata.max_history > 0, "max_history should be populated");
        assert!(metadata.max_edges > 0, "max_edges should be populated");
        assert!(metadata.max_nodes > 0, "max_nodes should be populated");
        assert!(metadata.max_graphs > 0, "max_graphs should be populated");
        assert!(!metadata.git_commit_hash.is_empty(), "git_commit_hash should not be empty");
        assert!(!metadata.signature_name.is_empty(), "sinature_name should not be empty");
        
        assert_eq!(metadata.batch_size, 1, "batch_size should be 1");
    }


    #[test]
    pub fn test_batch_config() {
        let config_path = get_ranker_metadata_batch_uri();
        let metadata = RankerModelMetadata::load_from_file(&config_path).unwrap();

        assert!(metadata.num_candidates > 0, "num_candidates should be populated");
        assert!(metadata.embed_len > 0, "embed_len should be populated");
        assert!(metadata.max_history > 0, "max_history should be populated");
        assert!(metadata.max_edges > 0, "max_edges should be populated");
        assert!(metadata.max_nodes > 0, "max_nodes should be populated");
        assert!(metadata.max_graphs > 0, "max_graphs should be populated");
        assert!(!metadata.git_commit_hash.is_empty(), "git_commit_hash should not be empty");
        assert!(!metadata.signature_name.is_empty(), "sinature_name should not be empty");

        assert!(metadata.batch_size > 1, "batch_size should be  > 1, e.g. 256");
        
    }
}