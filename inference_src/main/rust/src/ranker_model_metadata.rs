use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

/// Command-line arguments and environment variables
#[derive(Parser, Debug, Deserialize, Clone)]
#[command(name = "recommender-grpc")]
#[command(about = "gRPC Recommender Service Orchestrator", long_about = None)]
pub struct RankerModelMetadata {

    /// saved_model signature name
    #[arg(long, env = "SIGNATURE_NAME", default_value = "serving_default")]
    pub signature_name: String,

    #[arg(long, env = "BATCH_SIZE",  default_value_t = 1)]
    pub batch_size : usize,

    #[arg(long, env = "MAX_NODES",  default_value_t = 192)]
    pub max_nodes : usize,

    #[arg(long, env = "MAX_EDGES",  default_value_t = 192)]
    pub max_edges : usize,

    #[arg(long, env = "MAX_GRAPHS",  default_value_t = 3)]
    pub max_graphs : usize,

    #[arg(long, env = "MAX_HISTORY",  default_value_t = 80)]
    pub max_history : usize,

    #[arg(long, env = "NUM_CANDIDATES",  default_value_t = 70)]
    pub num_candidates : usize,

    #[arg(long, env = "EMBED_LEN",  default_value_t = 32)]
    pub embed_len : usize,

    #[arg(long, env = "NUM_CATALOG_USERS",  default_value_t = 6040)]
    pub num_catalog_users : usize,

    #[arg(long, env = "NUM_CATALOG_MOVIES",  default_value_t = 3883)]
    pub num_catalog_movies : usize,

    /// git commit hash for code used to train the model
    #[arg(long, env = "GIT_COMMIT_HASH")]
    pub git_commit_hash : String
}

impl RankerModelMetadata {
    /// Loads the configuration from a given file path
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: RankerModelMetadata = serde_json::from_reader(reader)?;
        Ok(config)
    }
}