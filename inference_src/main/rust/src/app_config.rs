use std::net::SocketAddr;
use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;
use std::fs::File;
use std::io::BufReader;


/// Command-line arguments and environment variables
#[derive(Parser, Debug, Deserialize, Clone)]
#[command(name = "recommender-grpc")]
#[command(about = "gRPC Recommender Service Orchestrator", long_about = None)]
pub struct AppConfig {
    /// The IP and port for the gRPC server to bind to
    #[arg(long, env = "SERVER_ADDR", default_value = "0.0.0.0:50051")]
    pub server_addr: SocketAddr,

    /// URI for the TFS Query Model
    #[arg(long, env = "QUERY_URI", default_value = "http://172.17.0.1:8500")]
    pub query_uri: String,

    /// URI for the TFS Ranker Model
    #[arg(long, env = "RANKER_URI", default_value = "http://172.17.0.1:8510")]
    pub ranker_uri: String,

    /// Path to the hyperparameters JSON file
    #[arg(long, env = "PARAMS_JSON_PATH")]
    pub params_json_path: String,

    /// Path to the movie embeddings binary
    #[arg(long, env = "MOVIE_EMBEDDINGS_PATH")]
    pub movie_embeddings_path: String,

    /// Comma-separated list of paths to ratings parquet/csv files
    #[arg(long, env = "RATINGS_URIS", value_delimiter = ',')]
    pub ratings_uris: Vec<String>,

    //on the ranker serving machine, the number of jax local devices, that is, the number
    // of GPUs or TPUs that the data will be partitioned over.
    // for single inference this is probably 1, but for batch inference there would be a performance gain
    // in using a machine with attached accelerators.
    #[arg(long, env = "RANKER_N_LOCAL_DEVICES",  default_value_t = 1)]
    pub ranker_n_local_devices : usize,

    // the number of candidates to choose from the ANN search and to rank
    #[arg(long, env = "TOP_K", default_value_t = 20)]
    pub top_k : usize,

    // path to use for the ANN embeddings indexer persistence.  the directory of the file must
    // already exist and be writable by this app
    #[arg(long, env = "PERSISTED_INDEX_PATH", default_value = "./target/movie_embeddings_indexer")]
    pub persisted_index_path : PathBuf,

}

impl AppConfig {
    /// Loads the configuration from a given file path
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: AppConfig = serde_json::from_reader(reader)?;
        Ok(config)
    }
}