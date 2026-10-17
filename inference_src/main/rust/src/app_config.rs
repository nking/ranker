use std::net::SocketAddr;
use clap::Parser;
use serde::Deserialize;
use std::io::BufReader;
use std::env;
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};

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

    /// URI for the TFS Query Model
    #[arg(long, env = "CANDIDATE_URI", default_value = "http://172.17.0.1:8520")]
    pub candidate_uri: String,

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

    // path to the users catalog file
    #[arg(long, env = "USER_DB_PATH")]
    pub user_db_path : PathBuf,

    // path to the movies catalog file
    #[arg(long, env = "MOVIES_PATH")]
    pub movies_path : String,

    #[arg(long, env = "MOVIE_TIERS_PATH")]
    pub movie_tiers_path : String,

    // path to the query saved_models directory holding the query saved_models..   expecting to see
    //     sub-directory of version numbers e.g. 1,2 and those subdirectories have the contents
    //     of saved model directories.  one of those is assets.extra and that directory contains
    //     the hyperparameters.json file for the model metadata
    #[arg(long, env = "QUERY_SAVED_MODELS_URI")]
    pub query_saved_models_uri : String,

    // path to the ranker saved_models directory holding the query saved_models..   expecting to see
    //   sub-directory of version numbers e.g. 1,2 and those subdirectories have the contents
    //   of saved model directories.  one of those is assets.extra and that directory contains
    //   the hyperparameters.json file for the model metadata
    #[arg(long, env = "RANKER_SAVED_MODELS_URI")]
    pub ranker_saved_models_uri : String,

    // whether the ranker saved models are serveving the batch signatures else single inference signatures
    #[arg(long, env = "RANKER_SERVING_IS_BATCHED")]
    pub ranker_serving_is_batched : bool,
}

impl AppConfig {
    /// Loads the configuration from a given file path
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {

        let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("<unknown>"));
        let rust_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        println!("[AppConfig] Current working directory: {:?}", cwd);
        println!("[AppConfig] Rust root (CARGO_MANIFEST_DIR): {:?}", rust_root);

        // Allow the config file path itself to be resolved relative to CWD first,
        // falling back to the Rust root if CWD is different (e.g. in IDE test runners).
        let config_path = {
            let p = Path::new(path);
            if p.exists() {
                Self::resolve_path(&cwd, path)
            } else {
                Self::resolve_path(&rust_root, path)
            }
        };
        println!("[AppConfig] Loading config from: {:?}", config_path);

        let file = File::open(&config_path)?;
        let reader = BufReader::new(file);
        let mut config: AppConfig = serde_json::from_reader(reader)?;

        // Resolve all relative paths in default.json against the Rust root directory
        config.resolve_all_paths(&rust_root);

        Ok(config)
    }

    fn resolve_all_paths(&mut self, base_dir: &Path) {
        self.params_json_path = Self::resolve_string_path(base_dir, &self.params_json_path);
        self.movie_embeddings_path = Self::resolve_string_path(base_dir, &self.movie_embeddings_path);
        self.movies_path = Self::resolve_string_path(base_dir, &self.movies_path);
        self.movie_tiers_path = Self::resolve_string_path(base_dir, &self.movie_tiers_path);
        self.query_saved_models_uri = Self::resolve_string_path(base_dir, &self.query_saved_models_uri);
        self.ranker_saved_models_uri = Self::resolve_string_path(base_dir, &self.ranker_saved_models_uri);

        self.ratings_uris = self
            .ratings_uris
            .iter()
            .map(|u| Self::resolve_string_path(base_dir, u))
            .collect();

        self.persisted_index_path = Self::resolve_path(base_dir, &self.persisted_index_path);
        self.user_db_path = Self::resolve_path(base_dir, &self.user_db_path);
    }

    fn resolve_string_path(base_dir: &Path, raw_path: &str) -> String {
        // Leave http:// or https:// URIs untouched if any ever pass through
        if raw_path.starts_with("http://") || raw_path.starts_with("https://") {
            return raw_path.to_string();
        }
        Self::resolve_path(base_dir, raw_path)
            .to_string_lossy()
            .into_owned()
    }

    /// Joins a relative path onto `base_dir` and normalizes `.` and `..` segments.
    /// Uses `fs::canonicalize` if the path exists on disk, or lexical resolution if it doesn't yet.
    fn resolve_path<P: AsRef<Path>>(base_dir: &Path, raw_path: P) -> PathBuf {
        let p = raw_path.as_ref();
        let joined = if p.is_absolute() {
            p.to_path_buf()
        } else {
            base_dir.join(p)
        };

        // If the file/dir already exists, canonicalize resolves symlinks and `..`
        if let Ok(canon) = fs::canonicalize(&joined) {
            return canon;
        }

        // Fallback lexical cleanup for paths that don't exist yet (like persisted_index_path)
        let mut normalized = PathBuf::new();
        for component in joined.components() {
            match component {
                Component::CurDir => {}
                Component::ParentDir => {
                    normalized.pop();
                }
                c => normalized.push(c.as_os_str()),
            }
        }
        normalized
    }

}