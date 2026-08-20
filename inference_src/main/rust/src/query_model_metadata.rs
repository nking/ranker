use clap::Parser;
use serde::Deserialize;
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;

/// Command-line arguments and environment variables
#[derive(Parser, Debug, Deserialize, Clone)]
#[command(name = "recommender-grpc")]
#[command(about = "gRPC Recommender Service Orchestrator", long_about = None)]
pub struct QueryModelMetadata {

    /// lngth of the output embedding, parsed from keywork "layer_sizes"
    #[arg(long, env = "EMBED_LEN")]
    pub embed_len : usize,

    #[arg(long, env = "NUM_CATALOG_USERS",  default_value_t = 6040)]
    #[serde(alias = "n_users")] // Automatically pulls from "n_users" if "num_catalog_users" isn't found
    pub num_catalog_users : usize,

    #[arg(long, env = "NUM_CATALOG_MOVIES",  default_value_t = 3883)]
    #[serde(alias = "n_movies")] // Automatically pulls from "n_movies" if "num_catalog_users" isn't found
    pub num_catalog_movies : usize,

    /// git commit hash for code used to train the model
    #[arg(long, env = "GIT_COMMIT_HASH")]
    #[serde(alias = "git_hash")] // Automatically pulls from "git_hash" if "num_catalog_users" isn't found
    pub git_commit_hash : String
}

impl QueryModelMetadata {
    /// Loads the configuration from a given file path
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut raw_json: Value = serde_json::from_reader(reader)?;
        let mut target_section = raw_json
            .get_mut("values")
            .map(Value::take)
            .ok_or("Missing 'values' key in JSON configuration")?;

        // extract layer_sizes
        if let Some(Value::String(layer_sizes_str)) = target_section.get("layer_sizes") {
            // Parse the string context (e.g. "[64, 32]") into an actual JSON array
            let sizes_array: Value = serde_json::from_str(layer_sizes_str)?;
            if let Some(sizes) = sizes_array.as_array() {
                // Grab the very last element of the array
                if let Some(last_val) = sizes.last() {
                    // Insert it directly into the target map under the "embed_len" key
                    if let Some(map) = target_section.as_object_mut() {
                        map.insert("embed_len".to_string(), last_val.clone());
                    }
                }
            }
        }

        let config: QueryModelMetadata = serde_json::from_value(target_section)?;
        Ok(config)
    }
}