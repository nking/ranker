use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

/// Command-line arguments and environment variables
#[derive(Parser, Debug, Deserialize, Clone, Default)]
#[command(name = "recommender-grpc")]
#[command(about = "gRPC Recommender Service Orchestrator", long_about = None)]
pub struct RerankerModelMetadata {

    /*
    #[arg(long, env = "LORA_R")]
    pub lora_r: usize,
    #[arg(long, env = "LORA_DROPOUT")]
    pub lora_dropout: f32,
    #[arg(long, env = "LEARNING_RATE")]
    pub learning_rate: f32,
    #[arg(long, env = "WEIGHT_DECAY")]
    pub weight_decay: f32,
    */
    #[arg(long, env = "MODEL_NAME", default_value = "qwen-reranker")]
    pub model_name: String,

    #[arg(long, env="INPUT_IDS_NAME", default_value="input-ids")]
    pub input_ids_name: String,

    #[arg(long, env="ATTENTION_MASK_NAME", default_value="attention-mask")]
    pub attention_mask_name: String,

    #[arg(long, env="RANKER_EMBS_NAME", default_value="ranker_embs")]
    pub ranker_embs_name: String,

    #[arg(long, env="OUTPUT_NAME", default_value="candidate_log_probs")]
    pub output_name: String,

    #[arg(long, env="EMB_PLACEHOLDER_TOKEN", default_value="<|vision_pad|>")]
    pub emb_placeholder_token: String,

    #[arg(long, env = "NUM_CANDIDATES")]
    pub num_candidates : usize,

    #[arg(long, env = "NUM_CATALOG_USERS")]
    pub num_catalog_users : usize,

    #[arg(long, env = "NUM_CATALOG_MOVIES")]
    pub num_catalog_movies : usize,

    #[arg(long, env = "EVAL_TOP_K")]
    pub eval_top_k : usize,

    #[arg(long, env = "DISTILL_TEMPERATURE")]
    pub distill_temperature : f32,

    #[arg(long, env = "PER_DEVICE_TRAIN_BATCH_SIZE")]
    pub per_device_train_batch_size : usize,

    #[arg(long, env = "PER_DEVICE_EVAL_BATCH_SIZE")]
    pub per_device_eval_batch_size : usize,

    #[arg(long, env = "GRADIENT_ACCUMULATION_STEPS")]
    pub gradient_accumulation_steps : usize,

    #[arg(long, env = "NUM_PROCESSES")]
    pub num_processes : usize,

    #[arg(long, env = "EFFECTIVE_BATCH_SIZE")]
    pub effective_batch_size : usize,

    #[arg(long, env = "NUM_TRAIN_SAMPLES")]
    pub num_train_samples : usize,

    #[arg(long, env = "NUM_EVAL_SAMPLES")]
    pub num_eval_samples : usize,

    #[arg(long, env = "INCLUDE_RANKER_EMBEDDINGS")]
    pub include_ranker_embeddings : bool,

    #[arg(long, env = "RANKER_EMB_DIM")] // same as ranker out_features
    pub ranker_emb_dim : usize,

    #[arg(long, env = "MAX_SEQ_TOKEN_LENGTH")]
    pub max_seq_token_length : usize,

    #[arg(long, env = "MAX_PROMPT_HISTORY")]
    pub max_prompt_history : usize,

    /// saved_model signature name
    //#[arg(long, env = "SIGNATURE_NAME", default_value = "serving_default")]
    //pub signature_name: String,

    /// git commit hash for code used to train the model
    #[arg(long, env = "GIT_COMMIT_HASH")]
    pub git_commit_hash : String,

    #[arg(long, env = "RERANKER_PACKAGE_VERSION")]
    pub reranker_package_version : String
}

impl RerankerModelMetadata {
    /// Loads the configuration from a given file path
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: RerankerModelMetadata = serde_json::from_reader(reader)?;
        Ok(config)
    }
}