use std::env;
use std::path::{PathBuf};
use std::collections::HashMap;

use inference_engine::app_config::AppConfig;

pub fn get_project_dir() -> Option<PathBuf> {
    let cwd = env::current_dir().ok()?;

    // Check if the current directory itself is "ranker"
    if cwd.file_name() == Some("ranker".as_ref()) {
        return Some(cwd);
    }

    // Traverse upwards to find a folder named "ranker"
    for ancestor in cwd.ancestors() {
        if ancestor.file_name() == Some("ranker".as_ref()) {
            return Some(ancestor.to_path_buf());
        }
    }

    None
}

#[allow(dead_code)]
pub fn get_bin_dir() -> Option<PathBuf> {
    // Call the previous function and map the path if found
    get_project_dir().map(|proj_dir| proj_dir.join("bin"))
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataSize {
    Full,
    Small,
    Tiny,
    Tiny3
}

#[allow(dead_code)]
pub fn get_train_val_test_liked_uris(
    data_size: DataSize,
    use_gcs_uri: bool,
) -> HashMap<String, String> {
    let base_uri = if use_gcs_uri {
        "gs://data/".to_string()
    } else {
        let mut path : Option<PathBuf> = get_project_dir();
        if let Some(ref mut p) = path {
            p.push("src/test/resources/data");

            match data_size {
                DataSize::Small => p.push("small"),
                DataSize::Tiny => p.push("tiny"),
                DataSize::Tiny3 => p.push("tiny3"),
                DataSize::Full => {} // No subfolder needed
            }
        }

        path.unwrap().to_string_lossy().into_owned()
    };

    let keys = [
        "train_3", "val_3", "test_3",
        "train_liked", "val_liked", "test_liked",
        "train_disliked", "val_disliked", "test_disliked",
    ];

    let mut out = HashMap::with_capacity(keys.len());
    for key in keys {
        let file_name = format!("ratings_{}.parquet", key);
        let full_path = format!("{}/{}", base_uri, file_name);
        out.insert(key.to_string(), full_path);
    }

    out
}

#[allow(dead_code)]
pub fn get_embeddings_uris(two_tower_version: Option<i32>) -> (String, String) {

    let base_dir = get_project_dir().expect("Failed to get project directory");
    let data_dir = match two_tower_version {
        Some(version) => format!("src/test/resources/data/tower_versions/{}/", version),
        None => "src/test/resources/data/tower_versions/1/".to_string(),
    };

    let user_embedding_uri = base_dir
        .join(&data_dir)
        .join("user_emb-00000-of-00001.parquet");

    let movie_embedding_uri = base_dir
        .join(&data_dir)
        .join("movie_emb-00000-of-00001.parquet");

    (
        user_embedding_uri.to_string_lossy().into_owned(),
        movie_embedding_uri.to_string_lossy().into_owned(),
    )
}

#[allow(dead_code)]
pub fn get_embeddings_metadata_uris(two_tower_version: Option<i32>) -> (String, String) {
    let base_dir = get_project_dir().expect("Failed to get project directory");
    let data_dir = match two_tower_version {
        Some(version) => format!("src/test/resources/data/tower_versions/{}/", version),
        None => "src/test/resources/data/tower_versions/1/".to_string(),
    };

    let user_uri = base_dir
        .join(&data_dir)
        .join("user_emb_metadata.json");

    let movie_uri = base_dir
        .join(&data_dir)
        .join("movie_emb_metadata.json");

    (
        user_uri.to_string_lossy().into_owned(),
        movie_uri.to_string_lossy().into_owned(),
    )
}

#[allow(dead_code)]
pub fn get_recommended_movies_uris(two_tower_version: Option<i32>) -> (String, String) {

    let base_dir = get_project_dir().expect("Failed to get project directory");
    let data_dir = match two_tower_version {
        Some(version) => format!("src/test/resources/data/tower_versions/{}/", version),
        None => "src/test/resources/data/tower_versions/1/".to_string(),
    };

    let movies_rec_uri = base_dir
        .join(&data_dir)
        .join("recommended_movies.parquet");

    let movies_rec_ts_uri = base_dir
        .join(&data_dir)
        .join("recommended_movies_timestamps.parquet");

    (
        movies_rec_uri.to_string_lossy().into_owned(),
        movies_rec_ts_uri.to_string_lossy().into_owned(),
    )
}

#[allow(dead_code)]
pub fn get_movies_uri() -> String {
    let movies_uri = get_project_dir()
        .map(|p| p.join("src/test/resources/data/movies.parquet"))
        .map(|p| p.to_string_lossy().into_owned())
        .expect("Project directory not found");

    movies_uri
}

#[allow(dead_code)]
pub fn get_ranker_metadata_uri(config: AppConfig) -> String {
    let filename = if config.ranker_serving_is_batched {
        "metadata_batch.json"
    } else {
        "metadata_single.json"
    };
    let ranker_metadata_uri = format!(
        "{}/{}/assets.extra/{}",
        config.ranker_saved_models_uri.trim_end_matches('/'), 1, filename);

    ranker_metadata_uri
}

#[allow(dead_code)]
pub fn get_ranker_training_hyperparameters_uri(config: AppConfig) -> String {
    let filename = "training_hyperparameters.json";
    let ranker_metadata_uri = format!(
        "{}/{}/assets.extra/{}",
        config.ranker_saved_models_uri.trim_end_matches('/'), 1, filename);

    ranker_metadata_uri
}

#[allow(dead_code)]
pub fn get_ranker_metadata_single_uri(cross_encoder_version: Option<i32>) -> String {

    let base_dir = get_project_dir().expect("Failed to get project directory");
    let data_dir = match cross_encoder_version {
        Some(version) => format!("src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/{}/", version),
        None => "src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/1/".to_string(),
    };
    let file_uri = base_dir
        .join(&data_dir)
        .join("assets.extra/metadata_single.json");

    file_uri.to_string_lossy().into_owned()

}

#[allow(dead_code)]
pub fn get_ranker_metadata_batch_uri(cross_encoder_version: Option<i32>) -> String {

    let base_dir = get_project_dir().expect("Failed to get project directory");
    let data_dir = match cross_encoder_version {
        Some(version) => format!("src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/{}/", version),
        None => "src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/1/".to_string(),
    };
    let file_uri = base_dir
        .join(&data_dir)
        .join("assets.extra/metadata_batch.json");

    file_uri.to_string_lossy().into_owned()
}

#[allow(dead_code)]
pub fn get_query_metadata_uri(cross_encoder_version: Option<i32>) -> String {
    let base_dir = get_project_dir().expect("Failed to get project directory");
    let data_dir = match cross_encoder_version {
        Some(version) => format!("src/test/resources/model_repositories/saved_model_formats/bi-encoder/query/{}/", version),
        None => "src/test/resources/model_repositories/saved_model_formats/bi-encoder/query/1/".to_string(),
    };
    let file_uri = base_dir
        .join(&data_dir)
        .join("assets.extra/hyperparameters.json");

    file_uri.to_string_lossy().into_owned()

}

#[allow(dead_code)]
pub fn get_model_param_json_uri(cross_encoder_version: Option<i32>) -> String {

    let base_dir = get_project_dir().expect("Failed to get project directory");
    let data_dir = match cross_encoder_version {
        Some(version) => format!("src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/{}/", version),
        None => "src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/1/".to_string(),
    };
    let file_uri = base_dir
        .join(&data_dir)
        .join("assets.extra/metadata_single.json");

    file_uri.to_string_lossy().into_owned()
}

#[allow(dead_code)]
pub fn get_config_json_uri() -> String {
    let params_uri = get_project_dir()
        .map(|p| p.join("./inference_src/main/rust/config/default.json"))
        .map(|p| p.to_string_lossy().into_owned())
        .expect("Project directory not found");
    params_uri
}
#[allow(dead_code)]
pub fn get_tiny_config_json_uri() -> String {
    let params_uri = get_project_dir()
        .map(|p| p.join("./inference_src/main/rust/config/default_tiny.json"))
        .map(|p| p.to_string_lossy().into_owned())
        .expect("Project directory not found");
    params_uri
}
#[allow(dead_code)]
pub fn assert_slices_nearly_equal(a: &[f32], b: &[f32], epsilon: f32) {
    assert_eq!(a.len(), b.len(), "Slices have different lengths");
    for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (val_a - val_b).abs();
        assert!(
            diff < epsilon,
            "At index {}: {} and {} are not close (diff: {})",
            i, val_a, val_b, diff
        );
    }
}

#[allow(dead_code)]
pub fn get_python_path() -> PathBuf {
    // Get the user's home directory from environment
    let home_dir = env::var("HOME")
        .or_else(|_| env::var("USERPROFILE"))
        .expect("Could not determine user home directory");

    // Construct <home_dir>/miniconda3/envs/ranker_py312/bin/python3
    let python_path = PathBuf::from(home_dir)
        .join("miniconda3")
        .join("envs")
        .join("ranker_py312")
        .join("bin")
        .join("python3");

    if !python_path.exists() {
        panic!(
            "Python binary not found at {:?}. Please ensure the ranker_py312 environment is created or edit this method for your venv.",
            python_path
        );
    }

    python_path
}
