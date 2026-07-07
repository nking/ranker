use std::env;
use std::path::{Path, PathBuf};
use std::collections::HashMap;

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

pub fn get_bin_dir() -> Option<PathBuf> {
    // Call the previous function and map the path if found
    get_project_dir().map(|proj_dir| proj_dir.join("bin"))
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataSize {
    Full,
    Small,
    Tiny,
}

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
        let file_name = format!("ratings_{}-00000-of-00001.parquet", key);
        let full_path = format!("{}/{}", base_uri, file_name);
        out.insert(key.to_string(), full_path);
    }

    out
}

pub fn get_embeddings_uris() -> (String, String) {
    let mut user_embedding_uri : Option<PathBuf> = get_project_dir();
    if let Some(ref mut p) = user_embedding_uri {
        p.push("src/test/resources/data/user_emb-00000-of-00001.parquet");
    }
    let mut movie_embedding_uri : Option<PathBuf> = get_project_dir();
    if let Some(ref mut p) = user_embedding_uri {
        p.push("src/test/resources/data/movie_emb-00000-of-00001.parquet");
    }

    (movie_embedding_uri.unwrap().to_string_lossy().into_owned(),
    user_embedding_uri.unwrap().to_string_lossy().into_owned())
}

