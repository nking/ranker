use std::io::Result;
use std::path::Path;
use std::sync::Arc;

use reranker_prompt::movie_catalog::MovieCatalog;
use tonic::Request;

use crate::pb::MoviesRequest;
use crate::tensorflow::serving::model_spec::VersionChoice;

/// Extension trait adapting `MovieCatalog` to build gRPC `MoviesRequest` payloads
/// without coupling the lean `reranker_prompt` crate to `tonic` or protobufs.
pub trait MovieCatalogRequestExt {
    fn build_movies_request(
        &self,
        movie_ids: &[i32],
        model_version: Option<VersionChoice>,
    ) -> Option<Request<MoviesRequest>>;
}

impl MovieCatalogRequestExt for MovieCatalog {
    fn build_movies_request(
        &self,
        movie_ids: &[i32],
        model_version: Option<VersionChoice>,
    ) -> Option<Request<MoviesRequest>> {
        if movie_ids.is_empty() {
            return None;
        }

        let capacity = movie_ids.len();
        let mut valid_movie_ids: Vec<i32> = Vec::with_capacity(capacity);
        let mut genres: Vec<String> = Vec::with_capacity(capacity);

        for &id in movie_ids {
            if let Some(genre_str) = self.get_genres(id) {
                valid_movie_ids.push(id);
                genres.push(genre_str.to_owned());
            }
        }

        if valid_movie_ids.is_empty() {
            return None;
        }

        let n_movies = valid_movie_ids.len() as u32;
        let version_num = match model_version {
            Some(VersionChoice::Version(v)) => Some(v),
            _ => None,
        };

        Some(Request::new(MoviesRequest {
            movie_ids: valid_movie_ids,
            genres,
            n_movies,
            candidate_model_version: version_num,
        }))
    }
}

/// Thin shared-ownership facade over `Arc<MovieCatalog>` preserving the existing
/// `MovieDb` API while sharing a single in-memory catalog with `RerankerPromptFormatter`.
#[derive(Debug, Clone)]
pub struct MovieDb {
    catalog: Arc<MovieCatalog>,
}

impl MovieDb {
    /// Loads `movies.parquet` directly (convenient for standalone tests and callers).
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let catalog = MovieCatalog::from_parquet(path)?;
        Ok(Self {
            catalog: Arc::new(catalog),
        })
    }

    /// Reuses an existing `Arc<MovieCatalog>` with zero duplicate Parquet parsing or memory.
    pub fn from_catalog(catalog: Arc<MovieCatalog>) -> Self {
        Self { catalog }
    }

    /// Exposes the underlying shared catalog handle for downstream components (e.g., reranker).
    pub fn catalog(&self) -> Arc<MovieCatalog> {
        Arc::clone(&self.catalog)
    }

    pub fn get_request(
        &self,
        movie_ids: &[i32],
        model_version: Option<VersionChoice>,
    ) -> Option<Request<MoviesRequest>> {
        self.catalog.build_movies_request(movie_ids, model_version)
    }
}