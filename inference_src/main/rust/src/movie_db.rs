use std::collections::HashMap;
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use tonic::Request;

use crate::tensorflow::serving::model_spec::VersionChoice;
use crate::pb::MoviesRequest;

#[derive(Debug)]
pub struct MovieDb {
    // In-memory lookup table for O(1) access
    // Maps movie_id -> genres
    movies: HashMap<i32, String>,
}

impl MovieDb {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;

        // Initialize the Parquet reader
        let reader = SerializedFileReader::new(file)
            .map_err(|e| Error::new(ErrorKind::InvalidData, format!("Failed to read Parquet: {}", e)))?;

        let mut movies = HashMap::new();

        // Iterate through all rows in the Parquet file
        let iter = reader.get_row_iter(None)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        for record_result in iter {
            let row = record_result.map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

            let mut movie_id = 0;
            let mut genres = String::new();

            // Robust column extraction by name (ignores column order)
            for (name, field) in row.get_column_iter() {
                match name.as_str() {
                    "movie_id" => {
                        // Handle both standard integer types just in case
                        match field {
                            Field::Int(val) => movie_id = *val,
                            Field::Long(val) => movie_id = *val as i32,
                            _ => {}
                        }
                    }
                    "genres" => {
                        if let Field::Str(val) = field {
                            genres = val.clone();
                        }
                    }
                    _ => {} // Ignore any extra columns
                }
            }

            movies.insert(movie_id, genres);
        }

        Ok(Self { movies })
    }

    pub fn get_request(
        &self,
        movie_ids: &[i32],
        model_version: Option<VersionChoice>,
    ) -> Option<Request<MoviesRequest>> {
        if movie_ids.is_empty() {
            return None;
        }

        // Pre-allocate vectors to avoid reallocation overhead during the loop
        let capacity: usize = movie_ids.len();
        let mut valid_movie_ids: Vec<i32> = Vec::with_capacity(capacity);
        let mut genres: Vec<String> = Vec::with_capacity(capacity);

        // Fetch genres from our in-memory HashMap
        for id in movie_ids {
            if let Some(genre_str) = self.movies.get(id) {
                valid_movie_ids.push(*id);
                genres.push(genre_str.clone()); // We must clone to own the string for the gRPC request
            }
        }

        // If none of the requested IDs were valid, return None
        if valid_movie_ids.is_empty() {
            return None;
        }

        let n_movies = valid_movie_ids.len() as u32;

        let version_num: i64 = match model_version {
            Some(VersionChoice::Version(v)) => v,
            _ => 1, // Default fallback version if None
        };

        Some(Request::new(MoviesRequest {
            movie_ids: valid_movie_ids,
            genres, // moved into the struct
            n_movies,
            candidate_model_version: version_num,
        }))
    }
}