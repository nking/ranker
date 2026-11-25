use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use rustc_hash::FxHashMap;

/// Lightweight in-memory movie catalog mapping `movie_id -> (title, genres)`
/// for prompt rendering without pulling in `tonic`, `arrow`, or `object_store`.
#[derive(Debug, Clone, Default)]
pub struct MovieCatalog {
    movies: FxHashMap<i32, (String, String)>,
}

impl MovieCatalog {
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = SerializedFileReader::new(file).map_err(|e| {
            Error::new(ErrorKind::InvalidData, format!("Failed to read Parquet: {e}"))
        })?;

        let mut movies = FxHashMap::default();
        let iter = reader
            .get_row_iter(None)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        for record_result in iter {
            let row = record_result.map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

            let mut movie_id: i32 = 0;
            let mut title = String::new();
            let mut genres = String::new();

            for (name, field) in row.get_column_iter() {
                match name.as_str() {
                    "movie_id" => match field {
                        Field::Int(val) => movie_id = *val,
                        Field::Long(val) => movie_id = *val as i32,
                        _ => {}
                    },
                    "title" => {
                        if let Field::Str(val) = field {
                            title = val.clone();
                        }
                    }
                    "genres" => {
                        if let Field::Str(val) = field {
                            genres = val.clone();
                        }
                    }
                    _ => {}
                }
            }

            movies.insert(movie_id, (title, genres));
        }

        Ok(Self { movies })
    }

    pub fn from_map(movies: FxHashMap<i32, (String, String)>) -> Self {
        Self { movies }
    }

    #[inline]
    pub fn get(&self, movie_id: i32) -> Option<(&str, &str)> {
        self.movies
            .get(&movie_id)
            .map(|(t, g)| (t.as_str(), g.as_str()))
    }

    #[inline]
    pub fn get_genres(&self, movie_id: i32) -> Option<&str> {
        self.movies.get(&movie_id).map(|(_, g)| g.as_str())
    }

    #[inline]
    pub fn get_titles(&self, movie_id: i32) -> Option<&str> {
        self.movies.get(&movie_id).map(|(g, _)| g.as_str())
    }

    #[inline]
    pub fn get_title_and_genres(&self, movie_id: i32) -> (&str, &str) {
        self.get(movie_id).unwrap_or(("", ""))
    }

    pub fn get_all_movie_ids(&self) -> Vec<i32> {
        let mut keys: Vec<i32> = self.movies.keys().copied().collect();
        keys.sort_unstable();
        keys
    }
}