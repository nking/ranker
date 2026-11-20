use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Result};
use std::path::Path;

use rustc_hash::FxHashMap;
use serde::Deserialize;

#[derive(Deserialize)]
struct MovieTierRecord {
    movie_id: i32,
    tier: i32,
}

/// Loads NDJSON `movie_tiers.json` (`{"movie_id": ..., "tier": ...}` per line).
pub fn load_from_file<P: AsRef<Path>>(file_path: P) -> Result<FxHashMap<i32, i32>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut tiers_map: FxHashMap<i32, i32> = FxHashMap::default();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: MovieTierRecord = serde_json::from_str(trimmed)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        tiers_map.insert(record.movie_id, record.tier);
    }

    Ok(tiers_map)
}