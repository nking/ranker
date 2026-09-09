use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;
use crate::util::check_path;

#[derive(Deserialize)]
struct MovieTierRecord {
    movie_id: i32,
    tier: i32,
}

pub fn load_from_file(file_path: &str) -> Result<HashMap<i32, i32>, Box<dyn Error>> {

    let _r = check_path(&file_path, "movie tiers JSON");

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut tiers_map = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue; // Skip empty lines
        }

        let record: MovieTierRecord = serde_json::from_str(trimmed)?;
        tiers_map.insert(record.movie_id, record.tier);
    }

    Ok(tiers_map)
}