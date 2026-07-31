use std::collections::HashMap;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;
use std::path::Path;
use crate::app_config::AppConfig;

// New struct to hold the cached Bayesian scores alongside global parameters.
// this is the population
pub struct CatalogStats {
    pub bayesian_scores: HashMap<i32, f32>,
    pub global_mean_c: f32,
    pub prior_m: f32,
    // population standard deviation.  useful to measure effect size, that is diffs/stddev
    pub std_dev : f32,
}

pub struct Movie {
    pub movie_id: i32,
    pub title: String,
    pub genres: Vec<String>,
    pub rating_counts: [u32; 5],
}

// Assuming a simplified representation of the user's historical ratings
pub struct UserHistory {
    pub movie_id: i32,
    pub rating: f32,
}

///
///
/// # Arguments
///
/// * `movies`: hashmap of all movies in the catalog with ratings_counts populated by user ratings.
///
/// returns: CatalogStats
///
/// # Examples
///
/// ```
///
/// ```
pub fn build_bayesian_catalog(movies: &HashMap<i32, Movie>) -> CatalogStats {
    let mut total_ratings_global: u64 = 0;
    let mut sum_ratings_global: f64 = 0.0;

    // Vector to collect rating volumes so we can find the 25th percentile (m)
    let mut item_rating_volumes: Vec<u32> = Vec::with_capacity(movies.len());

    // 1. First Pass: Gather global statistics for C and m
    for movie in movies.values() {
        let mut movie_total_ratings = 0;
        let mut movie_rating_sum = 0.0;

        // Assuming index 0 is 1-star, index 4 is 5-stars
        for (i, &count) in movie.rating_counts.iter().enumerate() {
            let rating_val = (i + 1) as f64;
            movie_total_ratings += count;
            movie_rating_sum += rating_val * (count as f64);
        }

        total_ratings_global += movie_total_ratings as u64;
        sum_ratings_global += movie_rating_sum;

        if movie_total_ratings > 0 {
            item_rating_volumes.push(movie_total_ratings);
        }
    }

    // Calculate C (Global Mean)
    let global_mean_c = if total_ratings_global > 0 {
        (sum_ratings_global / total_ratings_global as f64) as f32
    } else {
        3.0 // Fallback for empty catalog
    };

    // Calculate m (25th percentile of rating volume)
    item_rating_volumes.sort_unstable();
    let prior_m = if !item_rating_volumes.is_empty() {
        let p25_index = (item_rating_volumes.len() as f32 * 0.25).floor() as usize;
        item_rating_volumes[p25_index] as f32
    } else {
        1.0 // Fallback
    };

    // 2. Second Pass: Calculate Bayesian Score (S_i) for each item
    let mut bayesian_scores = HashMap::with_capacity(movies.len());

    let mut s_i_avg: f64 = 0.;

    for movie in movies.values() {
        let mut v = 0;
        let mut rating_sum = 0.0;

        for (i, &count) in movie.rating_counts.iter().enumerate() {
            v += count;
            rating_sum += (i + 1) as f32 * count as f32;
        }

        let v_f32 = v as f32;

        let s_i = if v == 0 {
            // No ratings? Shrink completely to global mean
            global_mean_c
        } else {
            let r = rating_sum / v_f32; // Naive mean
            ((v_f32 / (v_f32 + prior_m)) * r) + ((prior_m / (v_f32 + prior_m)) * global_mean_c)
        };

        bayesian_scores.insert(movie.movie_id, s_i);

        s_i_avg += s_i as f64;
    }
    s_i_avg = s_i_avg/(bayesian_scores.len() as f64);
    let mut s_i_sqsum : f64 = 0.0;
    for (_id, s_i) in &bayesian_scores {
        let diff = (*s_i as f64) - s_i_avg;
        s_i_sqsum += diff * diff;
    }
    let stdev : f64 = (s_i_sqsum/((bayesian_scores.len() + 1) as f64)).sqrt();

    CatalogStats {
        bayesian_scores,
        global_mean_c,
        prior_m,
        std_dev : stdev as f32
    }
}

pub fn calculate_user_mainstreamness(
    user_history: &[UserHistory],
    catalog_stats: &CatalogStats
) -> Option<f32> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for item in user_history {
        // Only score items that exist in our Bayesian cache
        if let Some(&s_i) = catalog_stats.bayesian_scores.get(&item.movie_id) {
            let r_ui = item.rating; // User's actual rating (1.0 - 5.0)

            numerator += r_ui * s_i;
            denominator += r_ui;
        }
    }

    if denominator > 0.0 {
        Some(numerator / denominator)
    } else {
        None // User has no valid rated history
    }
}

pub fn load_and_count_movies(config: &AppConfig) -> HashMap<i32, Movie> {

    let mut movies_map: HashMap<i32, Movie> = HashMap::new();

    // ==========================================
    // 1. Load Movies and Initialize the HashMap
    // ==========================================
    let movies_path = &config.movies_path;
    let movies_file = File::open(movies_path).expect("Failed to open movies parquet");
    let movies_reader = SerializedFileReader::new(movies_file).expect("Failed to create parquet reader");

    for row_result in movies_reader.get_row_iter(None).expect("Failed to get row iterator") {
        let row = row_result.expect("Failed to read row");

        let movie_id = row.get_long(0).expect("Missing movie_id") as i32;
        let title = row.get_string(1).expect("Missing title").to_string();

        // Assuming genres are stored as a delimited string in the parquet file (e.g., "Action|Sci-Fi")
        let genres_str = row.get_string(2).expect("Missing genres").to_string();
        let genres: Vec<String> = genres_str.split('|').map(|s| s.to_string()).collect();

        movies_map.insert(movie_id, Movie {
            movie_id,
            title,
            genres,
            rating_counts: [0; 5], // Initialize all 5 buckets to 0
        });
    }

    // ==========================================
    // 2. Filter URIs to Exclude the Test Set
    // ==========================================
    let ratings_uris_refs: &Vec<String> = &config.ratings_uris;

    println!("Updating rating counts using files: {:?}", ratings_uris_refs);

    for uri in ratings_uris_refs {
        let ratings_file = File::open(Path::new(uri)).expect("Failed to open ratings parquet");
        let ratings_reader = SerializedFileReader::new(ratings_file).expect("Failed to create parquet reader");

        // ratings:  user_id, movie_id, rating, timestamp
        for row_result in ratings_reader.get_row_iter(None).expect("Failed to get row iterator") {
            let row = row_result.expect("Failed to read row");

            let movie_id = row.get_int(1).expect("Missing movie_id") as i32;

            // Extract rating (handling potential f32/f64 types)
            // If your parquet stores ratings as ints, use `get_int` instead
            let rating_val = row.get_int(2).expect("Missing rating");

            // Map a rating like 4.5 to index 3 (which represents the 4-star bucket)
            // clamp(1, 5) ensures we don't go out of bounds on the array
            let bucket_idx: usize = (rating_val.clamp(1, 5) - 1) as usize;

            // Update the count if the movie exists in our catalog
            if let Some(movie) = movies_map.get_mut(&movie_id) {
                movie.rating_counts[bucket_idx] += 1;
            }
        }
    }

    movies_map
}
