//use polars::prelude::*;

use polars::df;
use polars::prelude::{concat, LazyFrame, PlRefPath, PolarsResult, ScanArgsParquet, UnionArgs, DataFrame, SortMultipleOptions, IdxSize, col, len, JoinArgs, JoinType, IntoLazy, DataType, QuantileMethod, when, lit};
use rustc_hash::FxHashMap;

#[allow(dead_code)]
pub fn load_and_concat_parquet(paths: &[&str]) -> PolarsResult<LazyFrame> {
    let frames: Result<Vec<LazyFrame>, _> = paths
        .iter()
        .map(|&path| {
            let pl_path = PlRefPath::from(path);
            LazyFrame::scan_parquet(pl_path, ScanArgsParquet::default())
        })
        .collect();
    concat(frames?, UnionArgs::default())
}

/// Given a Polars DataFrame with ['user_id', ...],
/// returns DataFrame with columns 'user_id', 'user_tier' where tier is 0, 1, or 2 for
/// head, torso, and tail of the distribution of the number of users ratings.
pub fn get_user_tiers_df(
    ratings_history_df: &LazyFrame,
    user_catalog_df: &LazyFrame
) -> LazyFrame {
    _get_key_tiers_df(ratings_history_df, user_catalog_df, "user")
}


/// Given a Polars DataFrame with ['movie_id', ...],
/// returns DataFrame with columns 'movie_id', 'movie_tier' where tier is 0, 1, or 2 for
/// head, torso, and tail of the distribution of the number of users ratings.
pub fn get_movie_tiers_df(
    ratings_history_df: &LazyFrame,
    movie_catalog_df: &LazyFrame
) -> LazyFrame {
    _get_key_tiers_df(ratings_history_df, movie_catalog_df, "movie")
}


pub fn _get_key_tiers_df(
    ratings_history_df: &LazyFrame,
    user_catalog_df: &LazyFrame,
    key : &str
) -> LazyFrame {

    let key_id = format!("{}_id", key);
    let key_counts = format!("{}_counts", key);
    let key_tier = format!("{}_tier", key);

    // Count history length per user
    let counts_lf = ratings_history_df.clone()
        .group_by([col(&key_id)])
        .agg([len().alias(&key_counts)]);

    // Define pure lazy expressions for the 80th and 20th percentiles.
    // By filtering for strictly > 0, we exactly replicate the Python logic
    // of computing percentiles BEFORE the 0-count users are joined in.
    let non_zero_counts = col(&key_counts)
        .filter(col(&key_counts).gt(lit(0)))
        .cast(DataType::Float64); // Cast to float for accurate quantile interpolation

    // If ratings_lf is completely empty, quantile() returns null.
    // .fill_null(0.0) safely handles the fallback matching your Python code.
    let head_min_expr = non_zero_counts.clone()
        .quantile(lit(0.80), QuantileMethod::Linear)
        .fill_null(lit(0.0));

    let torso_min_expr = non_zero_counts
        .quantile(lit(0.20), QuantileMethod::Linear)
        .fill_null(lit(0.0));

    // Construct the final LazyFrame graph
    user_catalog_df.clone()
        .select([col(&key_id)])
        .left_join(
            counts_lf,
            col(&key_id),
            col(&key_id)
        )
        .with_columns([
            col(&key_counts).fill_null(lit(0))
        ])
        .with_columns([
            // Use the lazily evaluated quantile expressions directly
            when(
                col(&key_counts).eq(lit(0))
                    .or(col(&key_counts).cast(DataType::Float64).lt(torso_min_expr))
            )
                .then(lit(2))  // Tail
                .when(col(&key_counts).cast(DataType::Float64).gt_eq(head_min_expr))
                .then(lit(0))  // Head
                .otherwise(lit(1)) // Torso
                .alias(&key_tier)
        ])
        .select([col(&key_id), col(&key_tier)])
}

///
/// load the ratings files from paths and return the n_users with the largest number of ratings.
///
/// # Arguments
///
/// * `paths`: paths to ratings parquet files
/// * `n_users`: the number of users to return
///
/// returns: Result<Vec<i32, Global>, Box<dyn Error, Global>>
///
#[allow(dead_code)]
pub fn get_most_frequent_users(paths: &[&str], n_users: usize) -> Result<Vec<i32>, Box<dyn std::error::Error>> {

    let df: LazyFrame = load_and_concat_parquet(&paths)?;

    let top_users_df: DataFrame = df
        .group_by([col("user_id")])
        .agg([len().alias("len")]) // alias added for safety
        .sort_by_exprs(
            vec![col("len")],
            SortMultipleOptions {
                descending: vec![true],
                nulls_last: vec![true],
                multithreaded: true,
                maintain_order: false,
                limit: None,
            },
        )
        .limit(n_users as IdxSize)
        .collect()?;

    let user_ids: Vec<i32> = top_users_df
        .column("user_id")?
        .as_materialized_series()
        .i32()?
        .into_no_null_iter()
        .collect();

    Ok(user_ids)

}


/// given the ratings dataframe, get unique user_ids and their first timestamps.
/// the ratings dataframe has columns user_id, movie_id, rating, timestamp.
///
/// # Arguments
///
/// * `df`: ratings file loaded into a LazyFrame
///
/// returns: (Vec<i32, Global>, Vec<i64, Global>)
#[allow(dead_code)]
pub fn get_unique_user_and_first_timestamp(df: LazyFrame) -> PolarsResult<(Vec<i32>, Vec<i64>)> {

    let collected_df = df
        .group_by([col("user_id")])
        .agg([col("timestamp").min().alias("timestamp")])
        .collect()
        .expect("failed to execute lazy query");

    let user_ca = collected_df.column("user_id")?.i32()?;
    let time_ca = collected_df.column("timestamp")?.i64()?;

    // 3. Collect into vectors using the fast no-null iterator
    let users: Vec<i32> = user_ca.into_no_null_iter().collect();
    let timestamps: Vec<i64> = time_ca.into_no_null_iter().collect();

    Ok((users, timestamps))

}

/// given the ratings dataframe and a movie_tiers lookup hashmap,
/// create a vectors of hashmaps where each vector element is the hashmap
/// for a tier and the hashmap holds key=user_id, value=movie_ids for that tier.
///
/// the ratings dataframe
///
/// # Arguments
///
/// * `df`: ratings file loaded into a PolarsResult<LazyFrame>.  it has
///     columns user_id, movie_id, rating, timestamp
/// * `movie_tier_map_ref`:   reference to hashmap with key=movie_id, value= movie_tier
///
/// returns: Vec<HashMap<i32, HashSet<i32>>>
#[allow(dead_code)]
pub fn get_user_movie_tier_map(lf: LazyFrame,
    movie_tier_map_ref: &FxHashMap<i32, i32>,
) -> PolarsResult<Vec<HashMap<i32, std::collections::HashSet<i32>>>> {

    // Convert the HashMap into a DataFrame
    let movie_ids: Vec<i32> = movie_tier_map_ref.keys().copied().collect();
    let tiers: Vec<i32> = movie_tier_map_ref.values().copied().collect();
    let tier_df = df!["movie_id" => movie_ids, "movie_tier" => tiers,]?
        .lazy();

    // Join the data and collect
    let collected_df = lf
        .join( tier_df, [col("movie_id")],[col("movie_id")],
            JoinArgs::new(JoinType::Inner),
        )
        .select([col("user_id"), col("movie_id"), col("movie_tier")])
        .collect()?;

    // Initialize the output
    let mut tier_user_gt_maps: Vec<HashMap<i32, std::collections::HashSet<i32>>> = vec![HashMap::new(); 3];

    // Extract the columns as fast Int32Chunked arrays.  these maintain the sam row ordering:
    let user_ca = collected_df.column("user_id")?.i32()?;
    let movie_ca = collected_df.column("movie_id")?.i32()?;
    let tier_ca = collected_df.column("movie_tier")?.i32()?;

    // Populate the maps in a single, lightning-fast O(N) pass
    // Zip all three iterators together
    for ((u, m), t) in user_ca
        .into_no_null_iter()
        .zip(movie_ca.into_no_null_iter())
        .zip(tier_ca.into_no_null_iter())
    {
        // Safety check to ensure tier is valid (0, 1, or 2)
        if t >= 0 && t < 3 {
            tier_user_gt_maps[t as usize]
                .entry(u)
                .or_default()
                .insert(m);
        }
    }

    Ok(tier_user_gt_maps)
}

#[allow(dead_code)]
pub fn create_user_movie_map(df_ratings: LazyFrame) -> PolarsResult<HashMap<i32, std::collections::HashSet<i32>>> {

    let df = df_ratings.collect()?;

    let user_series = df.column("user_id")?;
    let movie_series = df.column("movie_id")?;

    let user_chunked = user_series.i32()?;
    let movie_chunked = movie_series.i32()?;

    let mut h: HashMap<i32, std::collections::HashSet<i32>> = HashMap::new();

    for (user_opt, movie_opt) in user_chunked.iter().zip(movie_chunked.iter()) {
        if let (Some(user_id), Some(movie_id)) = (user_opt, movie_opt) {
            h.entry(user_id).or_default().insert(movie_id);
        }
    }

    Ok(h)
}

#[allow(dead_code)]
pub fn calc_normalized_emd_3(hist: &[f64], rec: &[f64]) -> f64 {

    if hist.len() != 3 || rec.len() != 3 {
        panic!("hist and rec need to be length 3, received {}, {}", hist.len(), rec.len());
    }

    // CDF for bin 0
    let hist_cdf_0 = hist[0];
    let rec_cdf_0 = rec[0];

    // CDF for bin 1 (bin 0 + bin 1)
    let hist_cdf_1 = hist[0] + hist[1];
    let rec_cdf_1 = rec[0] + rec[1];

    // Sum of absolute differences between CDFs, divided by theoretical max (2.0)
    let emd = (hist_cdf_0 - rec_cdf_0).abs() + (hist_cdf_1 - rec_cdf_1).abs();
    emd / 2.0
}

#[allow(dead_code)]
/// Helper to calculate mean and standard deviation
pub fn mean_and_std(data: &[f64]) -> (f64, f64) {
    if data.len() <= 1 {
        return (data.first().copied().unwrap_or(0.0), 0.0);
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    (mean, variance.sqrt())
}
