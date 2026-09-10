use polars::prelude::*;

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
pub fn get_user_movie_tier_map(
    lf: LazyFrame,
    movie_tier_map_ref: &HashMap<i32, i32>,
) -> PolarsResult<Vec<HashMap<i32, HashSet<i32>>>> { // Return a PolarsResult to use '?'

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
    let mut tier_user_gt_maps: Vec<HashMap<i32, HashSet<i32>>> = vec![HashMap::new(); 3];

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

