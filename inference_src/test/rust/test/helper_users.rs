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
/// * `df`: ratings file loaded into a PolarsResult<LazyFrame>
///
/// returns: (Vec<i32, Global>, Vec<i64, Global>)
pub fn get_unique_user_and_first_timestamp(df: PolarsResult<LazyFrame>) -> (Vec<i32>, Vec<i64>) {

    let lf = df.expect("failed to unwrap LazyFrame result");

    let collected_df = lf
        .group_by([col("user_id")])
        .agg([col("timestamp").min().alias("timestamp")])
        .collect()
        .expect("failed to execute lazy query");

    let user_s = collected_df.column("user_id").expect("user_id column missing");
    let time_s = collected_df.column("timestamp").expect("timestamp column missing");

    let users = user_s
        .i32()
        .expect("user_id type mismatch")
        .into_no_null_iter()
        .collect();

    let timestamps = time_s
        .i64()
        .expect("timestamp type mismatch")
        .into_no_null_iter()
        .collect();

    (users, timestamps)

}

/// given the ratings dataframe, get a hashmap wih key=movie_tier, value=polars dataframe.
/// the polars dataframes returned has columns user_id, movie_ids
///
/// the ratings dataframe
///
/// # Arguments
///
/// * `df`: ratings file loaded into a PolarsResult<LazyFrame>.  it has
///     columns user_id, movie_id, rating, timestamp
/// * `movie_tier_map`:   hashmap with key=movie_id, value= movie_tier
///
/// returns: HashMap<i32, DataFrame>
pub fn get_user_movie_tier_map(df: PolarsResult<LazyFrame>, movie_tier_map: HashMap<i32, i32>)
    -> HashMap<i32, DataFrame> {

    let lf = df.expect("failed to unwrap LazyFrame result");

    // Convert the HashMap into a temporary DataFrame and make it lazy
    let movie_ids: Vec<i32> = movie_tier_map.keys().copied().collect();
    let tiers: Vec<i32> = movie_tier_map.values().copied().collect();

    let tier_df = df![
        "movie_id" => movie_ids,
        "movie_tier" => tiers,
    ].expect("failed to create tier dataframe")
        .lazy();

    let collected_df = lf
        .join(
            tier_df,
            [col("movie_id")],
            [col("movie_id")],
            JoinArgs::new(JoinType::Inner),
        )
        .select([col("user_id"), col("movie_id"), col("movie_tier")])
        .collect()
        .expect("failed to execute lazy query");

    let mut user_tier_map: HashMap<i32, DataFrame> = HashMap::new();

    for movie_tier in 0..3 {
        // df_tier = filter collected_df for movie_tier, and group by user, aggregate the movie_ids
        let df_tier = collected_df
            .clone()
            .lazy() // Switch back to lazy for easy filtering and aggregation
            .filter(col("movie_tier").eq(lit(movie_tier)))
            .group_by([col("user_id")])
            .agg([col("movie_id").alias("movie_ids")]) // Automatically collects the movie_ids into a ListChunked array
            .collect()
            .expect("failed to group tier");

        if df_tier.height() == 0 {
            continue;
        }

        println!("df_tier.head(): {}", df_tier.head(Some(5)));

        user_tier_map.insert(movie_tier, df_tier);

    }

    user_tier_map

}

