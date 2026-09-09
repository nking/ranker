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
