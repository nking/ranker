#[cfg(test)]
mod user_history_tests {

    // In src/test/rust/integration_test.rs

    // Import the functions/structs you want to test from your main code
    //
    // To run:
    //   cd src/main/rust
    //   cargo test

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    use polars::prelude::*;
    use helper::{get_train_val_test_liked_uris, DataSize};

    use inference_engine::user_history::{build_user_history, UserHistory};

    fn load_and_concat_parquet(paths: &[&str]) -> PolarsResult<LazyFrame> {
        let frames: Result<Vec<LazyFrame>, _> = paths
            .iter()
            .map(|&path| {
                let pl_path = PlRefPath::from(path);
                LazyFrame::scan_parquet(pl_path, ScanArgsParquet::default())
            })
            .collect();
        concat(frames?, UnionArgs::default())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    pub async fn test_user_history_load() -> Result<(), Box<dyn std::error::Error>> {
        let ratings_map = get_train_val_test_liked_uris(DataSize::Tiny3, false);
        let max_history = 2048;

        let ratings_uris: Vec<&str> = vec![
            ratings_map.get("train_liked").unwrap(),
            ratings_map.get("train_3").unwrap(),
            ratings_map.get("train_disliked").unwrap(),
        ];

        let user_history: UserHistory = build_user_history(&ratings_uris, max_history).await;

        let df: LazyFrame = load_and_concat_parquet(&ratings_uris)?;

        // =======================================================
        // gather the test data
        // =======================================================

        // select the 2 user_ids with the most ratings
        let top_users_df: DataFrame = df.clone()
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
            .limit(2)
            .collect()?;

        let user_ids: Vec<i32> = top_users_df
            .column("user_id")?
            .as_materialized_series()
            .i32()?
            .into_no_null_iter()
            .collect();

        // Data structures to hold aggregated test conditions
        let mut all_movie_ids: Vec<Vec<i32>> = Vec::new();
        let mut row_nums: Vec<usize> = Vec::new();
        let mut test_timestamps: Vec<i64> = Vec::new();

        // Filter, sort, and extract data for each top user dynamically
        for &u_id in &user_ids {
            let user_df = df.clone()
                .filter(col("user_id").eq(lit(u_id)))
                .sort_by_exprs(
                    vec![col("timestamp")],
                    SortMultipleOptions {
                        descending: vec![false],
                        nulls_last: vec![true],
                        multithreaded: true,
                        maintain_order: false,
                        limit: None,
                    },
                )
                .collect()?;

            let movie_ids: Vec<i32> = user_df
                .column("movie_id")?
                .as_materialized_series()
                .i32()?
                .into_no_null_iter()
                .collect();

            let timestamps: Vec<i64> = user_df
                .column("timestamp")?
                .as_materialized_series()
                .i64()?
                .into_no_null_iter()
                .collect();

            let mid_point = timestamps.len() / 2;

            row_nums.push(mid_point);
            test_timestamps.push(timestamps[mid_point]);
            all_movie_ids.push(movie_ids);
        }

        // =======================================================
        // Assertions
        // =======================================================

        let (movie_hist, _ratings_hist) = user_history.get_history_before_timestamp(
            &user_ids,
            &test_timestamps,
            max_history
        );

        assert_eq!(movie_hist.len(), user_ids.len() * max_history);

        for (i, &_user_id) in user_ids.iter().enumerate() {
            let start = i * max_history;
            let end = start + max_history;
            let movies = &movie_hist[start..end];

            let expected_movie_ids = &all_movie_ids[i];
            let valid_count = row_nums[i];
            let valid_end = std::cmp::min(valid_count, max_history);

            // Assert valid elements match exactly
            assert_eq!(&expected_movie_ids[0..valid_end], &movies[0..valid_end]);

            // Assert that the trailing elements are filled with the expected pad value
            if valid_count < max_history {
                let expected_padding = vec![user_history.pad_value; max_history - valid_count];
                assert_eq!(&movies[valid_count..max_history], &expected_padding[..]);
            }
        }

        Ok(())
    }

}