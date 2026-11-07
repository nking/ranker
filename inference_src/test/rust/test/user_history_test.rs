#[cfg(test)]
mod user_history_tests {

    mod helper {
        include!("helper.rs");
        include!("helper_users.rs");
    }

    use polars::prelude::*;
    use helper::{get_train_val_test_liked_uris, DataSize, load_and_concat_parquet};

    use inference_engine::user_history::{build_user_history, UserHistory};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    pub async fn test_user_history_load0() -> Result<(), Box<dyn std::error::Error>> {
        let ratings_map = get_train_val_test_liked_uris(DataSize::Full, false);
        let max_history = 2048;

        let ratings_uris: Vec<&str> = vec![
            ratings_map.get("train_liked").unwrap(),
            ratings_map.get("train_3").unwrap(),
            ratings_map.get("train_disliked").unwrap(),
        ];

        let user_history: UserHistory = build_user_history(&ratings_uris, max_history).await;

        for u_id in 1..=user_history.n_users {
            let i0 = u_id * user_history.max_history;
            let i1 = i0 + user_history.max_history;
            let m_i = &user_history.movie_ids[i0..i1];
            let r_i = &user_history.ratings[i0..i1];
            let t_i = &user_history.timestamps[i0..i1];

            // Indices where padding is NOT present:
            let i_t: Vec<usize> = t_i
                .iter()
                .enumerate()
                .filter_map(|(idx, val)| (*val != user_history.ts_pad_value).then_some(idx))
                .collect();
            let i_m: Vec<usize> = m_i
                .iter()
                .enumerate()
                .filter_map(|(idx, val)| (*val != user_history.pad_value).then_some(idx))
                .collect();
            let i_r: Vec<usize> = r_i
                .iter()
                .enumerate()
                .filter_map(|(idx, val)| (*val != user_history.pad_value).then_some(idx))
                .collect();
            assert_eq!(i_t, i_m);
            assert_eq!(i_t, i_r);
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    pub async fn test_user_history() -> Result<(), Box<dyn std::error::Error>> {
        let ratings_map = get_train_val_test_liked_uris(DataSize::Full, false);
        let max_history = 2048;

        let ratings_uris: Vec<&str> = vec![
            ratings_map.get("train_liked").unwrap(),
            ratings_map.get("train_3").unwrap(),
            ratings_map.get("train_disliked").unwrap(),
        ];

        let user_history: UserHistory = build_user_history(&ratings_uris, max_history).await;

        let user_ids = vec![6040, 6039];
        let timestamps: Vec<i64> = vec![956705600, 956705600];

        let fixed_size = 200;

        let (m_1, r_1, t_1) = user_history.get_all_history_before_timestamp(
            &user_ids,
            &timestamps,
            fixed_size,
            true,
        );

        let mut timestamps2: Vec<i64> = Vec::with_capacity(timestamps.len());
        for i in 0..user_ids.len() {
            let i0 = i * fixed_size;
            let i1 = i0 + fixed_size;
            let m_1_i = &m_1[i0..i1];
            let r_1_i = &r_1[i0..i1];
            let t_1_i = &t_1[i0..i1];

            let i_t: Vec<usize> = t_1_i
                .iter()
                .enumerate()
                .filter_map(|(idx, val)| (*val != user_history.ts_pad_value).then_some(idx))
                .collect();
            let i_m: Vec<usize> = m_1_i
                .iter()
                .enumerate()
                .filter_map(|(idx, val)| (*val != user_history.pad_value).then_some(idx))
                .collect();
            let i_r: Vec<usize> = r_1_i
                .iter()
                .enumerate()
                .filter_map(|(idx, val)| (*val != user_history.pad_value).then_some(idx))
                .collect();
            assert_eq!(i_t, i_m);
            assert_eq!(i_t, i_r);

            let i10 = i_t[10];
            timestamps2.push(t_1_i[i10]);
        }

        let (m_2, r_2, t_2) = user_history.get_all_history_before_timestamp(
            &user_ids,
            &timestamps2,
            fixed_size,
            true,
        );

        for i in 0..user_ids.len() {
            let i0 = i * fixed_size;
            let i1 = i0 + fixed_size;
            let m_1_i = &m_1[i0..i1];
            let r_1_i = &r_1[i0..i1];
            let t_1_i = &t_1[i0..i1];

            let m_2_i = &m_2[i0..i1];
            let r_2_i = &r_2[i0..i1];
            let t_2_i = &t_2[i0..i1];

            // Extract all items from m_1_i whose timestamp is strictly < timestamps2[i]
            let m1_strictly_before_ts2: Vec<i32> = m_1_i
                .iter()
                .zip(t_1_i.iter())
                .filter_map(|(&m, &t)| (m != user_history.pad_value && t < timestamps2[i]).then_some(m))
                .collect();

            let r1_strictly_before_ts2: Vec<i32> = r_1_i
                .iter()
                .zip(t_1_i.iter())
                .filter_map(|(&r, &t)| (r != user_history.pad_value && t < timestamps2[i]).then_some(r))
                .collect();

            let m2_compare: Vec<i32> = m_2_i
                .iter()
                .copied()
                .filter(|&val| val != user_history.pad_value)
                .collect();

            let r2_compare: Vec<i32> = r_2_i
                .iter()
                .copied()
                .filter(|&val| val != user_history.pad_value)
                .collect();

            let t2_compare: Vec<i64> = t_2_i
                .iter()
                .copied()
                .filter(|&val| val != user_history.ts_pad_value)
                .collect();

            assert_eq!(m2_compare.len(), r2_compare.len());
            assert_eq!(m2_compare.len(), t2_compare.len());
            assert!(t2_compare.iter().all(|&t| t < timestamps2[i]));

            // The elements of m_1 that occurred before timestamps2[i] must match the prefix of m_2
            // (and if m_1 did not hit fixed_size, m2_compare.len() == overlap_len < m1_compare.len())
            let overlap_len = m1_strictly_before_ts2.len();
            assert!(overlap_len > 0);
            assert_eq!(&m1_strictly_before_ts2[..], &m2_compare[0..overlap_len]);
            assert_eq!(&r1_strictly_before_ts2[..], &r2_compare[0..overlap_len]);
        }

        Ok(())
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

        // Select the 2 user_ids with the most ratings
        let top_users_df: DataFrame = df
            .clone()
            .group_by([col("user_id")])
            .agg([len().alias("len")])
            .sort_by_exprs(
                vec![col("len")],
                SortMultipleOptions {
                    descending: vec![true],
                    nulls_last: vec![true],
                    multithreaded: true,
                    maintain_order: true,
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

        let mut all_movie_ids: Vec<Vec<i32>> = Vec::new();
        let mut row_nums: Vec<usize> = Vec::new();
        let mut test_timestamps: Vec<i64> = Vec::new();

        for &u_id in &user_ids {
            let user_df = df
                .clone()
                .filter(col("user_id").eq(lit(u_id)))
                .sort_by_exprs(
                    vec![col("timestamp")],
                    SortMultipleOptions {
                        descending: vec![false], // sorted ascending by timestamp
                        nulls_last: vec![true],
                        multithreaded: true,
                        maintain_order: true,    // preserve file order on timestamp ties
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
            let target_ts = timestamps[mid_point];
            // Exact count of interactions strictly < target_ts (handles ties at mid_point)
            let valid_before_count = timestamps.partition_point(|&ts| ts < target_ts);

            row_nums.push(valid_before_count);
            test_timestamps.push(target_ts);
            all_movie_ids.push(movie_ids);
        }

        // 1. Test most_recent_first = true (Newest -> Oldest)
        let (movie_hist_recent, _ratings_hist_recent) = user_history.get_history_before_timestamp(
            &user_ids,
            &test_timestamps,
            max_history,
            true,
        );

        // 2. Test most_recent_first = false (Oldest -> Newest within the most-recent window)
        let (movie_hist_chrono, _ratings_hist_chrono) = user_history.get_history_before_timestamp(
            &user_ids,
            &test_timestamps,
            max_history,
            false,
        );

        // 3. Test get_history_count_before_timestamp
        let counts = user_history.get_history_count_before_timestamp(&user_ids, &test_timestamps);

        assert_eq!(movie_hist_recent.len(), user_ids.len() * max_history);
        assert_eq!(movie_hist_chrono.len(), user_ids.len() * max_history);

        for (i, &_user_id) in user_ids.iter().enumerate() {
            let start = i * max_history;
            let end = start + max_history;
            let movies_recent = &movie_hist_recent[start..end];
            let movies_chrono = &movie_hist_chrono[start..end];

            let expected_movie_ids = &all_movie_ids[i];
            let valid_count = row_nums[i];
            assert_eq!(counts[i], valid_count);

            let valid_end = std::cmp::min(valid_count, max_history);
            let window_start = valid_count - valid_end;

            // Expected slice for most_recent_first = true (reversed: newest -> oldest)
            let expected_reversed: Vec<i32> = expected_movie_ids[window_start..valid_count]
                .iter()
                .rev()
                .copied()
                .collect();
            assert_eq!(&expected_reversed[..], &movies_recent[0..valid_end]);

            // Expected slice for most_recent_first = false (chronological: oldest -> newest)
            assert_eq!(
                &expected_movie_ids[window_start..valid_count],
                &movies_chrono[0..valid_end]
            );

            // Assert that the trailing elements are filled with pad_value (-1)
            if valid_end < max_history {
                let expected_padding = vec![user_history.pad_value; max_history - valid_end];
                assert_eq!(&movies_recent[valid_end..max_history], &expected_padding[..]);
                assert_eq!(&movies_chrono[valid_end..max_history], &expected_padding[..]);
            }
        }

        Ok(())
    }
}