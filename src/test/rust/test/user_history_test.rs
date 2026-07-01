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
    use helper::{get_train_val_test_liked_uris, DataSize};

    use prep_inputs_for_graphranker::user_history::{build_user_history, UserHistory, UserMapEntry};
    use prep_inputs_for_graphranker::user_history::_testable_build_map_async;

    #[test]
    pub fn test_user_history_load() {
        let ratings_map = get_train_val_test_liked_uris(DataSize::Tiny, false);

        let max_history = 2048;
        // storing the items as references
        let ratings_uris: Vec<&String> = vec![
            ratings_map.get("train_liked").unwrap(),
            ratings_map.get("train_3").unwrap(),
            ratings_map.get("train_disliked").unwrap(),
        ];

        let user_history : UserHistory = build_user_history(&ratings_uris, max_history);

        let user_ids : Vec<i32> = vec![6040, 6039];

        let ts : Vec<i64> = vec![956705600, 956705600];

        let (movie_hist, ratings_hist) = user_history.get_history_before_timestamp(user_ids.clone(), ts.clone(), max_history);

        assert_eq!(movie_hist.len(), user_ids.len() * max_history);



        let (lookup, max_history_found) = _testable_build_map_async(&ratings_uris);

        let fixed_length = max_history;

        for (i, &user_id) in user_ids.iter().enumerate() {
            // 1. Get the slice for this specific user from the flattened result
            let start = i * fixed_length;
            let end = start + fixed_length;
            let movies = &movie_hist[start..end];

            // ssert length
            assert_eq!(fixed_length, movies.len());

            // Count non-padding values (equivalent to np.sum(movies != -1))
            let count = movies.iter().filter(|&&m| m != -1).count();

            // Look up the raw data for this user
            let entry = &lookup[&user_id];

            // Binary search for the insertion point (equivalent to np.searchsorted)
            // partition_point returns the index of the first element that is NOT < target_ts
            let target_ts = ts[i];
            let end_idx = entry.timestamps.partition_point(|&t| t < target_ts);

            assert_eq!(end_idx, count);

            //  Specific unit test case for user 6039
            if user_id == 6039 {
                let test_ts = 956705636;
                let test_idx = entry.timestamps.partition_point(|&t| t < test_ts);

                // item at index 44 is 6940. test_idx=43
                // Ensure we don't go out of bounds
                if test_idx < entry.movie_ids.len() {
                    let test_movie = entry.movie_ids[test_idx];

                    let allowed_values = [8146, 6940, 7906];
                    assert!(allowed_values.contains(&test_movie), "Value {} was not one of {:?}", test_movie, allowed_values);

                    // assert that test_movie is NOT in the retrieved movies
                    assert!(!movies.contains(&test_movie), "Test movie found in history when it shouldn't be");
                }
            }
        }

        let tt = 2;

    }

}