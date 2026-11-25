#[cfg(test)]
mod tests {
    use inference_engine::util_sampling::*;
    use std::collections::HashSet;

    // -------------------------------------------------------------------------
    // Ported tests from test_util_numba.py
    // -------------------------------------------------------------------------

    #[test]
    fn test_row_wise_intersect() {
        let pad_value = -1;
        let a = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        // np.roll(a, shift=1, axis=1) with last element replaced by pad_value
        let b = vec![vec![3, 0, 1, pad_value], vec![7, 4, 5, pad_value]];

        let c = row_wise_intersect(&a, &b, pad_value).unwrap();
        let expected = vec![vec![0, 1, 3], vec![4, 5, 7]];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_row_wise_subtract() {
        let pad_value = -1;
        let a = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let b = vec![vec![3, 0, 1, pad_value], vec![7, 4, 5, pad_value]];

        let c = row_wise_sortedset_subtract(&a, &b, pad_value).unwrap();
        // {0,1,2,3} \ {3,0,1} = {2}; {4,5,6,7} \ {7,4,5} = {6}
        let expected = vec![vec![2], vec![6]];
        assert_eq!(c, expected);
    }

    #[test]
    fn test_simultaneous_shuffle() {
        let seed = 0;
        let a_orig = vec![
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![8, 9, 10, 11, 12, 13, 14, 15],
        ];
        // Roll each row right by 1
        let b_orig: Vec<Vec<i32>> = a_orig
            .iter()
            .map(|row| {
                let mut rolled = row.clone();
                rolled.rotate_right(1);
                rolled
            })
            .collect();

        let mut a = a_orig.clone();
        let mut b = b_orig.clone();
        simultaneous_shuffle(&mut a, &mut b, seed).unwrap();

        assert_ne!(a, a_orig);
        assert_ne!(b, b_orig);

        // Verify paired alignment: wherever a_orig[row][col] landed, b_orig[row][col] landed too
        for row in 0..a.len() {
            for col in 0..a[row].len() {
                let new_col = a[row]
                    .iter()
                    .position(|&x| x == a_orig[row][col])
                    .expect("Every original element must remain present after shuffle");
                assert_eq!(b[row][new_col], b_orig[row][col]);
            }
        }
    }

    #[test]
    fn test_generate_type_4_negatives() {
        let seed = 0;
        let pad_value = -1;
        let num_movies = 20;
        let n_negs = 10;

        let all_movie_ids: Vec<i32> = (0..num_movies).collect();
        let mut hist_row: Vec<i32> = (0..num_movies).filter(|i| (i & 1) == 0).collect();
        hist_row.extend_from_slice(&[pad_value, pad_value]);
        let movie_histories = vec![hist_row];

        let expected_pool =
            row_wise_sortedset_subtract(&[all_movie_ids.clone()], &movie_histories, pad_value)
                .unwrap();
        assert_eq!(expected_pool[0].len(), 10);
        assert!(expected_pool[0].iter().all(|&m| m % 2 == 1));

        let type_4_neg =
            generate_type_4_negatives(&all_movie_ids, &movie_histories, n_negs, pad_value, seed);

        assert_eq!(type_4_neg.len(), 1);
        assert_eq!(type_4_neg[0].len(), n_negs);
        assert!(type_4_neg[0].iter().all(|&m| m % 2 == 1));

        let hist_set: HashSet<i32> = movie_histories[0].iter().copied().collect();
        let type_4_set: HashSet<i32> = type_4_neg[0].iter().copied().collect();
        assert_eq!(type_4_set.len(), n_negs);
        assert_eq!(type_4_set.intersection(&hist_set).count(), 0);
    }

    #[test]
    fn test_build_negative_pool() {
        let pad_value = -1;
        let arr1 = vec![vec![1, 2, pad_value], vec![pad_value, pad_value, pad_value]];
        let arr2 = vec![
            vec![3, 4, pad_value, pad_value],
            vec![9, pad_value, pad_value, pad_value],
        ];
        let arr3 = vec![
            vec![pad_value, pad_value, pad_value, pad_value],
            vec![10, pad_value, pad_value, pad_value],
        ];
        let arr4 = vec![vec![5, 6, 7, 8], vec![20, 21, 22, 23]];

        let negatives = build_negative_pool(
            &arr1, &arr2, &arr3, &arr4, 1, 1, 1, 4, pad_value, 12345,
        )
            .unwrap();

        assert_eq!(negatives.len(), 2);
        assert_eq!(negatives[0].len(), 4);
        assert_eq!(negatives[1].len(), 4);

        let arr1_r0: HashSet<i32> = [1, 2].into_iter().collect();
        let arr2_r0: HashSet<i32> = [3, 4].into_iter().collect();
        let arr4_r0: HashSet<i32> = [5, 6, 7, 8].into_iter().collect();
        let arr4_r1: HashSet<i32> = [20, 21, 22, 23].into_iter().collect();

        assert!(arr1_r0.contains(&negatives[0][0]));
        assert!(arr2_r0.contains(&negatives[0][1]));
        assert!(arr4_r0.contains(&negatives[0][2]));
        assert!(arr4_r0.contains(&negatives[0][3]));
        assert_eq!(negatives[0].iter().copied().collect::<HashSet<_>>().len(), 4);

        assert_eq!(negatives[1][0], 9);
        assert_eq!(negatives[1][1], 10);
        assert!(arr4_r1.contains(&negatives[1][2]));
        assert!(arr4_r1.contains(&negatives[1][3]));
        assert_eq!(negatives[1].iter().copied().collect::<HashSet<_>>().len(), 4);
    }

    #[test]
    fn test_extract_shuffle_and_append() {
        let pad_value = -1;
        let num_negatives = 7;
        let source_row = vec![1, 2, 3, 4, 5];
        let max_take = 3;
        let source_set: HashSet<i32> = source_row.iter().copied().collect();

        let mut rng = SplitMix64::new(99);
        let mut pool_row = vec![pad_value; num_negatives];
        let mut buff = Vec::new();

        let write_to_idx = _extract_shuffle_and_append(
            &source_row,
            max_take,
            &mut pool_row,
            0,
            num_negatives,
            pad_value,
            &mut buff,
            &mut rng,
        );

        assert_eq!(write_to_idx, max_take);
        for &val in &pool_row[..write_to_idx] {
            assert!(source_set.contains(&val));
        }
        assert_eq!(
            pool_row[..write_to_idx].iter().copied().collect::<HashSet<_>>().len(),
            max_take
        );
        assert!(pool_row[write_to_idx..].iter().all(|&v| v == pad_value));

        let mut pool_row2 = vec![pad_value; num_negatives];
        let write_to_idx2 = _extract_shuffle_and_append(
            &source_row,
            num_negatives,
            &mut pool_row2,
            write_to_idx,
            num_negatives,
            pad_value,
            &mut buff,
            &mut rng,
        );

        assert_eq!(write_to_idx2, num_negatives);
        assert!(pool_row2[..write_to_idx].iter().all(|&v| v == pad_value));
        for &val in &pool_row2[write_to_idx..num_negatives] {
            assert!(source_set.contains(&val));
        }
        assert_eq!(
            pool_row2[write_to_idx..num_negatives]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            num_negatives - write_to_idx
        );
    }

    #[test]
    fn test_pad_value_and_deduplication_regression() {
        let custom_pad = -999;
        let arr1 = vec![vec![10, 20, 30, custom_pad]];
        let arr2 = vec![vec![20, custom_pad, custom_pad, custom_pad]];

        let sub = row_wise_sortedset_subtract(&arr1, &arr2, custom_pad).unwrap();
        assert_eq!(sub, vec![vec![10, 30]]);

        // arr4 contains 2, 3, 6 already picked in arr1..arr3; pool must remain 4 unique IDs
        let pool1 = vec![vec![2, -1]];
        let pool2 = vec![vec![6, -1]];
        let pool3 = vec![vec![3, -1]];
        let pool4 = vec![vec![2, 3, 6, 8]];

        let neg_pool =
            build_negative_pool(&pool1, &pool2, &pool3, &pool4, 1, 1, 1, 4, -1, 42).unwrap();

        let row_set: HashSet<i32> = neg_pool[0].iter().copied().collect();
        let expected: HashSet<i32> = [2, 6, 3, 8].into_iter().collect();
        assert_eq!(row_set.len(), 4);
        assert_eq!(row_set, expected);
    }

    // -------------------------------------------------------------------------
    // Additional Rust-specific edge-case, determinism, and graph tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parallel_determinism_with_same_seed() {
        let all_movies: Vec<i32> = (1..=200).collect();
        let exclude: Vec<Vec<i32>> = (0..32)
            .map(|u| vec![u + 1, u + 2, u + 3, -1])
            .collect();

        let run1 = generate_type_4_negatives(&all_movies, &exclude, 19, -1, 777);
        let run2 = generate_type_4_negatives(&all_movies, &exclude, 19, -1, 777);
        assert_eq!(run1, run2);

        let empty = vec![vec![]; 32];
        let p1 = build_negative_pool(&empty, &empty, &empty, &run1, 3, 10, 3, 19, -1, 888).unwrap();
        let p2 = build_negative_pool(&empty, &empty, &empty, &run2, 3, 10, 3, 19, -1, 888).unwrap();
        assert_eq!(p1, p2);

        let mut c1 = p1.clone();
        let mut l1 = vec![vec![0; 19]; 32];
        let mut c2 = p2.clone();
        let mut l2 = vec![vec![0; 19]; 32];
        simultaneous_shuffle(&mut c1, &mut l1, 999).unwrap();
        simultaneous_shuffle(&mut c2, &mut l2, 999).unwrap();
        assert_eq!(c1, c2);
        assert_eq!(l1, l2);
    }

    #[test]
    fn test_generate_type_4_negatives_small_catalog_fallback_does_not_hang() {
        let pad = -1;
        // Only 3 valid movies left after excluding [1, 2], but requesting n_negs = 6
        let all_movies = vec![1, 2, 3, 4, 5];
        let exclude = vec![vec![1, 2, pad]];

        let out = generate_type_4_negatives(&all_movies, &exclude, 6, pad, 42);
        assert_eq!(out[0].len(), 6);

        let valid: HashSet<i32> = out[0].iter().copied().filter(|&v| v != pad).collect();
        let expected_valid: HashSet<i32> = [3, 4, 5].into_iter().collect();
        assert_eq!(valid, expected_valid);
        assert_eq!(out[0].iter().filter(|&&v| v == pad).count(), 3);
    }

    #[test]
    fn test_extract_deduplicates_within_source_row_and_stops_at_pad() {
        let pad = -1;
        // Duplicate 2s before pad, and 99 after pad (must be ignored)
        let source_row = vec![2, 2, 3, 2, pad, 99];
        let mut pool_row = vec![pad; 5];
        let mut buff = Vec::new();
        let mut rng = SplitMix64::new(7);

        let written = _extract_shuffle_and_append(
            &source_row,
            5,
            &mut pool_row,
            0,
            5,
            pad,
            &mut buff,
            &mut rng,
        );

        assert_eq!(written, 2);
        let picked: HashSet<i32> = pool_row[..written].iter().copied().collect();
        assert_eq!(picked, [2, 3].into_iter().collect());
        assert!(!pool_row.contains(&99));
    }

    #[test]
    fn test_shape_mismatch_errors() {
        assert!(row_wise_intersect(&[vec![1]], &[vec![1], vec![2]], -1).is_err());
        assert!(row_wise_sortedset_subtract(&[vec![1]], &[], -1).is_err());
        assert!(
            build_negative_pool(&[vec![1]], &[], &[vec![1]], &[vec![1]], 1, 0, 0, 1, -1, 0)
                .is_err()
        );

        let mut c = vec![vec![1, 2]];
        let mut l_bad_rows = vec![vec![1, 0], vec![0, 1]];
        assert!(simultaneous_shuffle(&mut c, &mut l_bad_rows, 0).is_err());

        let mut l_bad_cols = vec![vec![1]];
        assert!(simultaneous_shuffle(&mut c, &mut l_bad_cols, 0).is_err());
    }

    #[test]
    fn test_build_graph_arrays_with_and_without_history() {
        let g = build_graph_arrays(
            7,
            2,
            &[201, 202, 203],
            &[5, 3, -1],
            &[101, 102, -1],
            &[0, 1, 0],
        );

        assert_eq!(g.total_nodes, 6);
        assert_eq!(g.total_edges, 5);
        assert_eq!(g.senders, vec![1, 2, 0, 0, 0]);
        assert_eq!(g.receivers, vec![0, 0, 3, 4, 5]);
        assert_eq!(g.edge_features, vec![5, 3, 0, 0, 0]);
        assert_eq!(g.node_ids, vec![7, 101, 102, 201, 202, 203]);
        assert_eq!(g.node_labels, vec![0, 0, 0, 0, 1, 0]);
        assert_eq!(g.node_types, vec![1, 2, 2, 3, 3, 3]);
        assert_eq!(g.candidate_mask, vec![false, false, false, true, true, true]);

        // Zero-history (cold-start) edge case
        let g_cold = build_graph_arrays(9, 0, &[50, 60], &[], &[], &[1, 0]);
        assert_eq!(g_cold.total_nodes, 3);
        assert_eq!(g_cold.total_edges, 2);
        assert_eq!(g_cold.senders, vec![0, 0]);
        assert_eq!(g_cold.receivers, vec![1, 2]);
        assert_eq!(g_cold.edge_features, vec![0, 0]);
        assert_eq!(g_cold.node_ids, vec![9, 50, 60]);
        assert_eq!(g_cold.node_labels, vec![0, 1, 0]);
        assert_eq!(g_cold.node_types, vec![1, 3, 3]);
        assert_eq!(g_cold.candidate_mask, vec![false, true, true]);
    }
}