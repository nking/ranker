
#[cfg(test)]
mod hard_negative_sampling_test {
    use std::sync::Arc;
    use inference_engine::hard_negative_sampling::{HardNegativeSamplingTransform, SamplingStrategy};
    use inference_engine::recommended_movies::RecommendedMovies;
    use inference_engine::user_history::UserHistory;

    //use super::*;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    const PAD: i32 = -1;
    const TS_PAD: i64 = 2_524_608_000;

    #[test]
    fn sample_negative_candidates_returns_only_negatives() {
        let (history, disliked, recommends, catalog)
            = make_fixture();

        let transform = HardNegativeSamplingTransform::new(
            history,
            disliked,
            catalog,
            recommends,
            20,
            SamplingStrategy::MixedHard,
        )
            .unwrap();

        let pos_ids = [99, 98];
        let negs_excl_pos = transform
            .sample_negative_candidates(&[1, 2], Some(&pos_ids), &[1000, 2000], 42, true)
            .unwrap();

        assert_eq!(negs_excl_pos.len(), 2);
        for (row_idx, &pos_id) in pos_ids.iter().enumerate() {
            let row = &negs_excl_pos[row_idx];
            assert_eq!(row.len(), 19);
            assert!(!row.contains(&pos_id));
            assert!(!row.contains(&PAD));
        }

        let negs_no_pos = transform
            .sample_negative_candidates(&[1, 2], None, &[1000, 2000], 42, false)
            .unwrap();
        assert_eq!(negs_no_pos.len(), 2);
        assert_eq!(negs_no_pos[0].len(), 19);
        assert_eq!(negs_no_pos[1].len(), 19);
        assert!(!negs_no_pos[0].contains(&PAD));
        assert!(!negs_no_pos[1].contains(&PAD));
    }

    fn make_user_history(
        rows_by_uid_1based: &[&[(i32, i32, i64)]],
        max_history: usize,
    ) -> UserHistory {
        let n_users = rows_by_uid_1based.len();
        let total_rows = n_users + 1; // row 0 is reserved sentinel
        let mut movie_ids = vec![PAD; total_rows * max_history];
        let mut ratings = vec![PAD; total_rows * max_history];
        let mut timestamps = vec![TS_PAD; total_rows * max_history];

        for (i, entries) in rows_by_uid_1based.iter().enumerate() {
            let user_idx = i + 1;
            let offset = user_idx * max_history;
            for (k, &(mid, r, ts)) in entries.iter().take(max_history).enumerate() {
                movie_ids[offset + k] = mid;
                ratings[offset + k] = r;
                timestamps[offset + k] = ts;
            }
        }

        UserHistory {
            movie_ids,
            ratings,
            timestamps,
            n_users,
            max_history,
            pad_value: PAD,
            ts_pad_value: TS_PAD,
        }
    }

    fn make_recommended_movies(
        rows_by_uid_1based: &[&[(i32, i64)]],
        row_len: usize,
    ) -> RecommendedMovies {
        let num_users = rows_by_uid_1based.len();
        let mut movie_ids = vec![PAD; num_users * row_len];
        let mut timestamps = vec![TS_PAD; num_users * row_len];

        for (user_idx, entries) in rows_by_uid_1based.iter().enumerate() {
            let offset = user_idx * row_len;
            for (k, &(mid, ts)) in entries.iter().take(row_len).enumerate() {
                movie_ids[offset + k] = mid;
                timestamps[offset + k] = ts;
            }
        }

        RecommendedMovies {
            movie_ids,
            timestamps,
            num_users,
            pad_value: PAD,
            ts_pad_value: TS_PAD,
        }
    }

    fn make_fixture() -> (Arc<UserHistory>, Arc<UserHistory>, RecommendedMovies, Vec<i32>) {
        let history = Arc::new(make_user_history(
            &[
                &[(1, 5, 100), (2, 4, 200), (3, 5, 300), (10, 1, 400)],
                &[(4, 5, 150), (5, 4, 250), (20, 2, 350)],
            ],
            8,
        ));
        let disliked = Arc::new(make_user_history(
            &[
                &[(10, 1, 400), (11, 2, 500), (12, 1, 600)],
                &[(20, 2, 350), (21, 1, 450)],
            ],
            8,
        ));
        let recs = make_recommended_movies(
            &[
                &[
                    (10, 400),
                    (30, 1500),
                    (31, 1600),
                    (32, 1700),
                    (33, 1800),
                    (34, 1900),
                ],
                &[(20, 350), (40, 2500), (41, 2600), (42, 2700), (43, 2800)],
            ],
            8,
        );
        let catalog: Vec<i32> = (1..=100).collect();
        (history, disliked, recs, catalog)
    }

    #[test]
    fn sample_candidates_with_pos_preserves_positive_and_no_padding() {
        let (history, disliked, recommends, catalog) = make_fixture();

        for strategy in [SamplingStrategy::MixedHard, SamplingStrategy::WorstRecommended] {
            let transform = HardNegativeSamplingTransform::new(
                Arc::clone(&history),
                Arc::clone(&disliked),
                catalog.clone(),
                recommends.clone(),
                20,
                strategy,
            )
                .unwrap();

            let out = transform
                .sample_candidates_with_pos(&[1, 2], &[99, 98], &[1000, 2000], 42, true)
                .unwrap();

            assert_eq!(out.candidate_ids.len(), 2);
            for (row_idx, &pos_id) in [99, 98].iter().enumerate() {
                let cands = &out.candidate_ids[row_idx];
                let lbls = &out.labels[row_idx];
                assert_eq!(cands.len(), 20);
                assert_eq!(lbls.iter().sum::<i32>(), 1);
                let pos_slot = lbls.iter().position(|&l| l == 1).unwrap();
                assert_eq!(cands[pos_slot], pos_id);
                assert!(!cands.contains(&PAD));
            }
        }
    }

}