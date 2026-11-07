#[cfg(test)]
mod recommended_movies_tests {

    mod helper {
        include!("helper.rs");
    }
    use helper::get_recommended_movies_uris;
    use inference_engine::recommended_movies::build_recommended_movies;

    #[test]
    pub fn test_movie_recomendations_build() {
        let (movies_rec_uri, movies_rec_ts_uri) = get_recommended_movies_uris(Some(1));

        let num_users: usize = 6040;

        let recommended_movies =
            build_recommended_movies(num_users, &movies_rec_uri, &movies_rec_ts_uri);

        assert!(recommended_movies.movie_ids.len() > num_users);
        let num_movies = recommended_movies.movie_ids.len() / num_users;
        assert_eq!(recommended_movies.movie_ids.len(), num_users * num_movies);

        let top_k: usize = 20;
        let user_ids = vec![1, 3];
        let timestamps: Vec<i64> = vec![978300760, 978298147];

        let movies = recommended_movies.get_unseen_movies(&user_ids, &timestamps, top_k);
        assert_eq!(movies.len(), user_ids.len() * top_k);

        let worst_movies =
            recommended_movies.get_lowest_scoring_unseen_movies(&user_ids, &timestamps, top_k);
        assert_eq!(worst_movies.len(), user_ids.len() * top_k);

        // Verify all returned non-padded movies for each user are genuinely unseen (> target_ts)
        for (i, (&uid, &target_ts)) in user_ids.iter().zip(timestamps.iter()).enumerate() {
            let user_idx = (uid - 1) as usize;
            let row_movies =
                &recommended_movies.movie_ids[user_idx * num_movies..(user_idx + 1) * num_movies];
            let row_ts =
                &recommended_movies.timestamps[user_idx * num_movies..(user_idx + 1) * num_movies];

            let expected_unseen_top: Vec<i32> = row_movies
                .iter()
                .zip(row_ts.iter())
                .filter_map(|(&m, &t)| (t > target_ts).then_some(m))
                .take(top_k)
                .collect();

            let actual_top = &movies[i * top_k..(i + 1) * top_k];
            assert_eq!(
                &actual_top[0..expected_unseen_top.len()],
                &expected_unseen_top[..]
            );
        }
    }
}