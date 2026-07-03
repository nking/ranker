#[cfg(test)]
mod recommended_movies_tests {

    mod helper {
        // Tell Rust to literally include the code from helper.rs
        include!("helper.rs");
    }
    use helper::{get_project_dir};

    use inference_engine::recommended_movies::{build_recommended_movies};

    #[test]
    pub fn test_movie_recomendations_build() {

        let movies_rec_uri = get_project_dir()
            .map(|p| p.join("src/test/resources/data/recommended_movies.parquet"))
            .map(|p| p.to_string_lossy().into_owned())
            .expect("Project directory not found");

        let movies_rec_ts_uri = get_project_dir()
            .map(|p| p.join("src/test/resources/data/recommended_movies_timestamps.parquet"))
            .map(|p| p.to_string_lossy().into_owned())
            .expect("Project directory not found");

        let num_users: usize = 6040;

        let recommended_movies = build_recommended_movies(num_users, &movies_rec_uri, &movies_rec_ts_uri);

        assert!(recommended_movies.movie_ids.len() > num_users);
        let num_movies = recommended_movies.movie_ids.len() / num_users;
        assert_eq!(recommended_movies.movie_ids.len(), num_users * num_movies);

        let ts : i64 = 978133414; //first timestamp from test dataset
        let top_k : usize = 20;

        let user_ids = vec![1, 3];
        let timestamps : Vec<i64> = vec![978300760, 978298147];

        let movies = recommended_movies.get_unseen_movies(&user_ids, &timestamps, top_k);

        assert_eq!(movies.len(), user_ids.len() * top_k);


    }

}