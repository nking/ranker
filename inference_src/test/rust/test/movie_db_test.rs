#[cfg(test)]
mod movie_db_tests {

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    use inference_engine::app_config::AppConfig;
    use inference_engine::model_client::tf_serving::model_spec::VersionChoice;
    use inference_engine::movie_db::MovieDb;
    use inference_engine::user_db::{UserDb};

    #[tokio::test]
    pub async fn test_movie_db_load() {

        let config_path = "./config/default.json";
        let config = AppConfig::load_from_file(config_path).unwrap();
        let movie_db_path = config.movies_path;

        let movie_db : MovieDb = MovieDb::new(movie_db_path).unwrap();

        let model_version = Some(VersionChoice::Version(1));

        //UserID::Gender::Age::Occupation::Zip-code
        //1::F::1::10::48067
        let movie_ids = vec![6041];
        let req_opt = movie_db.get_request(&movie_ids, model_version);
        assert!(req_opt.is_some(), "Movie ID {} should exist in database", movie_ids[0]);
        let tonic_req = req_opt.unwrap();

        // Extract the inner MoviesRequest from tonic::Request using .get_ref()
        let movie_req = tonic_req.get_ref();
        assert_eq!(movie_req.movie_ids[0], movie_ids[0]);
        assert_eq!(movie_req.genres[0], "Animation|Children's|Comedy");


    }

}