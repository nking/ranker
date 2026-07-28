#[cfg(test)]
mod user_db_tests {

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    use inference_engine::app_config::AppConfig;
    use inference_engine::user_db::{UserDb};

    #[tokio::test]
    pub async fn test_user_db_load() {

        let config_path = "./config/default.json";
        let config = AppConfig::load_from_file(config_path).unwrap();
        let user_db_path = config.user_db_path;

        let user_db : UserDb = UserDb::new(user_db_path).unwrap();

        //UserID::Gender::Age::Occupation::Zip-code
        //1::F::1::10::48067
        let user_id : i64 = 1;
        let user_req_opt = user_db.get_request(user_id);
        assert!(user_req_opt.is_some(), "User ID {} should exist in database", user_id);
        let tonic_req = user_req_opt.unwrap();

        // Extract the inner UserRequest from tonic::Request using .get_ref()
        let user_req = tonic_req.get_ref();
        assert_eq!(user_req.user_id, user_id);
        assert_eq!(user_req.gender, "F");
        assert_eq!(user_req.age, 1);
        assert_eq!(user_req.occupation, 10);

        //6040::M::25::6::11106
        let user_id = 6040;
        let user_req_opt = user_db.get_request(user_id);
        assert!(user_req_opt.is_some(), "User ID {} should exist in database", user_id);
        let tonic_req = user_req_opt.unwrap();

        // Extract the inner UserRequest from tonic::Request using .get_ref()
        let user_req = tonic_req.get_ref();
        assert_eq!(user_req.user_id, user_id);
        assert_eq!(user_req.gender, "M");
        assert_eq!(user_req.age, 25);
        assert_eq!(user_req.occupation, 6);


    }

}