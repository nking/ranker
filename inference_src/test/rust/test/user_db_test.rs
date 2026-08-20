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
        let user_ids = vec![1];
        let timestamps = vec![978300719];
        let user_req_opt = user_db.get_request(&user_ids, &timestamps);
        assert!(user_req_opt.is_some(), "User ID {} should exist in database", user_ids[0]);
        let tonic_req = user_req_opt.unwrap();

        // Extract the inner UserRequest from tonic::Request using .get_ref()
        let user_req = tonic_req.get_ref();
        assert_eq!(user_req.user_ids[0], user_ids[0]);
        assert_eq!(user_req.genders[0], "F");
        assert_eq!(user_req.ages[0], 1);
        assert_eq!(user_req.occupations[0], 10);

        //6040::M::25::6::11106
        let user_ids = vec![6040];
        let timestamps = vec![956716207];
        let user_req_opt = user_db.get_request(&user_ids, &timestamps);
        assert!(user_req_opt.is_some(), "User ID {} should exist in database", user_ids[0]);
        let tonic_req = user_req_opt.unwrap();

        // Extract the inner UserRequest from tonic::Request using .get_ref()
        let user_req = tonic_req.get_ref();
        assert_eq!(user_req.user_ids[0], user_ids[0]);
        assert_eq!(user_req.genders[0], "M");
        assert_eq!(user_req.ages[0], 25);
        assert_eq!(user_req.occupations[0], 6);

        let user_ids = vec![1, 6040];
        let timestamps = vec![978300719, 956716207];
        let batch_user_req_opt = user_db.get_request(&user_ids, &timestamps);
        assert!(batch_user_req_opt.is_some(), "User IDs {:?} should exist in database", user_ids.clone());
        let tonic_req = batch_user_req_opt.unwrap();
        let user_req = tonic_req.get_ref();
        assert_eq!(user_req.n_users as usize, user_ids.len());
        assert_eq!(user_req.timestamps.len() as usize, timestamps.len());
        for i in 0..user_ids.len() {
            if i == 0 {
                assert_eq!(user_req.user_ids[i], 1);
                assert_eq!(user_req.genders[i].clone(), "F");
                assert_eq!(user_req.ages[i], 1);
                assert_eq!(user_req.occupations[i], 10);
            } else {
                assert_eq!(user_req.user_ids[i], 6040);
                assert_eq!(user_req.genders[i].clone(), "M");
                assert_eq!(user_req.ages[i], 25);
                assert_eq!(user_req.occupations[i], 6);
            }
        }

    }

}