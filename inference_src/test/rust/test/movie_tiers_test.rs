#[cfg(test)]
mod movie_tiers_tests {
    use std::collections::HashMap;
    use inference_engine::app_config::AppConfig;
    use inference_engine::movie_tiers::load_from_file;
    //use super::*;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    #[test]
    pub fn test_load() -> Result<(), Box<dyn std::error::Error>>{

        let config_path = "./config/default.json";
        let config = AppConfig::load_from_file(config_path).unwrap();

        let movie_tiers : HashMap<i32, i32> = load_from_file(&config.movie_tiers_path)?;

        assert!(movie_tiers.len() > 1);

        Ok(())
    }

}