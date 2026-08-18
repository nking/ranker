#[cfg(test)]
mod embedding_ann_tests {

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    use std::collections::HashMap;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::PathBuf;
    use serde_json::Value;
    use usearch::ffi::Matches;
    use inference_engine::app_config::AppConfig;
    use inference_engine::embeddings_ann::Searcher;
    use inference_engine::embeddings_util::read_user_embeddings;
    use crate::embedding_ann_tests::helper::{get_embeddings_uris};

    #[tokio::test]
    async fn test_search() {

        // get num_candidates from the default config for this model
        let config_path = "./config/default_tiny.json";
        let config = AppConfig::load_from_file(config_path).unwrap();
        let file = File::open(&config.params_json_path).unwrap();
        let reader = BufReader::new(file);
        let dict: HashMap<String, Value> = serde_json::from_reader(reader).unwrap();

        let _max_history = dict.get("max_history").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let num_candidates = dict.get("num_candidates").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let _num_catalog_users = dict.get("num_catalog_users").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let (_user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris();

        let persisted_index_path: PathBuf = PathBuf::from("./target/movie_embeddings_indexer");

        let search = Searcher::new(&movie_embeddings_uri, num_candidates, persisted_index_path).unwrap();

        /*
        //copied from loadiing user embeddingss into polars ahd printing head(1) to get embedding for user_id=1
        let query_embedding: Vec<f32> = vec![
            0.335328, -0.010079, 0.216715, -0.306481, 0.30437, 0.070854, 0.124016, -0.126012, -0.15604,
            -0.132381, 0.04622, -0.030724, -0.031848, 0.022452, 0.148558, -0.372832, 0.096318, -0.187772,
            0.032677, 0.043376, 0.169722, 0.338766, -0.113468, -0.016541, 0.110968, 0.132685, 0.025607,
            -0.222184, -0.078174, 0.158795, 0.096952, -0.291931
        ];*/

        let (query_embeddings_vec, num_catalog_users, embed_len) = read_user_embeddings(&*_user_embeddings_uri);

        let results : Result<Matches, Box<dyn std::error::Error>> = search.search(&query_embeddings_vec[0..embed_len], None);

        let m : Matches = results.unwrap();
        let candidate_ids = &m.keys;
        let _distances = &m.distances;

        assert_eq!(num_candidates,  candidate_ids.len());

        // check that restore doesn't  throw errors
        let _indexer = search.restore().unwrap();
    }

    #[tokio::test]
    async fn test_search_batch() {

        // get num_candidates from the default config for this model
        let config_path = "./config/default_tiny.json";
        let config = AppConfig::load_from_file(config_path).unwrap();
        let file = File::open(&config.params_json_path).unwrap();
        let reader = BufReader::new(file);
        let dict: HashMap<String, Value> = serde_json::from_reader(reader).unwrap();

        let _max_history = dict.get("max_history").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let num_candidates = dict.get("num_candidates").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let _num_catalog_users = dict.get("num_catalog_users").and_then(|v| v.as_u64()).unwrap_or(0) as usize;


        let (_user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris();

        let persisted_index_path: std::path::PathBuf = PathBuf::from("./target/movie_embeddings_indexer");

        let search = Searcher::new(&movie_embeddings_uri, num_candidates, persisted_index_path).unwrap();

        let query_embedding: Vec<f32> = vec![
            0.117549196, 0.238659769, -0.215364203, -0.0403997824, 0.315108567, -0.468034804,
            -0.188685074, -0.0422358438, -0.0276149735, 0.021486342, -0.518427193, -0.194741741,
            0.139777973, 0.0450548381, -0.294477165, 0.108183414,
            0.312671244, 0.777051866, -0.589712679, -0.474950016, 0.872076154, 0.300890654,
            -0.489465088, 0.140587822, 0.0260277539, 0.32351321, -0.159624249, 0.587409139,
            -0.11216034, -0.650176942, -0.309607685, 0.14945665
        ];

        let results : Result<Vec<Matches>, Box<dyn std::error::Error>>
            = search.search_batch(&query_embedding, None);

        let m : Vec<Matches> = results.unwrap();
        for i in 0..m.len() {
            let candidate_ids = &m[i].keys;
            let _distances = &m[i].distances;
            assert_eq!(num_candidates,  candidate_ids.len());
        }


    }

}