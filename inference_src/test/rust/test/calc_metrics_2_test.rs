#[cfg(test)]
mod calc_metrics_2_tests {
    use tokio::sync::oneshot;
    use tokio::task::JoinHandle;

    struct TestServerGuard {
        tx_shutdown: Option<oneshot::Sender<()>>,
        server_handle: Option<JoinHandle<()>>,
    }

    impl Drop for TestServerGuard {
        fn drop(&mut self) {
            // Trigger the shutdown signal
            if let Some(tx) = self.tx_shutdown.take() {
                let _ = tx.send(());
            }

            // Safely block and join the background server thread using block_in_place
            if let Some(handle) = self.server_handle.take() {
                let _ = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(handle)
                });
            }
        }
    }

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
        include!("helper_users.rs");
    }

    use helper::{get_project_dir, load_and_concat_parquet, get_unique_user_and_first_timestamp,
        get_user_movie_tier_map};
    use std::path::PathBuf;
    use std::collections::{HashMap};
    use polars::prelude::DataFrame;
    use tonic::Request;
    use inference_engine::app_config::AppConfig;
    use inference_engine::app_runner::AppRunner;
    use inference_engine::model_client::RankerModelClient;
    use inference_engine::movie_tiers::load_from_file;
    use inference_engine::pb::UsersRequest;
    use inference_engine::ranker_model_metadata::RankerModelMetadata;
    use inference_engine::user_db::UserDb;
    use crate::calc_metrics_2_tests::helper::get_config_json_uri;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    pub async fn test_calc_test_metrics() {

        // ======= setup server ======================
        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        // the default confg is for the batch ranker model, so change to the single inference model:
        let ranker_metadat_uri = config.ranker_metadata_uri.clone();
        let single_uri = ranker_metadat_uri.replace("batch", "single");

        let ranker_metadata = RankerModelMetadata::load_from_file(&single_uri).unwrap();

        let client = RankerModelClient::new(config.ranker_uri.clone(), ranker_metadata.clone()).await;

        let _top_k = config.top_k;
        let _user_db_path: PathBuf = config.user_db_path.clone();
        let _persisted_index_path : PathBuf = config.persisted_index_path.clone();

        let _max_history = ranker_metadata.max_history;
        let _num_candidates = ranker_metadata.num_candidates;
        let _num_catalog_users = ranker_metadata.num_catalog_users;

        let runner = AppRunner::new(config.to_owned());

        let (tx_shutdown, rx_shutdown) = oneshot::channel::<()>();
        let (tx_addr, rx_addr) = oneshot::channel::<std::net::SocketAddr>();
        // Spawn the server in a background Tokio task
        let server_handle = tokio::spawn(async move {
            let shutdown_future = async {
                rx_shutdown.await.ok();
            };
            runner.run(shutdown_future, Some(tx_addr)).await.expect("Server crashed");
        });
        let addr = rx_addr.await.expect("Failed to receive server address");
        // Initialize the RAII Guard.
        // It will automatically trigger shutdown and join if the test finishes or panics.
        let _server_guard = TestServerGuard {
            tx_shutdown: Some(tx_shutdown),
            server_handle: Some(server_handle),
        };

        let endpoint = format!("http://{}", addr);

        // ===== run tests ==========

        if true {
            calc_tier_stratified_metrics(config.clone(), endpoint).await;
        }


        // server is shutdown by the guard when this method is out of scope

    }

    async fn calc_tier_stratified_metrics(config: AppConfig,  endpoint: String) -> Result<(), Box<dyn std::error::Error>> {

        // get the movie_tiers file
        let movie_tiers : HashMap<i32, i32> = load_from_file(&config.movie_tiers_path)?;

        // get the ground truth
        let proj_dir : String = get_project_dir().unwrap().to_string_lossy().into_owned();
        let test_liked = vec![
            format!("{}/src/test/resources/data/ratings_test_liked.parquet", proj_dir)];

        let (movie_tier_df_map, user_ids, timestamps) : (HashMap<i32, DataFrame>,Vec<i32>, Vec<i64> )
               = get_user_datastructures(&[&test_liked[0]], movie_tiers);

        let user_db : UserDb = UserDb::new(&config.user_db_path).expect("Failed to initialize UserDb from binary path");

        let ranker_metadata = RankerModelMetadata::load_from_file(&config.ranker_metadata_uri)?;

        let ranker_batch_size = ranker_metadata.batch_size;

        for (chunk_users, chunk_timestamps) in user_ids.chunks(ranker_batch_size).zip(timestamps.chunks(ranker_batch_size)) {
            // chunks are &[i32] slices
            let res :  Option<Request<UsersRequest>> = user_db.get_request(chunk_users, chunk_timestamps);

        }


        Ok(())
    }

    fn get_user_datastructures(ratings_uris: &[&str], movie_tiers : HashMap<i32, i32>)
        -> (HashMap<i32, DataFrame>, Vec<i32>, Vec<i64>) {

        let df_gt = load_and_concat_parquet(ratings_uris);

        let movie_tier_df_map = get_user_movie_tier_map(df_gt.clone(), movie_tiers);

        // these are needed for the UsersRequest
        let (user_ids, timestamps) : (Vec<i32>, Vec<i64>) = get_unique_user_and_first_timestamp(df_gt.clone());

        (movie_tier_df_map, user_ids, timestamps)

    }
}