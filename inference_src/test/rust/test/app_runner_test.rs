#[cfg(test)]
mod app_runner_tests {
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio::time::{sleep, Duration};
    use inference_engine::pb::recommender_service_client::RecommenderServiceClient;
    use inference_engine::pb::UserRequest;

    use inference_engine::app_config::AppConfig;
    use inference_engine::app_runner::AppRunner;
    use crate::app_runner_tests::helper::get_config_json_uri;

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    #[tokio::test]
    async fn test_app_runner_lifecycle_and_request() {
        // Setup channels for synchronization
        // tx_shutdown acts as our remote kill switch
        let (tx_shutdown, rx_shutdown) = oneshot::channel::<()>();
        // Create a new channel to receive the bound address
        let (tx_addr, rx_addr) = oneshot::channel::<std::net::SocketAddr>();

        //  Configure a test instance binding to port (dynamic port allocation)
        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let runner = AppRunner::new(config.clone());

        // Spawn the server in a background Tokio task
        let server_handle = tokio::spawn(async move {
            // The server will run until rx_shutdown receives a message
            let shutdown_future = async {
                rx_shutdown.await.ok(); // Ignore errors if the sender drops early
            };
            // pas the transmitter to the run method
            runner.run(shutdown_future, Some(tx_addr)).await.expect("Server crashed");
        });

        // Synchronization Wait:
        let actual_addr = rx_addr.await.expect("Failed to receive bound address from server");

        // Fire a real gRPC request from the test (client) task
        // Note: Because we used port 0, we'd ideally extract the bound port.
        // For simplicity, assuming a fixed test port like 50052 if you don't extract the OS port.
        // If you hardcode a test port in your config, connect to that:
        let test_uri = format!("http://{}", config.server_addr);

        let mut client = RecommenderServiceClient::connect(test_uri)
            .await
            .expect("Failed to connect to test server");

        let request = tonic::Request::new(UserRequest {
            user_id: 42,
            gender: "M".to_string(),
            occupation: 10,
            age: 25,
            timestamp: 1620000000,
        });

        // Assert success
        let response = client.predict(request).await;
        assert!(response.is_ok(), "gRPC request failed");

        // Trigger Shutdown
        println!("Sending shutdown signal...");
        let _ = tx_shutdown.send(());

        // Await the server handle to ensure it exited cleanly without panicking
        // If the server task panicked, this await will propagate the panic to fail the test.
        server_handle.await.expect("Server task panicked");
        println!("Server shut down cleanly.");
    }

}