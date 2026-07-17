use tokio::sync::oneshot;
use tokio::signal;
use inference_engine::app_config::AppConfig;
use inference_engine::app_runner::AppRunner;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

    let config_path = "./config/default.json";

    let config = AppConfig::load_from_file(config_path)?;

    let runner = AppRunner::new(config);

    // tx_shutdown acts as our remote kill switch
    let (tx_shutdown, rx_shutdown) = oneshot::channel::<()>();
    // Create a new channel to receive the bound address
    let (tx_addr, rx_addr) = oneshot::channel::<std::net::SocketAddr>();

    // Spawn the server in a background Tokio task
    let server_handle = tokio::spawn(async move {
        let shutdown_future = async {
            rx_shutdown.await.ok();
        };
        runner.run(shutdown_future, Some(tx_addr)).await.expect("Server crashed");
    });

    //  Await the bound address to guarantee the service is up and listening
    if let Ok(addr) = rx_addr.await {
        println!("Production server successfully bound to {}", addr);
    }

    // Block the main thread until an OS signal is received (e.g., Ctrl+C)
    signal::ctrl_c().await.expect("Failed to listen for event");
    println!("\nShutdown signal received. Initiating graceful shutdown...");

    // Trigger the remote kill switch
    let _ = tx_shutdown.send(());

    // Await the server handle to ensure all in-flight requests finish cleanly
    server_handle.await?;
    println!("Server exited successfully.");

    Ok(())
}