//use tonic::transport::Server;
//use std::sync::Arc;
//use notify::{Watcher, RecursiveMode, Result};

// Now orchestrator.rs can see it via crate::pb

#[tokio::main]
async fn main() {

}
/*
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // TODO: construct Arc<RwLock<AppState>> and pass to Orchestrator

    // Initialize shared orchestrator with clients
    let orchestrator = Arc::new(Orchestrator::new().await?);

    // Start gRPC Server
    let addr = "[::1]:50051".parse()?;
    Server::builder()
        .add_service(RecommenderService::new(orchestrator))
        .serve(addr)
        .await?;
    Ok(())
}*/
