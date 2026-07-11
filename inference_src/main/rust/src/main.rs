use tonic::transport::Server;
use std::sync::Arc;
use notify::{Watcher, RecursiveMode, Result};
use crate::orchestrator::Orchestrator;
use crate::states::AppState;
use crate::pb::{UserRequest, RankedMovies, recommender_service_server::RecommenderService};

// Now orchestrator.rs can see it via crate::pb

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // TODO: impl ...
}
