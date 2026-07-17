
pub mod user_history;
pub mod recommended_movies;
pub mod graph_builder;

pub mod embeddings_util;

pub mod util;
pub mod model_client;
pub mod orchestrator;
pub mod embeddings_ann;
pub mod app_runner;
pub mod app_config;

pub mod pb {
    tonic::include_proto!("recommender");
}