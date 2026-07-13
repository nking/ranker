use arc_swap::ArcSwap;
use crate::embeddings_ann::Searcher;
use tokio::sync::RwLock;

pub struct AppState {
    pub state: ServiceState,
    pub indexer: ArcSwap<Searcher>, // The atomic pointer for hot-swapping
    pub state_lock: RwLock<ServiceState>,
}

#[derive(PartialEq, Clone)]
pub enum ServiceState {
    Loading,
    Ready,
    Error(String),
}
