use std::ptr::null;
use std::sync::Arc;
use arc_swap::ArcSwap;
use crate::client::{QueryModelClient, RankerModelClient};
use crate::embeddings_ann::Searcher;
use crate::graph_builder::{build_enriched_padded_supergraph, JraphGraph};
use crate::states::{AppState, ServiceState};
use crate::user_history::{build_user_history, UserHistory};

// Now you can use them directly!
use crate::pb::{UserRequest, RankedMovies, recommender_service_server::RecommenderService};
use tonic::{Request, Response, Status};
use usearch::ffi::Matches;

// the number of local_devices attached to the ranker TFS.  e.g. = 2 for the kaggle T4x2 GPUs
const ranker_n_local_devices : usize = 1;

// max_history, num_candidates are hyper-parameters of the ranker_model
pub struct Orchestrator {
    query_model: QueryModelClient,
    ranker_model: RankerModelClient,
    searcher: ArcSwap<Searcher>, // updatable
    user_history: UserHistory,  // can be made updatable in future
    max_history: usize,
    num_candidates: usize,
    num_catalog_users: usize,
}

impl Orchestrator {
    // Note: We make this async because connecting to gRPC takes time
    pub async fn new(query_uri: &'static str, ranker_uri: &'static str,
        movie_embeddings_uri: &str,
        ratings_uris: Vec<&str>,
        max_history: usize, num_candidates: usize,
        num_catalog_users: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {

        let initial_searcher = Searcher::new(movie_embeddings_uri)?;

        let query_client = QueryModelClient::new(query_uri).await;
        let ranker_client = RankerModelClient::new(ranker_uri).await;

        let user_history: UserHistory = build_user_history(&ratings_uris, 2048);

        Ok(Self {
            query_model: query_client,
            ranker_model: ranker_client,
            max_history: max_history,
            num_candidates: num_candidates,
            num_catalog_users: num_catalog_users,
            searcher: ArcSwap::from_pointee(initial_searcher),
            user_history: user_history,
        })
    }

    pub async fn reload_embeddings(&self, movie_embeddings_uri: &str) -> Result<(), Box<dyn std::error::Error>> {

        let uri_owned: String = movie_embeddings_uri.to_string();

        // CPU Bound: Building a usearch index is heavy math.
        // We MUST offload this to tokio's blocking thread pool so we don't
        // starve the async workers handling incoming gRPC requests.
        let new_searcher = tokio::task::spawn_blocking(move || {
            // temporarily, consuming twice as much RAM with current and new indexer
            Searcher::new(&uri_owned.as_str()).map_err(|e| e.to_string())
        }).await??;

        // The Swap: This takes nanoseconds.
        // We drop the new instance into an Arc and swap the pointer.
        // Any request that starts *after* this line uses the new index.
        // Any request currently executing keeps using the old index until it finishes.
        self.searcher.store(Arc::new(new_searcher));

        Ok(())
    }
}

#[tonic::async_trait]
impl RecommenderService for Orchestrator {
    async fn predict(&self, req: Request<UserRequest>) ->Result<Response<RankedMovies>, tonic::Status> {

        let inner_req = req.into_inner();
        let user_ids : Vec<i32> = vec![inner_req.user_id as i32];
        let timestamps = vec![inner_req.timestamp];

        // Get user_embedding from TFS Query model
        let user_embedding = self.query_model.get_user_embedding(&inner_req).await
            .map_err(|e| tonic::Status::internal(format!("user embedding: {}", e)))?;
        let user_embeddings = user_embedding;

        let searcher = self.searcher.load();
        let nearest : Vec<Matches> = searcher.search(&user_embeddings)
            .map_err(|e| tonic::Status::internal(format!("Vector search failed: {}", e)))?;
        let candidate_ids: Vec<i32> = nearest
            .into_iter()
            // Flatten the nested collections
            .flat_map(|m| m.keys)
            // Cast the USearch u64 ID to your Protobuf i32 ID
            .map(|key| key as i32)
            .collect();

        let labels: Vec<i32> = vec![1; candidate_ids.len()];

        let padded_super_graph_arrays : JraphGraph = build_enriched_padded_supergraph(
            &user_ids,
            &timestamps,
            &candidate_ids,
            &labels, &self.user_history, self.max_history,
            self.num_catalog_users, searcher.get_num_catalog_movies(),
            searcher.get_embed_len(),
            searcher.get_movies_embedding_catalog_ref(),
            &user_embeddings,  ranker_n_local_devices);

        // Send to TFS Ranker model
        let final_response = self.ranker_model.get_candidate_ranks(
            padded_super_graph_arrays, searcher.get_embed_len()).await;

        //TODO: process the response

        match final_response {
            Ok(ranks) => {
                let r = RankedMovies{
                    movie_ids: candidate_ids,
                    scores: ranks,
                };
                Ok(Response::new(r))
            },
            Err(e) => {
                Err(tonic::Status::internal(format!("ranking request failed: {}", e)))
            }
        }
    }
}
