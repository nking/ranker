use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use arc_swap::ArcSwap;
use crate::model_client::{QueryModelClient, RankerModelClient};
use crate::embeddings_ann::Searcher;
use crate::graph_builder::{build_enriched_padded_supergraph, JraphGraph};
use crate::user_history::{build_user_history, UserHistory};

// Now you can use them directly!
use crate::pb::{UserRequest, RankedMovies, RankOnlyRequest, ApproxNearestNeighborsResponse};
use tonic::{Request, Response, Status};
use usearch::ffi::Matches;
use crate::pb::recommender_service_server::RecommenderService;
use crate::user_db::UserDb;
use crate::util::sort_by_scores;

// the number of local_devices attached to the ranker TFS.  e.g. = 2 for the kaggle T4x2 GPUs
// max_history, num_candidates are hyper-parameters of the ranker_model
pub struct Orchestrator {
    query_model: QueryModelClient,
    ranker_model: RankerModelClient,
    searcher: ArcSwap<Searcher>, // updatable
    user_history: UserHistory,  // can be made updatable in future
    max_history: usize,
    user_db: UserDb,
    #[allow(dead_code)]
    num_candidates: usize,
    num_catalog_users: usize,
    ranker_n_local_devices : usize,
    persisted_index_path: PathBuf,
    top_k : usize
}

impl Orchestrator {
    // Note: We make this async because connecting to gRPC takes time
    pub async fn new(
        query_uri: String,
        ranker_uri: String,
        movie_embeddings_uri: &str,
        ratings_uris: Vec<&str>,
        max_history: usize,
        num_candidates: usize,
        num_catalog_users: usize,
        ranker_n_local_devices : usize,
        top_k : usize,
        persisted_index_path: impl AsRef<Path>,
        user_db_path : impl AsRef<Path>
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {

        let initial_searcher = Searcher::new(movie_embeddings_uri, num_candidates, &persisted_index_path)?;
        let query_client = QueryModelClient::new(query_uri).await;
        let ranker_client = RankerModelClient::new(ranker_uri).await;

        let user_history: UserHistory = build_user_history(&ratings_uris, 2048).await;
        let user_db : UserDb = UserDb::new(user_db_path).expect("Failed to initialize UserDb from binary path");

        Ok(Self {
            query_model: query_client,
            ranker_model: ranker_client,
            max_history: max_history,
            num_candidates: num_candidates,
            num_catalog_users: num_catalog_users,
            searcher: ArcSwap::from_pointee(initial_searcher),
            user_history: user_history,
            ranker_n_local_devices: ranker_n_local_devices,
            persisted_index_path: persisted_index_path.as_ref().to_path_buf(),
            top_k: top_k,
            user_db: user_db,
        })
    }

    pub async fn reload_embeddings(&self, movie_embeddings_uri: &str, num_candidates: usize) -> Result<(), Box<dyn std::error::Error>> {

        let uri_owned: String = movie_embeddings_uri.to_string();
        let path = self.persisted_index_path.to_owned();
        // CPU Bound: Building a usearch index is heavy math.
        // We MUST offload this to tokio's blocking thread pool so we don't
        // starve the async workers handling incoming gRPC requests.
        let new_searcher = tokio::task::spawn_blocking(move || {
            // temporarily, consuming twice as much RAM with current and new indexer
            Searcher::new(&uri_owned.as_str(), num_candidates, &path).map_err(|e| e.to_string())
        }).await??;

        // The Swap: This takes nanoseconds.
        // We drop the new instance into an Arc and swap the pointer.
        // Any request that starts *after* this line uses the new index.
        // Any request currently executing keeps using the old index until it finishes.
        self.searcher.store(Arc::new(new_searcher));

        Ok(())
    }

    async fn make_ranker_request(&self, user_id : i32, timestamp: i64,
        user_embedding : Vec<f32>, candidate_ids : Vec<i32>) ->Result<RankedMovies, Status> {

        if candidate_ids.len() != self.num_candidates {
            println!(
                "[Warning] Expected candidate_ids length to be {}, but got {}",
                self.num_candidates,
                candidate_ids.len()
            );
        }

        let user_ids : Vec<i32> = vec![user_id];
        let timestamps: Vec<i64> = vec![timestamp];

        // finds num_candidates approx nearest neighbors
        let searcher = self.searcher.load();

        let labels: Vec<i32> = vec![1; candidate_ids.len()];

        let padded_super_graph_arrays : JraphGraph = build_enriched_padded_supergraph(
            &user_ids,
            &timestamps,
            &candidate_ids,
            &labels,
            &self.user_history,
            self.max_history,
            self.num_catalog_users,
            searcher.get_num_catalog_movies(),
            searcher.get_embed_len(),
            searcher.get_movies_embedding_catalog_ref(),
            &user_embedding,  self.ranker_n_local_devices);

        // Send to TFS Ranker model
        let final_response = self.ranker_model.get_candidate_ranks(
            padded_super_graph_arrays, searcher.get_embed_len()).await;

        match final_response {
            Ok(ranks) => {
                Ok(
                    RankedMovies{
                        movie_ids: candidate_ids,
                        scores: ranks
                    }
                )
            },
            Err(e) => {
                Err(Status::internal(format!("ranking request failed: {}", e)))
            }
        }
    }

}

#[tonic::async_trait]
impl RecommenderService for Orchestrator {
    /// given USerRequest, gets the num_candidates approximate nearest neighbors for user,
    /// scores then and returns them sorted.
    ///
    /// # Arguments
    ///
    /// * `req`:
    ///
    /// returns: Result<Response<RankedMovies>, Status>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    async fn approx_nearest_neighbors(
        &self,
        req: Request<UserRequest>,
    ) -> Result<Response<ApproxNearestNeighborsResponse>, Status> {

        let user_req = req.into_inner();

        // Get user_embedding from TFS Query model
        let user_embedding = self.query_model.get_user_embedding(&user_req).await
            .map_err(|e| Status::internal(format!("user embedding: {}", e)))?;

        let user_ids: Vec<i32> = vec![user_req.user_id as i32];
        let timestamps: Vec<i64> = vec![user_req.timestamp];
        let n_hist = self.user_history.get_history_count_before_timestamp(
            &user_ids, &timestamps
        );
        // choose more than num_candidates unseen movies to rank and take only the top_k from
        let n_srch = Some(self.num_candidates + n_hist[0]);

        // finds num_candidates approx nearest neighbors
        let searcher = self.searcher.load();
        let nearest: Matches = searcher.search(&user_embedding, n_srch)
            .map_err(|e| Status::internal(format!("Vector search failed: {}", e)))?;

        // candidate_ids are in "reference frame" of 0 to num_catalog_movies  - 1, so translate to
        // reference frame num_catalog_users + 1 to num_catalog_users + 1 + num_catalog_movies
        let mut candidate_ids: Vec<i32> = nearest.keys
            .into_iter()
            .map(|x| x as i32 + 1 + self.num_catalog_users as i32)
            .collect();

        // filter to keep only unseen movies

        let (history, _ratings) = self.user_history.get_history_before_timestamp(
            &user_ids, &timestamps, n_hist[0]
        );
        let watched_set: HashSet<i32> = history.iter().copied().collect();

        candidate_ids.retain(|id| !watched_set.contains(id));

        let n_backfill = self.num_candidates.saturating_sub(candidate_ids.len());
        if n_backfill > 0 {
            candidate_ids.extend(
                history.iter()
                    .take(n_backfill)
                    .copied() // or .cloned() depending on the type inside history
            );
        } else {
            candidate_ids.truncate(self.num_candidates);
        }

        Ok(Response::new(ApproxNearestNeighborsResponse {
            user_embedding,
            candidate_ids,
        }))
    }

    async fn predict(&self, req: Request<UserRequest>) -> Result<Response<RankedMovies>, Status> {

        let user_req = req.into_inner();

        let ann_req = Request::new(user_req.clone());
        let ann_res: ApproxNearestNeighborsResponse = self.approx_nearest_neighbors(ann_req).await?.into_inner();

        // Extract the generated fields from the new protobuf response message
        let user_embedding = ann_res.user_embedding;
        let candidate_ids = ann_res.candidate_ids;

        let ranked_movies = self.make_ranker_request(user_req.user_id as i32,
            user_req.timestamp, user_embedding, candidate_ids).await?;

        let (sorted_ids, sorted_scores) = sort_by_scores(
            &ranked_movies.movie_ids, &ranked_movies.scores);

        Ok(Response::new( RankedMovies{
            movie_ids: sorted_ids[0..self.top_k].to_vec(),
            scores: sorted_scores[0..self.top_k].to_vec(),
        }))
    }

    async fn rank_only_return_all(&self, request: Request<RankOnlyRequest>) -> Result<Response<RankedMovies>, Status> {

        /*
        RankOnlyRequest has:
            pub user_id: i32,
            pub timestamp: i64,
            pub candidate_ids: ::prost::alloc::vec::Vec<i32>,
         */
        let rank_req = request.into_inner();

        // populate a UserRequest with age, gender and occupation.  The UserRequest is needed to get a user_embedding
        let user_req_opt = self.user_db.get_request(rank_req.user_id as i64);
        assert!(user_req_opt.is_some(), "User ID {} should exist in database", rank_req.user_id);
        let tonic_req = user_req_opt.unwrap();
        // Extract the inner UserRequest from tonic::Request using .get_ref()
        let user_req = tonic_req.get_ref();

        let user_embedding = self.query_model.get_user_embedding(&user_req).await
            .map_err(|e| Status::internal(format!("user embedding: {}", e)))?;

        let ranked_movies = self.make_ranker_request(rank_req.user_id, rank_req.timestamp,
            user_embedding, rank_req.candidate_ids).await?;

        Ok(Response::new( ranked_movies))
    }

    async fn rank_only(&self, request: Request<RankOnlyRequest>) -> Result<Response<RankedMovies>, Status> {

        /*
        RankOnlyRequest has:
            pub user_id: i32,
            pub timestamp: i64,
            pub candidate_ids: ::prost::alloc::vec::Vec<i32>,
         */
        let rank_req = request.into_inner();

        // populate a UserRequest with age, gender and occupation.  The UserRequest is needed to get a user_embedding
        let user_req_opt = self.user_db.get_request(rank_req.user_id as i64);
        assert!(user_req_opt.is_some(), "User ID {} should exist in database", rank_req.user_id);
        let tonic_req = user_req_opt.unwrap();
        // Extract the inner UserRequest from tonic::Request using .get_ref()
        let user_req = tonic_req.get_ref();

        let user_embedding = self.query_model.get_user_embedding(&user_req).await
            .map_err(|e| Status::internal(format!("user embedding: {}", e)))?;

        let ranked_movies = self.make_ranker_request(rank_req.user_id, rank_req.timestamp,
            user_embedding, rank_req.candidate_ids).await?;

        let (sorted_ids, sorted_scores) = sort_by_scores(
            &ranked_movies.movie_ids, &ranked_movies.scores);

        Ok(Response::new( RankedMovies{
            movie_ids: sorted_ids[0..self.top_k].to_vec(),
            scores: sorted_scores[0..self.top_k].to_vec(),
        }))
    }
}
