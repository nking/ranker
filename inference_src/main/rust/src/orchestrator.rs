use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use arc_swap::ArcSwap;
use crate::model_client::{QueryModelClient, RankerModelClient};
use crate::embeddings_ann::Searcher;
use crate::graph_builder::{build_enriched_padded_supergraph, JraphGraph};
use crate::user_history::{build_user_history, UserHistory};

use crate::pb::{UserRequest, RankedMovies, BatchRankedMovies, RankOnlyRequest, ApproxNearestNeighborsResponse, BatchUserRequest};
use tonic::{Request, Response, Status};
use usearch::ffi::Matches;
use crate::pb::recommender_service_server::RecommenderService;
use crate::query_model_metadata::QueryModelMetadata;
use crate::ranker_model_metadata::RankerModelMetadata;
use crate::user_db::UserDb;
use crate::util::sort_by_scores;

// the number of local_devices attached to the ranker TFS.  e.g. = 2 for the kaggle T4x2 GPUs
// max_history, num_candidates are hyper-parameters of the ranker_model
#[derive(Debug)]
pub struct Orchestrator {
    query_model: QueryModelClient,
    ranker_model: RankerModelClient,
    query_model_metadata: QueryModelMetadata,
    ranker_model_metadata: RankerModelMetadata,
    searcher: ArcSwap<Searcher>, // updatable
    user_history: UserHistory,  // can be made updatable in future
    user_db: UserDb,
    #[allow(dead_code)]
    ranker_n_local_devices : usize,
    persisted_index_path: PathBuf,
    top_k : usize,
}

impl Orchestrator {
    ///
    ///
    /// # Arguments
    ///
    /// * `query_uri`: endpoint uri of the dpeloyed twotower query model
    /// * `ranker_uri`: endpoint uri of the deployed graphranker model
    /// * `query_metadata_uri`: uri for the query model metadata and hyperparameters json file
    /// * `ranker_metadata_uri`: uri for the ranker model metadata json file
    /// * `movie_embeddings_uri`: uri to the movie_embeddings parquet file
    /// * `ratings_uris`: Vector of ratings file uris usad to construct user histories
    /// * `ranker_n_local_devices`:
    /// * `top_k`:
    /// * `persisted_index_path`:
    /// * `user_db_path`:
    ///
    /// returns: Result<Orchestrator, Box<dyn Error+Send+Sync, Global>>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    // Note: We make this async because connecting to gRPC takes time
    pub async fn new(
        query_uri: String,
        ranker_uri: String,
        query_metadata_uri : String,
        ranker_metadata_uri : String,
        movie_embeddings_uri: &str,
        ratings_uris: Vec<&str>,
        ranker_n_local_devices : usize,
        top_k : usize,
        persisted_index_path: impl AsRef<Path>,
        user_db_path : impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {

        let query_metadata = QueryModelMetadata::load_from_file(&query_metadata_uri).unwrap();
        let ranker_metadata = RankerModelMetadata::load_from_file(&ranker_metadata_uri).unwrap();

        if query_metadata.embed_len != ranker_metadata.embed_len {
            return Err(format!(
                "Query model embed_len {} does not match the Ranker model embed_len {}",
                query_metadata.embed_len, ranker_metadata.embed_len
            ).into()); // .into() converts the String into Box<dyn std::error::Error + Send + Sync>
        }

        // these are in ranker_metadata
        //`max_history`
        /// * `num_candidates`
        /// * `num_catalog_users`:

        let initial_searcher = Searcher::new(movie_embeddings_uri, ranker_metadata.num_candidates, &persisted_index_path)?;
        let query_client = QueryModelClient::new(query_uri).await;
        let ranker_client = RankerModelClient::new(ranker_uri, ranker_metadata.clone()).await;

        let user_history: UserHistory = build_user_history(&ratings_uris, 2048).await;
        let user_db : UserDb = UserDb::new(user_db_path).expect("Failed to initialize UserDb from binary path");

        Ok(Self {
            query_model: query_client,
            ranker_model: ranker_client,
            query_model_metadata : query_metadata,
            ranker_model_metadata: ranker_metadata,
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

        if candidate_ids.len() != self.ranker_model_metadata.embed_len {
            println!(
                "[Warning] Expected candidate_ids length to be {}, but got {}",
                self.ranker_model_metadata.num_candidates,
                candidate_ids.len()
            );
        }

        let user_ids : Vec<i32> = vec![user_id];
        let timestamps: Vec<i64> = vec![timestamp];

        // finds num_candidates approx nearest neighbors
        let searcher = self.searcher.load();

        //target_movie_id should == 1
        let labels: Vec<i32> = vec![1; candidate_ids.len()];

        let padded_super_graph_arrays : JraphGraph = build_enriched_padded_supergraph(
            &user_ids,
            &timestamps,
            &candidate_ids,
            &labels,
            &self.user_history,
            self.ranker_model_metadata.max_history,
            self.ranker_model_metadata.num_catalog_users,
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
                        user_id: user_id as i64,
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
        // choose more than num_candidates to choose only unseen from them, then rank for top num_candidates
        let n_srch = Some(self.ranker_model_metadata.num_candidates + n_hist[0]);

        // finds num_candidates approx nearest neighbors
        let searcher = self.searcher.load();
        let nearest: Matches = searcher.search(&user_embedding, n_srch)
            .map_err(|e| Status::internal(format!("Vector search failed: {}", e)))?;

        // candidate_ids are in "reference frame" of 0 to num_catalog_movies  - 1, so translate to
        // reference frame num_catalog_users + 1 to num_catalog_users + 1 + num_catalog_movies
        let mut candidate_ids: Vec<i32> = nearest.keys
            .into_iter()
            .map(|x| x as i32 + 1 + self.ranker_model_metadata.num_catalog_users as i32)
            .collect();

        // filter to keep only unseen movies
        let (history, _ratings) = self.user_history.get_history_before_timestamp(
            &user_ids, &timestamps, n_hist[0]
        );
        let watched_set: HashSet<i32> = history.iter().copied().collect();

        (&mut candidate_ids).retain(|id: &i32| !watched_set.contains(id));

        let n_backfill = self.ranker_model_metadata.num_candidates.saturating_sub(candidate_ids.len());
        if n_backfill > 0 {
            (&mut candidate_ids).extend(
                history.iter()
                    .take(n_backfill)
                    .copied() // or .cloned() depending on the type inside history
            );
        } else {
            (&mut candidate_ids).truncate(self.ranker_model_metadata.num_candidates);
        }

        Ok(Response::new(ApproxNearestNeighborsResponse {
            user_embedding,
            candidate_ids,
        }))
    }

    /// predicts top_k movies for a user and returs them as a pairs of movie_id and
    /// cosine similarity distance sorted ascending by increasing distances.
    ///
    /// # Arguments
    ///
    /// * `req`: user identity and timestamp with enrichment such as gender, age, and occupation.
    ///
    /// returns: Result<Response<RankedMovies>, Status>
    ///   top_k  pairs of movie_id and cosine similarity distance sorted ascending by increasing distances
    ///
    /// # Examples
    ///
    /// Example call using client:
    /// `let user_req_opt = user_db.get_request(user_id as i64);
    //   let tonic_req = user_req_opt.unwrap();
    //   let mut active_client = client.clone();  //RecommenderServiceClient
    //   let response_response = active_client
    //       .predict(tonic_req)
    //       .await
    //       .map_err(|err| Status::internal(format!("ranking request failed: {}", err)))
    //       .unwrap();
    //   let response = response_response.into_inner();
    //   let retrieved_ids = response.movie_ids;
    ///
    /// Example call inside Orchestrator:
    /// let tonic_req = tonic::Request::new(mock_request);
    //  let results : Result<Response<RankedMovies>, tonic::Status>
    //      = orchestrator.predict(tonic_req).await;
    //  let response = results.unwrap().into_inner();
    async fn predict(&self, req: Request<UserRequest>) -> Result<Response<RankedMovies>, Status> {

        let user_req = req.into_inner();

        let ann_req = Request::new(user_req.clone());
        let ann_res: ApproxNearestNeighborsResponse = self.approx_nearest_neighbors(ann_req).await?.into_inner();

        // Extract the generated fields from the new protobuf response message
        let user_embedding = ann_res.user_embedding;
        let candidate_ids = ann_res.candidate_ids;
        
        println!("user embed_len{}", user_embedding.len());

        let ranked_movies = self.make_ranker_request(user_req.user_id as i32,
            user_req.timestamp, user_embedding, candidate_ids).await?;

        let (sorted_ids, sorted_scores) = sort_by_scores(
            &ranked_movies.movie_ids, &ranked_movies.scores);

        Ok(Response::new( RankedMovies{
            user_id: user_req.user_id,
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
            user_id: user_req.user_id,
            movie_ids: sorted_ids[0..self.top_k].to_vec(),
            scores: sorted_scores[0..self.top_k].to_vec(),
        }))
    }

    async fn batch_predict(&self, _req: Request<BatchUserRequest>) -> Result<Response<BatchRankedMovies>, Status> {
        Err(tonic::Status::unimplemented("BatchPredict is not yet implemented"))
    }
}
