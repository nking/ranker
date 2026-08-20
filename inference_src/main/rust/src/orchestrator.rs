use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use arc_swap::ArcSwap;
use crate::model_client::{QueryModelClient, RankerModelClient};
use crate::embeddings_ann::Searcher;
use crate::graph_builder::{build_enriched_padded_supergraph, JraphGraph};
use crate::user_history::{build_user_history, UserHistory};

use crate::pb::{UsersRequest, RankedMovies, RankOnlyRequest, ApproxNearestNeighborsResponse};
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
    #[allow(dead_code)]
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
        
        let initial_searcher = Searcher::new(movie_embeddings_uri, ranker_metadata.num_candidates, &persisted_index_path)?;
        let query_client = QueryModelClient::new(query_uri, query_metadata.embed_len).await;
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

    async fn make_ranker_request(&self, user_ids : Vec<i32>, timestamps: Vec<i64>,
        user_embeddings : Vec<f32>, candidate_ids : Vec<i32>) ->Result<RankedMovies, Status> {

        let n_users = user_ids.len();

        if n_users > self.ranker_model_metadata.batch_size {
            return Err(Status::invalid_argument(format!(
                "n_users {} must be <= batch_size {}",
                n_users, self.ranker_model_metadata.batch_size
            )));
        }

        // finds num_candidates approx nearest neighbors
        let searcher = self.searcher.load();

        // target_movie_id should == 1
        let labels: Vec<i32> = vec![1; candidate_ids.len()];

        let batch_size : usize = self.ranker_model_metadata.batch_size;

        let padded_super_graph_arrays: JraphGraph = build_enriched_padded_supergraph(
            batch_size,
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
            &user_embeddings,
            self.ranker_n_local_devices
        );

        // Send to TFS Ranker model
        let final_response = self.ranker_model.get_candidate_ranks(
            padded_super_graph_arrays,searcher.get_embed_len()).await;

        match final_response {
            Ok(mut ranks) => {
                // The JAX model returns statically shaped output (max_graphs).
                // Truncate the padded scores to match the actual number of valid inputs in this chunk.
                ranks.truncate(candidate_ids.len());

                Ok(RankedMovies {
                    user_ids,
                    movie_ids: candidate_ids,
                    scores: ranks,
                    num_candidates: self.ranker_model_metadata.num_candidates as u32
                })
            },
            Err(e) => {
                Err(Status::internal(format!("ranking request failed: {}", e)))
            }
        }
    }

    async fn _predict(&self, req: Request<UsersRequest>) -> Result<Response<RankedMovies>, Status> {

        let user_reqs = req.into_inner();

        let ann_reqs = Request::new(user_reqs.clone());
        let ann_res: ApproxNearestNeighborsResponse = self._approx_nearest_neighbors(ann_reqs).await?.into_inner();

        // Extract the generated fields from the new protobuf response message
        let user_ids = ann_res.user_ids;
        // length: n_users * embed_len
        let user_embeddings = ann_res.user_embeddings;
        // length: n_users * num_candidates
        let candidate_ids = ann_res.candidate_ids;

        let ranked_movies = self.make_ranker_request(user_ids.clone(),
            user_reqs.timestamps, user_embeddings, candidate_ids).await?;

        let (sorted_ids, sorted_scores) = sort_by_scores(
            &ranked_movies.movie_ids, &ranked_movies.scores);

        Ok(Response::new( RankedMovies{
            user_ids: user_ids,
            movie_ids: sorted_ids[0..self.top_k].to_vec(),
            scores: sorted_scores[0..self.top_k].to_vec(),
            num_candidates: ranked_movies.num_candidates,
        }))
    }

    async fn _approx_nearest_neighbors(&self, req: Request<UsersRequest>) -> Result<Response<ApproxNearestNeighborsResponse>, Status> {

        let users_req = req.into_inner();

        // Get user_embeddings from TFS Query model.  flattenend into a single array
        // length is n_users * embed_len
        let user_embeddings : Vec<f32> = self.query_model.get_users_embeddings(users_req.clone()).await
            .map_err(|e| Status::internal(format!("user embedding: {}", e)))?;

        // length is n_users
        let n_hists : Vec<usize> = self.user_history.get_history_count_before_timestamp(
            &users_req.user_ids, &users_req.timestamps
        );

        //TODO: consider limiting this to keep the search quick:
        let max_n_hist : usize = *(n_hists.iter().max().unwrap());

        // choose more than num_candidates to choose only unseen from them, then rank for top num_candidates
        let _n_srch : usize = self.ranker_model_metadata.num_candidates + max_n_hist;
        let n_srch = Some(_n_srch);

        // finds num_candidates approx nearest neighbors
        let searcher = self.searcher.load();
        // returns Match{ keys:Vec<n_users*n_srch:u64>, distances: Vec<n_users*n_srch:f32>)
        let nearest: Matches = searcher.search(&user_embeddings, n_srch)
            .map_err(|e| Status::internal(format!("Vector search failed: {}", e)))?;

        // candidate_ids length is n_users * n_srch
        // candidate_ids are in "reference frame" of 0 to num_catalog_movies  - 1, so translate to
        // reference frame num_catalog_users + 1 to num_catalog_users + 1 + num_catalog_movies
        let raw_candidate_ids: Vec<i32> = nearest.keys
            .into_iter()
            .map(|x| x as i32 + 1 + self.ranker_model_metadata.num_catalog_users as i32)
            .collect();

        // history is length n_users * max_n_hist
        let (history, _ratings) : (Vec<i32>, Vec<i32>) = self.user_history.get_history_before_timestamp(
            &users_req.user_ids, &users_req.timestamps, max_n_hist);

        // filter candidate_ids to keep only unseen movies
        let target_len = self.ranker_model_metadata.num_candidates;

        // Pre-allocate the exact size needed for the final flattened candidates
        let mut final_candidates: Vec<i32> = Vec::with_capacity(users_req.user_ids.len() * target_len);

        // Filter and backfill
        for i in 0..users_req.user_ids.len() {
            let hist_start = i * max_n_hist;
            let hist_end = hist_start + max_n_hist;

            let watched_set: HashSet<i32> = history[hist_start..hist_end].iter().copied().collect();

            let cand_start = i * _n_srch;
            let cand_end = cand_start + _n_srch;
            let user_raw_candidates = &raw_candidate_ids[cand_start..cand_end];

            // Filter out watched movies
            let mut unwatched: Vec<i32> = user_raw_candidates.iter()
                .copied()
                .filter(|id| !watched_set.contains(id))
                .collect();

            // Backfill if we filtered out too many
            if unwatched.len() < target_len {
                let n_backfill = target_len - unwatched.len();

                // NOTE: Backfilling with history means injecting watched movies.
                // If you have a padding ID (like 0) or popular fallback catalog IDs, use those instead.
                unwatched.extend(
                    history[hist_start..hist_end].iter().take(n_backfill).copied()
                );
            }

            // Ensure we have exactly num_candidates items for this user
            unwatched.truncate(target_len);

            // Append this user's fixed-length candidates into the final flat vector
            final_candidates.extend_from_slice(&unwatched);
        }

        Ok(Response::new(ApproxNearestNeighborsResponse {
            user_ids: users_req.user_ids, // Use the extracted field directly
            user_embeddings,
            candidate_ids: final_candidates,
        }))
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
    async fn approx_nearest_neighbors(&self, req: Request<UsersRequest>) -> Result<Response<ApproxNearestNeighborsResponse>, Status> {
        return self._approx_nearest_neighbors(req).await;
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
    async fn predict(&self, req: Request<UsersRequest>) -> Result<Response<RankedMovies>, Status> {

        let batch_size : usize = self.ranker_model_metadata.batch_size;

        let user_reqs = req.into_inner();

        let n_users = user_reqs.user_ids.len();

        if n_users == batch_size {
            return self._predict(Request::new(user_reqs.clone())).await;
        }

        // reserve response arrays:
        let mut final_candidate_ids: Vec<i32> = Vec::with_capacity(n_users * self.ranker_model_metadata.num_candidates);
        let mut final_scores: Vec<f32> = Vec::with_capacity(n_users * self.ranker_model_metadata.num_candidates);

        for i0 in (0..n_users).step_by(batch_size) {

            let i1 = std::cmp::min(i0 + batch_size, n_users);

            // make a new UsersRequest from the user data from i0 to i1
            let req_i = UsersRequest {
                user_ids : user_reqs.user_ids[i0..i1].to_vec(),
                genders : user_reqs.genders[i0..i1].to_vec(),
                occupations : user_reqs.occupations[i0..i1].to_vec(),
                ages : user_reqs.ages[i0..i1].to_vec(),
                timestamps : user_reqs.timestamps[i0..i1].to_vec(),
                n_users : (i1 - i0) as u32
            };

            let resp_i = self._predict(Request::new(req_i)).await?;
            let ranked_movies_i = resp_i.into_inner();

            final_candidate_ids.extend(ranked_movies_i.movie_ids);
            final_scores.extend(ranked_movies_i.scores);
        }

        Ok(Response::new( RankedMovies{
            user_ids: user_reqs.user_ids,
            movie_ids: final_candidate_ids,
            scores: final_scores,
            num_candidates: self.ranker_model_metadata.num_candidates as u32
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

        let user_ids = vec![rank_req.user_id];
        let timestamps = vec![rank_req.timestamp];

        // populate a UserRequest with age, gender and occupation.  The UserRequest is needed to get a user_embedding
        let user_req_opt : Option<Request<UsersRequest>> = self.user_db.get_request(
            &user_ids, &timestamps);
        assert!(user_req_opt.is_some(), "User ID {} should exist in database", rank_req.user_id);

        let request = user_req_opt.ok_or_else(|| {
            Status::invalid_argument("None of the requested user IDs were found or valid")
        })?;

        // 3. Extract the underlying UsersRequest message from the gRPC wrapper
        let users_request = request.into_inner();

        let user_embeddings = self.query_model.get_users_embeddings(users_request).await
            .map_err(|e| Status::internal(format!("user embedding: {}", e)))?;

        let ranked_movies =
            self.make_ranker_request(user_ids, timestamps,
            user_embeddings, rank_req.candidate_ids).await?;

        Ok(Response::new( ranked_movies))
    }

    async fn rank_only(&self, request: Request<RankOnlyRequest>) -> Result<Response<RankedMovies>, Status> {

        // get response.
        // unroll the results by user, sort the movie_ids and scores by scores dsending,
        // and pack up the top_k movie_ids and scores for each user
        /*
        message RankedMovies {
              repeated int32 user_ids = 1;
              repeated int32 movie_ids = 2; // "repeated" means it's a Vec in Rust
              repeated float scores = 3;
              uint32 num_candidates = 4;
            }
         */
        let ranked_all = self.rank_only_return_all(request).await?.into_inner();

        let num_candidates = self.ranker_model_metadata.num_candidates as usize;
        let n_users = ranked_all.user_ids.len();

        // Early return if empty to prevent divide-by-zero later
        if n_users == 0 {
            return Ok(Response::new(RankedMovies {
                user_ids: vec![],
                movie_ids: vec![],
                scores: vec![],
                num_candidates: num_candidates as u32,
            }));
        }

        // Determine how many movies were returned per user in the flattened payload
        let movies_per_user = ranked_all.movie_ids.len() / n_users;

        debug_assert!(movies_per_user == num_candidates);

        // Safety check to ensure perfectly rectangular tensors
        if ranked_all.movie_ids.len() % n_users != 0 {
            return Err(Status::internal(
                "Mismatched columnar data: movie_ids length is not a clean multiple of user_ids"
            ));
        }

        // Pre-allocate the exact capacities for the final response
        let mut final_movie_ids: Vec<i32> = Vec::with_capacity(n_users * num_candidates);
        let mut final_scores: Vec<f32> = Vec::with_capacity(n_users * num_candidates);

        for i in 0..n_users {
            let start = i * movies_per_user;
            let end = start + movies_per_user;

            let user_movies = &ranked_all.movie_ids[start..end];
            let user_scores = &ranked_all.scores[start..end];

            // Zip scores and IDs together so they sort as a pair
            let mut pairs: Vec<(f32, i32)> = user_scores.iter().copied()
                .zip(user_movies.iter().copied())
                .collect();

            //TODO: revisit this for caring about order for ties during sort:

            // Sort descending by score.
            // f32 cannot use `.sort()` directly due to NaN ambiguity, so we use `partial_cmp`.
            // `sort_unstable_by` is used over `sort_by` because it is significantly faster and
            // we don't care about preserving the original order of duplicate scores.
            pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Unpack the top `num_candidates` back into the flattened arrays
            for (score, movie_id) in pairs.into_iter().take(num_candidates) {
                final_scores.push(score);
                final_movie_ids.push(movie_id);
            }
        }

        // Return the assembled response
        Ok(Response::new(RankedMovies {
            user_ids: ranked_all.user_ids, // Reuse the original user_ids vector directly
            movie_ids: final_movie_ids,
            scores: final_scores,
            num_candidates: num_candidates as u32,
        }))
    }

}
