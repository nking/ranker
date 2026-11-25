use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;
use arc_swap::ArcSwap;
use tonic::transport::{Channel, Endpoint};
// use crate:: means look inside this project
use crate::ranker_graph_builder::{JraphGraph};
use crate::pb::{UsersRequest, MoviesRequest};
use crate::query_model_metadata::QueryModelMetadata;
use crate::ranker_model_metadata::RankerModelMetadata;
use crate::reranker_model_metadata::RerankerModelMetadata;
use crate::tensorflow::{
    TensorProto, DataType, TensorShapeProto,
    tensor_shape_proto::Dim,
    serving::{
        prediction_service_client::PredictionServiceClient,
        PredictRequest, GetModelStatusRequest, ModelSpec,
    },
    error::Code,
};
use crate::tensorflow::serving::model_service_client::ModelServiceClient;
use crate::tensorflow::serving::model_spec::VersionChoice;
use crate::tensorflow::serving::model_version_status::State;

use crate::triton::{grpc_inference_service_client::GrpcInferenceServiceClient,
    model_infer_request::{InferInputTensor, InferRequestedOutputTensor},
    InferTensorContents, ModelInferRequest, ModelReadyRequest, ModelMetadataRequest};
const CLIENT_CONNECT_TIMEOUT: u64 = 5;

#[derive(Debug)]
pub struct QueryModelClient {
    pub client: PredictionServiceClient<Channel>,
    pub model_service_client: ModelServiceClient<Channel>,
    model_metadata_dict: ArcSwap<HashMap<i64, Arc<QueryModelMetadata>>>,
    metadata_write_lock: Mutex<()>,
    model_latest_version: AtomicI64,
    models_path : String,
}

#[derive(Debug)]
pub struct CandidateModelClient {
    // because candidate is a complement tower model to QueryModelClient,
    // the QueryModelClient metadata dict is usable for this client too
    pub client: PredictionServiceClient<Channel>,
    pub model_service_client: ModelServiceClient<Channel>,
    model_metadata_dict: ArcSwap<HashMap<i64, Arc<QueryModelMetadata>>>,
    metadata_write_lock: Mutex<()>,
    model_latest_version: AtomicI64,
    models_path : String,
}

#[derive(Debug)]
pub struct RankerModelClient {
    pub client: PredictionServiceClient<Channel>,
    pub model_service_client: ModelServiceClient<Channel>,
    model_metadata_dict: ArcSwap<HashMap<i64, Arc<RankerModelMetadata>>>,
    metadata_write_lock: Mutex<()>,
    pub model_latest_version: AtomicI64,
    models_path : String,
    ranker_serving_is_batched: bool,
}

#[derive(Debug)]
pub struct RerankerModelClient {
    pub client: GrpcInferenceServiceClient<Channel>,
    model_metadata_dict: ArcSwap<HashMap<i64, Arc<RerankerModelMetadata>>>,
    metadata_write_lock: Mutex<()>,
    pub model_latest_version: AtomicI64,
    models_path : String,
}

impl QueryModelClient {
    pub async fn new(uri: String, query_models_path : String) -> Self {

        let endpoint = Endpoint::from_shared(uri).expect("Invalid URI format");

        let channel = endpoint
            .connect_timeout(Duration::from_secs(CLIENT_CONNECT_TIMEOUT))
            .connect()
            .await
            .expect("Failed to connect to TFS for query model");

        let client = ModelServiceClient::new(channel.clone());
        let mut versions = get_all_model_versions("query", client).await
            .expect("Failed to fetch query model versions");
        versions.sort_unstable();
        let query_latest_ver = versions[versions.len() - 1];

        let query_metadata_uri = format!(
            "{}/{}/assets.extra/{}",
            query_models_path.trim_end_matches('/'), query_latest_ver, "hyperparameters.json");
        let query_metadata = QueryModelMetadata::load_from_file(&query_metadata_uri)?;
        let mut initial_query_dict = HashMap::new();
        initial_query_dict.insert(query_latest_ver, Arc::new(query_metadata.clone()));

        Self {
            // Channel is an Arc-backed handle; cloning it shares the exact same TCP connection
            client: PredictionServiceClient::new(channel.clone()),
            model_service_client: client,
            model_metadata_dict : ArcSwap::from_pointee(initial_query_dict),
            metadata_write_lock: Mutex::new(()),
            model_latest_version: AtomicI64::new(query_latest_ver),
            models_path: query_models_path.clone()
        }

    }

    pub async fn get_latest_model_version(
        &self,
    ) -> Result<i64, Box<dyn Error + Send + Sync>> {

        let client = self.model_service_client.clone();

        let mut versions = get_all_model_versions("query", client).await
            .expect("Failed to fetch query model versions");

        versions.sort_unstable();
        Ok(versions[versions.len() - 1])
    }

    pub async fn get_or_fetch_metadata(
        &self,
        query_model_version_choice: Option<VersionChoice>,
    ) -> Result<Arc<QueryModelMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let query_version_num: i64 = match query_model_version_choice {
            Some(VersionChoice::Version(v)) => v,
            _ => self.get_latest_model_version().await?,
        };

        // FAST Lock-free read.
        // 99.99% of requests will hit this and return instantly.
        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&query_version_num) {
            return Ok(meta.clone());
        }

        // ACQUIRE LOCK
        // Only one thread gets past this line at a time.
        // this is a heavy lock and is dropped when the variable goes out of scope when method is done
        // but below we add a drop (unlock) for good code form.
        let _guard = self.metadata_write_lock.lock().unwrap();

        // single or batched TFS serving?
        let filename = "hyperparameters.json";

        let query_metadata_uri = build_tfs_metadata_uri(&self.models_path,
            query_version_num, filename);
        let new_metadata = QueryModelMetadata::load_from_file(&query_metadata_uri)?;

        // FETCH (CPU or I/O bound): Do this outside the map update
        let new_metadata_arc = Arc::new(new_metadata);

        // COPY-ON-WRITE: Safely update the dictionary
        // `rcu` automatically loops if another thread modified the map while we were working,
        // ensuring we don't accidentally overwrite someone else's concurrent insertion.
        self.model_metadata_dict.rcu(|current_map| {
            // "Shallow clone": copies the i64 keys and Arc pointers. Takes nanoseconds.
            let mut new_map = (**current_map).clone();

            // Insert our new item
            new_map.insert(query_version_num, new_metadata_arc.clone());

            Arc::new(new_map)
        });

        drop(_guard);

        Ok(new_metadata_arc)
    }

    pub async fn get_users_embeddings(&self, request: UsersRequest,
        query_model_version_choice: Option<VersionChoice>) -> Result<Vec<f32>, Box<dyn Error>> {

        let n_valid_users : usize = request.n_users as usize;

        let predict_req: PredictRequest = build_query_model_inputs(request, query_model_version_choice);

        let query_metadata = self.get_or_fetch_metadata(query_model_version_choice.clone())
            .await.expect("error fetching query metadata");
        let embed_len = query_metadata.embed_len;

        // Send the request
        let response = self.client.clone().predict(predict_req).await?;
        let inner_response = response.into_inner();

        if let Some((_key, tensor_proto)) = inner_response.outputs.into_iter().next() {

            // Extract the flat f32 vector
            let mut flat_embedding = if !tensor_proto.float_val.is_empty() {
                tensor_proto.float_val
            } else if !tensor_proto.tensor_content.is_empty() {
                let raw_bytes = tensor_proto.tensor_content;
                let mut embedding = Vec::with_capacity(raw_bytes.len() / 4);
                for chunk in raw_bytes.chunks_exact(4) {
                    let val = f32::from_le_bytes(chunk.try_into()?);
                    embedding.push(val);
                }
                embedding
            } else {
                return Err("Error: Tensor contained neither float_val nor tensor_content".into());
            };

            // Reshape into batched vectors
            let valid_len = n_valid_users * embed_len;

            // 3. Truncate the flat vector to drop padding graphs at the end
            if flat_embedding.len() > valid_len {
                flat_embedding.truncate(valid_len);
            }

            Ok(flat_embedding)
        } else {
            Err("Error: TFS response from query model contained no outputs".into())
        }
    }

    pub async fn update_latest_model_version(
        &self,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        // Query both TFS endpoints concurrently
        let query_ver = self.get_latest_model_version().await.expect("Error getting latest model version");

        self.model_latest_version.store(query_ver, Ordering::Release);

        Ok(query_ver)
    }

}

impl RankerModelClient {

    pub async fn new(uri: String, ranker_models_path : String,  ranker_serving_is_batched: bool) -> Self {

        let endpoint = Endpoint::from_shared(uri).expect("Invalid URI format");

        let channel = endpoint
            //.max_decoding_message_size(10 * 1024 * 1024) // 10MB example
            .connect_timeout(Duration::from_secs(CLIENT_CONNECT_TIMEOUT))
            .connect()
            .await.expect("Failed to connect to TFS for ranker model");

        // this can handle a RankedMovies response from a UsersRequest of 700_000 user_ids
        const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;
        let tfs_client = PredictionServiceClient::new(channel.clone())
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);

        let client = ModelServiceClient::new(channel.clone());
        let mut versions = get_all_model_versions("graph-ranker", client).await
            .expect("Failed to fetch graph-ranker model versions");
        versions.sort_unstable();
        let latest_ver = versions[versions.len() - 1];

        let filename = if ranker_serving_is_batched {
            "metadata_batch.json"
        } else {
            "metadata_single.json"
        };

        let ranker_metadata_uri = format!(
            "{}/{}/assets.extra/{}",
            ranker_models_path.trim_end_matches('/'), latest_ver, filename);
        let ranker_metadata = RankerModelMetadata::load_from_file(&ranker_metadata_uri)?;
        let mut initial_ranker_dict = HashMap::new();
        initial_ranker_dict.insert(latest_ver, Arc::new(ranker_metadata.clone()));

        // Channel is an Arc-backed handle; cloning it shares the exact same TCP connection
        Self {
            client: tfs_client,
            model_service_client: ModelServiceClient::new(channel),
            model_metadata_dict : ArcSwap::from_pointee(initial_ranker_dict),
            metadata_write_lock: Mutex::new(()),
            model_latest_version: AtomicI64::new(latest_ver),
            models_path: ranker_models_path.clone(),
            ranker_serving_is_batched: ranker_serving_is_batched
        }
    }

    pub async fn get_latest_model_version(
        &self,
    ) -> Result<i64, Box<dyn Error + Send + Sync>> {

        let client = self.model_service_client.clone();

        let mut versions = get_all_model_versions("graph-ranker", client).await
            .expect("Failed to fetch graph-ranker model versions");

        versions.sort_unstable();
        Ok(versions[versions.len()-1])
    }

    pub async fn update_latest_model_version(
        &self,
    ) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        // Query both TFS endpoints concurrently
        let query_ver = self.get_latest_model_version().await.expect("Error getting latest model version");

        self.model_latest_version.store(query_ver, Ordering::Release);

        Ok(query_ver)
    }

    pub async fn get_or_fetch_metadata(&self, ranker_model_version_choice: Option<VersionChoice>)
        -> Result<Arc<RankerModelMetadata>, Box<dyn std::error::Error + Send + Sync>> {

        let ranker_version_num: i64 = match ranker_model_version_choice {
            Some(VersionChoice::Version(v)) => v,
            _ => self.get_latest_model_version().await?,
        };

        // FAST Lock-free read.
        // 99.99% of requests will hit this and return instantly.
        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&ranker_version_num) {
            return Ok(meta.clone());
        }

        // ACQUIRE LOCK
        // Only one thread gets past this line at a time.
        // this is a heavy lock and is dropped when the variable goes out of scope when method is done
        // but below we add a drop (unlock) for good code form.
        let _guard = self.metadata_write_lock.lock().unwrap();

        // DOUBLE CHECK
        // If we were waiting for the lock, another thread might have just loaded it.
        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&ranker_version_num) {
            return Ok(meta.clone());
        }

        // single or batched TFS serving?
        let filename = if self.ranker_serving_is_batched {
            "metadata_batch.json"
        } else {
            "metadata_single.json"
        };

        let ranker_metadata_uri = build_tfs_metadata_uri(&self.models_path,
            ranker_version_num, filename);
        let new_metadata = RankerModelMetadata::load_from_file(&ranker_metadata_uri)?;

        // FETCH (CPU or I/O bound): Do this outside the map update
        let new_metadata_arc = Arc::new(new_metadata);

        // COPY-ON-WRITE: Safely update the dictionary
        // `rcu` automatically loops if another thread modified the map while we were working,
        // ensuring we don't accidentally overwrite someone else's concurrent insertion.
        self.model_metadata_dict.rcu(|current_map| {
            // "Shallow clone": copies the i64 keys and Arc pointers. Takes nanoseconds.
            let mut new_map = (**current_map).clone();
            // Insert our new item
            new_map.insert(ranker_version_num, new_metadata_arc.clone());
            Arc::new(new_map)
        });

        drop(_guard);

        Ok(new_metadata_arc)
    }


    pub async fn get_candidate_ranks(&self, padded_super_graph: JraphGraph, embed_len : usize,
        ranker_model_version_choice: Option<VersionChoice>) -> Result<Vec<f32>, Box<dyn Error>> {

        let metadata = self.get_or_fetch_metadata(
            ranker_model_version_choice.clone())
            .await.expect("Error fetching ranker metadata");
        let predict_req : PredictRequest = build_graph_ranker_proto_inputs(padded_super_graph,
            embed_len, metadata.batch_size > 1, ranker_model_version_choice);

        //println!("Sending gRPC request to TF Serving for GraphRanker...");

        let response = self.client.clone().predict(predict_req).await?;

        let inner_response = response.into_inner();

        // 32-bit floats for the scores
        if let Some((_key, tensor_proto)) = inner_response.outputs.into_iter().next() {
            if !tensor_proto.float_val.is_empty() {
                Ok(tensor_proto.float_val)
            } else if !tensor_proto.tensor_content.is_empty() {
                let raw_bytes = tensor_proto.tensor_content;
                let mut ranks = Vec::with_capacity(raw_bytes.len() / 4);
                for chunk in raw_bytes.chunks_exact(4) {
                    // a panic here if use .unwrap() instead, is inside tokio async so it kills the task executing this future, not the entire container or process
                    let val = f32::from_le_bytes(
                        chunk.try_into().map_err(|_| "Failed to parse float bytes")?
                    );
                    ranks.push(val);
                }
                Ok(ranks)
            } else {
                Err("Error: Tensor contained neither float_val nor tensor_content".into())
            }
        } else {
            Err("Error: TFS response contained no outputs".into())
        }
    }

    pub async fn get_candidate_diagnostics(
        &self,
        padded_super_graph: JraphGraph,
        embed_len: usize,
        ranker_model_version_choice: Option<VersionChoice>
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn Error>> {

        let predict_req: PredictRequest = build_graph_ranker_diagnostics_proto_inputs(
            padded_super_graph,
            embed_len,
            ranker_model_version_choice
        );

        // println!("Sending gRPC diagnostic request to TF Serving for GraphRanker...");

        let response = self.client.clone().predict(predict_req).await?;
        let mut inner_response = response.into_inner();

        // DEBUG
        // println!("TFS Response for ranker model: diagnostic  {:#?}", inner_response);

        /*
        diag_outputs = response_diag['outputs']   is shape (max_graphs, num_candidates)
        diag_user_reprs = response_diag['user_reprs'] is shape (max_graphs, out_features)
        diag_cand_reprs = response_diag['cand_reprs'] is shape (max_graphs * num_candidates, out_features)
         */


        // Helper closure to extract and parse a tensor by its exact key name
        let extract_float_tensor = |tensor_map: &mut std::collections::HashMap<String, TensorProto>, key: &str| -> Result<Vec<f32>, Box<dyn Error>> {
            let tensor_proto = tensor_map.remove(key)
                .ok_or_else(|| format!("Error: TFS response missing expected key '{}'", key))?;

            if !tensor_proto.float_val.is_empty() {
                Ok(tensor_proto.float_val)
            } else if !tensor_proto.tensor_content.is_empty() {
                let raw_bytes = tensor_proto.tensor_content;
                let mut vec = Vec::with_capacity(raw_bytes.len() / 4);

                for chunk in raw_bytes.chunks_exact(4) {
                    let val = f32::from_le_bytes(
                        chunk.try_into().map_err(|_| format!("Failed to parse float bytes in '{}'", key))?
                    );
                    vec.push(val);
                }
                Ok(vec)
            } else {
                Err(format!("Error: Tensor '{}' contained neither float_val nor tensor_content", key).into())
            }
        };

        // Extract the three distinct tensors requested by the diagnostic signature
        let outputs: Vec<f32> = extract_float_tensor(&mut inner_response.outputs, "outputs")?;
        let user_reprs: Vec<f32> = extract_float_tensor(&mut inner_response.outputs, "user_reprs")?;
        let cand_reprs: Vec<f32> = extract_float_tensor(&mut inner_response.outputs, "cand_reprs")?;

        Ok((outputs, user_reprs, cand_reprs))
    }
}

impl CandidateModelClient {
    pub async fn new(uri: String, models_path: String) -> Self {
        let endpoint = Endpoint::from_shared(uri).expect("Invalid URI format");

        let channel = endpoint
            .connect_timeout(Duration::from_secs(CLIENT_CONNECT_TIMEOUT))
            .connect()
            .await
            .expect("Failed to connect to TFS for candidate model");

        let client = ModelServiceClient::new(channel.clone());
        let mut versions = get_all_model_versions("candidate", client).await
            .expect("Failed to fetch candidate model versions");
        versions.sort_unstable();
        let latest_ver = versions[versions.len() - 1];

        let metadata_uri = format!(
            "{}/{}/assets.extra/{}",
            models_path.trim_end_matches('/'), latest_ver, "hyperparameters.json");
        let query_metadata = QueryModelMetadata::load_from_file(&metadata_uri)?;
        let mut initial_query_dict = HashMap::new();
        initial_query_dict.insert(latest_ver, Arc::new(query_metadata.clone()));

        Self {
            client: PredictionServiceClient::new(channel),
            model_service_client: client,
            model_metadata_dict: ArcSwap::from_pointee(initial_query_dict),
            metadata_write_lock: Mutex::new(()),
            model_latest_version: AtomicI64::new(latest_ver),
            models_path: models_path.clone()
        }
    }


    pub async fn get_or_fetch_metadata(
        &self,
        model_version_choice: Option<VersionChoice>,
    ) -> Result<Arc<QueryModelMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let version_num: i64 = match model_version_choice {
            Some(VersionChoice::Version(v)) => v,
            _ => self.get_latest_model_version().await?,
        };

        // FAST Lock-free read.
        // 99.99% of requests will hit this and return instantly.
        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&version_num) {
            return Ok(meta.clone());
        }

        // ACQUIRE LOCK
        // Only one thread gets past this line at a time.
        // this is a heavy lock and is dropped when the variable goes out of scope when method is done
        // but below we add a drop (unlock) for good code form.
        let _guard = self.metadata_write_lock.lock().unwrap();

        // single or batched TFS serving?
        let filename = "hyperparameters.json";

        let metadata_uri = build_tfs_metadata_uri(&self.models_path,
            version_num, filename);
        // same metadata as Query model so use template:
        let new_metadata = QueryModelMetadata::load_from_file(&metadata_uri)?;

        // FETCH (CPU or I/O bound): Do this outside the map update
        let new_metadata_arc = Arc::new(new_metadata);

        // COPY-ON-WRITE: Safely update the dictionary
        // `rcu` automatically loops if another thread modified the map while we were working,
        // ensuring we don't accidentally overwrite someone else's concurrent insertion.
        self.model_metadata_dict.rcu(|current_map| {
            // "Shallow clone": copies the i64 keys and Arc pointers. Takes nanoseconds.
            let mut new_map = (**current_map).clone();

            // Insert our new item
            new_map.insert(version_num, new_metadata_arc.clone());

            Arc::new(new_map)
        });

        drop(_guard);

        Ok(new_metadata_arc)
    }

    pub async fn get_latest_model_version(
        &self,
    ) -> Result<i64, Box<dyn Error + Send + Sync>> {

        let client = self.model_service_client.clone();

        let mut versions = get_all_model_versions("candidate", client).await
            .expect("Failed to fetch candidate model versions");

        versions.sort_unstable();
        Ok(versions[versions.len() - 1])
    }

    pub async fn get_movies_embeddings(&self, request: MoviesRequest,
        candidate_model_version_choice: Option<VersionChoice>) -> Result<Vec<f32>, Box<dyn Error>> {

        let n_valid_movies : usize = request.n_movies as usize;

        let predict_req: PredictRequest = build_candidate_model_inputs(request, candidate_model_version_choice);

        let metadata = self.get_or_fetch_metadata(candidate_model_version_choice.clone())
            .await.expect("Error getting metadata for candidate model");

        // Send the request
        let response = self.client.clone().predict(predict_req).await?;
        let inner_response = response.into_inner();

        if let Some((_key, tensor_proto)) = inner_response.outputs.into_iter().next() {

            // Extract the flat f32 vector
            let mut flat_embedding = if !tensor_proto.float_val.is_empty() {
                tensor_proto.float_val
            } else if !tensor_proto.tensor_content.is_empty() {
                let raw_bytes = tensor_proto.tensor_content;
                let mut embedding = Vec::with_capacity(raw_bytes.len() / 4);
                for chunk in raw_bytes.chunks_exact(4) {
                    let val = f32::from_le_bytes(chunk.try_into()?);
                    embedding.push(val);
                }
                embedding
            } else {
                return Err("Error: Tensor contained neither float_val nor tensor_content".into());
            };

            // Reshape into batched vectors
            let valid_len = n_valid_movies * metadata.embed_len;

            // 3. Truncate the flat vector to drop padding graphs at the end
            if flat_embedding.len() > valid_len {
                flat_embedding.truncate(valid_len);
            }

            Ok(flat_embedding)
        } else {
            Err("Error: TFS response from candidate model contained no outputs".into())
        }
    }
}

impl RerankerModelClient {
    pub async fn new(uri: String, reranker_models_path : String) -> Self {

        let endpoint = Endpoint::from_shared(uri).expect("Invalid URI format");

        let channel = endpoint
            //.max_decoding_message_size(10 * 1024 * 1024) // 10MB example
            .connect_timeout(Duration::from_secs(CLIENT_CONNECT_TIMEOUT))
            .connect()
            .await.expect("Failed to connect to triton for reranker model");

        const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;
        let triton_client = GrpcInferenceServiceClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);

        let model_name = "qwen-reranker";
        // get latest model version
        let req = ModelMetadataRequest {
            name: model_name.clone().into(),
            version: String::new(), // Empty string queries all active versions
        };

        let resp = triton_client.clone().model_metadata(req)
            .await.expect("Failed to fetch qwen-reranker metadata")
            .into_inner();

        let mut versions: Vec<i64> = resp
            .versions
            .into_iter()
            .filter_map(|v| v.parse::<i64>().ok())
            .collect();

        if versions.is_empty() {
            return Err(format!(
                "No valid numeric versions found in Triton for model '{}'",
                model_name
            ).into());
        }

        versions.sort_unstable();
        let latest_ver = *versions.last().unwrap();


        let reranker_metadata_uri = format!(
            "{}/{}/{}",
            reranker_models_path.trim_end_matches('/'), latest_ver, "training_hyperparameters.json");
        let reranker_metadata = RerankerModelMetadata::load_from_file(&reranker_metadata_uri)?;
        let mut initial_reranker_dict = HashMap::new();
        initial_reranker_dict.insert(latest_ver, Arc::new(reranker_metadata.clone()));

        Self {
            client: triton_client,
            model_metadata_dict
        }
    }

    // Checks whether the Triton model (and optional specific version) is loaded and ready.
    pub async fn is_model_ready(
        &self,
        model_version: Option<i64>,
    ) -> Result<bool, Box<dyn Error + Send + Sync>> {
        editing
        let req = ModelReadyRequest {
            name: self.metadata.model_name.clone(),
            version: model_version.map(|v| v.to_string()).unwrap_or_default(),
        };
        let resp = self.client.clone().model_ready(req).await?;
        Ok(resp.into_inner().ready)
    }

    /// Queries Triton's ModelMetadata RPC to return the highest available numeric version.
    pub async fn get_latest_model_version(
        &self,
    ) -> Result<i64, Box<dyn Error + Send + Sync>> {
        let req = ModelMetadataRequest {
            name: self.metadata.model_name.clone(),
            version: String::new(), // Empty string queries all active versions
        };

        let resp = self.client.clone().model_metadata(req).await?.into_inner();

        let mut versions: Vec<i64> = resp
            .versions
            .into_iter()
            .filter_map(|v| v.parse::<i64>().ok())
            .collect();

        if versions.is_empty() {
            return Err(format!(
                "No valid numeric versions found in Triton for model '{}'",
                self.metadata.model_name
            )
                .into());
        }

        versions.sort_unstable();
        Ok(*versions.last().unwrap())
    }

    /// Sends tokenized `[batch_size, seq_len]` inputs to Triton ONNX Runtime and returns flat FP32 scores/logits.
    pub async fn get_reranked_scores(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        ranker_embs: &[f64],
        batch_size: usize,
        seq_len: usize,
        model_version: Option<i64>,
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        let expected_len = batch_size * seq_len;
        if input_ids.len() != expected_len || attention_mask.len() != expected_len {
            return Err(format!(
                "Token buffer length mismatch: expected {} ({}x{}), got input_ids={} attention_mask={}",
                expected_len,
                batch_size,
                seq_len,
                input_ids.len(),
                attention_mask.len()
            )
                .into());
        }

        let infer_req = build_triton_reranker_infer_request(
            &self.metadata,
            input_ids,
            attention_mask,
            batch_size,
            seq_len,
            model_version,
        );

        let response = self.client.clone().model_infer(infer_req).await?;
        let inner = response.into_inner();

        // 1. Fast path: Triton C++ backend places raw little-endian bytes in `raw_output_contents`
        if let Some(raw_bytes) = inner.raw_output_contents.first() {
            if !raw_bytes.is_empty() {
                let mut scores = Vec::with_capacity(raw_bytes.len() / 4);
                for chunk in raw_bytes.chunks_exact(4) {
                    let val = f32::from_le_bytes(
                        chunk.try_into().map_err(|_| "Failed to parse f32 bytes from Triton")?,
                    );
                    scores.push(val);
                }
                return Ok(scores);
            }
        }

        // 2. Fallback path: Protobuf `InferTensorContents.fp32_contents`
        if let Some(out_tensor) = inner.outputs.first() {
            if let Some(contents) = &out_tensor.contents {
                if !contents.fp32_contents.is_empty() {
                    return Ok(contents.fp32_contents.clone());
                }
            }
        }

        Err("Error: Triton ModelInferResponse contained neither raw_output_contents nor fp32_contents".into())
    }
}
pub fn build_triton_reranker_infer_request(
    metadata: &RerankerModelMetadata,
    input_ids: &[i64],
    attention_mask: &[i64],
    batch_size: usize,
    seq_len: usize,
    model_version: Option<i64>,
) -> ModelInferRequest {
    let shape = vec![batch_size as i64, seq_len as i64];

    let input_ids_tensor = InferInputTensor {
        name: metadata.input_ids_name.clone(),
        datatype: "INT64".to_string(),
        shape: shape.clone(),
        parameters: HashMap::new(),
        contents: None, // Sent via raw_input_contents for zero-varint overhead
    };

    let attention_mask_tensor = InferInputTensor {
        name: metadata.attention_mask_name.clone(),
        datatype: "INT64".to_string(),
        shape,
        parameters: HashMap::new(),
        contents: None,
    };

    let requested_output = InferRequestedOutputTensor {
        name: metadata.output_name.clone(),
        parameters: HashMap::new(),
    };

    let encode_i64_le = |slice: &[i64]| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(slice.len() * 8);
        for &val in slice {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    };

    ModelInferRequest {
        model_name: metadata.model_name.clone(),
        model_version: model_version.map(|v| v.to_string()).unwrap_or_default(),
        id: String::new(),
        parameters: HashMap::new(),
        inputs: vec![input_ids_tensor, attention_mask_tensor],
        outputs: vec![requested_output],
        raw_input_contents: vec![
            encode_i64_le(input_ids),
            encode_i64_le(attention_mask),
        ],
    }
}

async fn get_all_model_versions(
    model_name: &str,
    mut client:  ModelServiceClient<Channel>,
) -> Result<Vec<i64>, Box<dyn Error + Send + Sync>> {

    let request = tonic::Request::new(GetModelStatusRequest {
        model_spec: Some(ModelSpec {
            name: model_name.to_string(),
            version_choice: None, // Leaving this None fetches all version statuses
            signature_name: String::new(),
        }),
    });

    let status_resp =
        client.get_model_status(request).await
        .expect("Couldn't get model status");

    let resp = status_resp.into_inner();

    let mut versions: Vec<i64> = resp
        .model_version_status
        .into_iter()
        .filter(|v| {
            let is_available = v.state == (State::Available as i32);
            let is_ok = v
                .status
                .as_ref()
                .map(|s| s.error_code == (Code::Ok as i32))
                .unwrap_or(true);
            is_available && is_ok
        })
        .map(|v| v.version)
        .collect();

    versions.sort_unstable();
    Ok(versions)
}


///
///
/// # Arguments
///
/// * `req`: a UsersRequest instance that has everything needed to make ANN searches for the users.
/// * `model_version_choice`: the Query model version to use.  defaults to 1 if None is given, else
///     the value.
///     e.g. Some(VersionChoice::Version(2))
///     e.g. None
/// returns: PredictRequest
///
/// # Examples
///
/// ```
///
/// ```
pub fn build_query_model_inputs(req: UsersRequest, model_version_choice: Option<VersionChoice>) -> PredictRequest {

    let mut inputs : HashMap<String, TensorProto> = HashMap::new();

    //currently: signature needs 64-bit inputs

    inputs.insert("user_id".into(), req.user_ids.into_64bit_tensor2d());
    inputs.insert("age".into(), req.ages.into_64bit_tensor2d());
    inputs.insert("gender".into(), req.genders.into_tensor2d());
    inputs.insert("occupation".into(), req.occupations.into_64bit_tensor2d());
    inputs.insert("timestamp".into(), req.timestamps.into_64bit_tensor2d());

    PredictRequest {
        model_spec: Some(ModelSpec {
            name: "query".to_string(),
            signature_name: "serving_default".to_string(),
            version_choice: model_version_choice,
            ..Default::default()
        }),
        inputs: inputs,
        ..Default::default()
    }
}

pub fn build_candidate_model_inputs(req: MoviesRequest, model_version_choice: Option<VersionChoice>) -> PredictRequest {

    let mut inputs : HashMap<String, TensorProto> = HashMap::new();

    //currently: signature needs 64-bit inputs

    inputs.insert("movie_id".into(), req.movie_ids.into_64bit_tensor2d());
    inputs.insert("genres".into(), req.genres.into_tensor2d());

    PredictRequest {
        model_spec: Some(ModelSpec {
            name: "candidate".to_string(),
            signature_name: "serving_default".to_string(),
            version_choice: model_version_choice,
            ..Default::default()
        }),
        inputs: inputs,
        ..Default::default()
    }
}

pub fn build_graph_ranker_proto_inputs(padded_super_graph: JraphGraph, embed_len : usize,
    client_is_batch: bool, ranker_model_version_choice: Option<VersionChoice>) -> PredictRequest {
    /*
    //MAX_GRAPHS:
    padded_super_graph.n_node
    padded_super_graph.n_edge
    //MAX_EDGES:
    padded_super_graph.senders
    padded_super_graph.receivers
    padded_super_graph.edge_features
    //MAX_NODES:
    padded_super_graph.node_ids
    padded_super_graph.node_labels
    padded_super_graph.node_types
    padded_super_graph.candidate_mask
    */

    let inputs : HashMap<String, TensorProto> = _build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

    let signature_name = if client_is_batch {
        "serving_batch"
    } else {
        "serving_default"
    }.into();

    // using the batch_size=1 default signature:
    let model_spec = ModelSpec {
        name: "graph-ranker".into(),
        signature_name: signature_name,
        version_choice: ranker_model_version_choice,
        ..Default::default()
    };

    PredictRequest {
        model_spec: Some(model_spec),
        inputs,
        output_filter: Vec::new(),
        predict_streamed_options: None,
        client_id: None,
        request_options: None,
    }
}


/// build the inputs for a ranker diagnostic request.  It uses the ranker batch format
/// (e.g. batch_size=256).
///
/// # Arguments
///
/// * `padded_super_graph`:
/// * `embed_len`:
/// * `ranker_model_version_choice`:
///
/// returns: PredictRequest ``
pub fn build_graph_ranker_diagnostics_proto_inputs(padded_super_graph: JraphGraph, embed_len : usize,
    ranker_model_version_choice: Option<VersionChoice>) -> PredictRequest {

    /*
    //MAX_GRAPHS:
    padded_super_graph.n_node
    padded_super_graph.n_edge
    //MAX_EDGES:
    padded_super_graph.senders
    padded_super_graph.receivers
    padded_super_graph.edge_features
    //MAX_NODES:
    padded_super_graph.node_ids
    padded_super_graph.node_labels
    padded_super_graph.node_types
    padded_super_graph.candidate_mask
    */

    let inputs : HashMap<String, TensorProto> = _build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

    let signature_name = "serving_diagnostics".into();

    // using the batch_size=1 default signature:
    let model_spec = ModelSpec {
        name: "graph-ranker".into(),
        signature_name: signature_name,
        version_choice: ranker_model_version_choice,
        ..Default::default()
    };

    PredictRequest {
        model_spec: Some(model_spec),
        inputs,
        output_filter: Vec::new(),
        predict_streamed_options: None,
        client_id: None,
        request_options: None,
    }
}

fn _build_graph_ranker_proto_inputs(padded_super_graph: JraphGraph, embed_len: usize
) -> HashMap<String, TensorProto> {

    // Calculate total nodes from the super-graph
    let max_nodes = padded_super_graph.node_ids.len() as i64;

    let mut inputs : HashMap<String, TensorProto> = HashMap::new();

    // 1D Tensors: Graph Structure
    inputs.insert("n_node".into(), padded_super_graph.n_node.into_tensor());
    inputs.insert("n_edge".into(), padded_super_graph.n_edge.into_tensor());

    // 1D Tensors: Connectivity & Edge Features
    inputs.insert("senders".into(), padded_super_graph.senders.into_tensor());
    inputs.insert("receivers".into(), padded_super_graph.receivers.into_tensor());
    inputs.insert("edge_features".into(), padded_super_graph.edge_features.into_tensor());

    // 1D Tensors: Node Features & Masks
    inputs.insert("node_ids".into(), padded_super_graph.node_ids.into_tensor());
    inputs.insert("node_label".into(), padded_super_graph.node_labels.into_tensor());
    inputs.insert("node_type".into(), padded_super_graph.node_types.into_tensor());
    inputs.insert("node_candidate_mask".into(), padded_super_graph.candidate_mask.into_tensor());

    // 2D Tensor: Node Embeddings [total_nodes, embed_len]
    let mut emb_tensor = padded_super_graph.node_embeddings.into_tensor();
    emb_tensor.tensor_shape = Some(TensorShapeProto {
        dim: vec![
            Dim { size: max_nodes, name: String::new() },
            Dim { size: embed_len as i64, name: String::new() },
        ],
        unknown_rank: false,
    });
    inputs.insert("node_embeddings".into(), emb_tensor);

    inputs
}

pub fn build_batch_graph_ranker_proto_inputs(padded_super_graph: JraphGraph,
    embed_len: usize, ranker_model_version_choice: Option<VersionChoice>
) -> PredictRequest {

    let inputs : HashMap<String, TensorProto> = _build_graph_ranker_proto_inputs(
        padded_super_graph, embed_len);

    PredictRequest {
        model_spec: Some(ModelSpec {
            name: "graph-ranker".into(),
            signature_name: "serving_batch".into(), // Targets the batch signature
            version_choice: ranker_model_version_choice,
            ..Default::default()
        }),
        inputs,
        ..Default::default()
    }
}

pub trait IntoTensorProto {
    fn into_tensor(self) -> TensorProto;
    fn into_tensor2d(self) -> TensorProto;
    fn into_64bit_tensor2d(self) -> TensorProto;
}

// Implement for Vec<i32> (For nodes, edges, senders, receivers)
impl IntoTensorProto for Vec<i32> {
    fn into_tensor(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtInt32 as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![Dim { size, name: String::new() }],
                unknown_rank: false,
            }),
            version_number: 0,
            int_val: self, // Pack the data here
            // ... all other fields must be empty vectors
            ..Default::default()
        }
    }
    fn into_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtInt32 as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim { size: size, name: String::new() }, // Batch dimension (N)
                    Dim { size: 1, name: String::new() }, // Feature dimension (1)
                ],
                unknown_rank: false,
            }),
            version_number: 0,
            int_val: self, // Pack the data here
            ..Default::default()
        }
    }
    fn into_64bit_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;

        // Cast i32 elements into i64 values
        let int64_vals: Vec<i64> = self.into_iter().map(|x| x as i64).collect();

        TensorProto {
            dtype: DataType::DtInt64 as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim { size, name: String::new() }, // Batch dimension (N)
                    Dim { size: 1, name: String::new() }, // Feature dimension (1)
                ],
                unknown_rank: false,
            }),
            version_number: 0,
            int64_val: int64_vals, // Note: TensorFlow protobufs store 64-bit ints in int64_val
            ..Default::default()
        }
    }
}

impl IntoTensorProto for Vec<f32> {
    fn into_tensor(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtFloat as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![Dim { size, name: String::new() }],
                unknown_rank: false,
            }),
            version_number: 0,
            // f32 data goes into float_val, NOT int_val
            float_val: self,
            ..Default::default()
        }
    }
    fn into_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtFloat as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim { size: size, name: String::new() }, // Batch dimension (N)
                    Dim { size: 1, name: String::new() }, // Feature dimension (1)
                ],
                unknown_rank: false,
            }),
            version_number: 0,
            // f32 data goes into float_val, NOT int_val
            float_val: self,
            ..Default::default()
        }
    }
    fn into_64bit_tensor2d(self) -> TensorProto {
        unimplemented!("use default 32 bit Vec<f32>")
    }
}

// Implement for Vec<bool> (For candidate masks)
impl IntoTensorProto for Vec<bool> {
    fn into_tensor(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtBool as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![Dim { size, name: String::new() }],
                unknown_rank: false,
            }),
            version_number: 0,
            bool_val: self, // Pack the bools here
            ..Default::default()
        }
    }
    fn into_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtBool as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim { size: size, name: String::new() }, // Batch dimension (N)
                    Dim { size: 1, name: String::new() }, // Feature dimension (1)
                ],
                unknown_rank: false,
            }),
            version_number: 0,
            bool_val: self, // Pack the bools here
            ..Default::default()
        }
    }
    fn into_64bit_tensor2d(self) -> TensorProto {
        unimplemented!("Conversion to 64-bit tensor is invalid for Vec<bool>")
    }
}

impl IntoTensorProto for Vec<i64> {
    fn into_tensor(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtInt64 as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![Dim { size, name: String::new() }],
                unknown_rank: false,
            }),
            int64_val: self, // Pack into int64_val
            ..Default::default()
        }
    }
    fn into_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtInt64 as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim { size: size, name: String::new() }, // Batch dimension (N)
                    Dim { size: 1, name: String::new() }, // Feature dimension (1)
                ],
                unknown_rank: false,
            }),
            int64_val: self, // Pack into int64_val
            ..Default::default()
        }
    }
    fn into_64bit_tensor2d(self) -> TensorProto {
        self.into_tensor2d()
    }
}

impl IntoTensorProto for Vec<String> {
    fn into_tensor(self) -> TensorProto {
        let size = self.len() as i64;
        // Protobuf string_val expects Vec<Vec<u8>>
        let bytes_data: Vec<Vec<u8>> = self.into_iter().map(|s| s.into_bytes()).collect();
        TensorProto {
            dtype: DataType::DtString as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![Dim { size, name: String::new() }],
                unknown_rank: false,
            }),
            string_val: bytes_data, // Pack into string_val
            ..Default::default()
        }
    }
    fn into_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        // Protobuf string_val expects Vec<Vec<u8>>
        let bytes_data: Vec<Vec<u8>> = self.into_iter().map(|s| s.into_bytes()).collect();
        TensorProto {
            dtype: DataType::DtString as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim { size: size, name: String::new() }, // Batch dimension (N)
                    Dim { size: 1, name: String::new() }, // Feature dimension (1)
                ],
                unknown_rank: false,
            }),
            string_val: bytes_data, // Pack into string_val
            ..Default::default()
        }
    }
    fn into_64bit_tensor2d(self) -> TensorProto {
        unimplemented!("Conversion to 64-bit tensor is invalid for Vec<String>")
    }
}

// filename can be a &str, String, &Path, or anything implementing AsRef<str> or Display
fn build_tfs_metadata_uri(
    saved_models_uri: &str,
    model_version_num: i64,
    filename: &str,
) -> String {
    format!(
        "{}/{}/assets.extra/{}",
        saved_models_uri.trim_end_matches('/'),
        model_version_num,
        filename
    )
}

