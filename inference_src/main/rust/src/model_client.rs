use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arc_swap::ArcSwap;
use tonic::transport::{Channel, Endpoint};

use crate::pb::{MoviesRequest, UsersRequest};
use crate::query_model_metadata::QueryModelMetadata;
use crate::ranker_graph_builder::JraphGraph;
use crate::ranker_model_metadata::RankerModelMetadata;
use crate::reranker_model_metadata::RerankerModelMetadata;
use crate::tensorflow::{
    error::Code,
    serving::{
        model_service_client::ModelServiceClient,
        model_spec::VersionChoice,
        model_version_status::State,
        prediction_service_client::PredictionServiceClient,
        GetModelStatusRequest, ModelSpec, PredictRequest,
    },
    tensor_shape_proto::Dim,
    DataType, TensorProto, TensorShapeProto,
};
use crate::triton::{
    grpc_inference_service_client::GrpcInferenceServiceClient,
    model_infer_request::{InferInputTensor, InferRequestedOutputTensor},
    ModelInferRequest, ModelMetadataRequest, ModelReadyRequest,
};

const CLIENT_CONNECT_TIMEOUT: u64 = 5;
// Handles large batched responses (e.g., RankedMovies for 700_000 user_ids)
const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Debug)]
pub struct QueryModelClient {
    pub client: PredictionServiceClient<Channel>,
    pub model_service_client: ModelServiceClient<Channel>,
    model_metadata_dict: ArcSwap<HashMap<i64, Arc<QueryModelMetadata>>>,
    metadata_write_lock: Mutex<()>,
    model_latest_version: AtomicI64,
    models_path: String,
}

#[derive(Debug)]
pub struct CandidateModelClient {
    // Because candidate is a complement tower model to QueryModelClient,
    // the QueryModelMetadata schema is shared for this client too.
    pub client: PredictionServiceClient<Channel>,
    pub model_service_client: ModelServiceClient<Channel>,
    model_metadata_dict: ArcSwap<HashMap<i64, Arc<QueryModelMetadata>>>,
    metadata_write_lock: Mutex<()>,
    model_latest_version: AtomicI64,
    models_path: String,
}

#[derive(Debug)]
pub struct RankerModelClient {
    pub client: PredictionServiceClient<Channel>,
    pub model_service_client: ModelServiceClient<Channel>,
    model_metadata_dict: ArcSwap<HashMap<i64, Arc<RankerModelMetadata>>>,
    metadata_write_lock: Mutex<()>,
    pub model_latest_version: AtomicI64,
    models_path: String,
    ranker_serving_is_batched: bool,
}

#[derive(Debug)]
pub struct RerankerModelClient {
    pub client: GrpcInferenceServiceClient<Channel>,
    model_metadata_dict: ArcSwap<HashMap<i64, Arc<RerankerModelMetadata>>>,
    metadata_write_lock: Mutex<()>,
    pub model_latest_version: AtomicI64,
    models_path: String,
    model_name: String, // for Triton model latest-version / readiness calls
}

fn extract_f32_from_tensor_proto(
    tensor_proto: TensorProto,
    label: &str,
) -> Result<Vec<f32>, BoxError> {
    if !tensor_proto.float_val.is_empty() {
        Ok(tensor_proto.float_val)
    } else if !tensor_proto.tensor_content.is_empty() {
        let raw_bytes = tensor_proto.tensor_content;
        if raw_bytes.len() % 4 != 0 {
            return Err(format!(
                "Error: Tensor '{}' tensor_content byte length ({}) is not a multiple of 4",
                label,
                raw_bytes.len()
            )
                .into());
        }
        let mut values = Vec::with_capacity(raw_bytes.len() / 4);
        for chunk in raw_bytes.chunks_exact(4) {
            values.push(f32::from_le_bytes(chunk.try_into()?));
        }
        Ok(values)
    } else {
        Err(format!(
            "Error: Tensor '{}' contained neither float_val nor tensor_content",
            label
        ).into())
    }
}

impl QueryModelClient {
    pub async fn new(uri: String, query_models_path: String) -> Self {
        let endpoint = Endpoint::from_shared(uri).expect("Invalid URI format");

        let channel = endpoint
            .connect_timeout(Duration::from_secs(CLIENT_CONNECT_TIMEOUT))
            .connect()
            .await
            .expect("Failed to connect to TFS for query model");

        let model_service_client = ModelServiceClient::new(channel.clone());
        let versions = get_all_model_versions("query", model_service_client.clone())
            .await
            .expect("Failed to fetch query model versions");
        let query_latest_ver = *versions
            .last()
            .expect("No available versions found for query model");

        let query_metadata_uri =
            build_tfs_metadata_uri(&query_models_path, query_latest_ver, "hyperparameters.json");
        let query_metadata = QueryModelMetadata::load_from_file(&query_metadata_uri)
            .expect("Error while loading query metadata");
        let mut initial_query_dict = HashMap::new();
        initial_query_dict.insert(query_latest_ver, Arc::new(query_metadata));

        // Channel is an Arc-backed handle; cloning shares the same TCP connection
        let tfs_client = PredictionServiceClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);

        Self {
            client: tfs_client,
            model_service_client,
            model_metadata_dict: ArcSwap::from_pointee(initial_query_dict),
            metadata_write_lock: Mutex::new(()),
            model_latest_version: AtomicI64::new(query_latest_ver),
            models_path: query_models_path,
        }
    }

    pub fn get_models_path(&self) -> &str {
        self.models_path.as_str()
    }

    pub async fn get_latest_model_version(&self) -> Result<i64, BoxError> {
        let versions = get_all_model_versions("query", self.model_service_client.clone()).await?;
        versions
            .last()
            .copied()
            .ok_or_else(|| "No available versions found in TFS for model 'query'".into())
    }

    pub async fn get_or_fetch_metadata(
        &self,
        query_model_version_choice: Option<VersionChoice>,
    ) -> Result<Arc<QueryModelMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let query_version_num: i64 = match query_model_version_choice {
            Some(VersionChoice::Version(v)) => v,
            _ => {
                let cached = self.model_latest_version.load(Ordering::Acquire);
                if cached > 0 {
                    cached
                } else {
                    self.update_latest_model_version().await?
                }
            }
        };

        // Fast lock-free read
        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&query_version_num) {
            return Ok(meta.clone());
        }

        // Acquire write lock + double-check to avoid duplicate disk loads under concurrency
        let _guard = self
            .metadata_write_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&query_version_num) {
            return Ok(meta.clone());
        }

        let query_metadata_uri =
            build_tfs_metadata_uri(&self.models_path, query_version_num, "hyperparameters.json");
        let new_metadata_arc = Arc::new(QueryModelMetadata::load_from_file(&query_metadata_uri)?);

        self.model_metadata_dict.rcu(|current_map| {
            let mut new_map = (**current_map).clone();
            new_map.insert(query_version_num, new_metadata_arc.clone());
            Arc::new(new_map)
        });

        drop(_guard);
        Ok(new_metadata_arc)
    }

    pub async fn get_users_embeddings(
        &self,
        request: UsersRequest,
        query_model_version_choice: Option<VersionChoice>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let n_valid_users: usize = request.n_users as usize;

        let predict_req =
            build_query_model_inputs(request, query_model_version_choice.clone());
        let query_metadata = self
            .get_or_fetch_metadata(query_model_version_choice)
            .await?;
        let embed_len = query_metadata.embed_len;

        let response = self.client.clone().predict(predict_req).await?;
        let inner_response = response.into_inner();

        let (key, tensor_proto) = inner_response
            .outputs
            .into_iter()
            .next()
            .ok_or("Error: TFS response from query model contained no outputs")?;

        let mut flat_embedding = extract_f32_from_tensor_proto(tensor_proto, &key)?;
        let valid_len = n_valid_users * embed_len;
        if flat_embedding.len() > valid_len {
            flat_embedding.truncate(valid_len);
        }

        Ok(flat_embedding)
    }

    pub async fn update_latest_model_version(&self) -> Result<i64, BoxError> {
        let query_ver = self.get_latest_model_version().await?;
        self.model_latest_version
            .store(query_ver, Ordering::Release);
        Ok(query_ver)
    }
}

impl RankerModelClient {
    pub async fn new(
        uri: String,
        ranker_models_path: String,
        ranker_serving_is_batched: bool,
    ) -> Self {
        let endpoint = Endpoint::from_shared(uri).expect("Invalid URI format");

        let channel = endpoint
            .connect_timeout(Duration::from_secs(CLIENT_CONNECT_TIMEOUT))
            .connect()
            .await
            .expect("Failed to connect to TFS for ranker model");

        let tfs_client = PredictionServiceClient::new(channel.clone())
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);

        let model_service_client = ModelServiceClient::new(channel);
        let versions = get_all_model_versions("graph-ranker", model_service_client.clone())
            .await
            .expect("Failed to fetch graph-ranker model versions");
        let latest_ver = *versions
            .last()
            .expect("No available versions found for graph-ranker model");

        let filename = if ranker_serving_is_batched {
            "metadata_batch.json"
        } else {
            "metadata_single.json"
        };

        let ranker_metadata_uri =
            build_tfs_metadata_uri(&ranker_models_path, latest_ver, filename);
        let ranker_metadata = RankerModelMetadata::load_from_file(&ranker_metadata_uri)
            .expect("Error while loading ranker metadata");
        let mut initial_ranker_dict = HashMap::new();
        initial_ranker_dict.insert(latest_ver, Arc::new(ranker_metadata));

        Self {
            client: tfs_client,
            model_service_client,
            model_metadata_dict: ArcSwap::from_pointee(initial_ranker_dict),
            metadata_write_lock: Mutex::new(()),
            model_latest_version: AtomicI64::new(latest_ver),
            models_path: ranker_models_path,
            ranker_serving_is_batched,
        }
    }

    pub fn get_models_path(&self) -> &str {
        self.models_path.as_str()
    }

    pub async fn get_latest_model_version(&self) -> Result<i64, BoxError> {
        let versions =
            get_all_model_versions("graph-ranker", self.model_service_client.clone()).await?;
        versions
            .last()
            .copied()
            .ok_or_else(|| "No available versions found in TFS for model 'graph-ranker'".into())
    }

    pub async fn update_latest_model_version(&self) -> Result<i64, BoxError> {
        let ver = self.get_latest_model_version().await?;
        self.model_latest_version.store(ver, Ordering::Release);
        Ok(ver)
    }

    pub async fn get_or_fetch_metadata(
        &self,
        ranker_model_version_choice: Option<VersionChoice>,
    ) -> Result<Arc<RankerModelMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let ranker_version_num: i64 = match ranker_model_version_choice {
            Some(VersionChoice::Version(v)) => v,
            _ => {
                let cached = self.model_latest_version.load(Ordering::Acquire);
                if cached > 0 {
                    cached
                } else {
                    self.update_latest_model_version().await?
                }
            }
        };

        // Fast lock-free read
        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&ranker_version_num) {
            return Ok(meta.clone());
        }

        // Acquire write lock + double-check
        let _guard = self
            .metadata_write_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&ranker_version_num) {
            return Ok(meta.clone());
        }

        let filename = if self.ranker_serving_is_batched {
            "metadata_batch.json"
        } else {
            "metadata_single.json"
        };

        let ranker_metadata_uri =
            build_tfs_metadata_uri(&self.models_path, ranker_version_num, filename);
        let new_metadata_arc =
            Arc::new(RankerModelMetadata::load_from_file(&ranker_metadata_uri)?);

        self.model_metadata_dict.rcu(|current_map| {
            let mut new_map = (**current_map).clone();
            new_map.insert(ranker_version_num, new_metadata_arc.clone());
            Arc::new(new_map)
        });

        drop(_guard);
        Ok(new_metadata_arc)
    }

    pub async fn get_candidate_ranks(
        &self,
        padded_super_graph: JraphGraph,
        embed_len: usize,
        ranker_model_version_choice: Option<VersionChoice>,
    ) -> Result<Vec<f32>, BoxError> {
        let metadata = self
            .get_or_fetch_metadata(ranker_model_version_choice.clone())
            .await?;
        let predict_req = build_graph_ranker_proto_inputs(
            padded_super_graph,
            embed_len,
            metadata.batch_size > 1,
            ranker_model_version_choice,
        );

        let response = self.client.clone().predict(predict_req).await?;
        let inner_response = response.into_inner();

        let (key, tensor_proto) = inner_response
            .outputs
            .into_iter()
            .next()
            .ok_or("Error: TFS response from ranker model contained no outputs")?;

        extract_f32_from_tensor_proto(tensor_proto, &key)
    }

    pub async fn get_candidate_diagnostics(
        &self,
        padded_super_graph: JraphGraph,
        embed_len: usize,
        ranker_model_version_choice: Option<VersionChoice>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error + Send + Sync>> {
        let predict_req = build_graph_ranker_diagnostics_proto_inputs(
            padded_super_graph,
            embed_len,
            ranker_model_version_choice,
        );

        let response = self.client.clone().predict(predict_req).await?;
        let mut inner_response = response.into_inner();

        /*
        diag_outputs    = response_diag['outputs']    shape: (max_graphs, num_candidates)
        diag_user_reprs = response_diag['user_reprs'] shape: (max_graphs, out_features)
        diag_cand_reprs = response_diag['cand_reprs'] shape: (max_graphs * num_candidates, out_features)
        */
        let mut take_float_tensor = |key: &str| -> Result<Vec<f32>, BoxError> {
            let tensor_proto = inner_response
                .outputs
                .remove(key)
                .ok_or_else(|| format!("Error: TFS response missing expected key '{}'", key))?;
            extract_f32_from_tensor_proto(tensor_proto, key)
        };

        let outputs = take_float_tensor("outputs")?;
        let user_reprs = take_float_tensor("user_reprs")?;
        let cand_reprs = take_float_tensor("cand_reprs")?;

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

        let model_service_client = ModelServiceClient::new(channel.clone());
        let versions = get_all_model_versions("candidate", model_service_client.clone())
            .await
            .expect("Failed to fetch candidate model versions");
        let latest_ver = *versions
            .last()
            .expect("No available versions found for candidate model");

        let metadata_uri =
            build_tfs_metadata_uri(&models_path, latest_ver, "hyperparameters.json");
        let query_metadata = QueryModelMetadata::load_from_file(&metadata_uri)
            .expect("Error while reading candidate metadata");
        let mut initial_dict = HashMap::new();
        initial_dict.insert(latest_ver, Arc::new(query_metadata));

        let tfs_client = PredictionServiceClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);

        Self {
            client: tfs_client,
            model_service_client,
            model_metadata_dict: ArcSwap::from_pointee(initial_dict),
            metadata_write_lock: Mutex::new(()),
            model_latest_version: AtomicI64::new(latest_ver),
            models_path,
        }
    }

    pub fn get_models_path(&self) -> &str {
        self.models_path.as_str()
    }

    pub async fn get_or_fetch_metadata(
        &self,
        model_version_choice: Option<VersionChoice>,
    ) -> Result<Arc<QueryModelMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let version_num: i64 = match model_version_choice {
            Some(VersionChoice::Version(v)) => v,
            _ => {
                let cached = self.model_latest_version.load(Ordering::Acquire);
                if cached > 0 {
                    cached
                } else {
                    self.update_latest_model_version().await?
                }
            }
        };

        // Fast lock-free read
        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&version_num) {
            return Ok(meta.clone());
        }

        // Acquire write lock + double-check
        let _guard = self
            .metadata_write_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&version_num) {
            return Ok(meta.clone());
        }

        let metadata_uri =
            build_tfs_metadata_uri(&self.models_path, version_num, "hyperparameters.json");
        let new_metadata_arc = Arc::new(QueryModelMetadata::load_from_file(&metadata_uri)?);

        self.model_metadata_dict.rcu(|current_map| {
            let mut new_map = (**current_map).clone();
            new_map.insert(version_num, new_metadata_arc.clone());
            Arc::new(new_map)
        });

        drop(_guard);
        Ok(new_metadata_arc)
    }

    pub async fn update_latest_model_version(&self) -> Result<i64, BoxError> {
        let ver = self.get_latest_model_version().await?;
        self.model_latest_version.store(ver, Ordering::Release);
        Ok(ver)
    }

    pub async fn get_latest_model_version(&self) -> Result<i64, BoxError> {
        let versions =
            get_all_model_versions("candidate", self.model_service_client.clone()).await?;
        versions
            .last()
            .copied()
            .ok_or_else(|| "No available versions found in TFS for model 'candidate'".into())
    }

    pub async fn get_movies_embeddings(
        &self,
        request: MoviesRequest,
        candidate_model_version_choice: Option<VersionChoice>,
    ) -> Result<Vec<f32>, BoxError> {
        let n_valid_movies: usize = request.n_movies as usize;

        let predict_req =
            build_candidate_model_inputs(request, candidate_model_version_choice.clone());
        let metadata = self
            .get_or_fetch_metadata(candidate_model_version_choice)
            .await?;

        let response = self.client.clone().predict(predict_req).await?;
        let inner_response = response.into_inner();

        let (key, tensor_proto) = inner_response
            .outputs
            .into_iter()
            .next()
            .ok_or("Error: TFS response from candidate model contained no outputs")?;

        let mut flat_embedding = extract_f32_from_tensor_proto(tensor_proto, &key)?;
        let valid_len = n_valid_movies * metadata.embed_len;
        if flat_embedding.len() > valid_len {
            flat_embedding.truncate(valid_len);
        }

        Ok(flat_embedding)
    }
}

impl RerankerModelClient {
    pub async fn new(uri: String, reranker_models_path: String) -> Self {
        let endpoint = Endpoint::from_shared(uri).expect("Invalid URI format");

        let channel = endpoint
            .connect_timeout(Duration::from_secs(CLIENT_CONNECT_TIMEOUT))
            .connect()
            .await
            .expect("Failed to connect to Triton for reranker model");

        let triton_client = GrpcInferenceServiceClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);

        let model_name = "qwen-reranker".to_string();
        let req = ModelMetadataRequest {
            name: model_name.clone(),
            version: String::new(), // Empty string queries all active versions
        };

        let resp = triton_client
            .clone()
            .model_metadata(req)
            .await
            .expect("Failed to fetch qwen-reranker metadata")
            .into_inner();

        let mut versions: Vec<i64> = resp
            .versions
            .into_iter()
            .filter_map(|v| v.parse::<i64>().ok())
            .collect();

        if versions.is_empty() {
            panic!(
                "No valid numeric versions found in Triton for model '{}'",
                model_name
            );
        }

        versions.sort_unstable();
        let latest_ver = *versions.last().unwrap();

        let reranker_metadata_uri = resolve_triton_metadata_uri(
            &reranker_models_path,
            latest_ver,
            "training_hyperparameters.json",
        );
        let reranker_metadata = RerankerModelMetadata::load_from_file(&reranker_metadata_uri)
            .expect("Error while reading reranker metadata");
        let mut initial_reranker_dict = HashMap::new();
        initial_reranker_dict.insert(latest_ver, Arc::new(reranker_metadata));

        Self {
            client: triton_client,
            model_metadata_dict: ArcSwap::from_pointee(initial_reranker_dict),
            metadata_write_lock: Mutex::new(()),
            model_latest_version: AtomicI64::new(latest_ver),
            models_path: reranker_models_path,
            model_name,
        }
    }
    
    pub fn get_models_path(&self) -> &str {
        self.models_path.as_str()
    }

    pub async fn get_or_fetch_metadata(
        &self,
        model_version: Option<i64>,
    ) -> Result<Arc<RerankerModelMetadata>, Box<dyn std::error::Error + Send + Sync>> {
        let version_num: i64 = match model_version {
            Some(v) => v,
            None => {
                let cached = self.model_latest_version.load(Ordering::Acquire);
                if cached > 0 {
                    cached
                } else {
                    self.update_latest_model_version().await?
                }
            }
        };

        // Fast lock-free read
        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&version_num) {
            return Ok(meta.clone());
        }

        // Acquire write lock + double-check
        let _guard = self
            .metadata_write_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let map = self.model_metadata_dict.load();
        if let Some(meta) = map.get(&version_num) {
            return Ok(meta.clone());
        }

        let metadata_uri = resolve_triton_metadata_uri(
            &self.models_path,
            version_num,
            "training_hyperparameters.json",
        );
        let new_metadata_arc =
            Arc::new(RerankerModelMetadata::load_from_file(&metadata_uri)?);

        self.model_metadata_dict.rcu(|current_map| {
            let mut new_map = (**current_map).clone();
            new_map.insert(version_num, new_metadata_arc.clone());
            Arc::new(new_map)
        });

        drop(_guard);
        Ok(new_metadata_arc)
    }

    pub async fn update_latest_model_version(&self) -> Result<i64, BoxError> {
        let ver = self.get_latest_model_version().await?;
        self.model_latest_version.store(ver, Ordering::Release);
        Ok(ver)
    }

    /// Checks whether the Triton model (and optional specific version) is loaded and ready.
    pub async fn is_model_ready(&self, model_version: Option<i64>) -> Result<bool, BoxError> {
        let req = ModelReadyRequest {
            name: self.model_name.clone(),
            version: model_version.map(|v| v.to_string()).unwrap_or_default(),
        };
        let resp = self.client.clone().model_ready(req).await?;
        Ok(resp.into_inner().ready)
    }

    /// Queries Triton's ModelMetadata RPC to return the highest available numeric version.
    pub async fn get_latest_model_version(&self) -> Result<i64, BoxError> {
        let req = ModelMetadataRequest {
            name: self.model_name.clone(),
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
                self.model_name
            )
                .into());
        }

        versions.sort_unstable();
        Ok(*versions.last().unwrap())
    }

    /// Sends tokenized `[batch_size, seq_len]` inputs and optional FP32 `ranker_embs` of shape
    /// `[batch_size, 1 + num_candidates, ranker_emb_dim]` to Triton ONNX Runtime, returning
    /// flat FP32 `candidate_log_probs` of shape `[batch_size, num_candidates]`.
    ///
    /// When `metadata.include_ranker_embeddings == false`, a zero-filled FP32 tensor of shape
    /// `[batch_size, 1 + num_candidates, ranker_emb_dim]` is sent automatically to satisfy
    /// the ONNX / Triton `config.pbtxt` signature.
    pub async fn get_reranked_scores(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        ranker_embs: Option<&[f32]>,
        batch_size: usize,
        seq_len: usize,
        model_version: Option<i64>,
    ) -> Result<Vec<f32>, BoxError> {
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        if seq_len == 0 {
            return Err("seq_len must be > 0".into());
        }

        let model_metadata = self.get_or_fetch_metadata(model_version).await?;

        let infer_req = build_triton_reranker_infer_request(
            &model_metadata,
            input_ids,
            attention_mask,
            ranker_embs,
            batch_size,
            seq_len,
            model_version,
        )?;

        let response = self.client.clone().model_infer(infer_req).await?;
        let inner = response.into_inner();
        let expected_out_len = batch_size * model_metadata.num_candidates;

        // 1. Fast path: Triton C++ backend places raw little-endian bytes in `raw_output_contents`
        if let Some(raw_bytes) = inner.raw_output_contents.first() {
            if !raw_bytes.is_empty() {
                if raw_bytes.len() % 4 != 0 {
                    return Err(format!(
                        "Triton raw_output_contents byte length ({}) is not a multiple of 4",
                        raw_bytes.len()
                    )
                        .into());
                }
                let mut scores = Vec::with_capacity(raw_bytes.len() / 4);
                for chunk in raw_bytes.chunks_exact(4) {
                    scores.push(f32::from_le_bytes(
                        chunk
                            .try_into()
                            .map_err(|_| "Failed to parse f32 bytes from Triton")?,
                    ));
                }
                if scores.len() != expected_out_len {
                    return Err(format!(
                        "Triton output length mismatch: expected {} ({}x{}), got {}",
                        expected_out_len,
                        batch_size,
                        model_metadata.num_candidates,
                        scores.len()
                    )
                        .into());
                }
                return Ok(scores);
            }
        }

        // 2. Fallback path: Protobuf `InferTensorContents.fp32_contents`
        if let Some(out_tensor) = inner.outputs.first() {
            if let Some(contents) = &out_tensor.contents {
                if !contents.fp32_contents.is_empty() {
                    if contents.fp32_contents.len() != expected_out_len {
                        return Err(format!(
                            "Triton fp32_contents length mismatch: expected {} ({}x{}), got {}",
                            expected_out_len,
                            batch_size,
                            model_metadata.num_candidates,
                            contents.fp32_contents.len()
                        )
                            .into());
                    }
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
    ranker_embs: Option<&[f32]>,
    batch_size: usize,
    seq_len: usize,
    model_version: Option<i64>,
) -> Result<ModelInferRequest, BoxError> {
    let expected_tok_len = batch_size * seq_len;
    if input_ids.len() != expected_tok_len || attention_mask.len() != expected_tok_len {
        return Err(format!(
            "Token buffer length mismatch: expected {} ({}x{}), got input_ids={} attention_mask={}",
            expected_tok_len,
            batch_size,
            seq_len,
            input_ids.len(),
            attention_mask.len()
        )
            .into());
    }

    // Triton config.pbtxt always expects ranker_embs of shape [batch_size, 1 + num_candidates, ranker_emb_dim]
    let emb_slots = 1 + metadata.num_candidates;
    let emb_dim = metadata.ranker_emb_dim;
    let expected_emb_len = batch_size * emb_slots * emb_dim;

    let ranker_embs_bytes: Vec<u8> = if metadata.include_ranker_embeddings {
        let embs = ranker_embs.ok_or_else(|| {
            format!(
                "Reranker model '{}' requires ranker_embs (include_ranker_embeddings=true), but None was provided",
                metadata.model_name
            )
        })?;
        if embs.len() != expected_emb_len {
            return Err(format!(
                "ranker_embs length mismatch: expected {} ({}x{}x{}), got {}",
                expected_emb_len,
                batch_size,
                emb_slots,
                emb_dim,
                embs.len()
            )
                .into());
        }
        let mut bytes = Vec::with_capacity(embs.len() * 4);
        for &val in embs {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    } else {
        // Matches Python PipelineRunner: send an all-zeros FP32 tensor when embeddings are disabled
        vec![0u8; expected_emb_len * 4]
    };

    let tok_shape = vec![batch_size as i64, seq_len as i64];
    let emb_shape = vec![batch_size as i64, emb_slots as i64, emb_dim as i64];

    let input_ids_tensor = InferInputTensor {
        name: metadata.input_ids_name.clone(),
        datatype: "INT64".to_string(),
        shape: tok_shape.clone(),
        parameters: HashMap::new(),
        contents: None, // Sent via raw_input_contents for zero-varint overhead
    };

    let attention_mask_tensor = InferInputTensor {
        name: metadata.attention_mask_name.clone(),
        datatype: "INT64".to_string(),
        shape: tok_shape,
        parameters: HashMap::new(),
        contents: None,
    };

    let ranker_embs_tensor = InferInputTensor {
        name: metadata.ranker_embs_name.clone(),
        datatype: "FP32".to_string(),
        shape: emb_shape,
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

    Ok(ModelInferRequest {
        model_name: metadata.model_name.clone(),
        model_version: model_version.map(|v| v.to_string()).unwrap_or_default(),
        id: String::new(),
        parameters: HashMap::new(),
        inputs: vec![input_ids_tensor, attention_mask_tensor, ranker_embs_tensor],
        outputs: vec![requested_output],
        raw_input_contents: vec![
            encode_i64_le(input_ids),
            encode_i64_le(attention_mask),
            ranker_embs_bytes,
        ],
    })
}

async fn get_all_model_versions(
    model_name: &str,
    mut client: ModelServiceClient<Channel>,
) -> Result<Vec<i64>, BoxError> {
    let request = tonic::Request::new(GetModelStatusRequest {
        model_spec: Some(ModelSpec {
            name: model_name.to_string(),
            version_choice: None, // None fetches all version statuses
            signature_name: String::new(),
        }),
    });

    let resp = client.get_model_status(request).await?.into_inner();

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

    if versions.is_empty() {
        return Err(format!(
            "No available versions found in TFS for model '{}'",
            model_name
        )
            .into());
    }

    versions.sort_unstable();
    Ok(versions)
}

pub fn build_query_model_inputs(
    req: UsersRequest,
    model_version_choice: Option<VersionChoice>,
) -> PredictRequest {
    let mut inputs: HashMap<String, TensorProto> = HashMap::new();

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
        inputs,
        ..Default::default()
    }
}

pub fn build_candidate_model_inputs(
    req: MoviesRequest,
    model_version_choice: Option<VersionChoice>,
) -> PredictRequest {
    let mut inputs: HashMap<String, TensorProto> = HashMap::new();

    inputs.insert("movie_id".into(), req.movie_ids.into_64bit_tensor2d());
    inputs.insert("genres".into(), req.genres.into_tensor2d());

    PredictRequest {
        model_spec: Some(ModelSpec {
            name: "candidate".to_string(),
            signature_name: "serving_default".to_string(),
            version_choice: model_version_choice,
            ..Default::default()
        }),
        inputs,
        ..Default::default()
    }
}

pub fn build_graph_ranker_proto_inputs(
    padded_super_graph: JraphGraph,
    embed_len: usize,
    client_is_batch: bool,
    ranker_model_version_choice: Option<VersionChoice>,
) -> PredictRequest {
    let inputs = _build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

    let signature_name = if client_is_batch {
        "serving_batch"
    } else {
        "serving_default"
    }
        .into();

    let model_spec = ModelSpec {
        name: "graph-ranker".into(),
        signature_name,
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

pub fn build_graph_ranker_diagnostics_proto_inputs(
    padded_super_graph: JraphGraph,
    embed_len: usize,
    ranker_model_version_choice: Option<VersionChoice>,
) -> PredictRequest {
    let inputs = _build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

    let model_spec = ModelSpec {
        name: "graph-ranker".into(),
        signature_name: "serving_diagnostics".into(),
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

fn _build_graph_ranker_proto_inputs(
    padded_super_graph: JraphGraph,
    embed_len: usize,
) -> HashMap<String, TensorProto> {
    let max_nodes = padded_super_graph.node_ids.len() as i64;
    let mut inputs: HashMap<String, TensorProto> = HashMap::new();

    // 1D Tensors: Graph Structure
    inputs.insert("n_node".into(), padded_super_graph.n_node.into_tensor());
    inputs.insert("n_edge".into(), padded_super_graph.n_edge.into_tensor());

    // 1D Tensors: Connectivity & Edge Features
    inputs.insert("senders".into(), padded_super_graph.senders.into_tensor());
    inputs.insert(
        "receivers".into(),
        padded_super_graph.receivers.into_tensor(),
    );
    inputs.insert(
        "edge_features".into(),
        padded_super_graph.edge_features.into_tensor(),
    );

    // 1D Tensors: Node Features & Masks
    inputs.insert("node_ids".into(), padded_super_graph.node_ids.into_tensor());
    inputs.insert(
        "node_label".into(),
        padded_super_graph.node_labels.into_tensor(),
    );
    inputs.insert(
        "node_type".into(),
        padded_super_graph.node_types.into_tensor(),
    );
    inputs.insert(
        "node_candidate_mask".into(),
        padded_super_graph.candidate_mask.into_tensor(),
    );

    // 2D Tensor: Node Embeddings [total_nodes, embed_len]
    let mut emb_tensor = padded_super_graph.node_embeddings.into_tensor();
    emb_tensor.tensor_shape = Some(TensorShapeProto {
        dim: vec![
            Dim {
                size: max_nodes,
                name: String::new(),
            },
            Dim {
                size: embed_len as i64,
                name: String::new(),
            },
        ],
        unknown_rank: false,
    });
    inputs.insert("node_embeddings".into(), emb_tensor);

    inputs
}

pub fn build_batch_graph_ranker_proto_inputs(
    padded_super_graph: JraphGraph,
    embed_len: usize,
    ranker_model_version_choice: Option<VersionChoice>,
) -> PredictRequest {
    let inputs = _build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

    PredictRequest {
        model_spec: Some(ModelSpec {
            name: "graph-ranker".into(),
            signature_name: "serving_batch".into(),
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

impl IntoTensorProto for Vec<i32> {
    fn into_tensor(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtInt32 as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![Dim {
                    size,
                    name: String::new(),
                }],
                unknown_rank: false,
            }),
            version_number: 0,
            int_val: self,
            ..Default::default()
        }
    }
    fn into_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtInt32 as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim {
                        size,
                        name: String::new(),
                    },
                    Dim {
                        size: 1,
                        name: String::new(),
                    },
                ],
                unknown_rank: false,
            }),
            version_number: 0,
            int_val: self,
            ..Default::default()
        }
    }
    fn into_64bit_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        let int64_vals: Vec<i64> = self.into_iter().map(|x| x as i64).collect();
        TensorProto {
            dtype: DataType::DtInt64 as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim {
                        size,
                        name: String::new(),
                    },
                    Dim {
                        size: 1,
                        name: String::new(),
                    },
                ],
                unknown_rank: false,
            }),
            version_number: 0,
            int64_val: int64_vals,
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
                dim: vec![Dim {
                    size,
                    name: String::new(),
                }],
                unknown_rank: false,
            }),
            version_number: 0,
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
                    Dim {
                        size,
                        name: String::new(),
                    },
                    Dim {
                        size: 1,
                        name: String::new(),
                    },
                ],
                unknown_rank: false,
            }),
            version_number: 0,
            float_val: self,
            ..Default::default()
        }
    }
    fn into_64bit_tensor2d(self) -> TensorProto {
        unimplemented!("use default 32 bit Vec<f32>")
    }
}

impl IntoTensorProto for Vec<bool> {
    fn into_tensor(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtBool as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![Dim {
                    size,
                    name: String::new(),
                }],
                unknown_rank: false,
            }),
            version_number: 0,
            bool_val: self,
            ..Default::default()
        }
    }
    fn into_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtBool as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim {
                        size,
                        name: String::new(),
                    },
                    Dim {
                        size: 1,
                        name: String::new(),
                    },
                ],
                unknown_rank: false,
            }),
            version_number: 0,
            bool_val: self,
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
                dim: vec![Dim {
                    size,
                    name: String::new(),
                }],
                unknown_rank: false,
            }),
            int64_val: self,
            ..Default::default()
        }
    }
    fn into_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        TensorProto {
            dtype: DataType::DtInt64 as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim {
                        size,
                        name: String::new(),
                    },
                    Dim {
                        size: 1,
                        name: String::new(),
                    },
                ],
                unknown_rank: false,
            }),
            int64_val: self,
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
        let bytes_data: Vec<Vec<u8>> = self.into_iter().map(|s| s.into_bytes()).collect();
        TensorProto {
            dtype: DataType::DtString as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![Dim {
                    size,
                    name: String::new(),
                }],
                unknown_rank: false,
            }),
            string_val: bytes_data,
            ..Default::default()
        }
    }
    fn into_tensor2d(self) -> TensorProto {
        let size = self.len() as i64;
        let bytes_data: Vec<Vec<u8>> = self.into_iter().map(|s| s.into_bytes()).collect();
        TensorProto {
            dtype: DataType::DtString as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: vec![
                    Dim {
                        size,
                        name: String::new(),
                    },
                    Dim {
                        size: 1,
                        name: String::new(),
                    },
                ],
                unknown_rank: false,
            }),
            string_val: bytes_data,
            ..Default::default()
        }
    }
    fn into_64bit_tensor2d(self) -> TensorProto {
        unimplemented!("Conversion to 64-bit tensor is invalid for Vec<String>")
    }
}

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

fn build_triton_metadata_uri(
    triton_models_uri: &str,
    model_version_num: i64,
    filename: &str,
) -> String {
    format!(
        "{}/{}/{}",
        triton_models_uri.trim_end_matches('/'),
        model_version_num,
        filename
    )
}

/// Checks `{triton_models_uri}/{version}/{filename}` first; if not present on disk,
/// falls back to `{triton_models_uri}/{filename}` (the layout emitted by `export.py`).
fn resolve_triton_metadata_uri(
    triton_models_uri: &str,
    model_version_num: i64,
    filename: &str,
) -> String {
    let versioned = build_triton_metadata_uri(triton_models_uri, model_version_num, filename);
    if Path::new(&versioned).exists() {
        return versioned;
    }
    let root_level = format!("{}/{}", triton_models_uri.trim_end_matches('/'), filename);
    if Path::new(&root_level).exists() {
        return root_level;
    }
    versioned
}