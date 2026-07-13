use std::collections::HashMap;
use std::error::Error;
use tonic::transport::Channel;
use crate::graph_builder::{JraphGraph};

// use crate:: means look inside this projct

use crate::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::triton::{ModelInferRequest, model_infer_request::InferInputTensor};

use crate::pb::UserRequest;

pub struct QueryModelClient {
    pub client: GrpcInferenceServiceClient<Channel>,
}

pub struct RankerModelClient {
    pub client: GrpcInferenceServiceClient<Channel>,
}

impl QueryModelClient {

    pub async fn new(uri: &'static str) -> Self {
        let channel = Channel::from_static(uri).connect().await.expect("Failed to connect to TFS for query model");
        Self { client: GrpcInferenceServiceClient::new(channel) }
    }

    pub async fn get_user_embedding(&self, request: &UserRequest) -> Result<(Vec<f32>), Box<dyn Error>> {

        let predict_req : ModelInferRequest = build_query_model_inputs(request);

        println!("DEBUG: Sending request for model: {}, version: {}",
            predict_req.model_name,
            predict_req.model_version
        );

        println!("Sending gRPC request to TF Serving for Query Embedding Model...");

        // Send the request
        let response = self.client.clone().model_infer(predict_req).await?;

        let mut inner_response = response.into_inner();

        //debug:
        println!("Triton Response for query model: {:#?}", inner_response);

        if let Some(raw_bytes) = inner_response.raw_output_contents.into_iter().next() {

            // Convert the raw Little Endian bytes back to Vec<f32>
            // A 16-dimensional embedding (f32) will be 64 bytes long.
            let mut embedding = Vec::with_capacity(raw_bytes.len() / 4);

            for chunk in raw_bytes.chunks_exact(4) {
                let val = f32::from_le_bytes(chunk.try_into().unwrap());
                embedding.push(val);
            }

            Ok(embedding)
        } else {
            Err("Error: Triton response from query model contained no raw_output_contents".into())
        }

    }
}

impl RankerModelClient {

    pub async fn new(uri: &'static str) -> Self {
        let channel = Channel::from_static(uri).connect().await.expect("Failed to connect to TFS for ranker model");
        Self { client: GrpcInferenceServiceClient::new(channel) }
    }

    pub async fn get_candidate_ranks(&self, padded_super_graph: JraphGraph, embed_len : usize) -> Result<(Vec<f32>), Box<dyn Error>> {

        let predict_req : ModelInferRequest = build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

            //name: "graph-ranker".into(),
            //signature_name: "serving_default".into(),

        println!("Sending gRPC request to TF Serving for GraphRanker...");

        let response = self.client.clone().model_infer(predict_req).await?;

        let mut inner_response = response.into_inner();

        //debug
        println!("Triton Response for ranker model: {:#?}", inner_response);

        // 32-bit floats for the scores
        if let Some(raw_bytes) = inner_response.raw_output_contents.into_iter().next() {
            let mut ranks = Vec::with_capacity(raw_bytes.len() / 4);

            // 1. The loop populates the vector...
            for chunk in raw_bytes.chunks_exact(4) {
                let val = f32::from_le_bytes(chunk.try_into().unwrap());
                ranks.push(val);
            }

            // 2. We explicitly place the Result here as the final expression
            // to satisfy the compiler's expected return type.
            Ok(ranks)

        } else {
            // This branch also needs to return an Error type to match Result<..., Box<dyn Error>>
            Err("Error: Triton response contained no raw_output_contents".into())
        }

    }
}

pub fn build_query_model_inputs(req: &UserRequest) -> ModelInferRequest {

    let mut inputs = Vec::new();
    let mut raw_input_contents = Vec::new();

    inputs.push(InferInputTensor {
        name: "user_id".into(),
        datatype: "INT64".into(), // Triton string format for datatypes
        shape: vec![1, 1],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(req.user_id.to_le_bytes().to_vec());

    inputs.push(InferInputTensor {
        name: "age".into(),
        datatype: "INT64".into(),
        shape: vec![1, 1],
        parameters: HashMap::new(),
        contents: None,
    });
    raw_input_contents.push(req.age.to_le_bytes().to_vec());

    inputs.push(InferInputTensor {
        name: "occupation".into(),
        datatype: "INT64".into(),
        shape: vec![1, 1],
        parameters: HashMap::new(),
        contents: None,
    });
    raw_input_contents.push(req.occupation.to_le_bytes().to_vec());

    inputs.push(InferInputTensor {
        name: "timestamp".into(),
        datatype: "INT64".into(),
        shape: vec![1, 1],
        parameters: HashMap::new(),
        contents: None,
    });
    raw_input_contents.push(req.timestamp.to_le_bytes().to_vec());

    inputs.push(InferInputTensor {
        name: "gender".into(),
        datatype: "BYTES".into(),
        shape: vec![1, 1],
        parameters: HashMap::new(),
        contents: None,
    });
    let gender_bytes = req.gender.as_bytes();
    let mut packed_string = Vec::new();
    // Prefix with length as a 32-bit integer
    packed_string.extend_from_slice(&(gender_bytes.len() as u32).to_le_bytes());
    packed_string.extend_from_slice(gender_bytes);
    raw_input_contents.push(packed_string);

    ModelInferRequest {
        model_name: "two-tower".into(),
        model_version: "1".into(),
        id: "request_1".into(), // Optional, helpful for tracing
        parameters: HashMap::new(),
        inputs,
        outputs: Vec::new(), // Empty means "return all outputs defined by the model"
        raw_input_contents,
    }

}

pub fn build_graph_ranker_proto_inputs(padded_super_graph: JraphGraph, embed_len : usize) -> ModelInferRequest {
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

    let max_edges = padded_super_graph.senders.len() as i64;
    let max_nodes = padded_super_graph.node_ids.len() as i64;
    let max_graphs = padded_super_graph.n_edge.len() as i64;

    let mut inputs = Vec::new();
    let mut raw_input_contents = Vec::new();

    inputs.push(InferInputTensor {
        name: "edge_features".into(),
        datatype: "INT32".into(), // Triton string format for datatypes
        shape: vec![max_edges],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(i32_vec_to_bytes(&padded_super_graph.edge_features));

    inputs.push(InferInputTensor {
        name: "n_edge".into(),
        datatype: "INT32".into(), // Triton string format for datatypes
        shape: vec![max_graphs],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(i32_vec_to_bytes(&padded_super_graph.n_edge));

    inputs.push(InferInputTensor {
        name: "n_node".into(),
        datatype: "INT32".into(), // Triton string format for datatypes
        shape: vec![max_graphs],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(i32_vec_to_bytes(&padded_super_graph.n_node));

    inputs.push(InferInputTensor {
        name: "node_ids".into(),
        datatype: "INT32".into(), // Triton string format for datatypes
        shape: vec![max_nodes],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(i32_vec_to_bytes(&padded_super_graph.node_ids));

    inputs.push(InferInputTensor {
        name: "node_label".into(),
        datatype: "INT32".into(), // Triton string format for datatypes
        shape: vec![max_nodes],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(i32_vec_to_bytes(&padded_super_graph.node_labels));

    inputs.push(InferInputTensor {
        name: "node_type".into(),
        datatype: "INT32".into(), // Triton string format for datatypes
        shape: vec![max_nodes],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(i32_vec_to_bytes(&padded_super_graph.node_types));

    inputs.push(InferInputTensor {
        name: "receivers".into(),
        datatype: "INT32".into(), // Triton string format for datatypes
        shape: vec![max_edges],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(i32_vec_to_bytes(&padded_super_graph.receivers));

    inputs.push(InferInputTensor {
        name: "senders".into(),
        datatype: "INT32".into(), // Triton string format for datatypes
        shape: vec![max_edges],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(i32_vec_to_bytes(&padded_super_graph.senders));

    inputs.push(InferInputTensor {
        name: "node_candidate_mask".into(),
        datatype: "BOOL".into(), // Triton string format for datatypes
        shape: vec![max_nodes],
        parameters: HashMap::new(),
        contents: None, // We leave this None and use raw_input_contents instead (much faster)
    });
    raw_input_contents.push(bool_vec_to_bytes(&padded_super_graph.candidate_mask));

    //In Triton (and in C/C++ memory generally), all tensors are stored as flat, 1D contiguous arrays of bytes in memory,
    // so no need to reshape the node_embeddings here
    inputs.push(InferInputTensor {
        name: "node_embeddings".into(),
        datatype: "FP32".into(),
        shape: vec![max_nodes, embed_len as i64],
        parameters: HashMap::new(),
        contents: None,
    });
    raw_input_contents.push(f32_vec_to_bytes(&padded_super_graph.node_embeddings));

    ModelInferRequest {
        model_name: "two-tower".into(),
        model_version: "1".into(),
        id: "request_1".into(), // Optional, helpful for tracing
        parameters: HashMap::new(),
        inputs,
        outputs: Vec::new(), // Empty means "return all outputs defined by the model"
        raw_input_contents,
    }
}

fn i32_vec_to_bytes(vec: &[i32]) -> Vec<u8> {
    vec.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

fn f32_vec_to_bytes(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

fn bool_vec_to_bytes(vec: &[bool]) -> Vec<u8> {
    // Triton expects 1 byte per boolean value (0 or 1)
    vec.iter().map(|&v| if v { 1u8 } else { 0u8 }).collect()
}