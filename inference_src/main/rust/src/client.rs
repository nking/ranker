use std::collections::HashMap;
use std::error::Error;
use tonic::transport::Channel;
// use crate:: means look inside this project
use crate::graph_builder::{JraphGraph};
use crate::pb::UserRequest;

pub mod tf_serving {
    tonic::include_proto!("tensorflow.serving");
}
pub mod tf_core {
    tonic::include_proto!("tensorflow");
}
use tf_core::{Example, SignatureDef, TensorProto, DataType, TensorShapeProto, tensor_shape_proto::Dim};
use tf_serving::{prediction_service_client::PredictionServiceClient, PredictRequest, ModelSpec};

pub struct QueryModelClient {
    pub client: PredictionServiceClient<Channel>,
}

pub struct RankerModelClient {
    pub client: PredictionServiceClient<Channel>,
}

impl QueryModelClient {
    pub async fn new(uri: &'static str) -> Self {
        let channel = Channel::from_static(uri)
            .connect()
            .await
            .expect("Failed to connect to TFS for query model");
        Self {
            client: PredictionServiceClient::new(channel),
        }
    }

    pub async fn get_user_embedding(&self, request: &UserRequest) -> Result<Vec<f32>, Box<dyn Error>> {
        let predict_req: PredictRequest = build_query_model_inputs(request);

        println!(
            "DEBUG: Sending request for model: {}",
            predict_req.model_spec.as_ref().unwrap().name
        );
        println!("Sending gRPC request to TF Serving for Query Embedding Model...");

        // Send the request
        let response = self.client.clone().predict(predict_req).await?;

        let inner_response = response.into_inner();

        //debug:
        println!("serving Response for query model: {:#?}", inner_response);

        if let Some((_key, tensor_proto)) = inner_response.outputs.into_iter().next() {

            if !tensor_proto.float_val.is_empty() {
                // Easy path: It returned a pre-parsed float vector
                Ok(tensor_proto.float_val)
            } else if !tensor_proto.tensor_content.is_empty() {
                let raw_bytes = tensor_proto.tensor_content;
                let mut embedding = Vec::with_capacity(raw_bytes.len() / 4);
                for chunk in raw_bytes.chunks_exact(4) {
                    let val = f32::from_le_bytes(chunk.try_into().unwrap());
                    embedding.push(val);
                }
                Ok(embedding)
            } else {
                Err("Error: Tensor contained neither float_val nor tensor_content".into())
            }
        } else {
            Err("Error: TFS response from query model contained no outputs".into())
        }
    }
}

impl RankerModelClient {

    pub async fn new(uri: &'static str) -> Self {
        let channel = Channel::from_static(uri)
            //.max_decoding_message_size(10 * 1024 * 1024) // 10MB example
            .connect()
            .await.expect("Failed to connect to TFS for ranker model");
        Self { client: PredictionServiceClient::new(channel) }
    }

    pub async fn get_candidate_ranks(&self, padded_super_graph: JraphGraph, embed_len : usize) -> Result<Vec<f32>, Box<dyn Error>> {

        let predict_req : PredictRequest = build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

        println!("Sending gRPC request to TF Serving for GraphRanker...");

        let response = self.client.clone().predict(predict_req).await?;

        let inner_response = response.into_inner();

        //debug
        println!("Triton Response for ranker model: {:#?}", inner_response);

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
}

pub fn build_query_model_inputs(req: &UserRequest) -> PredictRequest {

    let mut inputs = HashMap::new();

    inputs.insert("user_id".into(), vec![req.user_id].into_tensor2d());

    inputs.insert("age".into(), vec![req.age].into_tensor2d());

    inputs.insert("gender".into(), vec![req.gender.clone()].into_tensor2d());

    inputs.insert("occupation".into(), vec![req.occupation].into_tensor2d());

    inputs.insert("timestamp".into(), vec![req.timestamp].into_tensor2d());

    let mut predict_req = PredictRequest::default();
    predict_req.model_spec = Some(ModelSpec {
        name: "query".to_string(),
        signature_name: "serving_default".to_string(),
        ..Default::default()
    });
    predict_req.inputs = inputs;

    predict_req

}

pub fn build_graph_ranker_proto_inputs(padded_super_graph: JraphGraph, embed_len : usize) -> PredictRequest {
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

    //let max_edges = padded_super_graph.senders.len() as i64;
    let max_nodes = padded_super_graph.node_ids.len() as i64;
    //let max_graphs = padded_super_graph.n_edge.len() as i64;

    let mut inputs = HashMap::new();

    // Graph Structure (Usually single-element Vecs or small arrays)
    inputs.insert("n_node".into(), padded_super_graph.n_node.into_tensor());
    inputs.insert("n_edge".into(), padded_super_graph.n_edge.into_tensor());

    // Connectivity & Edge Features
    inputs.insert("senders".into(), padded_super_graph.senders.into_tensor());
    inputs.insert("receivers".into(), padded_super_graph.receivers.into_tensor());
    inputs.insert("edge_features".into(), padded_super_graph.edge_features.into_tensor());

    // Node Features
    inputs.insert("node_ids".into(), padded_super_graph.node_ids.into_tensor());
    inputs.insert("node_label".into(), padded_super_graph.node_labels.into_tensor());
    inputs.insert("node_type".into(), padded_super_graph.node_types.into_tensor());
    // Masks
    inputs.insert("node_candidate_mask".into(), padded_super_graph.candidate_mask.into_tensor());

    // Node Embeddings (Requires 2D Shape Override)
    let mut emb_tensor = padded_super_graph.node_embeddings.into_tensor();
    emb_tensor.tensor_shape = Some(TensorShapeProto {
        dim: vec![
            Dim { size: max_nodes, name: String::new() },
            Dim { size: embed_len as i64, name: String::new() },
        ],
        unknown_rank: false,
    });
    inputs.insert("node_embeddings".into(), emb_tensor);

    // using the batch_size=1 default signature:
    let model_spec = ModelSpec {
        name: "graph-ranker".into(),
        signature_name: "serving_default".into(),
        version_choice: None,
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

pub trait IntoTensorProto {
    fn into_tensor(self) -> TensorProto;
    fn into_tensor2d(self) -> TensorProto;
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
            tensor_content: vec![], half_val: vec![], float_val: vec![],
            double_val: vec![], string_val: vec![], scomplex_val: vec![],
            int64_val: vec![], bool_val: vec![], dcomplex_val: vec![],
            resource_handle_val: vec![], variant_val: vec![],
            uint32_val: vec![], uint64_val: vec![],
            float8_val: vec![],
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
            // ... all other fields must be empty vectors
            tensor_content: vec![], half_val: vec![], float_val: vec![],
            double_val: vec![], string_val: vec![], scomplex_val: vec![],
            int64_val: vec![], bool_val: vec![], dcomplex_val: vec![],
            resource_handle_val: vec![], variant_val: vec![],
            uint32_val: vec![], uint64_val: vec![],
            float8_val: vec![],
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
            // ... empty out the rest just like above
            tensor_content: vec![], half_val: vec![], float_val: vec![],
            double_val: vec![], int_val: vec![], string_val: vec![], scomplex_val: vec![],
            int64_val: vec![], dcomplex_val: vec![], resource_handle_val: vec![],
            variant_val: vec![], uint32_val: vec![], uint64_val: vec![],
            float8_val: vec![],
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
            // ... empty out the rest just like above
            tensor_content: vec![], half_val: vec![], float_val: vec![],
            double_val: vec![], int_val: vec![], string_val: vec![], scomplex_val: vec![],
            int64_val: vec![], dcomplex_val: vec![], resource_handle_val: vec![],
            variant_val: vec![], uint32_val: vec![], uint64_val: vec![],
            float8_val: vec![],
        }
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
            // Keep all other fields empty
            int_val: vec![], float_val: vec![], bool_val: vec![],
            string_val: vec![], tensor_content: vec![],
            // ... (ensure all other fields like half_val, double_val, etc., are empty)
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
            // Keep all other fields empty
            int_val: vec![], float_val: vec![], bool_val: vec![],
            string_val: vec![], tensor_content: vec![],
            // ... (ensure all other fields like half_val, double_val, etc., are empty)
            ..Default::default()
        }
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
            // Keep all other fields empty
            int_val: vec![], int64_val: vec![], float_val: vec![], bool_val: vec![],
            tensor_content: vec![],
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
            // Keep all other fields empty
            int_val: vec![], int64_val: vec![], float_val: vec![], bool_val: vec![],
            tensor_content: vec![],
            ..Default::default()
        }
    }
}
