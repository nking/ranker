use std::collections::HashMap;
use std::error::Error;
use tonic::transport::{Channel, Endpoint};
// use crate:: means look inside this project
use crate::graph_builder::{JraphGraph};
use crate::pb::{UsersRequest};

pub mod tf_serving {
    tonic::include_proto!("tensorflow.serving");
}
pub mod tf_core {
    tonic::include_proto!("tensorflow");
}
use tf_core::{Example, SignatureDef, TensorProto, DataType, TensorShapeProto, tensor_shape_proto::Dim};
use tf_serving::{prediction_service_client::PredictionServiceClient, PredictRequest, ModelSpec};
use crate::ranker_model_metadata::RankerModelMetadata;

#[derive(Debug)]
pub struct QueryModelClient {
    pub client: PredictionServiceClient<Channel>,
    pub embed_len: usize
}

#[derive(Debug)]
pub struct RankerModelClient {
    pub client: PredictionServiceClient<Channel>,
    pub metadata: RankerModelMetadata,
}

impl QueryModelClient {
    pub async fn new(uri: String, embed_len: usize) -> Self {
        let endpoint = Endpoint::from_shared(uri).expect("Invalid URI format");

        let channel = endpoint
            .connect()
            .await
            .expect("Failed to connect to TFS for query model");
        Self {
            client: PredictionServiceClient::new(channel), embed_len: embed_len
        }
    }

    pub async fn get_users_embeddings(&self, request: UsersRequest) -> Result<Vec<f32>, Box<dyn Error>> {

        let n_valid_users : usize = request.n_users as usize;

        let predict_req: PredictRequest = build_query_model_inputs(request);

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
            let valid_len = n_valid_users * self.embed_len;

            // 3. Truncate the flat vector to drop padding graphs at the end
            if flat_embedding.len() > valid_len {
                flat_embedding.truncate(valid_len);
            }

            Ok(flat_embedding)
        } else {
            Err("Error: TFS response from query model contained no outputs".into())
        }
    }

}

impl RankerModelClient {

    pub async fn new(uri: String, metadata: RankerModelMetadata) -> Self {
        let endpoint = Endpoint::from_shared(uri).expect("Invalid URI format");

        let channel = endpoint
            //.max_decoding_message_size(10 * 1024 * 1024) // 10MB example
            .connect()
            .await.expect("Failed to connect to TFS for ranker model");

        // this can handle a RankedMovies response from a UsersRequest of 700_000 user_ids
        const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;
        let mut tfs_client = PredictionServiceClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);

        Self { client: tfs_client, metadata: metadata }
    }

    pub async fn get_candidate_ranks(&self, padded_super_graph: JraphGraph, embed_len : usize) -> Result<Vec<f32>, Box<dyn Error>> {

        let predict_req : PredictRequest = build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

        //println!("Sending gRPC request to TF Serving for GraphRanker...");

        let response = self.client.clone().predict(predict_req).await?;

        let inner_response = response.into_inner();

        //debug
        //println!("Triton Response for ranker model: {:#?}", inner_response);

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

pub fn build_query_model_inputs(req: UsersRequest) -> PredictRequest {

    let mut inputs : HashMap<String, TensorProto> = HashMap::new();

    // the TwoTowerDNN QueryModel saved model doesn't have a fixed batch_size
    let batch_size = req.user_ids.len();

    //currently: signature needs 64-bit inputs

    inputs.insert("user_id".into(), req.user_ids.into_64bit_tensor2d());
    inputs.insert("age".into(), req.ages.into_64bit_tensor2d());
    inputs.insert("gender".into(), req.genders.into_64bit_tensor2d());
    inputs.insert("occupation".into(), req.occupations.into_64bit_tensor2d());
    inputs.insert("timestamp".into(), req.timestamps.into_64bit_tensor2d());

    PredictRequest {
        model_spec: Some(ModelSpec {
            name: "query".to_string(),
            signature_name: "serving_default".to_string(),
            ..Default::default()
        }),
        inputs: inputs,
        ..Default::default()
    }
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

    let inputs : HashMap<String, TensorProto> = _build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

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

pub fn build_batch_graph_ranker_proto_inputs(padded_super_graph: JraphGraph, embed_len: usize
) -> PredictRequest {

    let inputs : HashMap<String, TensorProto> = _build_graph_ranker_proto_inputs(padded_super_graph, embed_len);

    PredictRequest {
        model_spec: Some(ModelSpec {
            name: "graph-ranker".into(),
            signature_name: "serving_batch".into(), // Targets the batch signature
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
