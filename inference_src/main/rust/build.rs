//! for building protocol buffers.  cargo build invokes it before any src code
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_server(true) // Force server generation!
        .build_client(true)
        .compile_protos(
            &[
                "proto/tensorflow/core/protobuf/error_codes.proto",
                "proto/tensorflow/core/example/example.proto",
                "proto/tensorflow/core/example/feature.proto",
                "proto/tensorflow_serving/apis/prediction_service.proto",
                "proto/tensorflow/core/example/example.proto",
                "proto/tensorflow/core/protobuf/meta_graph.proto",
                "proto/tensorflow_serving/apis/model.proto",
                "proto/tensorflow_serving/config/model_server_config.proto",
                "proto/tensorflow_serving/apis/get_model_status.proto",
                "proto/tensorflow_serving/apis/model_service.proto",
                "proto/recommender/recommender.proto",
            ],
            &["proto/"], // The root include directory
        )?;
    Ok(())
}