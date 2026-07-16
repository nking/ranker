//! for building protocol buffers.  cargo build invokes it before any src code
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_server(true) // Force server generation!
        .build_client(true)
        .compile_protos(
            &[
                "proto/tensorflow_serving/apis/prediction_service.proto",
                "proto/tensorflow/core/example/example.proto",
                "proto/tensorflow/core/protobuf/meta_graph.proto",
                "proto/recommender/recommender.proto",
            ],
            &["proto/"], // The root include directory
        )?;
    Ok(())
}