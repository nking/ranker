Query Model deployment:
- the query model is a TF SavedModel format and can be deployed
  to TFS container tensorflow/serving:2.16.1 to match the TF
  that was used to build it.
- to deploy the model locally:
  cd to base of ranker directory
  docker compose --project-directory . -f deploy/compose/docker-compose-TFS.yaml up -d
  can check the deployment: 
    curl -X POST http://172.17.0.1:8501/v1/models/query-model:predict -H "Content-Type: application/json" -d '{
      "instances": [
            {
                "age": [25],
                "gender": ["F"],
                "occupation": [10],
                "timestamp": [1720880000],
                "user_id": [999]
            }
        ]
    }'
- to communicate w/ the TFS server over gRPC, the tf proto files were copied from
  https://github.com/tensorflow/serving/tree/master/tensorflow_serving
  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/framework
- to compile the protocol buffers,
  install protobuf-compiler
  e.g. sudo apt protobuf-compiler, brew install protobuf, etc.
i build.rs is invoked by cargo build, before the src files are loaded

Ranker model deployment:
see docker-compose-TFS.yaml

=======================================
graph-ranker bulk inferrence could benefit from a GPU
so in that case, use NVIDIA's triton container:
   nvcr.io/nvidia/tritonserver:26.04-py3

   it requires a different model repository structure and a config.pb.txt
   file and different docker compose flags,
   and different protocol buffer files that need to be added to
   proto directory and build.rs
    https://github.com/triton-inference-server/common/tree/main/protobuf

