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

exports to NVIDIA's triton server.

exporting to onxx

- the two-tower query model needs to be exported to onxx
  and that is an embedded model 

Then settled on NVIDIA's triton server.

triton versions 25.02 and earlier offer a tensorflow_savedmode backend
with container image pattern xx.yy-tf2-python-py3.
- for the two-tower model which was built with tf 2.16.1,
  the iamge is nvcr.io/nvidia/tritonserver:24.12-py3
- for the ranker model, no image contains tf and jax.
  so the ranker model needs to be exported to onxx format

to download the triton image for the two-tower model
to try to load the saved_model, do a docker pull ahead of
time because the images are so large:
e.g.:
export img=nvcr.io/nvidia/tritonserver:24.12-py3
docker pull $img
docker save $img > tritonserver_image.tar
docker load < tritonserver_image.tar

then from the ranker directory:
  docker compose --project-directory . -f deploy/compose/docker-compose-triton-query.yaml up -d

copied the protos into this project from
    https://github.com/triton-inference-server/common/tree/main/protobuf

