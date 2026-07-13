
helpful information while trying different exports and deployment servings:

---------------------------------------------------------------------------
for the attempt to use saved_model and TFS serving:

git checkout f501da4e0ba0a4c9b5f659f8cf631be574aad623
git checkout development

the tf proto files were copied from
https://github.com/tensorflow/serving/tree/master/tensorflow_serving
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/framework

on your platform, install the protocol buffer compiler:
    sudo apt install protobuf-compiler
    or use brew install protobuf
    etc...

then, before cargo builds the src code, it invokes build.rs to compile the protocol buffers

------------------------------------------------------------------------------------

TFS container images were not able to load the entire saved_model graph.  it dropped the
query model variables.  note that a trace is performed with real and fake data before saving
the models and that the models work when loaded by python tensorflows.
The problem seems to be that the TFS is TF C++ and somehow the variables aren't traced
in a way it understands.

the deployment for those is in ranker/deploy/compose/docker-compose-TFS.yaml

-------------------------------------------------------------------------------

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
and modified build.rs for the grpc proto
