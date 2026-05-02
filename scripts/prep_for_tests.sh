#!/bin/bash
#from this directory, invoke:
mkdir -p ../fake_gcs_server_buckets/data
mkdir -p ../fake_gcs_server_buckets/checkpoint_bucket
mkdir -p ../fake_gcs_server_buckets/mlflow_artifact_bucket
cp -rf ../src/test/resources/data/* ../fake_gcs_server_buckets/data/
chmod -R 775 ../fake_gcs_server_buckets

#make sure docker is running
systemctl status docker.service
#sudo systemctl start docker.service

#this is now handled in docker-compose.yaml:
# start the fake_gcs_server with paths relative to this directory
#docker run -d --name fake-gcs-server \
#  -u $(id -u):$(id -g) \
#  -p 127.0.0.1:4443:4443 \
#  -v ${PWD}/../fake_gcs_server_buckets:/storage \
#  fsouza/fake-gcs-server \
#  -scheme http \
#  -backend filesystem \
#  -data /storage \
#  -public-host 127.0.0.1:4443

