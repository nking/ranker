#!/bin/bash
#from this directory, invoke:
mkdir -p ../fake-gcs-server/data
mkdir -p ../fake-gcs-server/checkpoint-bucket
mkdir -p ../fake-gcs-server/hpo-results-bucket
mkdir -p ../fake-gcs-server/mlflow_artifact_bucket
cp -rf ../src/test/resources/data/* ../fake-gcs-server/data/
chmod -R 775 ../fake-gcs-server
#additionally, might want to set the group to docker recursively

#make sure docker is running
systemctl status docker.service
#systemctl start docker.service

#this is now handled in docker-compose.yaml:
# start the fake_gcs_server with paths relative to this directory
#docker run -d --name fake-gcs-server \
#  -u $(id -u):$(id -g) \
#  -p 127.0.0.1:4443:4443 \
#  -v ${PWD}/../fake-gcs-server:/storage \
#  fsouza/fake-gcs-server \
#  -scheme http \
#  -backend filesystem \
#  -data /storage \
#  -public-host 127.0.0.1:4443

