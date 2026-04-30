#!/bin/bash
#from this directory, invoke:
mkdir -p ../fake_gcs_server_buckets/data
cp -rf ../src/test/resources/data/* ../fake_gcs_server_buckets/data/

#make sure docker is running
systemctl status docker.service
#sudo systemctl start docker.service

# start the fake_gcs_server with paths relative to this directory
docker run -d --name fake-gcs-server \
  -u $(id -u):$(id -g) \
  -p 127.0.0.1:4443:4443 \
  -v ${PWD}/../fake_gcs_server_buckets:/storage \
  fsouza/fake-gcs-server \
  -scheme http \
  -backend filesystem \
  -data /storage \
  -public-host 127.0.0.1:4443

