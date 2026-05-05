#!/bin/bash

export $(grep -v '^#' .env | xargs)

rm -rf ./fake_gcs_server_buckets/checkpoint_bucket/*

docker compose -f docker-compose-dbs.yaml run --rm gcs db
