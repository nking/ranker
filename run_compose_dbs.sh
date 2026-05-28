#!/bin/bash

export $(grep -v '^#' .env | xargs)

rm -rf ./fake-gcs-server/checkpoint-bucket/*

docker compose -f docker-compose-dbs.yaml run --rm gcs db
