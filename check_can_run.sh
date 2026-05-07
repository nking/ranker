#!/bin/bash

export $(grep -v '^#' .env | xargs)

rm -rf ./fake_gcs_server_buckets/checkpoint_bucket/*

docker compose run --rm app \
--study_name="GraphRanker_tuning_cli" \
--mlflow_experiment_name="GraphRanker_tuning_cli" \
--vizier_endpoint="vizier_server:8000" \
--mlflow_tracking_uri="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/mlflow_db" \
--latest_checkpoint_uri="gs://checkpoint_bucket/latest" \
--best_checkpoint_uri="gs://checkpoint_bucket/best" \
--movies_uri="gs://data/movies-00000-of-00001.array_record" \
--recommendations_uri="gs://data/recommended_movies.array_record" \
--recommendations_ts_uri="gs://data/recommended_movies_timestamps.array_record" \
--ratings_train_uri="gs://data/small/ratings_train_liked.array_record" \
--ratings_val_uri="gs://data/small/ratings_val_liked.array_record" \
--train_negatives_uri="gs://data/train_negatives.array_record" \
--val_negatives_uri="gs://data/val_negatives.array_record" \
--movie_embeddings_uri="gs://data/movie_emb-00000-of-00001.array_record" \
--user_embeddings_uri="gs://data/user_emb-00000-of-00001.array_record" \
--trial_id=0 \
--num_epochs=2 \
--batch_size=64 \
--seed=12345 \
--phase="train" 
#--debug=True
