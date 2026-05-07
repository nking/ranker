#!/bin/bash

#learnable params are handled by vizier internally, but katbi passes them in
#--max_history=200 \
#--num_candidates=40 \
#--learning_rate=4e-4 \
#--weight_decay=1e-4 = \
#--out_dim=2 \
#--hidden_dim=64 \
#--num_layers=2 \
#--num_heads=4 \
#--edge_embed_dim=8 \
#--droupout_rate=0.1

export $(grep -v '^#' .env | xargs)

docker compose run --rm app \
--study_name="GraphRanker_tuning_cli" \
--mlflow_experiment_name="GraphRanker_tuning_cli" \
--vizier_endpoint="vizier_server:8000" \
--mlflow_tracking_uri="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/mlflow_db" \
--latest_checkpoint_uri="http://gcs:4443/checkpoint_bucket/latest" \
--best_checkpoint_uri="http://gcs:4443/checkpoint_bucket/best" \
--movies_uri="http://gcs:4443/data/movies-00000-of-00001.array_record" \
--recommendations_uri="http://gcs:4443/data/recommended_movies.array_record" \
--recommendations_ts_uri="http://gcs:4443/data/recommended_movies_timestamps.array_record" \
--ratings_train_uri="http://gcs:4443/data/small/ratings_train_liked.array_record" \
--ratings_val_uri="http://gcs:4443/data/small/ratings_val_liked.array_record" \
--train_negatives_uri="http://gcs:4443/data/small/ratings_train_disliked.array_record" \
--val_negatives_uri="http://gcs:4443/data/small/ratings_val_disliked.array_record" \
--movie_embeddings_uri="http://gcs:4443/data/movie_emb-00000-of-00001.array_record" \
--user_embeddings_uri="http://gcs:4443/data/user_emb-00000-of-00001.array_record" \
--trial_id=1 \
--num_epochs=2 \
--batch_size=64 \
--seed=12345 \
--phase="tune"
#--debug=True
