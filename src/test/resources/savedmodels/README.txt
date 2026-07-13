The two-tower model was created with project 
git@github.com:nking/recommender_systems.git
https://github.com/nking/recommender_systems/blob/main/src/test/python/movie_lens_tfx/run_kaggle_pipelines.py

The graph_ranker model was create in this project.
The training on Kaggle can be seen at:
<add kaggle url>
The export from orbax model to TF savedmodel is from
<add directory and method here>.
  ** currently this directory has a placeholder
     not a trained model but will hold the trained model soon.

=====================================================================

two_tower_model:

signatures (use saved_model_cli to see more):
   - serving_candidate:
     serialized example with "movie_id" and "genres"

   - serving_query:
     serialized example with "user_id", "age", "occupation", "gender", "timestamp"

   - serving_candidate_dict
     expecting named inputs movie_id, genres

   - serving_query_dict:
     expecting named inputs user_id, age, occupation, gender, timestamp

The model is a TwoTower DNN, a bi-encoder trained with 
an in-batch softmax objective (contrastive, listwise loss)
and corrected for item sampling bias 
following Yi et al. 2019 "Sampling-bias-corrected neural modeling
for large corpus item recommendations".

These metrics are from the pipeline's evaluation on the test
dataset, and uses in-batch negatives.
The metrics performed on the full movie catalog with the
test dataset are performed in this project in the test 
directory.

val metrics from the pipeline:
<metric>          value     random (for comparison)
ndcg@20           0.049       0.005
mrr@20            0.048       0.0035
recall@20         0.055       0.02

hyper-parameters:
        "learning_rate": 0.00010260217616970745,
        "weight_decay": 0.00016785171923416138,
        "regl2": 0.0,
        "drop_rate": 0.1175417396617746,
        "embed_out_dim": 32,
        "layer_sizes": "[16]",
        "feature_acronym": "ahosy",
        "incl_genres": true,
        "BATCH_SIZE": 2048,
        "NUM_EPOCHS": 40,
        "use_bias_corr": true,
        "bias_corr_alpha": 0.01,
        "temperature": 0.1,
        "n_users": 6040,
        "n_movies": 3883,
        "n_genres": 18,
        "run_eagerly": false,
        "device": "CPU",
        "num_train": 370838,
        "num_eval": 46354,
        "version": "1.0.0",
        "model_name": "user_movie",

=========================================================
The graph-ranker model:

