from xmanager import xm
from xmanager import xm_local
from dotenv import dotenv_values

"""
start db services with:
    ./run_compose_dbs.sh
or:
    docker compose -f docker-compose-dbs.yaml up -d
"""

#TODO: switch to coding for a GCS Secret Manager instead of embedding
#passwords in uris. see todo.txt for API details
def main(_):
    num_trials = 1 #20
    
    #default gateway used by docker is 172.17.0.1
    # can verify that with ip addr show docker0 | grep "inet "
    docker_bridge_gateway = "172.17.0.1"
    
    env_config = {
        **dotenv_values(".env"),
    }
    
    # Add the explicit environment overrides from your yaml
    # We use os.getenv to grab values from your shell (like ${UID})
    run_config = {
        'LOGNAME': env_config.get('POSTGRES_USER'),
        'USER': env_config.get('POSTGRES_USER'),
        "study_name": "GraphRanker_tuning_xmngr",
        "mlflow_experiment_name": "GraphRanker_tuning_cli",
        "optuna_storage_uri": f"postgresql://{env_config.get('POSTGRES_USER')}:{env_config.get('POSTGRES_PASSWORD')}@{docker_bridge_gateway}:5432/optuna_db",
        "mlflow_tracking_uri": f"postgresql://{env_config.get('POSTGRES_USER')}:{env_config.get('POSTGRES_PASSWORD')}@{docker_bridge_gateway}:5432/mlflow_db",
        "latest_checkpoint_uri": "gs://checkpoint_bucket/latest",
        "best_checkpoint_uri": "gs://checkpoint_bucket/best",
        "movies_uri": "gs://data/movies-00000-of-00001.array_record",
        "recommendations_uri": "gs://data/recommended_movies.array_record",
        "recommendations_ts_uri": "gs://data/recommended_movies_timestamps.array_record",
        "ratings_train_uri": "gs://data/small/ratings_train_liked.array_record",
        "ratings_val_uri": "gs://data/small/ratings_val_liked.array_record",
        "train_negatives_uri": "gs://data/train_negatives.array_record",
        "val_negatives_uri": "gs://data/val_negatives.array_record",
        "movie_embeddings_uri": "gs://data/movie_emb-00000-of-00001.array_record",
        "user_embeddings_uri": "gs://data/user_emb-00000-of-00001.array_record",
        "num_epochs": 2,
        "batch_size": 64,
        "seed": 12345,
        "phase": "train"
    }
    
    with xm_local.create_experiment(experiment_title='optuna_hpo_run') as experiment:
        docker_packageable = xm.dockerfile_container(
            path='.',
            dockerfile='Dockerfile_cpu',
            executor_spec = xm_local.Local.Spec(),
        )
        [executable] = experiment.package([docker_packageable])

        # 2. Define the Resource Requirements
        # Adjust based on whether you need GPUs or specific CPU counts
        resources = xm.JobRequirements(cpu=1, ram=8 * xm.GiB)

        #can find network name in docker-compose-dbs.yaml
        for i in range(num_trials):
            experiment.add(
                xm.Job(
                    executable=executable,
                    executor=xm_local.Local(
                        requirements=resources,
                        #docker_options=xm_local.DockerOptions(  #?ports ? volumns
                        #    network='hpo_shared_network'
                        #)
                    ),
                    name=f"{env_config.get('study_name')}_trial_{i}",
                    env_vars=env_config,
                    args={
                        **run_config,
                        'trial_id': i,
                    },
                )
            )

if __name__ == '__main__':
    xm_local.run(main)