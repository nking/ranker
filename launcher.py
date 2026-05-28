from xmanager import xm
from xmanager import xm_local
from dotenv import dotenv_values
from json import dumps, loads

import logging
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

"""
start db services with:
    ./run_compose_dbs.sh
or:
    docker compose -f docker-compose-dbs.yaml up -d
"""

#TODO: switch to coding for a GCS Secret Manager instead of embedding
#passwords in uris. see todo.txt for API details
def main(_):
    num_trials = 4 #20
    num_trials_per_worker = 2
    num_workers = 1
    print(f'JAX_NUM_PROCESSES={num_workers}', flush=True)
    
    #default gateway used by docker is 172.17.0.1
    # can verify that with ip addr show docker0 | grep "inet "
    docker_bridge_gateway = "172.17.0.1"
    
    env_config = {
        **dotenv_values(".env_unittests"),
        'PYTHONUNBUFFERED': '1',
        #'JAX_COORDINATOR_ADDRESS': f'{docker_bridge_gateway}:8888',
        'JAX_NUM_PROCESSES': str(num_workers),
    }
    
    # Add the explicit environment overrides from your yaml
    # We use os.getenv to grab values from your shell (like ${UID})
    run_config = {
        'LOGNAME': env_config.get('POSTGRES_USER'),
        'USER': env_config.get('POSTGRES_USER'),
        "study_name": "GraphRanker_tuning_xmngr",
        "mlflow_experiment_name": "GraphRanker_tuning_xmngr",
        "mlflow_tracking_uri": f"postgresql://{env_config.get('POSTGRES_USER')}:{env_config.get('POSTGRES_PASSWORD')}@{docker_bridge_gateway}:5432/mlflow_db",
        "vizier_endpoint": f"{docker_bridge_gateway}:8000",
        "latest_checkpoint_uri": "gs://checkpoint-bucket/latest",
        "best_checkpoint_uri": "gs://checkpoint-bucket/best",
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
        "phase": "tune",
        'project_id': 'tune-xmngr-01',
    }
    
    # run the experiment locally w/ xm_local.Local.  also means this is a block and wait context.
    # also means of xm_local.Vertex or Kubernetes is used, this block does not wait
    # and any jobs must be stopped from a cluster control plane.
    # BUT, if for some reason you want this blobk to block and wait after launching to
    # cloud, then use job_group = experiment.add(...) and at end use job_group.wait_for_completion()
    with xm_local.create_experiment(experiment_title='vizier_hpo_run') as experiment:
        docker_packageable = xm.dockerfile_container(
            path='.',
            dockerfile='Dockerfile_offline',
            executor_spec = xm_local.Local.Spec(),
            env_vars=env_config,
            args={
                **run_config,
            },
        )
        #docker_packageable = xm.dockerfile_container(
        #    path='.',
        #    dockerfile='Dockerfile_cpu',
        #    executor_spec = xm_local.Local.Spec(),
        #)
        #docker_packageable = xm.container(
        #    image_path='ranker-app:latest',
        #    executor_spec = xm_local.Local.Spec(),
        #    env_vars=env_config,
        #    args={
        #        **run_config,
        #    },
        #)
        [executable] = experiment.package([docker_packageable])

        # 2. Define the Resource Requirements
        # Adjust based on whether you need GPUs or specific CPU counts
        resources = xm.JobRequirements(cpu=1, ram=8 * xm.GiB)

        print("begin jobs")
        #can find network name in docker-compose-dbs.yaml
        for i in range(0, num_trials, num_trials_per_worker):
            trial_ids = [ii for ii in range(i, i+num_trials_per_worker)]
            experiment.add(
                xm.Job(
                    executable=executable,
                    executor=xm_local.Local(
                        requirements=resources,
                    ),
                    name=f"{run_config.get('study_name')}_job_{i}",
                    env_vars={
                        **env_config,
                        'JAX_PROCESS_ID': 0,
                    },
                    args={
                        **run_config,
                        'trial_ids': dumps(list(trial_ids)),
                        "debug": True,
                    },
                )
            )
            
    print(f'xmanager is done running {num_trials} trials')

if __name__ == '__main__':
    xm_local.run(main)
