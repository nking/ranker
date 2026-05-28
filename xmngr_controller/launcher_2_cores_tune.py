from xmanager import xm
from xmanager import xm_local
from dotenv import dotenv_values
from json import dumps, loads
import os
import logging
from absl import logging as absl_logging, app

absl_logging.set_verbosity(absl_logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
import asyncio
from xmanager.contrib import parameter_controller

"""
launcher for simulating 2 jax processes running the trials.
1) cd to raker project base directory
2) start db services with:
    ./run_compose_dbs.sh
or:
    docker compose -f docker-compose-dbs.yaml up -d
3) activate conda virtual environment having xmanager
4) xmanager launch xmngr_controller/launcher_2_cores_tune.py -- \
--xm_db_yaml_config_path=db_config.yaml

"""

#TODO: switch to coding for a GCS Secret Manager instead of embedding
#passwords in uris. see todo.txt for API details
def main(_):
    
    async def run_experiment():
    
        async with xm_local.create_experiment(experiment_title='pipeline') as experiment:
            num_trials = 4 #20
            num_trials_per_worker = 2
            num_processes = 2
            print(f'JAX_NUM_PROCESSES={num_processes}', flush=True)
            
            #default gateway used by docker is 172.17.0.1
            # can verify that with ip addr show docker0 | grep "inet "
            docker_bridge_gateway = "172.17.0.1"
    
            env_config = {
                **dotenv_values(".env_unittests"),
                'PYTHONUNBUFFERED': '1',
                #'JAX_COORDINATOR_ADDRES.S': f'{docker_bridge_gateway}:8888',
                'JAX_NUM_PROCESSES': str(num_processes),
                'XLA_FLAGS': f'--xla_force_host_platform_device_count={num_processes}',
                # Add other flags like this:
                # 'XLA_FLAGS': '--xla_force_host_platform_device_count=2 --xla_cpu_enable_fast_math=true',
                'PYTHONIOENCODING': 'UTF-8',
                'TF_CPP_MIN_LOG_LEVEL': '0',
                'JAX_LOG_LEVEL': 'debug',
                "LOCAL_SIMULATION": "True"
            }
    
            # Add the explicit environment overrides from your yaml
            # We use os.getenv to grab values from your shell (like ${UID})
            run_config = {
                'LOGNAME': env_config.get('POSTGRES_USER'),
                'USER': env_config.get('POSTGRES_USER'),
                "study_name": "GraphRanker_tuning_xmngr_2",
                "mlflow_experiment_name": "GraphRanker_tuning_xmngr_2",
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
                "num_epochs":2,
                "batch_size": 64,
                "seed": 12345,
                "phase": "tune",
                'project_id': 'tune-xmngr-01',
            }
    
            xmngr_db_uri = f"postgresql+psycopg2://{env_config.get('POSTGRES_USER')}:{env_config.get('POSTGRES_PASSWORD')}@{docker_bridge_gateway}:5432/xmngr_db"
    
            docker_packageable = xm.dockerfile_container(
                path='..',
                dockerfile='Dockerfile_offline',
                executor_spec=xm_local.Local.Spec(),
                env_vars=env_config,
                args={
                    **run_config,
                },
            )
    
            [executable] = await experiment.package([docker_packageable])
    
            resources = xm.JobRequirements(cpu=2, ram=4 * xm.GiB)
    
            #https://github.com/google-deepmind/xmanager/blob/63a2ee86bca0fa847787f362f421b8bc4d2a6eb8/docs/parameter_controller.md
            @parameter_controller.controller(
                # The controller itself runs as a job. Use Local() for local runs.
                controller_name = "GraphRanker_pipeline_xmngr",
                executor=xm_local.Local(
                    docker_options=xm_local.DockerOptions(
                        volumes={ '/var/run/docker.sock': '/var/run/docker.sock'}
                    ),
                ),
                controller_args={
                    #'db_config': './db_config.yaml', #pass this in via command line flags
                    'controller_id': 'main_tuning_controller',
                    'worker_executable': executable,
                },
                controller_env_vars={
                    **dotenv_values(".env_unittests")
                },
                use_host_db_config=False,
                package_path='.', #relative to launcher
            )
    
            async def training_pipeline(experiment: xm.Experiment, executable):
                container_gateway = "0.0.0.0"
                jax_port = 8888
                work_unit_id = 0
                print("begin tune jobs")
                for i in range(0, num_trials, num_trials_per_worker):
                    work_unit_id += 1
                    group_jobs = {}
                    trial_ids = [ii for ii in range(i, i+num_trials_per_worker)]
                    group_coordinator_port = jax_port + i*num_processes
                    coordinator_name = f"{experiment.experiment_id}_{work_unit_id}_job_{i}_worker_0"
                    for rank in range(num_processes):
                        if rank == 0:
                            container_ip = f"{container_gateway}"
                        else:
                            container_ip=coordinator_name
                        docker_options = xm_local.DockerOptions()
                        coordinator_addr = f"{container_ip}:{group_coordinator_port}"
                        group_jobs[f"job_{i}_worker_{rank}"] = xm.Job(
                                executable=executable,
                                executor=xm_local.Local(
                                    requirements=resources,
                                    docker_options=docker_options
                                ),
                                name=f"job_{i}_worker_{rank}",
                                env_vars={
                                    **env_config,
                                    'JAX_PROCESS_ID': str(rank),
                                    'JAX_COORDINATOR_ADDRESS': coordinator_addr,
                                    #'JAX_COORDINATOR_IP': container_ip,
                                    'JAX_COORDINATOR_PORT': str(jax_port),
                                },
                                args={
                                    **run_config,
                                    'trial_ids': dumps(list(trial_ids)),
                                    "debug": True,
                                },
                            )
                    print(f'launching tuning job {i}')
                    tuning_handle = await experiment.add(xm.JobGroup(**group_jobs))
                    await tuning_handle.wait_until_complete()
                    print(f'finished tune job {i}')
            print(f'tuning phase finished for {num_trials} trials')
        
            await experiment.add(training_pipeline(experiment, executable))
    
    xm_local.run_async(run_experiment())
    print(f'xmanager is done with experiment pipeline')

if __name__ == '__main__':
    app.run(main)
