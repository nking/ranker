#adapted from https://github.com/google-deepmind/xmanager/blob/63a2ee86bca0fa847787f362f421b8bc4d2a6eb8/examples/parameter_controller/launcher.py#L86
# which has the following copyright:
#
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import os
from json import dumps

from xmanager import xm
from xmanager import xm_local
from xmanager.contrib import parameter_controller
from dotenv import dotenv_values

import logging
from absl import logging as absl_logging, app

absl_logging.set_verbosity(absl_logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

"""
launcher for simulating 2 jax processes running the pipeline tune, train, test

start db services with:
    ./run_compose_dbs.sh
or:
    docker compose -f docker-compose-dbs.yaml up -d

xmanager launch xmngr_controller/launcher_pipeline.py -- --xm_db_yaml_config_path=db_config.yaml
"""

#TODO: switch to coding for a GCS Secret Manager instead of embedding
#passwords in uris. see todo.txt for API details
def main(_):
    
    with xm_local.create_experiment(experiment_title='xmngr_pipeline') as experiment:
        
        num_trials = 4  # 20
        num_trials_per_worker = 2
        num_processes = 2
        print(f'JAX_NUM_PROCESSES={num_processes}', flush=True)
        
        # default gateway used by docker is 172.17.0.1
        # can verify that with ip addr show docker0 | grep "inet "
        docker_bridge_gateway = "172.17.0.1"
        env_config = {
            **dotenv_values(".env_unittests"),
            # relative to based dir where xmanager invoked
            'PYTHONUNBUFFERED': '1',
            # 'JAX_COORDINATOR_ADDRESS': f'{docker_bridge_gateway}:8888',
            'JAX_NUM_PROCESSES': str(num_processes),
            'XLA_FLAGS': f'--xla_force_host_platform_device_count={num_processes}',
            # Add other flags like this:
            # 'XLA_FLAGS': '--xla_force_host_platform_device_count=2 --xla_cpu_enable_fast_math=true',
            'PYTHONIOENCODING': 'UTF-8',
            'JAX_LOG_LEVEL': 'debug',
        }
        run_config = {
            'LOGNAME': env_config.get('POSTGRES_USER'),
            'USER': env_config.get('POSTGRES_USER'),
            "study_name": "GraphRanker_tuning_xmngr_2",
            "mlflow_experiment_name": "GraphRanker_tuning_xmngr_2",
            "mlflow_tracking_uri": f"postgresql://{env_config.get('POSTGRES_USER')}:{env_config.get('POSTGRES_PASSWORD')}@{docker_bridge_gateway}:5432/mlflow_db",
            "vizier_endpoint": f"{docker_bridge_gateway}:8000",
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
            "phase": "tune",
            'project_id': 'tune-xmngr-01',
        }
        
        executable = experiment.package([
            # docker tag ranker-app:latest localhost/ranker-app:latest
            xm.Packageable(
                executable_spec=xm.Dockerfile(
                    path=os.path.abspath('.'),
                    dockerfile='Dockerfile_offline',
                ),
                executor_spec=xm_local.Local.Spec()
            ),
        ])[0]
        
        @parameter_controller.controller(
            executor=xm_local.Local(
                docker_options=xm_local.DockerOptions(
                    # for local runs
                    volumes={'/var/run/docker.sock': '/var/run/docker.sock'}
                ),
            ),
            controller_args={},
            controller_env_vars=env_config,
            package_path='.',
        )
        async def run_pipeline(experiment: xm.Experiment):
            '''
            SPMD w/ grain dataloader:
            with num_processes = 2, we are partitioning the data betweewn worker,shard, process_id=0
            and worker,shard,process_id=1.
            if we set cpu=2, each of the worker, shard, process_id further partitions the data they
            receive into 2.
            NOTE that cpu=x cannot exceed the number of cores on the machine it is running on
            for this simulation.
            '''
            resources = xm.JobRequirements(cpu=2, ram=4 * xm.GiB)
            
            container_gateway = "0.0.0.0"
            jax_port = 8888
            work_unit_id = 0
            
            print("begin tune jobs")
            # can find network name in docker-compose-dbs.yaml
            for i in range(0, num_trials, num_trials_per_worker):
                work_unit_id += 1
                group_jobs = {}
                trial_ids = [ii for ii in range(i, i + num_trials_per_worker)]
                group_coordinator_port = jax_port + i * num_processes
                # this uses the name given to container for worker_0 by xmanager and jax uses dns to get the ip address
                coordinator_name = f"{experiment.experiment_id}_{work_unit_id}_job_{i}_worker_0"
                for rank in range(num_processes):
                    if rank == 0:
                        container_ip = f"{container_gateway}"
                    else:
                        container_ip = coordinator_name
                    docker_options = xm_local.DockerOptions()
                    coordinator_addr = f"{container_ip}:{group_coordinator_port}"
                    logging.info(
                        f'job={i}, rank={rank}, coordinator_addr={coordinator_addr}, coordinator_name={coordinator_name}')
                    '''
                    JAX: coordinator_address (str | None) – the IP address of process 0 and
                         a port on which that process should launch a coordinator service.
                         The choice of port does not matter, so long as the port is available
                         on the coordinator and all processes agree on the port.
                    '''
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
                            # 'JAX_COORDINATOR_IP': container_ip,
                            'JAX_COORDINATOR_PORT': str(jax_port),
                        },
                        args={
                            **run_config,
                            'trial_ids': dumps(list(trial_ids)),
                            "debug": False,
                        },
                    )
                
                logging.info(f'adding tuning job_{i}')
                tuning_handle = await experiment.add(xm.JobGroup(**group_jobs))
                await tuning_handle.wait_until_complete()
                logging.info(f'finished tuning job_{i}')
            logging.info(f"finished tuning {num_trials} trials")
            print('\a')
            
            # ===============  begin train  =======================
            jax_port = 8890
            print("begin train job")
            group_jobs = {}
            work_unit_id += 1
            group_coordinator_port = jax_port
            coordinator_name = f"{experiment.experiment_id}_{work_unit_id}_train_job_0_worker_0"
            for rank in range(num_processes):
                if rank == 0:
                    container_ip = f"{container_gateway}"
                else:
                    container_ip = coordinator_name
                docker_options = xm_local.DockerOptions()
                coordinator_addr = f"{container_ip}:{group_coordinator_port}"
                group_jobs[f"train_job_0_worker_{rank}"] = xm.Job(
                    executable=executable,
                    executor=xm_local.Local(
                        requirements=resources,
                        docker_options=docker_options
                    ),
                    name=f"train_job_0_worker_{rank}",
                    env_vars={
                        **env_config,
                        'JAX_PROCESS_ID': str(rank),
                        'JAX_COORDINATOR_ADDRESS': coordinator_addr,
                        # 'JAX_COORDINATOR_IP': container_ip,
                        'JAX_COORDINATOR_PORT': str(jax_port),
                    },
                    args={
                        **run_config,
                        'phase': 'train_best',
                        'validate_checkpoint_restores' : True,
                        "debug": True,
                    },
                )
            logging.info(f'adding train job')
            handle = await experiment.add(xm.JobGroup(**group_jobs))
            await handle.wait_until_complete()
            logging.info(f'finished train job')
            print('\a')
            print('\a')
            
            # ===============  begin test  =======================
            """
            jax_port = 8892
            print("begin test job")
            group_jobs = {}
            work_unit_id += 1
            group_coordinator_port = jax_port
            coordinator_name = f"{experiment.experiment_id}_{work_unit_id}_test_job_0_worker_0"
            for rank in range(num_processes):
                if rank == 0:
                    container_ip = f"{container_gateway}"
                else:
                    container_ip = coordinator_name
                docker_options = xm_local.DockerOptions()
                coordinator_addr = f"{container_ip}:{group_coordinator_port}"
                group_jobs[f"test_job_0_worker_{rank}"] = xm.Job(
                    executable=executable,
                    executor=xm_local.Local(
                        requirements=resources,
                        docker_options=docker_options
                    ),
                    name=f"test_job_0_worker_{rank}",
                    env_vars={
                        **env_config,
                        'JAX_PROCESS_ID': str(rank),
                        'JAX_COORDINATOR_ADDRESS': coordinator_addr,
                        # 'JAX_COORDINATOR_IP': container_ip,
                        'JAX_COORDINATOR_PORT': str(jax_port),
                    },
                    args={
                        **run_config,
                        'phase': 'test_best',
                        'validate_checkpoint_restores' : False,
                        "debug": True,
                        "ratings_test_uri": "gs://data/small/ratings_test_liked.array_record",
                        "test_negatives_uri": "gs://data/test_negatives.array_record",
                    },
                )
            logging.info(f'adding test job')
            handle = await experiment.add(xm.JobGroup(**group_jobs))
            await handle.wait_until_complete()
            logging.info(f'finished test job')
            print('\a')
            print('\a')
            print('\a')
            """
            
        experiment.add(run_pipeline())
        
        logging.info("pipeline done.")
        
if __name__ == '__main__':
    app.run(main)