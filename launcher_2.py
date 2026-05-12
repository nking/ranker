from xmanager import xm
from xmanager import xm_local
from dotenv import dotenv_values
from json import dumps, loads

import logging
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

"""
launcher for simulating 2 jax processes running the trials

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
    num_processes = 2
    print(f'JAX_NUM_PROCESSES={num_processes}', flush=True)
    
    #default gateway used by docker is 172.17.0.1
    # can verify that with ip addr show docker0 | grep "inet "
    docker_bridge_gateway = "172.17.0.1"
    
    env_config = {
        **dotenv_values(".env_unittests"),
        'PYTHONUNBUFFERED': '1',
        #'JAX_COORDINATOR_ADDRESS': f'{docker_bridge_gateway}:8888',
        'JAX_NUM_PROCESSES': str(num_processes),
        'XLA_FLAGS': f'--xla_force_host_platform_device_count={num_processes}',
        # Add other flags like this:
        # 'XLA_FLAGS': '--xla_force_host_platform_device_count=2 --xla_cpu_enable_fast_math=true',
        'PYTHONIOENCODING': 'UTF-8',
        'TF_CPP_MIN_LOG_LEVEL': '0',
        'JAX_LOG_LEVEL': 'debug',
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
        "num_epochs":2,
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
        print(f'Experiment={experiment}', flush=True)
        print(f'Experiment id={experiment.experiment_id}')
        #print(f'Experiment work_unit_id={experiment.work_unit_id}', flush=True)
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
        
        #container_gateway = "172.17.0.1"
        container_gateway = "0.0.0.0"
        #container_gateway = "127.0.0.1"
        jax_port = 8888
        
        work_unit_id = 0
        print("begin jobs")
        #can find network name in docker-compose-dbs.yaml
        for i in range(0, num_trials, num_trials_per_worker):
            work_unit_id += 1
            trial_ids = [ii for ii in range(i, i+num_trials_per_worker)]
            group_coordinator_port = jax_port + i*num_processes
            group_jobs = {}
            coordinator_name = f"{experiment.experiment_id}_{work_unit_id}_job_{i}_worker_0"
            for rank in range(num_processes):
                if rank == 0:
                    container_ip = f"{container_gateway}"
                else:
                    container_ip=coordinator_name
                docker_options = xm_local.DockerOptions()
                coordinator_addr = f"{container_ip}:{group_coordinator_port}"
                print(f'job={i}, rank={rank}, coordinator_addr={coordinator_addr}, coordinator_name={coordinator_name}')
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
                            #'JAX_COORDINATOR_IP': container_ip,
                            'JAX_COORDINATOR_PORT': str(jax_port),
                        },
                        args={
                            **run_config,
                            'trial_ids': dumps(list(trial_ids)),
                            "debug": True,
                        },
                    )
            #https://github.com/google-deepmind/xmanager/blob/c9c7a46957c052978b578411f3c385e47e663fc5/xmanager/xm/job_blocks.py#L421
            #launch the ranks together as a JobGroup:
            experiment.add(xm.JobGroup(**group_jobs))
           
    print(f'xmanager is done running {num_trials} trials')

if __name__ == '__main__':
    xm_local.run(main)
