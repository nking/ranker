import os
import logging

import grpc
import jax

def safe_jax_init():
    # Check if we are in a distributed environment (e.g., K8s, Vertex, Slurm)
    # Different orchestrators use different keys, but these are common:
    is_distributed = any(k in os.environ for k in [
        'JAX_COORDINATOR_ADDRESS', 'KUBERNETES_SERVICE_HOST',
        'SLURM_JOB_ID', 'PADDLE_TRAINER_ENDPOINTS'
    ])

    try:
        if is_distributed:
            # Let JAX auto-detect cluster settings
            jax.distributed.initialize()
        else:
            # Force local-only initialization for unit tests
            jax.distributed.initialize(
                coordinator_address="localhost:8888",
                num_processes=1,
                process_id=0
            )
    except RuntimeError as e:
        # Handle the "already initialized" error gracefully
        print(f'WARNING while trying to initialize jax distributed: {e}')
safe_jax_init()

from typing import Tuple
import json
import mlflow
from absl import flags
from vizier.service import pyvizier as vz
from vizier.service import clients as vz_clients

from movie_lens_ranker.train import train_fn, test_fn, \
    restore_items_from_checkpoint
from movie_lens_ranker.util import define_flags

FLAGS = flags.FLAGS

from absl import app

import urllib.request
import time
import sys

import fsspec
#load this globally:
fsspec.config.conf['gcs'] = {
    'requester_pays': False,
    'token': 'anon',
    'endpoint_url': os.getenv('STORAGE_EMULATOR_HOST')
}

def wait_for_gcs(fake_gcs_uri, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(fake_gcs_uri) as response:
                if response.getcode() == 200:
                    print("Successfully connected to GCS emulator!")
                    return True
        except Exception:
            print("Waiting for GCS emulator...")
            time.sleep(2)
    print("GCS emulator connection timed out.")
    sys.exit(1)

def get_or_create_mlflow_experiment(experiment_name:str):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def _get_study_config(top_k:int=20, use_batching_alg:bool=False):
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_int_param("top_k", min_value=top_k, max_value=top_k, default_value=top_k)
    root.add_int_param("num_layers", min_value=2, max_value=2,
        default_value=2)
    num_heads_vals = [2, 4, 8]
    num_heads = root.add_discrete_param("num_heads", feasible_values=num_heads_vals)
    all_hidden_options = [64, 128, 256]
    for h_val in num_heads_vals:
        valid_dims = [d for d in all_hidden_options if d % h_val == 0]
        # Create a conditional branch for this specific value of num_heads
        hidden_dim = num_heads.select_values([h_val]).add_discrete_param(name="hidden_dim", feasible_values=valid_dims)
        # conditional max_history:
        for d_val in valid_dims:
            # Grandchild: max_history (Depends on hidden_dim)
            # We select the specific value of hidden_dim to define the next range
            hidden_dim.select_values([d_val]).add_int_param(
                name="max_history",
                min_value=2 * top_k,
                max_value=5 * d_val,
                scale_type=vz.ScaleType.LINEAR)
    root.add_discrete_param("num_candidates", feasible_values=[i for i in range(2*top_k, 500 + 1, 10)])
    
    #if want a linear relationship between lr and wd, setup a dependency:
    # wd_ratio = trial.suggest_float("wd_ratio", 0.01, 1.0, log=True)
    # config['weight_decay'] = config['learning_rate'] * wd_ratio
    root.add_float_param("learning_rate", min_value=1e-4, max_value=1e-2, default_value=1e-3,
        scale_type=vz.ScaleType.LOG)
    root.add_float_param("weight_decay", min_value=1e-4, max_value=1e-2,
        default_value=1e-3,
        scale_type=vz.ScaleType.LOG)
    root.add_discrete_param("out_dim", feasible_values=[16, 32])
    root.add_discrete_param("edge_embed_dim", feasible_values=[8, 16])
    root.add_discrete_param("dropout_rate", feasible_values=[i*0.05 for i in range(1, 7)])

    problem.metric_information.append(
        vz.MetricInformation(name=f'ndcg_{top_k}',
        goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )
    
    study_config = vz.StudyConfig.from_problem(problem)
    #if using 4+ GPUs concurrently, choose GP_UCB_PE instead:
    if use_batching_alg:
        study_config.algorithm = 'GP_UCB_PE'
    else:
        #bandit and eagle cannot support conditional search space:
        #study_config.algorithm = 'GAUSSIAN_PROCESS_BANDIT'
        #study_config.algorithm = 'EAGLE_STRATEGY'
        #study_config.algorithm = 'RANDOM_SEARCH'
        study_config.algorithm = 'DEFAULT'
    return study_config

def setup_vizier_study(project_id: str, study_name: str, endpoint: str,
        top_k:int=20, use_batching_alg:bool=False, waittime_sec:int=60)\
        -> vz_clients.Study:
    """
    get or create a vizier study
    :param project_id:
    :param study_name:
    :param endpoint:
    :raises RuntimeException: If a worker who is not jax process id 0 times out waiting for worker 0 to create the study.
    :raises grpc.RpcError for connection or other serivce errors ike permission errors
    to load
    :return: study owned by project_id and having name study_name
    """
    vz_clients.environment_variables.server_endpoint = endpoint
    resource_name = f"owners/{project_id}/studies/{study_name}"
    
    study_config = _get_study_config(top_k=top_k, use_batching_alg=use_batching_alg)
    
    if jax.process_index() == 0:
        # Now connects to the explicitly created server.
        #loads existing by owner_id for study_id and study_config, else creates new study
        study = vz_clients.Study.from_study_config(study_config,
            owner=project_id,
            study_id=study_name)
        return study
    #else other workers may need to wait for worker 0 to create the study
    #the other wokers might need to wait
    n_waits = waittime_sec//5
    # use the poll-retry pattern to wait until worker 0 creates study
    for _ in range(n_waits):
        try:
            return vz_clients.Study.from_owner_and_id(owner=project_id, study_id=study_name)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                print(f"Worker {jax.process_index()} waiting for study to be created...")
                time.sleep(5)  # Back off to avoid spamming the gRPC server
            else:
                # If it's a connection or permission error, crash early
                raise e
    raise RuntimeError(f"Worker {jax.process_index()} timed out waiting for study to be created by worker 0.")

def tune_run(config):
    
    if "debug" in config and config['debug']:
        print(f'args received: {config}', flush=True)
    
    if "phase" not in config:
        print("ERROR: expecting phase='tune'")
        return
    
    worker_rank = jax.process_index()
    
    study = None
    if worker_rank == 0:
        
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        #create an ML-Flow parent study if it does not exist
        experiment = mlflow.get_experiment_by_name(name=config['study_name'])
        if experiment is None:
            experiment = mlflow.set_experiment(experiment_name=config['study_name'])
            # Create the parent run and immediately get its ID
            try:
                parent_run = mlflow.start_run(run_name="tune")
                mlflow_parent_run_id = parent_run.info.run_id
            finally:
                mlflow.end_run()
        else:
            #get parent run id:
            runs = mlflow.search_runs(
                experiment_names=[config['study_name']],
                filter_string="attributes.run_name = 'tune'",
                output_format="list"
            )
            if runs:
                mlflow_parent_run_id = runs[0].info.run_id
            else:
                try:
                    parent_run = mlflow.start_run(run_name="tune")
                    mlflow_parent_run_id = parent_run.info.run_id
                finally:
                    mlflow.end_run()
        config['mlflow_experiment_id'] = experiment.experiment_id
        config['mlflow_parent_run_id'] = mlflow_parent_run_id
    
    config['mlflow_experiment_name'] = config['study_name']
    config['mlflow_experiment_id'] = get_or_create_mlflow_experiment(config['mlflow_experiment_name'])
    
    trial_ids = json.loads(config['trial_ids'])
    n_large = len(trial_ids) > 10
    
    study = setup_vizier_study(project_id=config['project_id'], study_name=config['study_name'],
        endpoint=config['vizier_endpoint'], top_k=config['top_k'], use_batching_alg=n_large)
    
    suggested_trials = study.suggest(count=len(trial_ids), client_id=study._client._client_id)

    for i, trial in enumerate(suggested_trials):
        
        hparams = {k: v.value for k, v in trial.parameters.items()}
        config2 = {
            **config,
            **hparams,
        }
        for k, v in config2.items():
            if k.find('?') > -1:
                print(f"problem key from trial: {k}={v}", flush=True)
        
        trial_id = trial_ids[i]
        config2['trial_id'] = trial_id
        
        best_val_ndcg_k, mlflow_run_id = train_fn(config2, trial=trial, save_checkpoints=False)
        
        trial.metadata.get_namespace('user')['mlflow_run_id'] = mlflow_run_id
        
        #NOTE: if have a comb of infeasible params or failure in which trial should not be
        # repeated, mark the trial using trial.infeasible()
        trial.complete(vz.Measurement(metrics={f'ndcg_{config["top_k"]}': float(best_val_ndcg_k)}))

def train_run(config):
   
    if "debug" in config and config['debug']:
        print(f'args received: {config}', flush=True)
    
    if "phase" not in config:
        print("ERROR: expecting phase='train_best' or 'train_given'")
        return
    
    worker_rank = jax.process_index()
    
    study = None
    if worker_rank == 0:
        
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        #create an ML-Flow parent study if it does not exist
        experiment = mlflow.get_experiment_by_name(name=config['study_name'])
        if experiment is None:
            experiment = mlflow.set_experiment(experiment_name=config['study_name'])
            # Create the parent run and immediately get its ID
            try:
                parent_run = mlflow.start_run(run_name="train")
                mlflow_parent_run_id = parent_run.info.run_id
            finally:
                mlflow.end_run()
        else:
            #get parent run id:
            runs = mlflow.search_runs(
                experiment_names=[config['study_name']],
                filter_string="attributes.run_name = 'train'",
                output_format="list"
            )
            if runs:
                mlflow_parent_run_id = runs[0].info.run_id
            else:
                try:
                    parent_run = mlflow.start_run(run_name="train")
                    mlflow_parent_run_id = parent_run.info.run_id
                finally:
                    mlflow.end_run()
        config['mlflow_experiment_id'] = experiment.experiment_id
        config['mlflow_parent_run_id'] = mlflow_parent_run_id
    
    config['mlflow_experiment_name'] = config['study_name']
    config['mlflow_experiment_id'] = get_or_create_mlflow_experiment(config['mlflow_experiment_name'])
    
    if config['phase'] == 'train_best':
        #get best hpo results and add to config
        vz_clients.environment_variables.server_endpoint = config['vizier_endpoint']
        study = vz_clients.Study.from_owner_and_id(owner=config['project_id'], study_id=config['study_name'])
        optimal_trials = study.optimal_trials()
        if optimal_trials is None:
            raise ValueError(f"No optimal trials found for project_id={config['project_id']}, study_name={config['study_name']}, endpoint={config['vizier_endpoint']}")
        best_trial = optimal_trials[0]
        best_trial_data = best_trial.materialize()
        best_params = dict(best_trial_data.parameters)
        best_value = best_trial_data.final_measurement.metrics[0].value
        
        print(f"Loaded Best Objective: {best_value}")
        print(f"Loaded Best Parameters: {best_params}")
        config.update(**best_params)
            
    best_val_ndcg_k, mlflow_run_id = train_fn(config, trial=None, save_checkpoints=True)

def test_run(config):
    if "debug" in config and config['debug']:
        print(f'args received: {config}', flush=True)
    
    if "phase" not in config:
        print("ERROR: expecting phase='test_best' or 'test_given'")
        return
    
    worker_rank = jax.process_index()
    
    study = None
    if worker_rank == 0:
        
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        # create an ML-Flow parent study if it does not exist
        experiment = mlflow.get_experiment_by_name(name=config['study_name'])
        if experiment is None:
            experiment = mlflow.set_experiment(
                experiment_name=config['study_name'])
            # Create the parent run and immediately get its ID
            try:
                parent_run = mlflow.start_run(run_name="test")
                mlflow_parent_run_id = parent_run.info.run_id
            finally:
                mlflow.end_run()
        else:
            # get parent run id:
            runs = mlflow.search_runs(
                experiment_names=[config['study_name']],
                filter_string="attributes.run_name = 'test'",
                output_format="list"
            )
            if runs:
                mlflow_parent_run_id = runs[0].info.run_id
            else:
                try:
                    parent_run = mlflow.start_run(run_name="test")
                    mlflow_parent_run_id = parent_run.info.run_id
                finally:
                    mlflow.end_run()
        config['mlflow_experiment_id'] = experiment.experiment_id
        config['mlflow_parent_run_id'] = mlflow_parent_run_id
    
    config['mlflow_experiment_name'] = config['study_name']
    config['mlflow_experiment_id'] = get_or_create_mlflow_experiment(config['mlflow_experiment_name'])
    
    if config['phase'] == 'test_best':
        restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['best_checkpoint_uri'])
    else:
        restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['test_checkpoint_uri'])
        
    restore_dict['config']['ratings_test_uri'] = config['ratings_test_uri']
    restore_dict['config']['train_negatives_uri'] = config['train_negatives_uri']
    
    restore_dict['config']['test_id'] = config.get('test_id', 0)
    
    test_metrics = test_fn(config=restore_dict['config'])
        
    print(f'TEST METRICS: {test_metrics}', flush=True)
    
def main(_):
    config = FLAGS.flag_values_dict()
    # removing problem key: '?'
    config = {k: v for k, v in config.items() if k.find('?') == -1}
    if config['phase'] == 'tune':
        tune_run(config)
    elif config['phase'] == 'test_best' or config['phase'] == 'test_given':
        test_run(config)
    else:
        train_run(config)
    
if __name__ == '__main__':
    define_flags()
    app.run(main)
    