import os
import uuid
from functools import partial
from typing import Dict, Union

import jax

def safe_jax_init():
    # Check if we are in a distributed environment (e.g., K8s, Vertex, Slurm)
    # Different orchestrators use different keys, but these are common:
    is_distributed = any(k in os.environ for k in [
        'KUBERNETES_SERVICE_HOST',
        'SLURM_JOB_ID', 'PADDLE_TRAINER_ENDPOINTS'
    ])
   
    try:
        if is_distributed:
            # Let JAX auto-detect cluster settings
            jax.distributed.initialize()
        else:
            # Force local-only initialization for unit tests
            jax.distributed.initialize(
                coordinator_address=os.environ.get('JAX_COORDINATOR_ADDRESS', 'localhost:8888'),
                num_processes=int(os.environ.get('JAX_NUM_PROCESSES', 1)),
                process_id=int(os.environ.get('JAX_PROCESS_ID', 0))
            )
    except RuntimeError as e:
        # Handle the "already initialized" error gracefully
        print(f'WARNING while trying to initialize jax distributed: {e}')
safe_jax_init()

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import grpc
from vizier._src.pyvizier.shared.trial import ParameterValue, ParameterDict
import json
import mlflow
from absl import flags
from vizier.service import pyvizier as vz
from vizier.service import clients as vz_clients

from movie_lens_ranker.train import train_fn, test_fn
from movie_lens_ranker.util import define_flags, get_recognized_keys

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

#devices = jax._src.mesh_utils.create_device_mesh((jax.device_count(),))
##devices = np.array(jax.devices())
#mesh2 = jax.sharding.Mesh(devices, axis_names=('processes',))

# Create a 2D grid of devices: (num_processes, devices_per_process)
# If you have 4 workers with 1 GPU each, this is (4, 1)
# If you have 2 workers with 8 GPUs each, this is (2, 8)
device_grid = jax._src.mesh_utils.create_device_mesh((jax.process_count(), jax.local_device_count()))
mesh2 = jax.sharding.Mesh(device_grid, axis_names=('processes', 'local_devices'))

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

def extract_correct_vizier_param_types_dict(params:Union[ParameterDict, Dict]):
    config = {}
    int_keys = {"top_k", "num_layers", "num_heads","hidden_dim","max_history","num_candidates","out_dim","edge_embed_dim"}
    for k, v in params.items():
        if k in int_keys:
            if isinstance(v, ParameterValue):
                config[k] = int(v.value)
            else:
                config[k] = int(v)
        else:
            if isinstance(v, ParameterValue):
                config[k] = float(v.value)
            else:
                config[k] = float(v)
    return config

def _get_study_config(top_k:int=20, use_batching_alg:bool=False):
    
    problem = vz.ProblemStatement()
    #https://oss-vizier.readthedocs.io/en/latest/guides/user/search_spaces.html#search-spaces
    root = problem.search_space.select_root()
    
    root.add_discrete_param("top_k", feasible_values=[top_k])
    root.add_discrete_param("num_layers", feasible_values=[2])
    #hidden_dim % num_heads == 0
    root.add_discrete_param("num_heads", feasible_values=[2, 4, 8])
    root.add_discrete_param("hidden_dim", feasible_values=[64, 128, 256])
    root.add_discrete_param("max_history", feasible_values=[i for i in range(2*top_k, 6*256, 248)])
    
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
        study_config.algorithm = 'GAUSSIAN_PROCESS_BANDIT'
        #study_config.algorithm = 'EAGLE_STRATEGY'
        #study_config.algorithm = 'RANDOM_SEARCH'
        #study_config.algorithm = 'DEFAULT'
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

@jax.jit
@partial(jax.shard_map, mesh=mesh2, in_specs=P('processes', None), out_specs=P())
def _sync_params(params_array:jnp.ndarray):
    # Sum the arrays across the 'processes' axis of the mesh.
    # Since others are 0, Sum(Value, 0, 0...) = Value on all workers.
    return jax.lax.psum(params_array, axis_name='processes')
    
def sync_hyperparams(params_dict) -> Dict[str, Union[int, float]]:
    # Convert dict to a fixed-order array on Process 0
    # Others initialize with zeros
    sync_keys = ["top_k", "num_layers", "num_heads", "hidden_dim",
        "max_history","num_candidates", "learning_rate", "weight_decay", "out_dim",
        "edge_embed_dim","dropout_rate"]
    if jax.process_index() == 0:
        #extract ParameterValue to primitives:
        params_dict = extract_correct_vizier_param_types_dict(params_dict)
        # everything is converted to float:
        params_arr = jnp.array([params_dict[k] for k in sync_keys])
    else:
        params_arr = jnp.zeros(len(sync_keys))  # Must match shape
        
    # Add a 'processes' dimension so it is (1, num_params) locally
    params_arr_2d = params_arr[None, :]

    # Create a Global Array view from this local piece
    # This tells JAX: 'My (1, 11) array is the i-th shard of a (TotalProcesses, 11) array'
    global_sharding = jax.sharding.NamedSharding(mesh2, P('processes', None))
    global_params = jax.make_array_from_single_device_arrays(
        shape=(jax.process_count(), len(sync_keys)),
        sharding=global_sharding,
        arrays=[params_arr_2d]
    )

    with mesh2:
        # 4. Sync via shard_map
        synced_global_2d = _sync_params(global_params)
    
    # 5. Return as a simple 1D array (all rows are now identical)
    final_params = synced_global_2d[0]
    # map back to dictionary
    final_params_dict = {k: v for k, v in zip(sync_keys, final_params)}
    # cast to int where needed:
    final_params_dict = extract_correct_vizier_param_types_dict(
        final_params_dict)
    return final_params_dict
    
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
                experiment_ids=[experiment.experiment_id],
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
    
    if "debug" in config and config['debug']:
        print(f'args received by tune_fn: {config}', flush=True)
        
    trial_ids = json.loads(config['trial_ids'])
    n_large = len(trial_ids) > 10
    
    if worker_rank == 0:
        study = setup_vizier_study(project_id=config['project_id'], study_name=config['study_name'],
            endpoint=config['vizier_endpoint'], top_k=config['top_k'], use_batching_alg=n_large)
        unique_id = uuid.uuid4().hex[:8]
        resource_name = f"owners_{config['project_id']}_studies_{config['study_name']}"
        client_id = f"{resource_name}_{unique_id}"
        #suggested_trials = study.suggest(count=len(trial_ids), client_id=study._client._client_id)
        suggested_trials = study.suggest(count=len(trial_ids), client_id=client_id)
    
    trial_suggestion = None
    hparams = {}
    for i in range(len(trial_ids)):
        if worker_rank == 0:
            trial_suggestion = suggested_trials[i]
            hparams = {k: v for k, v in trial_suggestion.parameters.items()}
        hparams = sync_hyperparams(hparams)
        config2 = {
            **config,
            **hparams,
        }
        for k, v in config2.items():
            if k.find('?') > -1:
                print(f"problem key from trial: {k}={v}", flush=True)
        trial_id = trial_ids[i]
        config2['trial_id'] = trial_id
        
        # NOTE: if have a comb of infeasible params or failure in which trial should not be
        # repeated, mark the trial using trial.infeasible() and continue w/o running train_fn
        os.environ.get("JAX_COORDINATOR_ADDRESS")
    
        best_val_ndcg_k, mlflow_run_id = train_fn(config2, trial=trial_suggestion, save_checkpoints=False)
        
        if worker_rank == 0:
            trial_suggestion.update_metadata(vz.Metadata({'mlflow_run_id': mlflow_run_id}))
            trial_suggestion.complete(vz.Measurement(metrics={f'ndcg_{config["top_k"]}': float(best_val_ndcg_k)}))
        
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
                experiment_ids = [experiment.experiment_id],
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
        best_trial = None
        for tr in optimal_trials:
            best_trial = tr
            break
        best_trial_data = best_trial.materialize()
        #best_params contains only the params being tuned, not all params needed for train_fn
        best_params = extract_correct_vizier_param_types_dict(best_trial_data.parameters)
        best_value = best_trial_data.final_measurement.metrics.get(f'ndcg_{config["top_k"]}')
        best_value = best_value.value
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
                experiment_ids=[experiment.experiment_id],
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
    
    test_metrics = test_fn(config=config)
        
    print(f'TEST METRICS: {test_metrics}', flush=True)
    
def main(_):
    config = FLAGS.flag_values_dict()
    
    if "debug" in config and config['debug']:
        print(f'all args received from flgs: {config}', flush=True)
    
    config = {k:v for k, v in config.items() if k in get_recognized_keys()}
    
    # work-around for xmanager encapsulating the json dumps string of array with extra quotes
    if "trial_ids" in config:
        # idempotent if doesn't have additional quotes:
        config['trial_ids'] = config['trial_ids'].strip("'").strip('"')
    
    if "debug" in config and config['debug']:
        print(f'recognized args: {config}', flush=True)
        
    # static top_k
    config['top_k'] = 20
    
    if config['phase'] == 'tune':
        tune_run(config)
    elif config['phase'].find('test') == 0:
        test_run(config)
    else:
        train_run(config)
    
    
if __name__ == '__main__':
    define_flags()
    app.run(main)
    