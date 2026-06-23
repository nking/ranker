
"""
main runner for the tuning, training, and testing of a Jax AI stack
model with JaxAI stack dataloader under SPMD paradigm with multi-host, multi-process
abilities.
"""

import os
import sys
import logging

import jax
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import uuid
from typing import Dict, Union, Any
from mlflow import MlflowClient
import jax.numpy as jnp

import grpc
from vizier._src.pyvizier.shared.trial import ParameterValue, ParameterDict
import json
import mlflow
from absl import flags
from vizier.service import pyvizier as vz
from vizier.service import clients as vz_clients
from jax.experimental import mesh_utils

from movie_lens_ranker.train import train_fn, test_fn
from movie_lens_ranker.util import define_flags, get_recognized_keys, \
    app_runner_is_missing_minimum_required_keys, \
    destringify_mlflow_params

FLAGS = flags.FLAGS

import urllib.request
import time

import fsspec
#load this globally:
fsspec.config.conf['gcs'] = {
    'requester_pays': False,
    'token': 'anon',
    'endpoint_url': os.getenv('STORAGE_EMULATOR_HOST')
}

#devices = mesh_utils.create_device_mesh((jax.device_count(),))
##devices = np.array(jax.devices())
#mesh2 = jax.sharding.Mesh(devices, axis_names=('processes',))

# Create a 2D grid of devices: (num_processes, devices_per_process)
# If you have 4 workers with 1 GPU each, this is (4, 1)
# If you have 2 workers with 8 GPUs each, this is (2, 8)
device_grid = mesh_utils.create_device_mesh((jax.process_count(), jax.local_device_count()))
mesh2 = jax.sharding.Mesh(device_grid, axis_names=('processes', 'local_devices'))

def wait_for_gcs(fake_gcs_uri, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(fake_gcs_uri) as response:
                if response.getcode() == 200:
                    logging.info("Successfully connected to GCS emulator!")
                    return True
        except Exception:
            logging.exception("Waiting for GCS emulator...")
            time.sleep(2)
    logging.exception("GCS emulator connection timed out.")
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
    root.add_discrete_param("hidden_dim", feasible_values=[64, 128])
    root.add_discrete_param("max_history", feasible_values=[i for i in range(2*top_k, 100, 10)])
    
    root.add_discrete_param("num_candidates", feasible_values=[i for i in range(2*top_k, 100, 10)])
    
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
                logging.exception(f"Worker {jax.process_index()} waiting for study to be created...")
                time.sleep(5)  # Back off to avoid spamming the gRPC server
            else:
                # If it's a connection or permission error, crash early
                raise e
    raise RuntimeError(f"Worker {jax.process_index()} timed out waiting for study to be created by worker 0.")

def sync_hyperparams(params_dict) -> Dict[str, Union[int, float]]:
    # Convert dict to a fixed-order array on Process 0
    # Others initialize with zeros
    sync_keys = ["top_k", "num_layers", "num_heads", "hidden_dim",
        "max_history","num_candidates", "learning_rate", "weight_decay", "out_dim",
        "edge_embed_dim","dropout_rate"]
    num_keys = len(sync_keys)
    if jax.process_index() == 0:
        #extract ParameterValue to primitives:
        params_dict = extract_correct_vizier_param_types_dict(params_dict)
        local_arr = jnp.array([float(params_dict[k]) for k in sync_keys],
            dtype=jnp.float32)
    else:
        local_arr = jnp.zeros((num_keys,), dtype=jnp.float32)
    
    gathered = jax.experimental.multihost_utils.process_allgather(local_arr)
    
    final_params = jnp.sum(gathered, axis=0)
    
    # map back to dictionary
    final_params_dict = {k: v for k, v in zip(sync_keys, final_params)}
    # cast to int where needed:
    final_params_dict = extract_correct_vizier_param_types_dict(
        final_params_dict)
    return final_params_dict
    
def run_tune(config):
    
    if "debug" in config and config['debug']:
        logging.info(f'tune_run config: {config}')
    
    if "phase" not in config:
        logging.error("ERROR: expecting phase='tune'")
        return
    
    req_keys = {'user_embeddings_uri', 'movie_embeddings_uri', 'movies_uri',
        'recommendations_uri', 'recommendations_ts_uri',
        'ratings_train_liked_uri',
        'ratings_train_3_uri', 'ratings_train_disliked_uri',
        'ratings_val_liked_uri', 'ratings_val_3_uri',
        'ratings_val_disliked_uri',
        'seed'}
    for key in req_keys:
        if key not in config:
            raise LookupError(f'missing key {key} in config')
    
    worker_rank = jax.process_index()
    
    experiment_name = config.get('mlflow_experiment_name', config['study_name'])
    config['mlflow_experiment_name'] = experiment_name
    
    study = None
    if worker_rank == 0:
        logging.info(f"worker_{worker_rank}: creating MLFlow parent run")
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        #create an ML-Flow parent study if it does not exist
        experiment = mlflow.get_experiment_by_name(name=experiment_name)
        if experiment is None:
            experiment = mlflow.set_experiment(experiment_name=experiment_name)
            # Create the parent run and get its ID
            try:
                parent_run = mlflow.start_run(run_name="tune", experiment_id=experiment.experiment_id)
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
                    parent_run = mlflow.start_run(run_name="tune", experiment_id=experiment.experiment_id)
                    mlflow_parent_run_id = parent_run.info.run_id
                finally:
                    mlflow.end_run()
        config['mlflow_experiment_id'] = experiment.experiment_id
        config['mlflow_parent_run_id'] = mlflow_parent_run_id
        logging.info(f"worker_{worker_rank}: done creating MLFlow parent run")
        
    trial_ids = json.loads(config['trial_ids'])
    n_large = len(trial_ids) > 10
    
    jax.experimental.multihost_utils.sync_global_devices( "sync_barrier_for_vizier")
    
    if worker_rank == 0:
        logging.info(f"worker_{worker_rank}: creating vizier study")
        study = setup_vizier_study(project_id=config['project_id'], study_name=config['study_name'],
            endpoint=config['vizier_endpoint'], top_k=config['top_k'], use_batching_alg=n_large)
        unique_id = uuid.uuid4().hex[:8]
        resource_name = f"owners_{config['project_id']}_studies_{config['study_name']}"
        client_id = f"{resource_name}_{unique_id}"
        #suggested_trials = study.suggest(count=len(trial_ids), client_id=study._client._client_id)
        suggested_trials = study.suggest(count=len(trial_ids), client_id=client_id)
        logging.info(f"worker_{worker_rank}: has suggested trials")
        
    trial_suggestion = None
    hparams = {}
    for i in range(len(trial_ids)):
        trial_id = trial_ids[i]
        if worker_rank == 0:
            trial_suggestion = suggested_trials[i]
            hparams = {k: v for k, v in trial_suggestion.parameters.items()}
            logging.info(f'suggested hparams: {hparams}')
        
        logging.info(f"worker_{worker_rank}: wait at barrier for trial_id={trial_id}")
        jax.experimental.multihost_utils.sync_global_devices(f"sync_barrier_for_trial_{{trial_id}}")
        logging.info(f"worker_{worker_rank}: passed barrier for trial_id={trial_id}")

        hparams = sync_hyperparams(hparams)
        
        logging.info(f"worker_{worker_rank}: synchronized params for trial_id={trial_id}")

        config2 = config.copy()
        config2.update(hparams)
        
        for k, v in config2.items():
            if k.find('?') > -1:
                logging.info(f"problem key from trial: {k}={v}")
        
        config2['trial_id'] = trial_id
        
        # NOTE: if have a comb of infeasible params or failure in which trial should not be
        # repeated, mark the trial using trial.infeasible() and continue w/o running train_fn
      
        # if worker_Rank !=0, then mlflow_run_id is ""
        best_val_ndcg_k, mlflow_run_id = train_fn(config2, trial=trial_suggestion, save_checkpoints=False)
        
        if worker_rank == 0:
            trial_suggestion.update_metadata(vz.Metadata({'mlflow_run_id': mlflow_run_id}))
            trial_suggestion.complete(vz.Measurement(metrics={f'ndcg_{config2["top_k"]}': float(best_val_ndcg_k)}))
        
def run_train(config):
   
    if "debug" in config and config['debug']:
        logging.info(f'train_run config: {config}')
    
    if "phase" not in config:
        logging.info("ERROR: expecting phase='train-best' or 'train-given'")
        return
    
    req_keys = {'user_embeddings_uri', 'movie_embeddings_uri', 'movies_uri',
        'recommendations_uri', 'recommendations_ts_uri',
        'ratings_train_liked_uri',
        'ratings_train_3_uri', 'ratings_train_disliked_uri',
        'ratings_val_liked_uri', 'ratings_val_3_uri',
        'ratings_val_disliked_uri',
        'seed'}
    for key in req_keys:
        if key not in config:
            raise LookupError(f'missing key {key} in config')
    
    worker_rank = jax.process_index()
    
    experiment_name = config.get('mlflow_experiment_name', config['study_name'])
    config['mlflow_experiment_name'] = experiment_name
    
    study = None
    if worker_rank == 0:
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        #create an ML-Flow parent study if it does not exist
        experiment = mlflow.get_experiment_by_name(name=experiment_name)
        if experiment is None:
            experiment = mlflow.set_experiment(experiment_name=experiment_name)
            # Create the parent run and immediately get its ID
            try:
                parent_run = mlflow.start_run(run_name="train", experiment_id=experiment.experiment_id)
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
                    parent_run = mlflow.start_run(run_name="train", experiment_id=experiment.experiment_id)
                    mlflow_parent_run_id = parent_run.info.run_id
                finally:
                    mlflow.end_run()
        config['mlflow_experiment_id'] = experiment.experiment_id
        config['mlflow_parent_run_id'] = mlflow_parent_run_id
    
    if config['phase'] == 'train-best':
        #worker==0 fetches the best parameters and then all workers synchronize to get best params
        best_params = {}
        if worker_rank == 0:
            best_params = get_best_parameters_for_training(config)
        best_params = sync_hyperparams(best_params)
        config.update(**best_params)
        
    best_val_ndcg_k, mlflow_run_id = train_fn(config, trial=None, save_checkpoints=True)

def get_best_parameters_for_training(config:Dict[str, Any]) -> Dict[str, Union[float, int]]:
    """
    get the best hyperparameter optimization (HPO) results given fonfig dictionary with keys "vizier_endpoint"
    "project_id", and "study_name"
    :param config:
    :return:
    """
    vz_clients.environment_variables.server_endpoint = config['vizier_endpoint']
    study = vz_clients.Study.from_owner_and_id(owner=config['project_id'],
        study_id=config['study_name'])
    optimal_trials = study.optimal_trials()
    if optimal_trials is None:
        raise ValueError(
            f"No optimal trials found for project_id={config['project_id']}, "
            f"study_name={config['study_name']}, endpoint={config['vizier_endpoint']}")
    best_trial = next(iter(optimal_trials), None)
    if best_trial is None:
        raise ValueError(f"No optimal trials found for project_id={config['project_id']},"
            f"study_name={config['study_name']}, endpoint={config['vizier_endpoint']}")
    best_trial_data = best_trial.materialize()
    # best_params contains only the params being tuned, not all params needed for train_fn
    best_params = extract_correct_vizier_param_types_dict( best_trial_data.parameters)
    return best_params

def get_best_checkpoint_uri_for_testing(config:Dict[str, Any]) -> str:
    """
    given a dictionary with keys "mlflow_tracking_uri", and "mlflow_experiment_name", find the
    latest train run's 'best_checkpoint_uri' tag and return it.
    :param config: a dictionary with keys "mlflow_tracking_uri", and "mlflow_experiment_name"
    :return: best_checkpoint_uri for latest train run of experiment having mlflow_experiment_name
    """
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    if 'mlflow_experiment_id' not in config:
        experiment = mlflow.get_experiment_by_name(name=config['mlflow_experiment_name'])
        if experiment is None:
            raise LookupError(f"Experiment {config['mlflow_experiment_name']} is not found")
        config['mlflow_experiment_id'] = experiment.experiment_id
    runs = mlflow.search_runs(
        experiment_ids=[config['mlflow_experiment_id']],
        filter_string="attributes.run_name LIKE 'train_%'",
        order_by=["attributes.end_time DESC"],
        max_results=1,
        output_format="list"
    )
    if runs is None or len(runs) == 0:
        raise ValueError(f"No runs found for train_* for MLFlow experiment name: {config['study_name']}")
    return runs[0].data.tags.get("best_checkpoint_uri")

def run_test(config):
    if "debug" in config and config['debug']:
        logging.info(f'test_run config: {config}')
    
    if "phase" not in config:
        logging.info("ERROR: expecting phase='test-best' or 'test-given'")
        return
    
    worker_rank = jax.process_index()
    
    if config['phase'] == 'test-best':
        #all worker ranks need this in order to get the checkpoint
        config['best_checkpoint_uri'] = get_best_checkpoint_uri_for_testing(config)
    
    experiment_name = config.get('mlflow_experiment_name', config['study_name'])
    config['mlflow_experiment_name'] = experiment_name
    
    study = None
    if worker_rank == 0:
        
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        # create an ML-Flow parent study if it does not exist
        experiment = mlflow.get_experiment_by_name(name=experiment_name)
        if experiment is None:
            experiment = mlflow.set_experiment(experiment_name=experiment_name)
            # Create the parent run and immediately get its ID
            try:
                parent_run = mlflow.start_run(run_name="test", experiment_id=experiment.experiment_id)
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
                    parent_run = mlflow.start_run(run_name="test", experiment_id=experiment.experiment_id)
                    mlflow_parent_run_id = parent_run.info.run_id
                finally:
                    mlflow.end_run()
        config['mlflow_experiment_id'] = experiment.experiment_id
        config['mlflow_parent_run_id'] = mlflow_parent_run_id
    
    test_metrics = test_fn(config=config)
        
    logging.info(f'TEST METRICS: {test_metrics}')
   
def run_export_results(config: Dict[str, Any]):
    """
    given a dictionary which includes vizier mappings: study_name, project_id, vizier_endpoint
    and MLFlow mappings: mlflow_tracking_uri, extract the best found hyperparameters from the
    HPO tuning and the resulting metrics and write to the given output uris output_hyperparams_uri and output_metrics_uri.
    Note that the files will be json dictionaries.
    :param config: a dictionary which includes vizier mappings: study_name, project_id, vizier_endpoint
    and MLFlow mappings: mlflow_tracking_uri
    :param output_hyperparams_uri: uri to write the best found hyper-parameters to
    :param output_metrics_uri: uri to write the metrics dictionary from the best-hyper parameters run.
    """
    
    for key in ("study_name", "project_id", "vizier_endpoint", "mlflow_tracking_uri", "output_hyperparams_uri", "output_metrics_uri"):
        if key not in config:
            raise ValueError(f"Missing key {key} in config in run_export_results")
    
    phase = config["phase"]
    logging.info(f'run {phase}')
    
    STUDY_NAME = config["study_name"]
    project_id = config['project_id']
    
    if phase == "export-hpo-results":
        pass
    elif phase == "export-train-results":
        srch = 'train_%'
    elif phase == "export-test-results":
        srch = 'test_%'
    else:
        raise ValueError(f"unrecognized phase for run_export_results: {phase}")
    
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    
    metrics_dict = {}
    
    if phase == "export-hpo-results":
        vz_clients.environment_variables.server_endpoint = config['vizier_endpoint']
        logging.info(f'looking for study_name {STUDY_NAME} at endpoint {config["vizier_endpoint"]} for phase={phase}')
        # resource_name = f"owners/{project_id}/studies/{STUDY_NAME}"
        
        study = vz_clients.Study.from_owner_and_id(owner=project_id, study_id=STUDY_NAME)
        
        optimal_trials = study.optimal_trials()
        best_trial = next(iter(optimal_trials), None)
        best_trial_data = best_trial.materialize()
        #best_params contains only the params being tuned, not all params needed for train_fn
        best_params = extract_correct_vizier_param_types_dict(best_trial_data.parameters)
        #logging.info("Available metrics:", list(best_trial_data.final_measurement.metrics.keys()))
        bfm = best_trial_data.final_measurement
        bfm = bfm.metrics.get(f'ndcg_20')
        best_value = bfm.value
    
        logging.info(f"Loaded Best Objective: {best_value}")
        logging.info(f"Loaded Best Parameters: {best_params}")
        
        # run_uuid is this:
        mlflow_run_id = best_trial_data.metadata.get('mlflow_run_id')
        if "debug" in config and config["debug"]:
            logging.info(f"mlflow_run_id={mlflow_run_id}")
        mlflow_run = mlflow.get_run(mlflow_run_id)
        hparams = destringify_mlflow_params(mlflow_run.data.params)
        hparams_json = json.dumps(hparams, indent=4, sort_keys=True)
        try:
            with fsspec.open(config['output_hyperparams_uri'], mode="w") as f:
                f.write(hparams_json)
        except Exception as e:
            logging.exception(
                f'ERROR while trying to write to {config["output_hyperparams_uri"]}: {e}.  Check for permission errors')
            raise e
    else:
        experiment = mlflow.get_experiment_by_name(config['mlflow_experiment_name'])
        if experiment is None:
            raise LookupError(f"Experiment {config['mlflow_experiment_name']} not found.")
        #default ordering is descending start_time, run_id
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.run_name LIKE '{srch}'",
            output_format="list"
        )
        if runs is None or len(runs) == 0:
            raise LookupError(f"No runs found for experiment_name={config['mlflow_experiment_name']} and run ={srch}")
        mlflow_run = runs[0]
        mlflow_run_id = mlflow_run.info.run_id
        if phase == "export-test-results":
            #there is only metrics associated with the run data for test phases
            metrics_dict = mlflow_run.data.metrics
   
    if phase != "export-test-results":
        #get the metrics:
        mlflow_client = MlflowClient(tracking_uri=config['mlflow_tracking_uri'])
        for key in ("loss", "ndcg_20", "recall_20", "mrr_20"):
            for key_t in (f"train_{key}", f"val_{key}"):
                metrics_dict[key_t] = {'x': [], 'y': []}
                m_dict = mlflow_client.get_metric_history(mlflow_run_id, key=key_t)
                for m in m_dict:
                    metrics_dict[key_t]['x'].append(int(m.step))
                    metrics_dict[key_t]['y'].append(float(m.value))
    
    metrics_json = json.dumps(metrics_dict, indent=4, sort_keys=True)
    try:
        with fsspec.open(config['output_metrics_uri'], mode="w") as f:
            f.write(metrics_json)
    except Exception as e:
        logging.exception(f'ERROR while trying to write to {config["output_metrics_uri"]}: {e}.  Check for permission errors')
        raise e

def _jax_sees_gpus() -> bool:
    gpu_count = 0
    gpus_expected = 0
    for device in jax.local_devices():
        if device.platform == 'gpu':
            gpu_count += 1
            if device.device_kind is not None:
                if device.device_kind.lower().find('t4') > -1:
                    gpus_expected = 2
                elif device.device_kind.lower().find('p100') > -1:
                    gpus_expected = 1
    return (gpu_count > 0 and gpus_expected == gpu_count)
    
def _vizier_connection(vizier_endpoint:str) -> bool:
    """
    check the vizier connection by creating a deleting a study
    :param config:
    :return: True if no exception raised
    """
    vz_clients.environment_variables.server_endpoint = vizier_endpoint
    problem = vz.ProblemStatement()
    study_config = vz.StudyConfig.from_problem(problem)
    study = vz_clients.Study.from_study_config(study_config,
        owner='tmp', study_id='tmp')
    study.delete()
    return True

def _mlflow_connection(mlflow_tracking_uri:str) -> bool:
    """
    check that can connect to MLFlow server
    :param mlflow_tracking_uri:
    :return: True if no exception raised
    """
    from mlflow.protos.service_pb2 import ViewType
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    client.search_experiments(view_type = ViewType.ACTIVE_ONLY, max_results=1)
    return True

def _fake_gcs_server_connection(config) -> bool:
    """
    check the fake gcs server connection
    :param config:
    :return: True if exception is not throw and a data file is successfully read
    """
    from movie_lens_ranker.util import read_movies_array_record
    gs_uri = "gs://data/movies-00000-of-00001.array_record"
    emb = read_movies_array_record(gs_uri, batch_size=1024)
    return emb is not None and len(emb) > 0
    
def connections_check(config):
    
    if os.environ.get('JAX_PLATFORM_NAME', '').lower().find('gpu') > -1:
        if not _jax_sees_gpus():
            raise EnvironmentError(f"could not find expected GPUs: {jax.devices()}")
    
    if jax.process_index() == 0:
        _vizier_connection(config['vizier_endpoint'])
        _mlflow_connection(config['mlflow_tracking_uri'])
        _fake_gcs_server_connection(config)
        logging.info('passed connections check')

def main(_):
    config = FLAGS.flag_values_dict()
    
    if "debug" in config and config['debug']:
        logging.info(f'all args received from flags: {config}')
    
    config = {k:v for k, v in config.items() if k in get_recognized_keys()}
    
    if "connections_check" in config and int(config["connections_check"])==1:
        connections_check(config)
        return
    
    if app_runner_is_missing_minimum_required_keys(config):
        logging.info(f'warning: missing the minimum required flags')
        return
    
    # work-around for xmanager encapsulating the json dumps string of array with extra quotes
    if "trial_ids" in config:
        # idempotent if doesn't have additional quotes:
        config['trial_ids'] = config['trial_ids'].strip("'").strip('"')
    
    if "debug" in config and config['debug']:
        logging.info(f'recognized args: {config}')
        
    # static top_k is throughout code
    config['top_k'] = 20
    
    logging.info(f'jax_process_index={jax.process_index()}; '
          f'jax.local_devices={jax.local_devices()}; '
          f'jax.devices={jax.devices()}, phase={config["phase"]}')

    if config['phase'] == 'tune':
        run_tune(config)
    elif config['phase'].find('test') == 0:
        run_test(config)
    elif config['phase'].find('train') == 0:
        run_train(config)
    elif config['phase'].find('export') == 0:
        run_export_results(config)
    else:
        raise ValueError('unknown phase: {config["phase"]}')