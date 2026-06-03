import os
import logging
from typing import Union, Dict, Any

## NOTE: must be run in a venv that contains vizier

import mlflow
from mlflow import MlflowClient
from vizier._src.pyvizier.shared.trial import ParameterValue, ParameterDict
from vizier.service import pyvizier as vz
from vizier.service import clients as vz_clients
import numpy as np
from dotenv import dotenv_values
from absl import flags
import json

import glob
import os.path

import psycopg2
import time

import jax.distributed
from array_record.python import array_record_module

from absl import flags

from movie_lens_ranker.util import destringify_mlflow_params

#found by ip addr show docker0
#base_url = "172.17.0.1"

def get_project_dir() -> str:
  cwd = os.getcwd()
  head = cwd
  proj_dir = ""
  while head and head != os.sep:
    head, tail = os.path.split(head)
    if tail:  # Add only if not an empty string (e.g., from root or multiple separators)
      if tail == "ranker":
        proj_dir = os.path.join(head, tail)
        break
  return proj_dir

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
    
def main():
    
    config = {}
    # === these are so that grain dataloader can read data from fake gcs server running in docker ====
    env_file = os.path.join(os.getcwd(), "run_config.env")
    for k, v in dotenv_values(env_file).items():
        v = v.replace("vizier-service:", "172.17.0.1:")
        v = v.replace("db-service:", "172.17.0.1:")
        #os.environ[k] = v
        config[k] = v
    env_file = os.path.join(get_project_dir(), ".env")
    for k, v in dotenv_values(env_file).items():
        v = v.replace("gcs:", "172.17.0.1:")
        #os.environ[k] = v
        config[k] = v
        
    STUDY_NAME = config["study_name"]
    project_id = config['project_id']
        
    vz_clients.environment_variables.server_endpoint = config['vizier_endpoint']
    print(f'looking for study_name {STUDY_NAME} at endpoint {config["vizier_endpoint"]}', flush=True)
    resource_name = f"owners/{project_id}/studies/{STUDY_NAME}"

    study = vz_clients.Study.from_owner_and_id(owner=project_id, study_id=STUDY_NAME)
    
    optimal_trials = study.optimal_trials()
    best_trial = None
    for tr in optimal_trials:
        best_trial = tr
        break
    best_trial_data = best_trial.materialize()
    #best_params contains only the params being tuned, not all params needed for train_fn
    best_params = extract_correct_vizier_param_types_dict(best_trial_data.parameters)
    print("Available metrics:", list(best_trial_data.final_measurement.metrics.keys()), flush=True)
    bfm = best_trial_data.final_measurement
    bfm = bfm.metrics.get(f'ndcg_20')
    best_value = bfm.value
        
    print(f"Loaded Best Objective: {best_value}")
    print(f"Loaded Best Parameters: {best_params}")
    
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    
    #mlflow table called runs has columns:
    #run_uuid | name | source_type | source_name | entry_point_name | user_id | status | start_time | end_time | source_version | lifecycle_stage | artifact_uri | experiment_id | deleted_time
    
    #run_uuid is this:
    mlflow_run_id = best_trial_data.metadata.get('mlflow_run_id')
    mlflow_run = mlflow.get_run(mlflow_run_id)
    
    config2 = destringify_mlflow_params(mlflow_run.data.params)
    
    print(f'\nconfig:')
    print(json.dumps(config2, indent=4, sort_keys=True))
    
    mlflow_client = MlflowClient(tracking_uri=config['mlflow_tracking_uri'])
    
    metrics_dict = {}
    for key in ("loss", "ndcg_20", "recall_20", "mrr_20"):
        for key_t in (f"train_{key}", f"val_{key}"):
            metrics_dict[key_t] = {'x': [], 'y': []}
            m_dict = mlflow_client.get_metric_history(mlflow_run_id, key=key_t)
            for m in m_dict:
                metrics_dict[key_t]['x'].append(int(m.step))
                metrics_dict[key_t]['y'].append(float(m.value))
    
    print(f'\nmetrics:')
    print(json.dumps(metrics_dict, indent=4, sort_keys=True))
    
if __name__ == '__main__':
    main()