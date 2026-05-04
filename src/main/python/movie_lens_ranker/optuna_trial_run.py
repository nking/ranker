import jax
import mlflow
from absl import flags
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler

from movie_lens_ranker.train import train_fn, test_fn, get_optuna_suggestions
from movie_lens_ranker.util import get_args_parser, define_flags

FLAGS = flags.FLAGS

import optuna
from absl import app
from optuna import Trial

import urllib.request
import time
import sys

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
    
def main(_):
    """
    set-up optuna trial
    :param _:
    :return:
    """
    config = FLAGS.flag_values_dict()
    
    if "debug" in config and config['debug']:
        print(f'args received: {config}', flush=True)
    
    if "phase" not in config:
        print("ERROR: expecting phase='test' or other such as 'train', 'tune/train'")
        return
    
    if config['phase'] == 'test':
        return test_fn(config)
    
    worker_rank = jax.process_index()
    
    study = None
    if worker_rank == 0:
        
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        #create an ML-Flow parent study if it does not exist
        experiment = mlflow.get_experiment_by_name(name=config['study_name'])
        if experiment is None:
            experiment = mlflow.set_experiment(experiment_name=config['study_name'])
            # Create the parent run and immediately get its ID
            parent_run = mlflow.start_run(run_name="Optuna_HPO")
            mlflow_parent_run_id = parent_run.info.run_id
            mlflow.end_run()
        else:
            #get parent run id:
            runs = mlflow.search_runs(
                experiment_names=[config['study_name']],
                filter_string="attributes.run_name = 'Optuna_HPO'",
                output_format="list"
            )
            if runs:
                mlflow_parent_run_id = runs[0].info.run_id
            else:
                parent_run = mlflow.start_run(run_name="Optuna_HPO")
                mlflow_parent_run_id = parent_run.info.run_id
                mlflow.end_run()
        config['mlflow_experiment_id'] = experiment.experiment_id
        config['mlflow_parent_run_id'] = mlflow_parent_run_id
        
        # Initialize the optuna study in the database if doesn't already exist
        study = optuna.create_study(
            study_name=config['study_name'],
            storage=config['optuna_storage_uri'],
            sampler=RandomSampler(),
            pruner=MedianPruner(),
            direction="maximize",
            load_if_exists=True
        )
    
    # Connect to the study created by the launcher
    if study is None:
        study = optuna.load_study(
            study_name=config['study_name'],
            storage=config['optuna_storage_uri'],
            sampler=RandomSampler(),
            pruner=MedianPruner(),
        )
    
    # Optuna's DB locking ensures each container gets unique params
    trial: Trial = study.ask()
    config.update(trial.params)
    
    config['mlflow_experiment_name'] = config['study_name']
    config['mlflow_experiment_id'] = get_or_create_mlflow_experiment(config['mlflow_experiment_name'])
    
    #NOTE: this is specific to disks and assumes have permission to mkdir...wold be different for cloud storage
    #append trial id to uris:
    config['best_checkpoint_uri'] = f"{config['best_checkpoint_uri']}/{config['study_name']}/trial_{config['trial_id']}"
    config['latest_checkpoint_uri'] = f"{config['latest_checkpoint_uri']}/{config['study_name']}/trial_{config['trial_id']}"
    
    # get trial suggestions
    optuna_params = get_optuna_suggestions(trial)
    config.update(optuna_params)
    
    print(f'begin train_fn')
    
    best_val_ndcg_k, STATE = train_fn(config, trial)
    
    study.tell(trial, values=float(best_val_ndcg_k), state=STATE)
    
if __name__ == '__main__':
    define_flags()
    app.run(main)
    