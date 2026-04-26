#see https://github.com/google-deepmind/xmanager/tree/v0.7.1/examples/vizier
# see gemini notes https://gemini.google.com/app/d3df1bcf5a80bc26
# see https://github.com/google-deepmind/xmanager/tree/v0.7.1/examples/dockerfile
import socket

import mlflow
import xmanager as xm
from optuna import create_study, load_study
from absl import flags
import os
import shutil
import sys
import subprocess

MLFLOW_DIR = os.path.join(os.getcwd(), "bin", "mlflow")

def get_best_config(study_name, storage_url):
    # Re-connect to the existing study
    study = load_study(study_name=study_name, storage=storage_url)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (NDCG): {study.best_value}")
    # Return the winning params
    return study.best_trial.params, study.best_trial.number

def main():
    
    STUDY_NAME = "GraphRanker_tuning_xmgr"
    
    # Initialize the optuna study in the database
    # This just "reserves the name" in your Postgres/MySQL DB
    create_study(
        study_name=STUDY_NAME,
        storage="postgresql://user:pass@host:5432/db", 
        load_if_exists=True
    )
    
    #init mlflow experiment
    try:
        mlflow.delete_experiment(STUDY_NAME)
    except Exception as e:
        print(f'error while deleting experiment: {e}')
    mlflow.set_experiment(STUDY_NAME)
    # Create the parent run and immediately get its ID
    parent_run = mlflow.start_run(run_name="Optuna_HPO")
    mlflow_parent_run_id = parent_run.info.run_id
    mlflow.end_run()
    
    # Tell XManager to launch N parallel trials
    # Each trial is a separate Docker container instance
    with xm.create_experiment(experiment_name='gatv2_search') as experiment:
        # Define your JAX training executable
        executable, = experiment.package([
            xm.python_executable(
                path='.',
                entrypoint=xm.ModuleName('optuna_trial_run'), # Points to optuna_trial_run.py
                docker_instructions=[...] 
            )
        ])
        
        # Launch 20 parallel trials
        hpo_jobs = [experiment.add(
            experiment.add(xm.Job(
                executable=executable,
                # args are read as FLAGs by optuna_trial_run
                args={
                    'study_name': STUDY_NAME,
                    'storage_url': "postgresql://user:pass@host:5432/db",
                    'phase': 'train/tune',
                    'mlflow_parent_run_id': mlflow_parent_run_id
                }
            ))) for _ in range(20)]
        
        # blocks launcher.py until all HPO trial workers exit
        experiment.wait_for_jobs(*hpo_jobs)
        
        FLAGS = flags.FLAGS
        best_params, best_trial_id = get_best_config(FLAGS.study_name, FLAGS.storage_url)
        
        # We pass 'phase=test' so train.py knows to run test_fn
        experiment.add(xm.Job(
            executable=executable,
            args={
                **FLAGS.flag_values_dict(),  # Global defaults
                **best_params,  # Winning overrides
                'phase': 'test',  # Signal to run evaluation
                'load_checkpoint_id': best_trial_id,
            }
        ))
        
