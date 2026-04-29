#see https://github.com/google-deepmind/xmanager/tree/v0.7.1/examples/vizier
# see gemini notes https://gemini.google.com/app/d3df1bcf5a80bc26
# see https://github.com/google-deepmind/xmanager/tree/v0.7.1/examples/dockerfile
import socket

import mlflow
import xmanager as xm
from optuna import create_study, load_study
from absl import flags
import os

from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler

MLFLOW_DIR = os.path.join(os.getcwd(), "bin", "mlflow")

def get_best_config(study_name, storage_url):
    # Re-connect to the existing study
    study = load_study(study_name=study_name, storage=storage_url)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (NDCG): {study.best_value}")
    # Return the winning params
    return study.best_trial.params, study.best_trial.number

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

def get_bin_dir() -> str:
  return os.path.join(get_project_dir(), "bin")

def main():
    
    STUDY_NAME = "GraphRanker_tuning_xmngr"
    optuna_db_path = os.path.join(get_bin_dir(), f"{STUDY_NAME}.db")
    optuna_storage_uri = f"sqlite:///{optuna_db_path}?mode=memory&cache=shared"
    if os.path.exists(optuna_db_path):
        os.remove(optuna_db_path)
        print(f"Deleted old database at {optuna_db_path}")
    
    # Initialize the optuna study in the database
    # This just "reserves the name" in your Postgres/MySQL DB
    create_study(
        study_name=STUDY_NAME,
        storage=optuna_storage_uri,
        sampler=RandomSampler(),
        pruner=MedianPruner(),
        direction="maximize",
        load_if_exists=True
    )
    
    #init mlflow experiment
    try:
        exp_id = mlflow.get_experiment_by_name(STUDY_NAME)
        if exp_id is not None:
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
                docker_instructions=[
                    'FROM python:3.11-slim AS builder',
                    'WORKDIR /app',
                    'RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential',
                    'COPY --from=builder /root/.local /root/.local',
                    'ENV PATH=/root/.local/bin:$PATH',
                    'COPY ../src/main/python/movie_lens_ranker ./src/main/python/movie_lens_ranker',
                    'COPY ../pyproject.toml ./',
                    'RUN pip install --user --no-cache-dir -e .',
                ]
            )
        ])
        
        # Launch 20 parallel trials
        hpo_jobs = [experiment.add(
            experiment.add(xm.Job(
                executable=executable,
                # args are read as FLAGs by optuna_trial_run
                args={
                    'study_name': STUDY_NAME,
                    'optuna_storage_uri': optuna_storage_uri,
                    'phase': 'train/tune',
                    'mlflow_parent_run_id': mlflow_parent_run_id,
                    'movies_uri': movies_uri,
                    'recommendations_uri': recommendations_uri,
                    'recommendations_ts_uri': recommendations_ts_uri,
                    'ratings_train_uri': ratings_train_uri,
                    'ratings_val_uri': ratings_val_uri,
                    'train_negatives_uri': train_negatives_uri,
                    'val_negatives_uri': val_negatives_uri,
                    'latest_checkpoint_dir': latest_checkpoint_dir,
                    'best_checkpoint_dir': best_checkpoint_dir,
                    'movie_embeddings_uri': movie_embeddings_uri,
                    'user_embeddings_uri': user_embeddings_uri,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'seed': seed,
                    "trial_id": 1,
                    'mlflow_tracking_uri': mlflow_dir,
                    'mlflow_registry_uri': mlflow_registry_dir,
                    'mlflow_experiment_id': mlflow.get_experiment_by_name(STUDY_NAME),
                    'mlflow_experiment_name': STUDY_NAME,
                    # 'mlflow_tracking_token': None,
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
        
