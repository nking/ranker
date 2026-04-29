#see https://github.com/google-deepmind/xmanager/tree/v0.7.1/examples/vizier
# see gemini notes https://gemini.google.com/app/d3df1bcf5a80bc26
# see https://github.com/google-deepmind/xmanager/tree/v0.7.1/examples/dockerfile
import socket
from typing import Dict, Tuple, Sequence
from absl import app

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

def get_train_val_test_liked_uris(use_small: bool = True) -> Tuple[str, str, str]:
    base_dir = os.path.join(get_project_dir(), "src/test/resources/data/")
    if use_small:
        base_dir = os.path.join(base_dir, "small")
    return (os.path.join(base_dir, "ratings_train_liked.array_record"),
        os.path.join(base_dir, "ratings_val_liked.array_record"),
        os.path.join(base_dir, "ratings_test_liked.array_record"))

def get_train_val_test_disliked_uris(use_small: bool = True) -> Tuple[
    str, str, str]:
    base_dir = os.path.join(get_project_dir(), "src/test/resources/data/")
    if use_small:
        base_dir = os.path.join(base_dir, "small")
    return (os.path.join(base_dir, "ratings_train_disliked.array_record"),
        os.path.join(base_dir, "ratings_val_disliked.array_record"),
        os.path.join(base_dir, "ratings_test_disliked.array_record"))

def get_args_dict() -> Dict:
    ratings_train_uri, ratings_val_uri, ratings_test_uri \
        = get_train_val_test_liked_uris(use_small=True)
    ratings_train_disliked_uri, ratings_val_disliked_uri, ratings_test_disliked_uri \
        = get_train_val_test_disliked_uris(use_small=True)
    return {
        'ratings_train_uri':ratings_train_uri,
        'ratings_val_uri':ratings_val_uri,
        'ratings_train_disliked_uri' : ratings_train_disliked_uri,
        'ratings_val_disliked_uri':ratings_val_disliked_uri,
        'movie_embeddings_uri' : os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record"),
        'user_embeddings_uri' : os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb-00000-of-00001.array_record"),
        'recommendations_uri' :  os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record"),
        'recommendations_ts_uri': os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies_timestamps.array_record"),
        'movies_uri' : os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record"),
        'train_negatives_uri' : os.path.join(get_project_dir(),
            "src/test/resources/data/train_negatives.array_record"),
        'val_negatives_uri' : os.path.join(get_project_dir(),
            "src/test/resources/data/val_negatives.array_record"),
        'test_negatives_uri' : os.path.join(get_project_dir(),
            "src/test/resources/data/test_negatives.array_record"),
        'train_val_negatives_uri' : os.path.join(get_project_dir(),
            "src/test/resources/data/train_val_negatives.array_record"),
        'train_val_test_negatives_uri' :  os.path.join(get_project_dir(),
            "src/test/resources/data/train_val_test_negatives.array_record"),
        'num_epochs' : 100,
        'batch_size' : 64,
        'seed' : 1234,
        'mlflow_dir' : os.path.join(get_bin_dir(), "mlflow"),
        'mlflow_registry_dir': os.path.join(get_bin_dir(), "mlflow_registry"),
    }

def main(argv: Sequence[str]):
    del argv
    
    STUDY_NAME = "GraphRanker_tuning_xmngr"
    optuna_db_path = os.path.join(get_bin_dir(), f"{STUDY_NAME}.db")
    optuna_storage_uri = f"sqlite:///{optuna_db_path}"
    
    print(f'xmanager for {STUDY_NAME} HPO')
    
    if os.path.exists(optuna_db_path):
        os.remove(optuna_db_path)
        print(f"Deleted old database at {optuna_db_path}")
    
    # Initialize the optuna study in the database
    # This just "reserves the name" in your Postgres/MySQL DB
    print(f'create optuna study {STUDY_NAME}')
    create_study(
        study_name=STUDY_NAME,
        storage=optuna_storage_uri,
        sampler=RandomSampler(),
        pruner=MedianPruner(),
        direction="maximize",
        load_if_exists=True
    )
    
    xm_exp_name = 'gatv2_search'
    print(f'create XManager experiment {xm_exp_name}', flush=True)

    # Tell XManager to launch N parallel trials
    # Each trial is a separate Docker container instance
    with xm.create_experiment(experiment_name=xm_exp_name) as experiment:
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
        
        job_args = get_args_dict()
        
        num_trials = 2 #20
        
        print(f'add HPO jobs to XManager experiment', flush=True)
        
        # Launch num_trial parallel trials
        hpo_jobs = [experiment.add(
            experiment.add(xm.Job(
                executable=executable,
                # args are read as FLAGs by optuna_trial_run
                args={
                    'study_name': STUDY_NAME,
                    'optuna_storage_uri': optuna_storage_uri,
                    'phase': 'train/tune',
                    'movies_uri': job_args['movies_uri'],
                    'recommendations_uri': job_args['recommendations_uri'],
                    'recommendations_ts_uri': job_args['recommendations_ts_uri'],
                    'ratings_train_uri': job_args['ratings_train_uri'],
                    'ratings_val_uri': job_args['ratings_val_uri'],
                    'train_negatives_uri': job_args['train_negatives_uri'],
                    'val_negatives_uri': job_args['val_negatives_uri,'],
                    'latest_checkpoint_dir': job_args['latest_checkpoint_dir'],
                    'best_checkpoint_dir': job_args['best_checkpoint_dir'],
                    'movie_embeddings_uri': job_args['movie_embeddings_uri'],
                    'user_embeddings_uri': job_args['user_embeddings_uri'],
                    'num_epochs': job_args['num_epochs'],
                    'batch_size': job_args['batch_size'],
                    'seed': job_args['seed'],
                    "trial_id": 1,
                    'mlflow_tracking_uri': job_args['mlflow_dir'],
                    'mlflow_registry_uri': job_args['mlflow_registry_dir'],
                    'mlflow_experiment_name': STUDY_NAME,
                    # 'mlflow_tracking_token': None,
                }
            ))) for _ in range(num_trials)]
        
        # blocks launcher.py until all HPO trial workers exit
        experiment.wait_for_jobs(*hpo_jobs)
        
        print(f'HPO done', flush=True)

        FLAGS = flags.FLAGS
        best_params, best_trial_id = get_best_config(FLAGS.study_name, FLAGS.storage_url)
        
        print(f'best config: trial_id={best_trial_id},\nparams=\n{best_params}', flush=True)

        #TODO: edit this
        '''
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
        '''
        
if __name__ == '__main__':
  app.run(main)