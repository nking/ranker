import datetime
import os
import logging
#to test for multiple devices before using on GPUs or TPUs:
#os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

# Force Python to spawn clean workers instead of cloning the GPU context.
import multiprocessing as mp
import os
import logging

from movie_lens_ranker.util_np import optimized_batch_and_pad


def init_multiprocessing():
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    try:
        mp.get_logger().setLevel(logging.DEBUG)
    except Exception:
        pass
    
    # If CUDA_VISIBLE_DEVICES is exactly "", we know this process is a Grain child worker
    # (because our wrapper in data_loading.py temporarily set it to "" before spawning).
    # We must remove JAX_PLATFORM_NAME=gpu from the child's environment so JAX doesn't
    # crash when it gracefully falls back to the CPU.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        os.environ.pop("JAX_PLATFORM_NAME", None)
    
    # Ensure PYTHONPATH is inherited by child processes
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = ":".join(sys.path)


init_multiprocessing()

import jax

def safe_jax_init():
    try:
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

import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*'\.value' access is now deprecated\..*"
)
import fsspec
import gcsfs
from mlflow import MlflowClient, config
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

from helper import *
from movie_lens_ranker.train import *
from movie_lens_ranker.util import set_flags_from_dict, \
    destringify_mlflow_params
from movie_lens_ranker.util_plots import plot_mlflow_metrics, \
    get_mlflow_metrics_by_exp_name

from movie_lens_ranker.app_runner_inner import main as app_runner
from movie_lens_ranker.app_runner_inner import extract_correct_vizier_param_types_dict, \
    get_best_checkpoint_uri_for_testing, get_best_parameters_for_training

import unittest
import subprocess
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)], # Route to stdout to avoid red text in PyCharm
    force=True # CRITICAL: Overrides any logging setups initialized by jax, grain, or mlflow
)

#found by ip addr show docker0
base_url = "172.17.0.1"

"""
this uses the docker compose-dbs.yaml fake-gcs-server and db (==postgres server)

the gcs uris are  formatted to gs://bucket_name/...

the postgres uris use postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
where netlo is 172.17.0.1

to start the fake gcs server and the postgres db:
    docker compose --project-directory . -f deploy/compose/docker-compose-dbs.yaml up -d
"""

def wait_for_postgres_vizier_mlflow_dbs(retries=5, delay=2):
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    s = [False, False]
    for ii, db in enumerate(["mlflow_db"]):
        dsn = f"host={base_url} user={user} password={password} dbname={db}"
        for i in range(retries):
            try:
                conn = psycopg2.connect(dsn)
                conn.close()
                s[ii] = True
                break
            except psycopg2.OperationalError as ex:
                print(
                    f"Database {db} not ready, retrying in {delay}s... ({i + 1}/{retries});  ex={ex}")
                time.sleep(delay)
    return s[0]==True and s[1]==True
    
def reset_mlflow_db():
    # The SQL command
    truncate_query = """
    TRUNCATE TABLE
        experiments,
        experiment_tags,
        datasets,
        endpoints,
        assessments
    CASCADE;
    """
    container_name = "local_db_store"
    db = "mlflow_db"
    env_file = os.path.join(get_project_dir(), ".env_unittests")
    env_dict = dotenv_values(env_file)
    
    # Construct the docker command
    command = [
        "docker", "exec", "-t", container_name,
        "psql", "-U", env_dict['POSTGRES_USER'], "-w", env_dict['POSTGRES_PASSWORD'],
        "-d", db, "-c", truncate_query
    ]
        
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"MLFlow Database reset successful")
    except subprocess.CalledProcessError as e:
        print(f"Error resetting database: {e.stderr}")

def reset_checkpoint_buckets():
    command = [
        "docker", "exec", "gcs_emulator",
        "sh", "-c", "rm -rf /storage/checkpoint-bucket/*"
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        print("empty checkpoint-bucket/* successful")
    except subprocess.CalledProcessError as e:
        print(f"Error resetting database: {e.stderr}")
        
def reset_hpo_results_bucket(project_id:str, study_name:str):
    command = [
        "docker", "exec", "gcs_emulator",
        "sh", "-c", f"rm -rf /storage/hpo-results-bucket/{project_id}/{study_name}"
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        print("empty checkpoint-bucket/* successful")
    except subprocess.CalledProcessError as e:
        print(f"Error resetting database: {e.stderr}")

class TestRanker(unittest.TestCase):
    
    def setUp(self):
        
        # === these are so that grain dataloader can read data from fake gcs server running in docker ====
        env_file = os.path.join(get_project_dir(), ".env_unittests")
        for k, v in dotenv_values(env_file).items():
            os.environ[k] = v
        
        ratings_uri_dict = get_train_val_test_liked_uris(data_size=DataSize.TINY, use_gcs_uri=True)
        
        self.ratings_train_liked_uri = ratings_uri_dict["train_liked"]
        self.ratings_val_liked_uri = ratings_uri_dict["val_liked"]
        self.ratings_test_liked_uri = ratings_uri_dict["test_liked"]
        
        self.ratings_train_3_uri = ratings_uri_dict["train_3"]
        self.ratings_val_3_uri = ratings_uri_dict["val_3"]
        self.ratings_test_3_uri = ratings_uri_dict["test_3"]
        
        self.ratings_train_disliked_uri = ratings_uri_dict["train_disliked"]
        self.ratings_val_disliked_uri = ratings_uri_dict["val_disliked"]
        self.ratings_test_disliked_uri = ratings_uri_dict["test_disliked"]
        
        # (movie_id, float array of embed_dim as a tuple)
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        
        # (user_id, float array of embed_dim as a tuple)
        self.user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb-00000-of-00001.array_record")
        
        # (user_id, int array of movie_ids as a tuple) is full catalog for each user, no history subtracted
        self.recommendations_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record")
        self.recommendations_ts_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies_timestamps.array_record")
        
        # (movie_id, title, genres)
        self.movies_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")
        
        STUDY_NAME = "GraphRanker_tuning_unittest3"
        
        num_epochs = 4  # keep this to > 2 and < 10 for the restore tests at end of this method
        batch_size = 64
        seed = 234
        
        # tensorstore keeps trying to authenticate with google so for tests we'll use the abs path to checkpoint dir
        checkpoint_dir = 'gs://checkpoint-bucket'
        latest_checkpoint_uri = f'{checkpoint_dir}/latest'
        best_checkpoint_uri = f'{checkpoint_dir}/best'
        
        # mflow_db_path = os.path.join(get_bin_dir(), f"{STUDY_NAME}_mlflow.db")
        # mflow_uri = f"sqlite:///{mflow_db_path}?mode=memory&cache=shared"
        ##    postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
        vizier_endpoint = f'{base_url}:8000'
        mlflow_uri = f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{base_url}:5432/mlflow_db"
        
        self.config = {
            'movies_uri': self.transform_to_gs_uri(self.movies_uri),
            'recommendations_uri': self.transform_to_gs_uri(
                self.recommendations_uri),
            'recommendations_ts_uri': self.transform_to_gs_uri(
                self.recommendations_ts_uri),
            
            'ratings_train_liked_uri': self.transform_to_gs_uri(
                self.ratings_train_liked_uri),
            'ratings_train_3_uri': self.transform_to_gs_uri(
                self.ratings_train_3_uri),
            'ratings_train_disliked_uri': self.transform_to_gs_uri(
                self.ratings_train_disliked_uri),
            
            'ratings_val_liked_uri': self.transform_to_gs_uri(
                self.ratings_val_liked_uri),
            'ratings_val_3_uri': self.transform_to_gs_uri(
                self.ratings_val_3_uri),
            'ratings_val_disliked_uri': self.transform_to_gs_uri(
                self.ratings_val_disliked_uri),
            
            'ratings_test_liked_uri': self.transform_to_gs_uri(
                self.ratings_test_liked_uri),
            'ratings_test_3_uri': self.transform_to_gs_uri(
                self.ratings_test_3_uri),
            'ratings_test_disliked_uri': self.transform_to_gs_uri(
                self.ratings_test_disliked_uri),
            
            'movie_embeddings_uri': self.transform_to_gs_uri(
                self.movie_embeddings_uri),
            'user_embeddings_uri': self.transform_to_gs_uri(
                self.user_embeddings_uri),
            
            'latest_checkpoint_uri': latest_checkpoint_uri,
            'best_checkpoint_uri': best_checkpoint_uri,
            
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'seed': seed,
            'study_name': STUDY_NAME,
            'project_id': 'tune-unittest-02',
            "trial_ids": json.dumps([0, 1]),
            'phase': 'tune',
            'top_k': 20,
            'vizier_endpoint': vizier_endpoint,
            'mlflow_tracking_uri': mlflow_uri,
            'mlflow_experiment_name': STUDY_NAME,
        }
        
        # check that docker fake gcs server is running
        try:
            response = requests.get(
                f"http://{base_url}:4443/storage/v1/b/data/o")
            if response.status_code == 200:
                data = response.json()
                print(data)
                self.assertTrue(len(data['items']) > 0)
            else:
                print(f"Failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                print(f'is the fake-gcs-server container not running?')
                return
            wait_for_postgres_vizier_mlflow_dbs()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return
        
        config = self.config
        
        set_flags_from_dict(config)
        
        # reset oss vizier db
        try:
            vz_clients.environment_variables.server_endpoint = config[
                'vizier_endpoint']
            resource_name = f"owners/{config['project_id']}/studies/{config['study_name']}"
            study = vz_clients.Study.from_owner_and_id(
                owner=config['project_id'],
                study_id=config['study_name'])
            study.delete()
        except Exception as ex:
            pass
        
        # reset mlflow db
        try:
            reset_mlflow_db()
        except Exception as ex:
            pass
        
        # reset orbax checkpoint-bucket
        try:
            reset_checkpoint_buckets()
        except Exception as ex:
            pass
        
        try:
            reset_hpo_results_bucket(config['project_id'],
                config['study_name'])
        except Exception as ex:
            pass
    
    def transform_to_gs_uri(self, file_path:str):
        idx = file_path.find("/data/")
        tr = f'gs://{file_path[idx+1:]}'
        return tr
    
    def test_local_info(self):
        print(f'local_devices={jax.local_devices()}') #[CpuDevice(id=0)]
        print(f'local device count={jax.local_device_count()}')
        print(f'process_count={jax.process_count()}')
        print(f'process_index={jax.process_index()}')
    
    def test_run_app_check(self):
        config = self.config.copy()
        config['connections_check'] = 1
        del config['phase']
        set_flags_from_dict(config)
        app_runner(None)
        pass
    
    def test_run_tune_train_test(self):
        """
        this uses the docker container fake-gcs-server
        and so all uris are gs:// and are transformed by the local google software to
        http://172.17.0.1:4443/ ... depending upon context
        
        it also uses a vizier service
        
        to start the fake gcs server and the postgres db and vizier server:
            docker compose --project-directory . -f deploy/compose/docker-compose-dbs.yaml up -d
        """
        
        config = self.config.copy()
        
        self._run_and_assert_hpo(config)
        
        config['phase'] = 'train-best'
        config['train_id'] = 1234567
        config['validate_checkpoint_restores'] = True
        
        restore_dict, train_run = self._run_train_and_restore_chkpoint_and_assert(config)
        
        config['phase'] = 'test-best'
        test_id = 234567
        config['test_id'] = test_id
        self._run_test_and_assert(config, restore_dict)
        
        ##====== load train_ for use in stats =======
        TRAIN_BATCH_SIZE = restore_dict['train_dataloader']._sampler.batch_size
        TOTAL_RECORDS = restore_dict['train_dataloader']._sampler.total_records
        STEPS_PER_EPOCH_GLOBAL = restore_dict['train_dataloader']._sampler.num_batches_per_epoch  # = 7234
        NUM_TRAIN_SHARDS = restore_dict['train_dataloader']._sampler._shard_options.shard_count
        STEPS_PER_EPOCH_LOCAL = STEPS_PER_EPOCH_GLOBAL // NUM_TRAIN_SHARDS
        
        ## ================ get the 2nd to last latest checkpoint and assert can continue training from it. ====
        best_checkpoint_uri = train_run.data.tags.get("best_checkpoint_uri")
        latest_checkpoint_uri = train_run.data.tags.get("latest_checkpoint_uri")
        earlier_restore_dict = restore_items_from_checkpoint(latest_checkpoint_uri, get_earliest=True)
        print(f'global_step next to last={earlier_restore_dict["global_step"]}')
        self.assertTrue(earlier_restore_dict['global_step'] > 0)
        epoch = (earlier_restore_dict['global_step']//TRAIN_BATCH_SIZE)//STEPS_PER_EPOCH_GLOBAL
        self.assertEqual(epoch, (config['num_epochs']-2))
        
        #because this is next to last epoch saved, we should see < STEPS_PER_EPOCH_LOCAL loops over the iterator
        start_step = earlier_restore_dict['global_step'] // NUM_TRAIN_SHARDS
        n_iter = 0
        try:
            for batch_idx, padded_super_graph in enumerate(earlier_restore_dict['train_dataloader_iter']):
                n_iter += 1
        except StopIteration:
            pass
        print(f"n_iter={n_iter}")
        #self.assertEqual(n_iter, 1)
        
        ## ==== get the last latest checkpoint and assert that doesn't continue training from it because number of epochs is reached. ====
        last_restored_dict = restore_items_from_checkpoint(latest_checkpoint_uri, get_earliest=False)
        print(f'global_step last epoch={last_restored_dict["global_step"]}')
        self.assertTrue(earlier_restore_dict['global_step'] > 0)
        epoch = (last_restored_dict['global_step'] // TRAIN_BATCH_SIZE) // STEPS_PER_EPOCH_GLOBAL
        self.assertEqual(epoch, (config['num_epochs'] - 1))
        
        start_step = last_restored_dict['global_step'] // NUM_TRAIN_SHARDS
        n_iter = 0
        try:
            for batch_idx, padded_super_graph in enumerate(
                    last_restored_dict['train_dataloader_iter']):
                n_iter += 1
        except StopIteration:
            pass
        print(f"n_iter2={n_iter}")
        # self.assertEqual(n_iter, 0)
        
        ## ====== assert that training continues ======
        print(f'BEGIN RESUME TRAIN 2nd to last')
        best_val_ndcg_k_2 = resume_train_fn(config=earlier_restore_dict['config'],
            trial=None, save_checkpoints=True)
        
        #already done, so no new value:
        print(f'best_val_ndcg_k from resume 2nd to last chkpt training={best_val_ndcg_k_2}')
        self.assertAlmostEqual(best_val_ndcg_k_2, -1.0, places=6)
        
        # ===== read mlflow db metrics ======
        #experiments: name, experiment_id
        #run: run_uuid|name|source_type|source_name|entry_point_name|user_id|status|start_time|end_time|source_version|lifecycle_stage|artifact_uri|experiment_id|deleted_time
        # metrics:  key|value|timestamp|run_uuid|step|is_nan
        '''
        with cte1 as (
            SELECT experiment_id from experiments
            where name = 'GraphRanker_tuning_unittest'
        ), cte2 as (
            select run_uuid from runs
            join cte1 on runs.experiment_id=cte1.experiment_id
            order by start_time desc limit 1
        ) select key, value, timestamp, step from metrics
          join cte2 on metrics.run_uuid==cte2.run_uuid
          order by key,timestamp;
        '''
        experiment = mlflow.get_experiment_by_name(name=config['mlflow_experiment_name'])
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.run_name LIKE 'train_%'",
            output_format="list"
        )
        expected_keys = {'loss', 'ndcg_20', 'mrr_20', 'recall_20'}
       
        metrics_dicts = get_mlflow_metrics_by_exp_name(
            mlflow_tracking_uri=config['mlflow_tracking_uri'],
            experiment_name=config['study_name'])
        
        print(f'metrics_dicts={metrics_dicts}')
        metrics_dict = None
        for run_id, d in metrics_dicts.items():
            if metrics_dict is None or (len(d['train_loss']['x']) > len(metrics_dict['train_loss']['x'])):
                metrics_dict = d
        plot_mlflow_metrics(metrics_dict)
        pngs = glob.glob(os.path.join(get_bin_dir(), "*.png"))
        self.assertIsNotNone(pngs)
        self.assertTrue(len(pngs) > 0)
        for png_file in pngs:
            self.assertTrue(os.path.exists(png_file))
            
        self._assert_export_methods(config)
    
    def test_run_train_test_given(self):
        config = self.config.copy()
        
        config['study_name'] = "GraphRanker_tuning_unittest4"
        config['project_id'] =  'tune-unittest-04'
        config['mlflow_experiment_name'] = config['study_name']
        
        hparams = {'top_k': 20, 'num_layers': 2, 'num_heads': 4, 'hidden_dim': 128,
            'max_history': 70, 'num_candidates': 70, 'learning_rate': 0.001,
            'weight_decay': 0.001, 'out_dim': 32, 'edge_embed_dim': 16, 'dropout_rate': 0.2}
        config.update(hparams)
        
        config['phase'] = 'train-given'
        config['train_id'] = 2345
        config['validate_checkpoint_restores'] = True
        
        restore_dict, train_run = self._run_train_and_restore_chkpoint_and_assert(config)
        
        config['phase'] = 'test-best'
        test_id = 234567
        config['test_id'] = test_id
        self._run_test_and_assert(config, restore_dict)
        
        self._assert_export_train_results(config)
        self._assert_export_test_results(config)
    
    def _assert_export_methods(self, config:Dict[str, Any]):
        
        # use the method and assert results
        # output_hp_path = os.path.join(get_bin_dir(), "output_hp.json")
        # output_metrics_path = os.path.join(get_bin_dir(), "output_metrics.json")
        output_hp_path = f"gs://hpo-results-bucket/{config['project_id']}/{config['study_name']}/tune/hparams.json"
        output_metrics_path = f"gs://hpo-results-bucket/{config['project_id']}/{config['study_name']}/tune/metrics.json"
        config['output_hyperparams_uri'] = output_hp_path
        config['output_metrics_uri'] = output_metrics_path
        config['phase'] = 'export-hpo-results'
        
        #run_export_results(config=config)
        set_flags_from_dict(config)
        app_runner(None)
        
        logging.info(f"assert {config['phase']} results")
        with fsspec.open(output_hp_path, mode='r') as f:
            content = f.read()
            output_hp_dict = json.loads(content)
        with fsspec.open(output_metrics_path, mode='r') as f:
            content = f.read()
            output_metrics_dict = json.loads(content)
        self.assertIsNotNone(output_hp_dict)
        self.assertIsNotNone(output_metrics_dict)
        
        for key in ("ndcg_20", 'mrr_20', 'recall_20', 'loss'):
            key1 = f'train_{key}'
            key2 = f'val_{key}'
            self.assertTrue(key1 in output_metrics_dict)
            self.assertTrue(key2 in output_metrics_dict)
            
        self._assert_export_train_results(config)
        self._assert_export_test_results(config)
        
    def _assert_export_train_results(self, config: Dict[str, Any]):
        
        # ===================================================
        output_metrics_path = f"gs://hpo-results-bucket/{config['project_id']}/{config['study_name']}/train/metrics.json"
        config['output_metrics_uri'] = output_metrics_path
        config['phase'] = 'export-train-results'
        
        #run_export_results(config=config)
        set_flags_from_dict(config)
        app_runner(None)
        
        logging.info(f"assert {config['phase']} results")
        with fsspec.open(output_metrics_path, mode='r') as f:
            content = f.read()
            output_metrics_dict = json.loads(content)
        self.assertIsNotNone(output_metrics_dict)
        for key in ("ndcg_20", 'mrr_20', 'recall_20', 'loss'):
            key1 = f'train_{key}'
            key2 = f'val_{key}'
            self.assertTrue(key1 in output_metrics_dict)
            self.assertTrue(key2 in output_metrics_dict)
        
    def _assert_export_test_results(self, config: Dict[str, Any]):
        # ===================================================
        output_metrics_path = f"gs://hpo-results-bucket/{config['project_id']}/{config['study_name']}/test/metrics.json"
        config['output_metrics_uri'] = output_metrics_path
        config['phase'] = 'export-test-results'
        
        #run_export_results(config=config)
        set_flags_from_dict(config)
        app_runner(None)
        
        logging.info(f"assert {config['phase']} results")
        with fsspec.open(output_metrics_path, mode='r') as f:
            content = f.read()
            output_metrics_dict = json.loads(content)
        self.assertIsNotNone(output_metrics_dict)
        for key in ("ndcg_20", 'mrr_20', 'recall_20', 'loss'):
            key1 = f'test_{key}'
            self.assertTrue(key1 in output_metrics_dict)
    
    def _run_and_assert_hpo(self, config):
        print(f'BEGIN TUNING')
        
        set_flags_from_dict(config)
        
        # run tune HPO
        app_runner(None)
        
        vz_clients.environment_variables.server_endpoint = config['vizier_endpoint']
        study = vz_clients.Study.from_owner_and_id(owner=config['project_id'], study_id=config['study_name'])
        self.assertIsNotNone(study)
        optimal_trials = study.optimal_trials()
        self.assertIsNotNone(optimal_trials)
        
        best_trial = None
        for tr in optimal_trials:
            best_trial = tr
            break
        self.assertIsNotNone(best_trial)
        best_trial_data = best_trial.materialize()
        # best_params contains only the params being tuned, not all params needed for train_fn
        best_params = extract_correct_vizier_param_types_dict(best_trial_data.parameters)
        print("Available metrics:",
            list(best_trial_data.final_measurement.metrics.keys()), flush=True)
        bfm = best_trial_data.final_measurement
        bfm = bfm.metrics.get(f'ndcg_{config["top_k"]}')
        best_value = bfm.value
        
        print(f"Loaded Best Objective: {best_value}")
        print(f"Loaded Best Parameters: {best_params}")
        self.assertTrue(best_value > 0)
        
        mlflow_run_id = best_trial_data.metadata.get('mlflow_run_id')
        self.assertIsNotNone(mlflow_run_id)
        
        best_params2 = get_best_parameters_for_training(config)
        
        # phase was tune, so no need to check for checkpoint paths
        
        mlflow_run = mlflow.get_run(mlflow_run_id)
        config = destringify_mlflow_params(mlflow_run.data.params)
        self.assertIsNotNone(config)
        self.assertTrue(isinstance(config['batch_size'], int))
        
        ## assert the values are the same
        for k, v in best_params.items():
            if isinstance(v, float):
                self.assertAlmostEqual(v, config[k], delta=0.01 * v)
                self.assertAlmostEqual(v, best_params2[k], delta=0.01 * v)
            else:
                self.assertEqual(v, config[k])
    
    def _run_train_and_restore_chkpoint_and_assert(self, config):
        #### ====================================================== ####
        print(f'BEGIN TRAINING')
        
        set_flags_from_dict(config)
       
        # run train using best found in HPO
        app_runner(None)
        
        #results are asserted in _assert_export_methods
        
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        experiment = mlflow.get_experiment_by_name(name=config['mlflow_experiment_name'])
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.run_name LIKE 'train_%'",
            output_format="list"
        )
        
        run_name = get_canonical_mlflow_run_name(config)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.run_name = '{run_name}'",
            output_format="list"
        )
        self.assertIsNotNone(runs)
        self.assertTrue(len(runs) == 1)
        
        # phase is train, so assert checkpoint paths are in mlflow
        # we have to fetch the checkpoint path from the mflow params or tags
        
        best_checkpoint_uri_tag = runs[0].data.tags.get("best_checkpoint_uri")
        print(f'best_checkpoint_uri_tag={best_checkpoint_uri_tag}')
        self.assertTrue(best_checkpoint_uri_tag.find('train_') > -1)
        best_checkpoint_uri = get_best_checkpoint_uri_for_testing(config)
        self.assertEqual(best_checkpoint_uri_tag, best_checkpoint_uri)
        
        # the train method stores checkpoints so assert checkpoints and restore and assert can resume training if not complete ==============================
        restore_dict = restore_items_from_checkpoint(checkpoint_uri=best_checkpoint_uri_tag)
        
        expected_keys = {'model', 'optimizer', 'train_dataloader',
            'train_dataloader_iter',
            'val_dataloader', 'rngs', 'global_step', 'config'}
        for key in expected_keys:
            self.assertTrue(key in restore_dict)
        
        return restore_dict, runs[0]
    
    def _run_test_and_assert(self, config, restore_dict):
        #=== operate test from entrypoint
        print(f'BEGIN TESTING')
        
        ## =============== add test uris to config and run tests.  also tests that restore works================
        
        #config['test_checkpoint_uri'] = best_checkpoint_uri_tag  #for use when phase is 'test-given'
        #config['best_checkpoint_uri'] = best_checkpoint_uri_tag #the method now looks this up in mlflow records
        
        set_flags_from_dict(config)
        app_runner(None)
        
        #results are asserted in _assert_export_methods
    
    def test_feed_fake_data(self):
        
        config = self.config
        
        embeddings, num_users = read_embeddings(
            user_embeddings_uri=config['user_embeddings_uri'],
            movie_embeddings_uri=config['movie_embeddings_uri'],
            batch_size=1024)
        
        num_movies = len(embeddings) - 1 - num_users
        
        max_history = 10
        num_candidates = 20
        user_id_range = (1, num_users)
        movie_id_range = (num_users + 1, num_users + num_movies)
        
        fake_data = create_dummy_super_padded_graph(batch_size = 1,
            max_history = max_history,
            num_candidates = num_candidates,
            user_id_range = user_id_range, movie_id_range = movie_id_range)
        
        rngs = nnx.Rngs(config.get('seed', 0))
        
        #these are assigned by HPO
        config['num_candidates'] = 2*config["top_k"]
        config['max_history'] = max_history
        config['hidden_dim'] = 64
        config['num_layers'] = 2
        config['out_dim'] = 32
        config['num_heads'] = 4
        config['edge_embed_dim'] = 8
        config['dropout_rate'] = 0.05
        
        model = GraphRanker(user_movie_embeds=embeddings,
            num_candidates=config['num_candidates'],
            hidden_features=config['hidden_dim'],
            num_layers=config['num_layers'],
            out_features=config['out_dim'],
            heads=config['num_heads'],
            edge_embed_dim=config['edge_embed_dim'],
            dropout_rate=config['dropout_rate'], rngs=rngs)
        
        model.eval()
        all_scores = model(fake_data)
        model.train()
        
        self.assertIsNotNone(all_scores)
        
        #==========
        batch_size=256
        max_history=200
        num_candidates=100
        
        jax_graph_comp_dict = calc_number_jax_graph_components(
            batch_size=batch_size, max_history=max_history, num_candidates=num_candidates,
            n_local_devices=jax.local_device_count())
        
        fake_batch = create_fake_jagged_batch(batch_size=batch_size,
            max_history=max_history,
            num_candidates=num_candidates, user_id_range=user_id_range,
            movie_id_range=movie_id_range)
        
        n_local_devices = jax.local_device_count()
        
        time0 = datetime.datetime.now()
        
        padded_super_graph_0 = pad_graph_tuple_batch(fake_batch,
            jax_graph_comp_dict)
        
        time1 = datetime.datetime.now()
        
        padded_super_graph_1, n_samples = optimized_batch_and_pad(
            batch=fake_batch,
            max_nodes=jax_graph_comp_dict['max_nodes'],
            max_edges=jax_graph_comp_dict['max_edges'],
            max_graphs=jax_graph_comp_dict['max_graphs'],
        )
        
        time2 = datetime.datetime.now()
        
        diff0 = time1 - time0
        diff1 = time2 - time1
        
        print(f'time0={diff0}\ntime1={diff1}', flush=True)
        
        ## compare the graphs
        self._dictionaries_are_same(padded_super_graph_1.edges, padded_super_graph_0.edges)
        np.testing.assert_array_equal(padded_super_graph_1.n_edge, padded_super_graph_0.n_edge)
        np.testing.assert_array_equal(padded_super_graph_1.n_node, padded_super_graph_0.n_node)
        self._tuples_are_same(padded_super_graph_1._fields, padded_super_graph_0._fields)
        self.assertIsNone(padded_super_graph_0.globals)
        self.assertIsNone(padded_super_graph_1.globals)
        
        np.testing.assert_array_equal(padded_super_graph_1.receivers, padded_super_graph_0.receivers)
        np.testing.assert_array_equal(padded_super_graph_1.senders, padded_super_graph_0.senders)
        
    def _dictionaries_are_same(self, d0:Dict[str, np.ndarray], d1:Dict[str, np.ndarray]):
        self.assertEqual(len(d0), len(d1))
        self.assertEqual(d0.keys(), d1.keys())
        for key in d0.keys():
            np.testing.assert_array_equal(d0[key], d1[key])

    def _tuples_are_same(self, t0:Tuple[str], t1:Tuple[str]):
        self.assertEqual(len(t0), len(t1))
        for i, v in enumerate(t0):
            self.assertEqual(v, t1[i])

if __name__ == '__main__':
    unittest.main()
