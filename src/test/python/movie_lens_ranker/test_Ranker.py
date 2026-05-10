import os
import logging

#to test for multiple devices before using on GPUs or TPUs:
#os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
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
        if "already initialized" in str(e).lower():
            pass
        else:
            raise e
safe_jax_init()

from mlflow import MlflowClient
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

from movie_lens_ranker.app_runner import main as app_runner, \
    extract_correct_vizier_param_types_dict

import unittest
import subprocess

#found by ip addr show docker0
base_url = "172.17.0.1"

"""
this uses the docker compose-dbs.yaml fake-gcs-server and db (==postgres server)

the gcs uris are  formatted to gs://bucket_name/...

the postgres uris use postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
where netlo is 172.17.0.1

to start the fake gcs server and the postgres db:
    docker compose -f docker-compose-dbs.yaml up -d
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
        "sh", "-c", "rm -rf /storage/checkpoint_bucket/*"
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        print("empty checkpoint_bucket/* successful")
    except subprocess.CalledProcessError as e:
        print(f"Error resetting database: {e.stderr}")
    
class TestRanker(unittest.TestCase):
    def setUp(self):
        
        # === these are so that grain dataloader can read data from fake gcs server running in docker ====
        env_file = os.path.join(get_project_dir(), ".env_unittests")
        for k, v in dotenv_values(env_file).items():
            os.environ[k] = v
            
        # user recommendations with each user history subtracted already:
        # (user id, (movie_ids))
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/recommended_movies.array_record")
        
        #(user_id, movie_id, rating, timestamp)
        self.ratings_train_uri, self.ratings_val_uri, self.ratings_test_uri \
            = get_train_val_test_liked_uris(use_small=True)
        
        # (user_id, movie_id, rating, timestamp)
        self.ratings_train_disliked_uri, self.ratings_val_disliked_uri, self.ratings_test_disliked_uri \
            = get_train_val_test_disliked_uris(use_small=True)
        
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
        
        #these are the "elite" hard negatives (=intersection between train_disliked and recommended movies)
        # + train disliked.
        self.train_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/train_negatives.array_record")
        self.val_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/val_negatives.array_record")
        self.test_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/test_negatives.array_record")
        self.train_val_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/train_val_negatives.array_record")
        self.train_val_test_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/train_val_test_negatives.array_record")
    
    def transform_to_gs_uri(self, file_path:str):
        idx = file_path.find("/data/")
        tr = f'gs://{file_path[idx+1:]}'
        return tr
    
    def test_local_info(self):
        print(f'local_devices={jax.local_devices()}') #[CpuDevice(id=0)]
        print(f'local device count={jax.local_device_count()}')
        print(f'process_count={jax.process_count()}')
        print(f'process_index={jax.process_index()}')
    
    def get_or_create_mlflow_experiment(self, experiment_name: str):
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)
        
    def test_run_tune_train_test(self):
        """
        this uses the docker container fake-gcs-server
        and so all uris are gs:// and are transformed by the local google software to
        http://172.17.0.1:4443/ ... depending upon context
        
        it also uses a vizier service
        
        to start the fake gcs server and the postgres db and vizier server:
            docker compose -f docker-compose-dbs.yaml up -d
        """
        
        # check that docker fake gcs server is running
        try:
            response = requests.get(f"http://{base_url}:4443/storage/v1/b/data/o")
            if response.status_code == 200:
                data = response.json()
                print(data)
                self.assertTrue(len(data['items']) > 0)
            else:
                print(f"Failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                print(f'is the fake_gcs_server container not running?')
                return
            wait_for_postgres_vizier_mlflow_dbs()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return
        
        STUDY_NAME = "GraphRanker_tuning_unittest3"
        
        num_epochs = 4 #keep this to > 2 and < 10 for the restore tests at end of this method
        batch_size = 64
        seed = 234
        
        # tensorstore keeps trying to authenticate with google so for tests we'll use the abs path to checkpoint dir
        checkpoint_dir = 'gs://checkpoint_bucket'
        latest_checkpoint_uri = f'{checkpoint_dir}/latest'
        best_checkpoint_uri = f'{checkpoint_dir}/best'
        
        #mflow_db_path = os.path.join(get_bin_dir(), f"{STUDY_NAME}_mlflow.db")
        #mflow_uri = f"sqlite:///{mflow_db_path}?mode=memory&cache=shared"
        ##    postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
        vizier_endpoint = f'{base_url}:8000'
        mlflow_uri = f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{base_url}:5432/mlflow_db"
        
        config = {
            'movies_uri': self.transform_to_gs_uri(self.movies_uri),
            'recommendations_uri': self.transform_to_gs_uri(self.recommendations_uri),
            'recommendations_ts_uri' : self.transform_to_gs_uri(self.recommendations_ts_uri),
            'ratings_train_uri' : self.transform_to_gs_uri(self.ratings_train_uri),
            'ratings_val_uri' :self.transform_to_gs_uri(self.ratings_val_uri),
            'train_negatives_uri': self.transform_to_gs_uri(self.train_negatives_uri),
            'val_negatives_uri': self.transform_to_gs_uri(self.val_negatives_uri),
            'latest_checkpoint_uri':latest_checkpoint_uri,
            'best_checkpoint_uri': best_checkpoint_uri,
            'movie_embeddings_uri' : self.transform_to_gs_uri(self.movie_embeddings_uri),
            'user_embeddings_uri': self.transform_to_gs_uri(self.user_embeddings_uri),
            'num_epochs' : num_epochs,
            'batch_size':batch_size, 'seed':seed,
            'study_name' : STUDY_NAME,
            'project_id' : 'tune-unittest-02',
            "trial_ids" : json.dumps([0, 1]),
            'phase' : 'tune',
            'top_k' : 20,
            'vizier_endpoint': vizier_endpoint,
            'mlflow_tracking_uri': mlflow_uri,
            'mlflow_experiment_id': self.get_or_create_mlflow_experiment(STUDY_NAME),
            'mlflow_experiment_name': STUDY_NAME,
        }
        set_flags_from_dict(config)
        
        try:
            vz_clients.environment_variables.server_endpoint = config['vizier_endpoint']
            resource_name = f"owners/{config['project_id']}/studies/{config['study_name']}"
            study = vz_clients.Study.from_owner_and_id(owner=config['project_id'],
                study_id=config['study_name'])
            study.delete()
        except Exception as ex:
            pass
        
        try:
            # empty MLFlow db:
            reset_mlflow_db()
        except Exception as ex:
            pass
        
        try:
            # epty the checkpoints bucket
            reset_checkpoint_buckets()
        except Exception as ex:
            pass
        
        # run tune HPO
        app_runner(None)
        
        vz_clients.environment_variables.server_endpoint = config['vizier_endpoint']
        study = vz_clients.Study.from_owner_and_id(owner=config['project_id'],
            study_id=config['study_name'])
        self.assertIsNotNone(study)
        optimal_trials = study.optimal_trials()
        self.assertIsNotNone(optimal_trials)
        
        best_trial = None
        for tr in optimal_trials:
            best_trial = tr
            break
        self.assertIsNotNone(best_trial)
        best_trial_data = best_trial.materialize()
        #best_params contains only the params being tuned, not all params needed for train_fn
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
        
        #phase was tune, so no need to check for checkpoint paths
        
        mlflow_run = mlflow.get_run(mlflow_run_id)
        config = destringify_mlflow_params(mlflow_run.data.params)
        self.assertIsNotNone(config)
        self.assertTrue(isinstance(config['batch_size'], int))
        
        ## assert the values are the same
        for k, v in best_params.items():
            if isinstance(v, float):
                self.assertAlmostEqual(v, config[k], delta=0.01*v)
            else:
                self.assertEqual(v, config[k])
        
        #### ====================================================== ####
        config['phase'] = 'train_best'
        train_id = 1234567
        config['train_id'] = train_id
        
        for k, v in config.items():
            if k.find('<=') > -1:
                print(f"problem key: {k}={v}", flush=True)
        
        set_flags_from_dict(config)
        # more debugging:
        for name, flag in flags.FLAGS._flags().items():
            try:
                flag.value #logger_levels
            except Exception as ex:
                pass

        #run train using best found in HPO
        app_runner(None)
        
        #results will be stored for study_name and run_name='train'
        run_name = f"train_{train_id}"
        runs = mlflow.search_runs(
            experiment_ids=[config['mlflow_experiment_id']],
            filter_string=f"attributes.run_name = '{run_name}'",
            output_format="list"
        )
        
        mlflow_client = MlflowClient(tracking_uri=config['mlflow_tracking_uri'])
        
        self.assertIsNotNone(runs)
        self.assertEqual(len(runs), 1)
        best_run = runs[0]
        run_id = best_run.info.run_id
        metrics_dict = {}
        for key in ("loss", "ndcg_20", "recall_20", "mrr_20"):
            for key_t in (f"train_{key}", f"val_{key}"):
                metrics_dict[key_t] = {'x': [], 'y': []}
                m_dict = mlflow_client.get_metric_history(run_id, key=key_t)
                for m in m_dict:
                    metrics_dict[key_t]['x'].append(int(m.step))
                    metrics_dict[key_t]['y'].append(float(m.value))
        self.assertTrue(len(metrics_dict), 8)
        
        #phase is train, so assert checkpoint paths are in mlflow
        # we have to fetch the checkpoint path from the mflow params or tags
        best_checkpoint_uri_tag = best_run.data.tags.get("best_checkpoint_uri")
        best_checkpoint_uri_param = json.loads(best_run.data.params.get("best_checkpoint_uri"))
        self.assertTrue(best_checkpoint_uri_tag == best_checkpoint_uri_param)
        print(f'best_checkpoint_uri={best_checkpoint_uri}')
        
        #the train method stores checkpoints so assert checkpoints and restore and assert can resume training if not complete ==============================
        restore_dict = restore_items_from_checkpoint(checkpoint_uri=best_checkpoint_uri_tag)
        
        expected_keys = {'model', 'optimizer', 'train_dataloader', 'train_dataloader_iter',
            'val_dataloader', 'rngs', 'global_step', 'config'}
        for key in expected_keys:
            self.assertTrue(key in restore_dict)
        
        ## =============== add test uris to config and run tests.  also tests that restore works================
        restore_dict['config']['ratings_test_uri'] = self.transform_to_gs_uri(self.ratings_test_uri)
        restore_dict['config']['train_negatives_uri'] = self.transform_to_gs_uri(self.test_negatives_uri)
        
        #test_metrics = test_fn(config=restore_dict['config'])
        #print(f'TEST METRICS: {test_metrics}', flush=True)
        
        #=== operate test from entrypoint
        config['ratings_test_uri'] = self.transform_to_gs_uri(self.ratings_test_uri)
        config['train_negatives_uri'] = self.transform_to_gs_uri(self.test_negatives_uri)
        #config['test_checkpoint_uri'] = best_checkpoint_uri_tag  #for use when phase is 'test_given'
        config['best_checkpoint_uri'] = best_checkpoint_uri_tag
        config['phase'] = 'test_best'
        test_id = 234567
        config['test_id'] = test_id
        set_flags_from_dict(config)
        app_runner(None)
    
        run_name = f'test_{config.get('test_id', 0)}'
        runs = mlflow.search_runs(
            experiment_ids=[config['mlflow_experiment_id']],
            filter_string=f"attributes.run_name = '{run_name}'",
            output_format="list"
        )
        self.assertIsNotNone(runs)
        self.assertEqual(len(runs), 1)
        run_id = runs[0].info.run_id
        metrics_dict = {}
        for key in ("loss", "ndcg_20", "recall_20", "mrr_20"):
            for key_t in (f"train_{key}", f"val_{key}"):
                metrics_dict[key_t] = {'x': [], 'y': []}
                m_dict = mlflow_client.get_metric_history(run_id, key=key_t)
                for m in m_dict:
                    metrics_dict[key_t]['x'].append(int(m.step))
                    metrics_dict[key_t]['y'].append(float(m.value))
        self.assertTrue(len(metrics_dict), 8)
        
        ##====== load train_ for use in stats =======
        TRAIN_BATCH_SIZE = restore_dict['train_dataloader']._sampler.batch_size
        TOTAL_RECORDS = restore_dict['train_dataloader']._sampler.total_records
        STEPS_PER_EPOCH_GLOBAL = restore_dict['train_dataloader']._sampler.num_batches_per_epoch  # = 7234
        NUM_TRAIN_SHARDS = restore_dict['train_dataloader']._sampler._shard_options.shard_count
        STEPS_PER_EPOCH_LOCAL = STEPS_PER_EPOCH_GLOBAL // NUM_TRAIN_SHARDS
        
        ## ================ get the 2nd to last latest checkpoint and assert can continue training from it. ====
        best_checkpoint_uri = best_run.data.tags.get("best_checkpoint_uri")
        latest_checkpoint_uri = best_run.data.tags.get("latest_checkpoint_uri")
        earlier_restore_dict = restore_items_from_checkpoint(latest_checkpoint_uri, get_earliest=True)
        print(f'global_step next to last={earlier_restore_dict["global_step"]}')
        self.assertTrue(earlier_restore_dict['global_step'] > 0)
        epoch = (earlier_restore_dict['global_step']//TRAIN_BATCH_SIZE)//STEPS_PER_EPOCH_GLOBAL
        self.assertEqual(epoch, (num_epochs-2))
        
        #because this is next to last epoch saved, we shoud see < STEPS_PER_EPOCH_LOCAL loops over the iterator
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
        self.assertEqual(epoch, (num_epochs - 1))
        
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
        best_val_ndcg_k_2 = resume_train_fn(config=earlier_restore_dict['config'],
            trial=None, save_checkpoints=True)
        
        print(f'best_val_ndcg_k from resume 2nd to last chkpt training={best_val_ndcg_k_2}')
        
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
        runs = mlflow.search_runs(
            experiment_ids=[config['mlflow_experiment_id']],
            filter_string="attributes.run_name LIKE 'train_%'",
            output_format="list"
        )
        expected_keys = {'loss', 'ndcg_20', 'mrr_20', 'recall_20'}
       
        metrics_dicts = get_mlflow_metrics_by_exp_name(
            mlflow_tracking_uri=mlflow_uri,
            experiment_name=STUDY_NAME)
        
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
    
if __name__ == '__main__':
    unittest.main()
