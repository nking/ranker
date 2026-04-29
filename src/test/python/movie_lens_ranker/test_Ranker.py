import glob
import os.path
import pathlib
import threading
import unittest

import jax.distributed
from array_record.python import array_record_module

import shutil
from absl import flags
from asyncssh.encryption import get_encryption_algs
from networkx.algorithms.centrality import information_centrality
from optuna import create_study, load_study
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler

from helper import *
from movie_lens_ranker.data_loading import *
from movie_lens_ranker.train import *
from movie_lens_ranker.util import set_flags_from_dict
from movie_lens_ranker.util_plots import plot_mlflow_metrics

from movie_lens_ranker.optuna_trial_run import main as run_optuna_main

class TestRanker(unittest.TestCase):
    def setUp(self):
        
        # user recommendations with each user history subtacted already:
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
    
    def test_local_info(self):
        print(f'local_devices={jax.local_devices()}') #[CpuDevice(id=0)]
        print(f'local device count={jax.local_device_count()}')
        print(f'process_count={jax.process_count()}')
        print(f'process_index={jax.process_index()}')

    def test_dp(self):
        print(f"devices = {jax.devices()}")
        mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('data',))
        jax.set_mesh(mesh)
        print(mesh)
    
    def get_or_create_mlflow_experiment(self, experiment_name: str):
        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)
        
    def test_run_train_without_optuna(self):
        
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
        
        max_history = 200
        num_candidates = 40
        batch_size = 64
        num_epochs = 3# 10#120
        seed = 1234
        
        top_k = 20
        learning_rate = 5e-4#1e-3
        weight_decay = 1e-4
        out_dim = 32
        hidden_dim = 64  # 2 * embed_in_dim is probably good
        num_layers = 2  # captures the 2-hop neighborhood.  3 tends to oversmooth
        num_heads = 4  # each head sees 64 hidden / 4 heads = 16 dimensional subspace
        edge_embed_dim = 8
        dropout_rate = 0.1
        
        checkpoint_dir = os.path.join(get_bin_dir(), "checkpoints")
        latest_checkpoint_dir = os.path.join(checkpoint_dir, "latest")
        best_checkpoint_dir = os.path.join(checkpoint_dir, "best")
        mlflow_dir = os.path.join(get_bin_dir(), "mlflow")
        mlflow_registry_dir = os.path.join(get_bin_dir(), "mlflow_registry")
        os.makedirs(latest_checkpoint_dir, exist_ok=True)
        os.makedirs(best_checkpoint_dir, exist_ok=True)
        os.makedirs(mlflow_dir, exist_ok=True)
        os.makedirs(mlflow_registry_dir, exist_ok=True)
        
        non_trainable_params = get_nontrainable_train_config(
            movies_uri=self.movies_uri,
            recommendations_uri=self.recommendations_uri,
            recommendations_ts_uri=self.recommendations_ts_uri,
            ratings_train_uri=self.ratings_train_uri,
            ratings_val_uri=self.ratings_val_uri,
            train_negatives_uri=self.train_negatives_uri,
            val_negatives_uri=self.val_negatives_uri,
            latest_checkpoint_dir =latest_checkpoint_dir,
            best_checkpoint_dir = best_checkpoint_dir,
            movie_embeddings_uri = self.movie_embeddings_uri,
            user_embeddings_uri = self.user_embeddings_uri,
            num_epochs=num_epochs, batch_size=batch_size, seed=seed
        )
        
        data_params_trainable = {'max_history':max_history, 'num_candidates':num_candidates,
            'num_epochs':num_epochs,
            'batch_size':batch_size}
        
        model_params_trainable = {'top_k':top_k, 'learning_rate':learning_rate, 'weight_decay':weight_decay,
            'out_dim':out_dim, 'hidden_dim':hidden_dim, 'num_layers':num_layers,
            'num_heads':num_heads, 'edge_embed_dim':edge_embed_dim, 'dropout_rate':dropout_rate,
        }
        
        STUDY_NAME = "GraphRanker_tuning_unittest"
        
        try:
            mlflow.delete_experiment(STUDY_NAME)
            print(f'Deleted experiment {STUDY_NAME}')
        except Exception as e:
            pass
        mlflow.set_experiment(STUDY_NAME)
        # Create the parent run and immediately get its ID
        parent_run = mlflow.start_run(run_name="unittest_train")
        mlflow_parent_run_id = parent_run.info.run_id
        mlflow.end_run()
    
        mlflow_config = {
            'mlflow_tracking_uri': mlflow_dir,
            'mlflow_registry_uri': mlflow_registry_dir,
            'mlflow_experiment_id': self.get_or_create_mlflow_experiment(STUDY_NAME),
            'mlflow_experiment_name': STUDY_NAME,
            #'mlflow_tracking_token': None,
            'mlflow_parent_run_id': mlflow_parent_run_id
        }
        
        config = {**non_trainable_params, **data_params_trainable,
            **model_params_trainable, **mlflow_config}
        
        config['study_name'] = STUDY_NAME
        config["trial_id"] = 1
        config['phase'] = 'train'
        
        config['best_checkpoint_dir'] = f"{config['best_checkpoint_dir']}/{config['study_name']}/trial_{config['trial_id']}"
        config['latest_checkpoint_dir'] = f"{config['latest_checkpoint_dir']}/{config['study_name']}/trial_{config['trial_id']}"

        os.makedirs(config['latest_checkpoint_dir'], exist_ok=True)
        os.makedirs(config['best_checkpoint_dir'], exist_ok=True)

        set_flags_from_dict(config)
        
        best_val_ndcg_k, STATE = train_fn(config)
        
        print(f'final best val ndcg@k={best_val_ndcg_k}')
        
        ## ======== validate mlflow and checkpoints ==========
        root = pathlib.Path(mlflow_dir)
        dirs = {'metrics':[], 'params':[], 'artifacts':[]}
        for srch_dir in dirs.keys():
            for dir_path in root.rglob(srch_dir):
                # dir_path is relative to root. Uncomment the next line for absolute paths
                # dir_path = filepath.resolve()
                dirs[srch_dir].append(os.path.join(dir_path.parent, dir_path.name))
            self.assertTrue(len(dirs[srch_dir]) > 0)
        a = None
        b = None
        for metric_file in dirs['metrics']:
            for file_path in glob.glob(f'{metric_file}/*'):
                with open(file_path, 'r') as f:
                    line = f.readline().strip()
                    ts, value, epoch = line.split()
                    self.assertIsNotNone(value)
                    self.assertIsNotNone(epoch)
                    a = value
                    b = epoch
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        a = None
        for param_file in dirs['params']:
            for file_path in glob.glob(f'{param_file}/*'):
                with open(file_path, 'r') as f:
                    line = f.readline().strip()
                    self.assertIsNotNone(line)
                    a = line
        self.assertIsNotNone(a)
        
        #
        #tag tags = {"mlflow.parentRunId" : config['mlflow_parent_run_id']}
        #get the mlflow directory
        experiment = mlflow.get_experiment_by_name(config['mlflow_experiment_name'])
        if experiment is None:
            raise ValueError(f"Experiment {config['mlflow_experiment_name']} not found.")
        path = experiment.artifact_location
        entries = os.listdir(path)
        subdirs = [os.path.join(path, e) for e in entries if os.path.isdir(os.path.join(path, e))]
        subdirs.sort(key=os.path.getctime, reverse=True)
        metrics_dir = f'{subdirs[0]}/metrics'
        self.assertTrue(os.path.exists(metrics_dir))
        plot_mlflow_metrics(metrics_dir)
        for key in ("loss", "ndcg_20", "recall_20", "mrr_20"):
            self.assertTrue(os.path.exists(os.path.join(get_bin_dir(), f"{key}.png")))
        
        if False:
            #add test_fn use
            pass
        
    def test_run_train_with_optuna(self):
        
        num_epochs = 4 #keep this to > 2 and < 10 for the restore tests at end of this method
        batch_size = 64
        seed = 234
        
        checkpoint_dir = os.path.join(get_bin_dir(), "checkpoints")
        latest_checkpoint_dir = os.path.join(checkpoint_dir, "latest")
        best_checkpoint_dir = os.path.join(checkpoint_dir, "best")
        mlflow_dir = os.path.join(get_bin_dir(), "mlflow")
        mlflow_registry_dir = os.path.join(get_bin_dir(), "mlflow_registry")
        os.makedirs(latest_checkpoint_dir, exist_ok=True)
        os.makedirs(best_checkpoint_dir, exist_ok=True)
        os.makedirs(mlflow_dir, exist_ok=True)
        os.makedirs(mlflow_registry_dir, exist_ok=True)
        
        STUDY_NAME = "GraphRanker_tuning_unittest"
        optuna_db_path = os.path.join(get_bin_dir(), f"{STUDY_NAME}.db")
        optuna_storage_uri = f"sqlite:///{optuna_db_path}?mode=memory&cache=shared"
        if os.path.exists(optuna_db_path):
            os.remove(optuna_db_path)
            print(f"Deleted old database at {optuna_db_path}")
        
        # Initialize the optuna study in the database
        # This just "reserves the name" in your Postgres/MySQL DB
        study = create_study(
            study_name=STUDY_NAME,
            storage=optuna_storage_uri,
            sampler=RandomSampler(),
            pruner=MedianPruner(),
            direction="maximize",
            load_if_exists=False
        )
        
        # init mlflow experiment
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
        
        set_flags_from_dict({
            'movies_uri': self.movies_uri,
            'recommendations_uri': self.recommendations_uri,
            'recommendations_ts_uri' : self.recommendations_ts_uri,
            'ratings_train_uri' : self.ratings_train_uri, 'ratings_val_uri' :self.ratings_val_uri,
            'train_negatives_uri': self.train_negatives_uri, 'val_negatives_uri': self.val_negatives_uri,
            'latest_checkpoint_dir':latest_checkpoint_dir,
            'best_checkpoint_dir': best_checkpoint_dir,
            'movie_embeddings_uri' : self.movie_embeddings_uri,
            'user_embeddings_uri': self.user_embeddings_uri,
            'num_epochs' : num_epochs, 'batch_size':batch_size, 'seed':seed,
            'study_name' : STUDY_NAME,
            "trial_id" : 1,
            'phase' : 'train',
            'optuna_storage_uri':optuna_storage_uri,
            'mlflow_tracking_uri': mlflow_dir,
            'mlflow_registry_uri': mlflow_registry_dir,
            'mlflow_experiment_id': self.get_or_create_mlflow_experiment(STUDY_NAME),
            'mlflow_experiment_name': STUDY_NAME,
            # 'mlflow_tracking_token': None,
            'mlflow_parent_run_id': mlflow_parent_run_id
        })
        
        run_optuna_main(None)
        
        ##  ====== assert optuna results were stored ======
        FLAGS = flags.FLAGS
        study = load_study(study_name=STUDY_NAME, storage=optuna_storage_uri)
        self.assertIsNotNone(study)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value (NDCG): {study.best_value}")
        self.assertTrue(study.best_value > 0)
        # Return the winning params
        self.assertIsNotNone(study.best_trial.params)
        self.assertIsNotNone(study.best_trial.number)
        optuna_params = study.best_trial.params
        print(f'Best params from optuna: {optuna_params}')
        
        #get config from mlflow.  itt was storead as mlflow.log_params(config)
        mlflow_run_name = f"trial_{study.best_trial.number}"
        optuna_attrs = study.best_trial.user_attrs
        self.assertIsNotNone(optuna_attrs)
        print(f'optuna_attrs={optuna_attrs}')
        for tr in study.trials:
            print(f'trial={tr}, user_attrs={tr.user_attrs}', flush=True)
       
        mlflow_run_id = optuna_attrs['mlflow_run_id']
        self.assertIsNotNone(mlflow_run_id)
        
        mlflow_run = mlflow.get_run(mlflow_run_id)
        #caveat: numbers are all strings in this:
        config = mlflow_run.data.params
        self.assertIsNotNone(config)
        
        #===========================  assert checkpoints and restore and resme training ==============================
        restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['best_checkpoint_dir'])
        
        expected_keys = {'model', 'optimizer', 'train_dataloader', 'train_dataloader_iter',
            'val_dataloader', 'rngs', 'global_step', 'config'}
        for key in expected_keys:
            self.assertTrue(key in restore_dict)
            
        # assert contents of config and restore_dict['config'] are the same
        '''
        # TODO: choose keys to compare.
        # TODO: consider filtering which flag parameters to save.  there are many not used by this code.
        for key, value in config.items():
            if not isinstance(value, float):
                self.assertEqual(value, restore_dict['config'][key])
            else:
                self.assertAlmostEqual(value, restore_dict['config'][key], delta=value/100.)
        '''
        
        ## =============== add test uris to config and run tests.  also tests that restore works================
        restore_dict['config']['ratings_test_uri'] = self.ratings_test_uri
        restore_dict['config']['train_negatives_uri'] = self.test_negatives_uri
        
        test_metrics = test_fn(config=restore_dict['config'])
        
        print(f'TEST METRICS: {test_metrics}', flush=True)
        
        ## load train_ for use in stats
        TRAIN_BATCH_SIZE = restore_dict['train_dataloader']._sampler.batch_size
        TOTAL_RECORDS = restore_dict['train_dataloader']._sampler.total_records
        STEPS_PER_EPOCH_GLOBAL = restore_dict['train_dataloader']._sampler.num_batches  # = 7234
        NUM_TRAIN_SHARDS = restore_dict['train_dataloader']._sampler._shard_options.shard_count
        STEPS_PER_EPOCH_LOCAL = STEPS_PER_EPOCH_GLOBAL // NUM_TRAIN_SHARDS
        
        ## ================ get the 2nd to last latest checkpoint and assert can continue training from it. ====
        earlier_restore_dict = restore_items_from_checkpoint(config['latest_checkpoint_dir'], get_earliest=True)
        print(f'global_step={earlier_restore_dict["global_step"]}')
        self.assertTrue(earlier_restore_dict['global_step'] > 0)
        epoch = (earlier_restore_dict['global_step']//TRAIN_BATCH_SIZE)//STEPS_PER_EPOCH_GLOBAL
        self.assertEqual(epoch, (num_epochs-2))
        
        #because this is next to last epoch saved, we shoud see < STEPS_PER_EPOCH_LOCAL loops over the iterator
        start_step = earlier_restore_dict['global_step'] // NUM_TRAIN_SHARDS
        n_iter = 0
        #TODO: follow up.  something is wrong here because it iterates over 4 epochs
        try:
            for batch_idx, padded_super_graph in enumerate(
                earlier_restore_dict['train_dataloader_iter'], start=start_step):
                n_iter += 1
        except StopIteration:
            pass
        print(f"n_iter={n_iter}")
        #self.assertEqual(n_iter, 1)
        
        ## ====== assert that training continues ======
        best_val_ndcg_k_2 = resume_train_fn(config=restore_dict['config'], trial=None)
        print(f'best_val_ndcg_k from resume 2nd to last chkpt training={best_val_ndcg_k_2}')
        
        ## ==== get the last latest checkpoint and assert that doesn't continue training from it because number of epochs is reached. ====
        last_restored_dict = restore_items_from_checkpoint(config['latest_checkpoint_dir'], get_earliest=False)
        self.assertTrue(earlier_restore_dict['global_step'] > 0)
        epoch = (last_restored_dict['global_step'] // TRAIN_BATCH_SIZE) // STEPS_PER_EPOCH_GLOBAL
        self.assertEqual(epoch, (num_epochs - 1))
        
        start_step = last_restored_dict['global_step'] // NUM_TRAIN_SHARDS
        n_iter = 0
        try:
            for batch_idx, padded_super_graph in enumerate(
                    last_restored_dict['train_dataloader_iter'], start=start_step):
                n_iter += 1
        except StopIteration:
            pass
        print(f"n_iter2={n_iter}")
        # self.assertEqual(n_iter, 0)

    if __name__ == '__main__':
        unittest.main()
