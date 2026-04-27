import glob
import os.path
import pathlib
import unittest

import jax.distributed
import jraph
import mlflow
import optax
from array_record.python import array_record_module
from flax import nnx

from absl import flags
from helper import *
from movie_lens_ranker.RandomAccessArrayRecordDataSource import *
from movie_lens_ranker.RatingsHistoryLookupTransform import *
from movie_lens_ranker.HardNegativeSamplingTransform import *
from movie_lens_ranker.SparseLocalSubgraphTransform import \
    SparseLocalSubgraphTransform
from movie_lens_ranker.JraphPaddedGraphTupleTransform import JraphPaddedGraphTupleTransform
from movie_lens_ranker.data_loading import *
from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.train import *
from movie_lens_ranker.util import read_embeddings, set_flags_from_dict
from movie_lens_ranker.util_plots import plot_mlflow_metrics


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
        tb_logs_uri = os.path.join(get_bin_dir(), "tb_logs")
        os.makedirs(latest_checkpoint_dir, exist_ok=True)
        os.makedirs(best_checkpoint_dir, exist_ok=True)
        os.makedirs(mlflow_dir, exist_ok=True)
        os.makedirs(mlflow_registry_dir, exist_ok=True)
        os.makedirs(tb_logs_uri, exist_ok=True)
        
        data_params_nontrainable = {
            'movies_uri':self.movies_uri,
            'recommendations_uri':self.recommendations_uri,
            'recommendations_ts_uri':self.recommendations_ts_uri,
            'ratings_train_uri':self.ratings_train_uri,
            'ratings_val_uri':self.ratings_val_uri,
            'train_negatives_uri':self.train_negatives_uri,
            'val_negatives_uri':self.val_negatives_uri,
            'seed':seed
        }
        data_params_trainable = {'max_history':max_history, 'num_candidates':num_candidates,
            'num_epochs':num_epochs,
            'batch_size':batch_size}
        model_params_nontrainable = {'latest_checkpoint_dir':latest_checkpoint_dir,
            'best_checkpoint_dir':best_checkpoint_dir,
            'movie_embeddings_uri': self.movie_embeddings_uri, 'user_embeddings_uri':self.user_embeddings_uri}
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
        tb_config = {'tb_logs_uri' : tb_logs_uri}
        
        config = {**data_params_nontrainable, **data_params_trainable,
            **model_params_nontrainable, **model_params_trainable,
            **mlflow_config, **tb_config}
        
        config['study_name'] = STUDY_NAME
        config["trial_id"] = 1
        config['phase'] = 'train'
        
        config['best_checkpoint_dir'] = f"{config['best_checkpoint_dir']}/{config['study_name']}/trial_{config['trial_id']}"
        config['latest_checkpoint_dir'] = f"{config['latest_checkpoint_dir']}/{config['study_name']}/trial_{config['trial_id']}"
        config['tb_logs_uri'] = f"{config['tb_logs_uri']}/{config['study_name']}/trial_{config['trial_id']}"

        os.makedirs(config['latest_checkpoint_dir'], exist_ok=True)
        os.makedirs(config['best_checkpoint_dir'], exist_ok=True)
        os.makedirs(config['tb_logs_uri'], exist_ok=True)

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
            print(f'run test metrics')
            #TOTO: needs a runner too
            test_dataloader = create_test_dataloader(
                movies_uri = self.movies_uri,
                recommendations_uri = self.recommendations_uri,
                recommendations_ts_uri = self.recommendations_ts_uri,
                ratings_uri = self.ratings_test_uri,
                negatives_uri = self.test_negatives_uri,
                max_history = max_history, num_candidates = num_candidates,
                batch_size = batch_size, seed = seed)
            eval_metrics = test_fn(model=model, test_dataloader=test_dataloader, top_k=top_k)
        
    if __name__ == '__main__':
        unittest.main()
