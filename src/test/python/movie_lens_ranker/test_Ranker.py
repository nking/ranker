import os.path
import unittest

import jax.distributed
import jraph
import optax
from array_record.python import array_record_module
from flax import nnx

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
from movie_lens_ranker.util import read_embeddings

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
    
    def print_local_info(self):
        print(f'local_devices={jax.local_devices()}') #[CpuDevice(id=0)]
        print(f'local device count={jax.local_device_count()}')
        print(f'process_count={jax.process_count()}')
        print(f'process_index={jax.process_index()}')

    def test_run_train(self):
        
        max_history = 200
        num_candidates = 40
        batch_size = 64
        num_epochs = 120
        seed = 1234
        
        top_k = 20
        learning_rate = 5e-4#1e-3
        weight_decay = 1e-4
        out_dim = 32
        hidden_dim = 64  # 2 * embed_in_dim is probably good
        num_layers = 2  # captures the 2-hop neighborhood.  3 tends to oversmooth
        num_heads = 4  # each head sees 64 hidden / 4 heads = 16 dimensional subspace
        dropout_rate = 0.1
        
        checkpoint_dir = os.path.join(get_bin_dir(), "checkpoints")
        latest_checkpoint_dir = os.path.join(checkpoint_dir, "latest")
        log_dir = os.path.join(get_bin_dir(), "logdir")
        mlflow_dir = os.path.join(get_bin_dir(), "mlflow")
        mlflow_artifacts_dir = os.path.join(get_bin_dir(), "mlflow_artifacts")
        os.makedirs(latest_checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(mlflow_dir, exist_ok=True)
        os.makedirs(mlflow_artifacts_dir, exist_ok=True)
        
        from movie_lens_ranker.train_ray import train_loop_per_worker
        
        nontrainable_data_config = {'movies_uri':self.movies_uri,
            'recommendations_uri':self.recommendations_uri,
            'recommendations_ts_uri':self.recommendations_ts_uri,
            'ratings_train_uri':self.ratings_train_uri,
            'ratings_val_uri':self.ratings_val_uri,
            'train_negatives_uri':self.train_negatives_uri,
            'val_negatives_uri':self.val_negatives_uri,
            'seed':seed
        }
        trainable_data_config = {'max_history':max_history, 'num_candidates':num_candidates,
            'num_epochs':num_epochs,
            'batch_size':batch_size}
        nontrainable_model_config = {'latest_checkpoint_dir':latest_checkpoint_dir,
            'log_dir':log_dir,
            'movie_embeddings_uri': self.movie_embeddings_uri, 'user_embeddings_uri':self.user_embeddings_uri}
        trainable_model_config = {'top_k':top_k, 'learning_rate':learning_rate, 'weight_decay':weight_decay,
            'out_dim':out_dim, 'hidden_dim':hidden_dim, 'num_layers':num_layers,
            'num_heads':num_heads, 'dropout_rate':dropout_rate,
        }
    
        mlflow_config = {
            'tracking_uri': mlflow_dir,
            'registry_uri': None,
            'experiment_id': None,
            'experiment_name': 'GraphRanker_dev',
            'tracking_token': None,
            'artifact_location': mlflow_artifacts_dir,
            'create_experiment_if_not_exists': True
        }
        
        config = {**nontrainable_data_config, **trainable_data_config, **nontrainable_model_config, **trainable_model_config}
        config["mlflow_config"] = mlflow_config
        
        ray_results_dir = os.path.join(get_bin_dir(), "ray_results")
        
        def get_env_resources():
            # 'cpu', 'gpu', or 'tpu'
            backend = jax.extend.backend.get_backend().platform
            num_local_devices = jax.local_device_count()
            devices = np.array(jax.devices())
            mesh = jax.sharding.Mesh(devices, axis_names=('data',))
            jax.set_mesh(mesh)
            device_dict = {}
            if backend == "tpu":
                jax.distributed.initialize()
                device_dict.update({"use_gpu": False, "use_tpu": True,
                    "resources_per_worker": {"TPU": num_local_devices}})
            elif backend == "gpu":
                # Usually, Ray handles GPU assignment automatically with use_gpu=True,
                # but specifying 1 GPU per worker ensures strict isolation.
                jax.distributed.initialize()
                device_dict.update({"use_gpu": True, "use_tpu": False,
                    "resources_per_worker": {"GPU": 1}})
            else:
                # CPU path
                device_dict.update({"use_gpu": False, "use_tpu": False,
                    "resources_per_worker": {"CPU": 1}})
            return device_dict
        
        env_resources = get_env_resources()
        
        train_loop_per_worker(config)
       
        if False:
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
