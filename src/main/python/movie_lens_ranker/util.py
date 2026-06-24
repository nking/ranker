import json
import os
import shutil
from collections import defaultdict
from typing import Tuple, Dict, List, Union, Any
import jax
import jax.numpy as jnp
from array_record.python import array_record_module
import msgpack
import numpy as np
from absl import flags
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
from jax.experimental import mesh_utils
import subprocess
# In JAX 0.8+, shard_map is typically in the main namespace

FLAGS = flags.FLAGS

data_params_nontrainable_keys = {
    'movies_uri', 'recommendations_uri',
    'recommendations_ts_uri',
    'ratings_train_3_uri', 'ratings_val_3_uri', 'ratings_test_3_uri',
    'ratings_train_liked_uri', 'ratings_val_liked_uri', 'ratings_test_liked_uri',
    'ratings_train_disliked_uri', 'ratings_val_disliked_uri', 'ratings_test_disliked_uri',
    'seed',
}
model_params_nontrainable_keys = {
    'latest_checkpoint_uri', 'best_checkpoint_uri',
    'movie_embeddings_uri', 'user_embeddings_uri', 'validate_checkpoint_restores'
}
mlflow_config_keys = {
    'mlflow_tracking_uri',
    'mlflow_experiment_id',
    'mlflow_experiment_name',
    # 'mlflow_tracking_token': None,
    'mlflow_parent_run_id'
}
hpo_config_keys = {'vizier_endpoint', 'trial_ids',
    'trial_id', 'train_id', 'test_id',
    'phase', 'LOGNAME', 'USER', 'debug', 'study_name', 'project_id', "debug",
    'output_hyperparams_uri', 'output_metrics_uri'
}
model_params_trainable_keys = {
    'top_k',
    'learning_rate',
    'weight_decay',
    'out_dim',
    'hidden_dim',
    'num_layers',
    'num_heads',
    'max_history',
    'num_candidates',
    'edge_embed_dim',
    'dropout_rate',
    'num_epochs',
    'batch_size',
}
def get_recognized_keys():
    return {
        *data_params_nontrainable_keys,
        *model_params_nontrainable_keys,
        *mlflow_config_keys,
        *model_params_trainable_keys,
        *hpo_config_keys,
        *{'connections_check', 'debug'}
    }

def app_runner_is_missing_minimum_required_keys(config: Dict[str, Any]) -> bool:
    """
    a method mostly for use to silently return from requests by docker compose polling
    :param config: dictionary of flags passed to app_runner
    :return: False if has minumum required keys, else returns False
    """
    for key in ("study_name", "phase",  "mlflow_tracking_uri"):
        if config.get(key, None) is None:
            logging.info(f'missing a key')
            return True
    return False
    
def get_canonical_mlflow_run_name(config: Dict[str, Any]) -> str:
    if config['phase'].find('tune') == 0:
        run_name = f"trial_{config.get('trial_id', 0)}"
    elif config['phase'].find('train') == 0:
        run_name = f"train_{config.get('train_id',0)}"
    elif config['phase'].find('test') == 0:
        run_name = f"test_{config.get('test_id',0)}"
    else:
        raise ValueError(f"Invalid phase={config['phase']}")
    return run_name
    
def get_model_mesh():
    device_grid = mesh_utils.create_device_mesh((jax.process_count(), jax.local_device_count()))
    model_mesh = jax.sharding.Mesh(device_grid, axis_names=('processes', 'local_devices'))
    #data_sharding = jax.sharding.NamedSharding(model_mesh, P('local_data'))
    return model_mesh

def set_flags_from_dict(params_dict, store_only_recognized:bool=True):
    """Sets absl FLAGS from a dictionary, ensuring they are marked as parsed."""
    recognized_keys = get_recognized_keys() if store_only_recognized else None
    for key, value in params_dict.items():
        if store_only_recognized and key not in recognized_keys:
            continue
        if hasattr(FLAGS, key):
            try:
                setattr(FLAGS, key, value)
            except Exception as e:
                pass
        else:
            # Optional: Define the flag on the fly if it's missing
            # This is helpful for dynamic HPO params
            try:
                flags.DEFINE_alias(key, key)  # Or use a generic DEFINE
            except Exception:
                if isinstance(value, str):
                    flags.DEFINE_string(key, key, f"{key}={value}")
                elif isinstance(value, bool):
                    #check for boolean must come before check for int because it is narrower type.  int includes int and bool
                    flags.DEFINE_bool(key, value, f"{key}={value}")
                elif isinstance(value, int):
                    flags.DEFINE_integer(key, value, f"{key}={value}")
                elif isinstance(value, float):
                    flags.DEFINE_float(key, value, f"{key}={value}")
                
            setattr(FLAGS, key, value)
    # Crucial for unit tests: tells absl it's safe to read these values
    if not FLAGS.is_parsed():
        FLAGS.mark_as_parsed()

def define_flags():
    """
    define global flags.  they are received in main method from command line arguments, xmanager arguments, params etc.
    Note: if using this in a jupyter notebook, it might need to be enclosed by a try/except to avoid errors when a cell containing
    this is reinvoked.
    :return:
    """
    if 'movies_uri' in FLAGS:
        return
    
    flags.DEFINE_string('movies_uri', None, 'uri for array_record containing movie ids')
    
    flags.DEFINE_string("recommendations_uri", default=None,
        help="uri for array_record containing, each row being [user_id, [movie_ids]]"
    )
    flags.DEFINE_string("recommendations_ts_uri", default=None,
        help="uri for array_record containing the timestamps for recommendations_uri, each row being [user_id, [timestamps]]"
    )
    
    flags.DEFINE_string("ratings_train_liked_uri", default=None,
        help="uri for array_record containing the ratings train dataset having ratings > 3, each row being [user_id, movie_id, rating, timestamp]. for this project the dataset should contain only positives"
    )
    flags.DEFINE_string("ratings_val_liked_uri", default=None,
        help="uri for array_record containing the ratings val dataset having ratings > 3, each row being [user_id, movie_id, rating, timestamp].  for this project the dataset should contain only positives"
    )
    flags.DEFINE_string("ratings_test_liked_uri", default=None,
        help="uri for array_record containing the ratings test dataset having ratings > 3, each row being [user_id, movie_id, rating, timestamp].  for this project the dataset should contain only positives"
    )
    
    flags.DEFINE_string("ratings_train_disliked_uri", default=None,
        help="uri for array_record containing the ratings train dataset having ratings < 3, each row being [user_id, movie_id, rating, timestamp]. for this project the dataset should contain only positives"
    )
    flags.DEFINE_string("ratings_val_disliked_uri", default=None,
        help="uri for array_record containing the ratings val dataset having ratings < 3, each row being [user_id, movie_id, rating, timestamp].  for this project the dataset should contain only positives"
    )
    flags.DEFINE_string("ratings_test_disliked_uri", default=None,
        help="uri for array_record containing the ratings test dataset having ratings < 3, each row being [user_id, movie_id, rating, timestamp].  for this project the dataset should contain only positives"
    )
    
    
    flags.DEFINE_string("ratings_train_3_uri", default=None,
        help="uri for array_record containing the ratings train dataset having ratings == 3, each row being [user_id, movie_id, rating, timestamp]. for this project the dataset should contain only positives"
    )
    flags.DEFINE_string("ratings_val_3_uri", default=None,
        help="uri for array_record containing the ratings val dataset having ratings == 3, each row being [user_id, movie_id, rating, timestamp].  for this project the dataset should contain only positives"
    )
    flags.DEFINE_string("ratings_test_3_uri", default=None,
        help="uri for array_record containing the ratings test dataset having ratings == 3, each row being [user_id, movie_id, rating, timestamp].  for this project the dataset should contain only positives"
    )
    

    flags.DEFINE_integer("seed", default=0,
        help="seed used for pseudo random number generator"
    )
    # ====== TRAINABLE DATA PARAMS ======
    flags.DEFINE_integer("max_history", default=200,
        help="maximum number per user of positive ratings to use for their graph"
    )
    flags.DEFINE_integer("num_candidates", default=40,
        help="number per user of negatives + positive to use for their final graph"
    )
    flags.DEFINE_integer("num_epochs", default=40,
        help="number of epochs to train"
    )
    flags.DEFINE_integer("batch_size", default=64,
        help="number of data examples to use at a time for training"
    )
    # ====== NON-TRAINABLE MODEL PARAMS ======
    flags.DEFINE_string("user_embeddings_uri", default=None,
        help="uri to read the retrieval written user embeddings. each row holds [user_id] [embeddings]]")
        
    flags.DEFINE_string("movie_embeddings_uri", default=None,
        help="uri to read the retrieval written movie embeddings. each row holds [movie_id] [embeddings]]"
    )
    flags.DEFINE_string("latest_checkpoint_uri", default=None,
        help="uri to write latest checkpoints too.  model, data, optimizer and seed state are saved.   the study_name and trial number will be appended to the given path"
    )
    flags.DEFINE_string("best_checkpoint_uri", default=None,
        help="uri to write checkpoints to for best model.  model, data, optimizer and seed state are saved.  it's also the uri to read best model from when phase='test-best'. the study_name and trial_id will be appended to the given path"
    )
    flags.DEFINE_string("test_checkpoint_uri", default=None,
        help="uri to read orbax checkpointed model for tests when phase='test-given'"
    )
    flags.DEFINE_bool("validate_checkpoint_restores", default=False, help="compares validation metrics of active model  "
        "with validation metrics for saved active model restored, but only if the phase is a train phase with save checkpoints enabled."
    )
    #TODO: remove mlflow_experiment_name and make it clear that internally mlflow_experiment_name==study_name
    flags.DEFINE_string("study_name", default=None,
        help="study name for use in vizier study and mflow experiment name.  must be same as 'mlflow_experiment_name'"
    )
    flags.DEFINE_string("project_id", default=None,
        help="project_id for use with vizier HPO"
    )
    flags.DEFINE_string("vizier_endpoint", default=None,
        help="endpoint for vizier server"
    )
    flags.DEFINE_string("trial_ids", default="[0]",
        help="a string serialization of array of integer trial ids for a worker, e.g. '[0, 1]' and 2 trials will be conducted"
    )
    flags.DEFINE_integer("test_id", default=0,
        help="an id to assign to test if phase is 'test-best' or 'test-given'"
    )
    flags.DEFINE_integer("train_id", default=0,
        help="an id to assign to train if phase is 'train-best' or 'train-given'"
    )
    flags.DEFINE_integer("trial_id", default=0,
        help="an id internally in a single trial _train_fn run"
    )
    flags.DEFINE_enum(
        'phase', 'train-best',
        ['tune', 'train-best', 'train-given', 'test-best', 'test-given',
            'export-hpo-results', 'export-train-results', 'export-test-results'],
        'mode for running the train_fn.  tune: HPO run; '
        'train-best: use best HPs from tune; '
        'train-given: use given HPs; '
        'test-best: use test_fn for best model for the given study_name and project_id;'
        'test-given: use test_fn with test_checkpoint_uri; '
        'export-hpo-results: extract the HPO best results into params and metrics json files; '
        'export-test-results: extract the test results into params and metrics json files; '
        'export-train-results: extract the train results into params and metrics json files'
    )
    flags.DEFINE_string("mlflow_tracking_uri", default=None,
        help="MLFlow tracking uri"
    )
    flags.DEFINE_string("mlflow_experiment_name", default=None,
        help="MLFlow experiment name.  must be the same as study_name"
    )
    flags.DEFINE_string("LOGNAME", default=None,
        help="linux env variable name"
    )
    flags.DEFINE_string("USER", default=None,
        help="linux env variable name"
    )
    # ====== TRAINABLE MODEL PARAMS, for tune phase, they're supplied by internal code ======
    flags.DEFINE_integer("top_k", default=20,
        help="used when calculating metrics NDCG@k, recal@k, MRR@k"
    )
    flags.DEFINE_float("learning_rate", default=5e-4,
        help="learning_rate for the AdamW optimizer"
    )
    flags.DEFINE_float("weight_decay", default=1e-4,
        help="weight_decay for the AdamW optimizer"
    )
    flags.DEFINE_integer("out_dim", default=32,
        help="output dimension of the score head dense layer in GraphRanker"
    )
    flags.DEFINE_integer("hidden_dim", default=64,
        help="size of hidden layers per head in the GATv2 layer of GraphRanker"
    )
    flags.DEFINE_integer("num_layers", default=2,
        help="number of layers in the GATv2 layer of the GraphRanker"
    )
    flags.DEFINE_integer("num_heads", default=4,
        help="number of attention heads in the GATv2 layer of the GraphRanker"
    )
    flags.DEFINE_integer("edge_embed_dim", default=8,
        help="size of output of the GATv2 layer of GraphRanker"
    )
    flags.DEFINE_float("dropout_rate", default=0.1,
        help="the dropout probability of a layer in the GATv2 layer of the GraphRanker"
    )
    flags.DEFINE_string("output_hyperparams_uri", default=None,
        help="uri to write the best hyperparameters from HPO in json format"
    )
    flags.DEFINE_string("output_metrics_uri", default=None,
        help="uri to write the metrics from best hyperparameters from HPO in json format"
    )
    flags.DEFINE_bool("debug", default=False,
        help="prints debug statements"
    )
    flags.DEFINE_integer("connections_check", default=0,
        help="set to 1 to run connections check before phase.  "
             "additionally, if JAX_PLATFORM_NAME=gpu there will be a check for expected number of GPUs found")

def stringify_mlflow_params(config:dict):
    return {k: json.dumps(v) for k, v in config.items() if
        k.find('?') == -1}

def destringify_mlflow_params(params:dict):
    config = {}
    for k, v in params.items():
        try:
            config[k] = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            # Fallback for plain strings that aren't valid JSON (like "adam")
            config[k] = v
    return config
    
def read_embeddings(user_embeddings_uri:str, movie_embeddings_uri:str, batch_size:int=1024) -> Tuple[jnp.ndarray, int]:
    """
    read the user and movie embeddings and concatenate them: [row of zeros, user embeddings, movie embeddings].  the
    row of zeros is needed because user_ids start with 1.
    :param user_embeddings_uri:
    :param movie_embeddings_uri:
    :param batch_size:
    :return: a tuple of:
        concatenated row of zeros, user embeddings, movie embeddings,
        num_users
    """
    user_emb = _read_embeddings(user_embeddings_uri, batch_size=batch_size)
    movie_emb = _read_embeddings(movie_embeddings_uri, batch_size=batch_size)
    zero_row = jnp.zeros((1, user_emb.shape[1]))
    emb = jnp.concatenate([zero_row, user_emb, movie_emb])
    return emb, len(user_emb)

    
def _read_embeddings(embeddings_uri:str, batch_size:int=1024) ->  jnp.ndarray:
    """
    given the embedding uri return and the embeddings
    as a 2D jnp array.
    :param embeddings_uri:
    :param batch_size:
    :return:
    """
    ids = []
    embeddings = []
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(embeddings_uri)
        n_records = reader.num_records()
        for i in range(0, n_records, batch_size):
            stop = min(i + batch_size, n_records)
            batch_bytes = reader.read([x for x in range(i, stop)])
            batch = [msgpack.unpackb(b, use_list=True) for b in batch_bytes]  # list of [int, [list]] of id and embedding
            for record in batch:
                ids.append(record[0])
                embeddings.append(record[1])
        #just in case the array_record was not written in order, sort by id
        embeddings = [val for _, val in sorted(zip(ids, embeddings))]
    except Exception as e:
        logging.exception(f'error in _read_embeddings: {e}')
        raise e
    finally:
        if reader is not None:
            reader.close()
    return jnp.array(embeddings)

def _read_user_recommendations_array_record(user_uri:str, batch_size:int=1048) -> np.ndarray:
    """
    read the array_record at user_uri into a dictionary with key user_id and values
    being the set of movie_ids.
    :param user_uri: the uri for the array_record of user recommendations
    :param batch_size: batch size to use in reading the array_record.
    :return: a dictionary with key user_id and values
    being the set of movies.
    """
    ids = []
    output = []
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(user_uri)
        n_records = reader.num_records()
        for i in range(0, n_records, batch_size):
            stop = min(i + batch_size, n_records)
            batch_bytes = reader.read([x for x in range(i, stop)])
            batch = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  # list of tuples user_id, movie_ids
            for record in batch:
                ids.append(record[0])
                output.append(record[1])
    except Exception as e:
        raise e
    finally:
        if reader is not None:
            reader.close()
    output = np.array([val for _, val in sorted(zip(ids, output))], dtype=np.int64)
    return output

def read_movies_array_record(movies_uri:str, batch_size:int=1048) -> List[int]:
    """
    read the array_record at movies_uri into a list of movie ids
    :param movies_uri: the uri for the array_record of movie ids
    :param batch_size: batch size to use in reading the array_record.
    :return: a list of movie_ids
    """
    movie_ids = []
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(movies_uri)
        n_records = reader.num_records()
        for i in range(0, n_records, batch_size):
            stop = min(i + batch_size, n_records)
            batch_bytes = reader.read([x for x in range(i, stop)])
            records = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  #(movie_id, title, genres)
            for record in records:
                movie_ids.append(record[0])
        #not necessary, but might as well sort in case written out of order
        movie_ids.sort()
    except Exception as e:
        logging.exception(f'Error in read_movies_array_record: {e}')
        raise e
    finally:
        if reader is not None:
            reader.close()
    return movie_ids

def read_recommendations(user_recommendations_uri:str, batch_size:int=1048) -> np.ndarray:
    """
    read the array_record at user_recommendations_uri into a ndarray where the indices are implied user_id-1
       and rows are the list of movie_ids recommended for the user, ordered
    :param user_recommendations_uri: the uri for the array_record of user recommendations that exclude their seen movies
    :param batch_size: batch size to use in reading the array_record.
    :return: a dictionary with key user_id and values
    being the set of movies that the user was recommended and hasn't seen.
    """
    return _read_user_recommendations_array_record(user_recommendations_uri, batch_size=batch_size)

def build_history_lookup(ratings_uri_list: Union[str, List[str]], batch_size: int = 1024) -> Tuple[Dict[int, Tuple[List, List, List]], int]:
    """
    Scans the training ratings once to build an O(1) user lookup.
    Arguments:
        ratings_uri_list: list of uri to ratings array_record holding tuples of user_id, movie_id, rating, timestamp
        batch_size: size of batch to use when reading.  does not affect returned data structure size
    returns: tuple of dictionary of { user_id: {ts, movie_id, rating} } in which ts, movie_id
    and rating values are numpy arrays sorted by timestamp, and returns the maximum number of movies seen by any user
    in this dataset.
    """
    if isinstance(ratings_uri_list, str):
        ratings_uri_list = [ratings_uri_list]
        
    lookup = defaultdict(
        lambda: {"ts": [], "movie_id": [], "rating": []})
    for ratings_uri in ratings_uri_list:
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(ratings_uri)
            n_records = reader.num_records()
            for i in range(0, n_records, batch_size):
                stop = min(i + batch_size, n_records)
                batch_bytes = reader.read([x for x in range(i, stop)])
                batch = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  # list of tuples
                for record in batch:
                    u = record[0]
                    lookup[u]["movie_id"].append(record[1])
                    lookup[u]["rating"].append(record[2])
                    lookup[u]["ts"].append(record[3])
        except Exception as e:
            logging.exception(f'Error in build_history_lookup: {e}')
            raise e
        finally:
            if reader is not None:
                reader.close()
    #logging.info(f'rewrite lookup size = {len(lookup)}')
    max_history = 0
    lookup2 = {}
    for u in lookup:
        # sort all lists by timestamp
        ts = lookup[u]["ts"]
        idx = sorted(range(len(ts)), key=lambda ii: ts[ii])
        m = lookup[u]["movie_id"]
        r = lookup[u]["rating"]
        max_history = max(max_history, len(m))
        lookup2[u] = (
            [ts[ii] for ii in idx], [m[ii] for ii in idx],
            [r[ii] for ii in idx])
    
    return lookup2, max_history


def calc_number_jax_graph_components(batch_size: int, max_history: int,
        num_candidates: int,  n_local_devices:int) -> Dict[str, int]:
    """
    calculate the padding for graph components.  note that the number of local devices is considered in order
    to make max_graphs divisible by jax.local_devices_count() to give an integer quotient.
    :param n_local_devices:
    :param batch_size:
    :param max_history:
    :param num_candidates:
    :return:
    """
    
    # 40->50, #123->200, #1234->2000, #12345->20000
    def next_64(x) -> int:
        return 64 * (1 + int(x // 64))
    
    max_nodes = next_64(batch_size * (1 + max_history + num_candidates))
    max_edges = next_64(batch_size * (max_history + num_candidates))
    
    #batch_size + 1 extra for every local device + padd up to integer quotient of local_devices
    add_to = n_local_devices - (batch_size % n_local_devices)
    max_graphs = batch_size + n_local_devices + add_to
    
    return {'max_nodes': max_nodes, 'max_edges': max_edges,
        'max_graphs': max_graphs}

def is_running_on_gpu() -> bool:
    for device in jax.local_devices():
        if device.platform == 'gpu':
            return True
    return False

def find_executable_path(binary_name: str):
    """Run a shell command and print output.
    :param binary_name: name of binary path to resolve
    """
    path = shutil.which(binary_name)
    if path:
        return path
    
    # If not found, explicitly check common installation locations
    home = os.path.expanduser("~")
    fallback_locations = [
        f"/usr/bin/{binary_name}",  # Alternate Linux path
        f"/usr/local/bin/{binary_name}",  # Standard Linux path
        f"/opt/bin/{binary_name}",
        f"/bin/{binary_name}",
        f"/snap/bin/{binary_name}",
        os.path.join(home, ".local", "bin", binary_name)  # Local user bin
    ]
    
    for path in fallback_locations:
        # os.path.exists checks if it's there, kindos.access checks if it is executable
        if os.path.exists(path) and os.access(path, os.X_OK):
            logging.info(f"⚠️ Found {binary_name} via fallback path: {path}")
            return path
    
    # If we exhaust all options, raise a clear error
    raise FileNotFoundError(
        f"Could not find the {binary_name} executable in PATH or fallback directories.")

def get_gpu_stats() -> str:
    """Fetches real-time GPU utilization and VRAM usage."""
    if not is_running_on_gpu():
        return ""
    if os.environ.get('NO_NVIDIA-SMI'):
        return ""
    try:
        nvidia_path = find_executable_path('nvidia-smi')
    except Exception as e:
        os.environ['NO_NVIDIA-SMI'] = "1"
        return ""
    try:
        # Queries index, compute util %, used VRAM, and total VRAM
        cmd = [
            nvidia_path,
            '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Format the output into a readable single-line string
        stats = result.stdout.strip().replace('\n', ' | ')
        return f"Hardware Stats [ID, Util%, Used, Total]: {stats}"
    except Exception as e:
        return f"Could not fetch GPU stats: {e}"