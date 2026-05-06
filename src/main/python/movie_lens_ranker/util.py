import argparse
from collections import defaultdict
from typing import Tuple, Dict, List, Union, Set
import jax
import jax.numpy as jnp
from array_record.python import array_record_module
import msgpack
import numpy as np
from absl import flags

# In JAX 0.8+, shard_map is typically in the main namespace

FLAGS = flags.FLAGS

data_params_nontrainable_keys = {'movies_uri', 'recommendations_uri',
    'recommendations_ts_uri',
    'ratings_train_uri', 'ratings_val_uri', 'train_negatives_uri',
    'val_negatives_uri',
    'seed'
}
model_params_nontrainable_keys = {'latest_checkpoint_uri',
    'best_checkpoint_uri', 'movie_embeddings_uri', 'user_embeddings_uri',
    'mlflow_config'}
mlflow_config_keys = {
    'mlflow_tracking_uri',
    'mlflow_experiment_id',
    'mlflow_experiment_name',
    # 'mlflow_tracking_token': None,
    'mlflow_parent_run_id'
}
optuna_config_keys = {'optuna_storage_uri'}
model_params_trainable_keys = {
    'top_k',
    'learning_rate',
    'weight_decay',
    'out_dim',
    'hidden_dim',
    'num_layers',
    'num_heads',
    'edge_embed_dim',
    'dropout_rate',
}
data_params_trainable_keys = {'max_history',
    'num_candidates',
    'num_epochs',
    'batch_size'}

def get_env_resources():
    # 'cpu', 'gpu', or 'tpu'
    backend = jax.extend.backend.get_backend().platform
    num_local_devices = jax.local_device_count()
    devices = np.array(jax.devices())
    mesh = jax.sharding.Mesh(devices, axis_names=('data',))
    jax.set_mesh(mesh)
    device_dict = {}
    if backend == "tpu":
        device_dict.update({"use_gpu": False, "use_tpu": True,
            "resources_per_worker": {"TPU": num_local_devices}})
    elif backend == "gpu":
        # Usually, Ray handles GPU assignment automatically with use_gpu=True,
        # but specifying 1 GPU per worker ensures strict isolation.
        device_dict.update({"use_gpu": True, "use_tpu": False,
            "resources_per_worker": {"GPU": 1}})
    else:
        # CPU path
        device_dict.update({"use_gpu": False, "use_tpu": False,
            "resources_per_worker": {"CPU": 1}})
    return device_dict, mesh


def parse_args_into_dict_with_exists_check():
    parser = get_args_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    #for key in {**data_params_nontrainable_keys, **data_params_trainable_keys,
    #    **model_params_nontrainable_keys, **model_params_trainable_keys,
    #    **optuna_config_keys}:
    #    if key not in args_dict:
    #        raise ValueError("missing required argument: {}".format(key))
    return args_dict

def set_flags_from_dict(params_dict):
    """Sets absl FLAGS from a dictionary, ensuring they are marked as parsed."""
    for key, value in params_dict.items():
        if hasattr(FLAGS, key):
            setattr(FLAGS, key, value)
        else:
            # Optional: Define the flag on the fly if it's missing
            # This is helpful for dynamic Optuna params
            try:
                flags.DEFINE_alias(key, key)  # Or use a generic DEFINE
            except Exception:
                if isinstance(value, str):
                    flags.DEFINE_string(key, key, f"{key}={value}")
                elif isinstance(value, int):
                    flags.DEFINE_integer(key, value, f"{key}={value}")
                elif isinstance(value, float):
                    flags.DEFINE_float(key, value, f"{key}={value}")
                elif isinstance(value, bool):
                    flags.DEFINE_bool(key, value, f"{key}={value}")
            setattr(FLAGS, key, value)
    # Crucial for unit tests: tells absl it's safe to read these values
    if not FLAGS.is_parsed():
        FLAGS.mark_as_parsed()


def define_flags():
    """
    define global flags.  they are received in main method from command line arguments, xmanager arguments, kaic params etc.
    Note: if using this in a jupyter notebook, it might need to be enclosed by a try/except to avoid errors when a cell contianing
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
    flags.DEFINE_string("ratings_train_uri", default=None,
        help="uri for array_record containing the ratings train dataset, each row being [user_id, movie_id, rating, timestamp]. for this project the dataset should contain only positives"
    )
    flags.DEFINE_string("ratings_val_uri", default=None,
        help="uri for array_record containing the ratings val dataset, each row being [user_id, movie_id, rating, timestamp].  for this project the dataset should contain only positives"
    )
    flags.DEFINE_string("train_negatives_uri", default=None,
        help="uri for array_record containing the ratings train dataset negatives, each row being [user_id, movie_id, rating, timestamp]"
    )
    flags.DEFINE_string("val_negatives_uri", default=None,
        help="uri for array_record containing the ratings val dataset negatives, each row being [user_id, movie_id, rating, timestamp]"
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
        help="uri to write latest checkpoints too.  model, data, optimizer and seed state are saved"
    )
    flags.DEFINE_string("best_checkpoint_uri", default=None,
        help="uri to write checkpoints to for best model.  model, data, optimizer and seed state are saved"
    )
    flags.DEFINE_string("study_name", default=None,
        help="study name for use in optuna study and mflow experiment name"
    )
    flags.DEFINE_string("optuna_storage_uri", default=None,
        help="uri for optuna db"
    )
    flags.DEFINE_integer("trial_id", default=1,
        help="trial id for use with optuna, and orbax checkpoints"
    )
    flags.DEFINE_string("phase", default="train",
        help="tag used with mlflow run.  e.g. train, e.g. test"
    )
    flags.DEFINE_string("mlflow_tracking_uri", default=None,
        help="MLFlow tracking uri"
    )
    flags.DEFINE_string("mlflow_experiment_name", default=None,
        help="MLFlow experiment name"
    )
    flags.DEFINE_string("LOGNAME", default=None,
        help="linux env variable name"
    )
    flags.DEFINE_string("USER", default=None,
        help="linux env variable name"
    )
    # ====== TRAINABLE MODEL PARAMS ======
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
        help="size of hidden layers per head in the GATv2 layer of GraphPranker"
    )
    flags.DEFINE_integer("num_layers", default=2,
        help="number of layers in the GATv2 layer of the GraphRanker"
    )
    flags.DEFINE_integer("num_heads", default=4,
        help="number of attention heads in the GATv2 layer of the GraphRanker"
    )
    flags.DEFINE_integer("edge_embed_dim", default=8,
        help="size of output of the GATv2 layer of GraphPranker"
    )
    flags.DEFINE_float("dropout_rate", default=0.1,
        help="the dropout probability of a layer in the GATv2 layer of the GraphRanker"
    )
    flags.DEFINE_bool("debug", default=False,
        help="prints debug statements"
    )

def get_args_parser():
    parser = argparse.ArgumentParser(description="parse for training run", )
    # ====== NON-TRAINABLE DATA PARAMS ======
    parser.add_argument("--movies_uri", type=str,
        help="uri for array_record containing movie ids"
    )
    parser.add_argument("--recommendations_uri", type=str,
        help="uri for array_record containing, each row being [user_id, [movie_ids]]"
    )
    parser.add_argument("--recommendations_ts_uri", type=str,
        help="uri for array_record containing the timestamps for recommendations_uri, each row being [user_id, [timestamps]]"
    )
    parser.add_argument("--ratings_train_uri", type=str,
        help="uri for array_record containing the ratings train dataset, each row being [user_id, movie_id, rating, timestamp]. for this project the dataset should contain only positives"
    )
    parser.add_argument("--ratings_val_uri", type=str,
        help="uri for array_record containing the ratings val dataset, each row being [user_id, movie_id, rating, timestamp].  for this project the dataset should contain only positives"
    )
    parser.add_argument("--train_negatives_uri", type=str,
        help="uri for array_record containing the ratings train dataset negatives, each row being [user_id, movie_id, rating, timestamp]"
    )
    parser.add_argument("--val_negatives_uri", type=str,
        help="uri for array_record containing the ratings val dataset negatives, each row being [user_id, movie_id, rating, timestamp]"
    )
    parser.add_argument("--seed", type=int, default=0,
        help="seed used for pseudo random number generator"
    )
    # ====== TRAINABLE DATA PARAMS ======
    parser.add_argument("--max_history", type=int,
        help="maximum number per user of positive ratings to use for their graph"
    )
    parser.add_argument("--num_candidates", type=int,
        help="number per user of negatives + positive to use for their final graph"
    )
    parser.add_argument("--num_epochs", type=int,
        help="number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int,
        help="number of data examples to use at a time for training"
    )
    # ====== NON-TRAINABLE MODEL PARAMS ======
    parser.add_argument("--user_embeddings_uri", type=str,
        help="uri to read the retrieval written user embeddings. each row holds [user_id] [embeddings]]"
    )
    parser.add_argument("--movie_embeddings_uri", type=str,
        help="uri to read the retrieval written movie embeddings. each row holds [movie_id] [embeddings]]"
    )
    parser.add_argument("--latest_checkpoint_uri", type=str,
        help="uri to write latest checkpoints too.  model, data, optimizer and seed state are saved"
    )
    parser.add_argument("--best_checkpoint_uri", type=str,
        help="uri to write checkpoints to for best model.  model, data, optimizer and seed state are saved"
    )
    parser.add_argument("--study_name", type=str,
        help="study name for use in optuna study and mflow experiment name"
    )
    parser.add_argument("--optuna_storage_uri", type=str,
        help="uri for optuna db"
    )
    parser.add_argument("--trial_id", type=int,
        help="trial id for use with optuna, and orbax checkpoints"
    )
    parser.add_argument("--phase", type=int,
        help="tag used with mlflow run.  e.g. train, e.g. test"
    )
    parser.add_argument("--mlflow_tracking_uri", type=str,
        help="MLFlow tracking uri"
    )
    parser.add_argument("--mlflow_experiment_name", type=str,
        help="MLFlow experiment name"
    )
    # ====== TRAINABLE MODEL PARAMS ======
    parser.add_argument("--top_k", type=int,
        help="used when calculating metrics NDCG@k, recal@k, MRR@k"
    )
    parser.add_argument("--learning_rate", type=float,
        help="learning_rate for the AdamW optimizer"
    )
    parser.add_argument("--weight_decay", type=float,
        help="weight_decay for the AdamW optimizer"
    )
    parser.add_argument("--out_dim", type=int,
        help="output dimension of the score head dense layer in GraphRanker"
    )
    parser.add_argument("--hidden_dim", type=int,
        help="size of hidden layers per head in the GATv2 layer of GraphPranker"
    )
    parser.add_argument("--num_layers", type=int,
        help="number of layers in the GATv2 layer of the GraphRanker"
    )
    parser.add_argument("--num_heads", type=int,
        help="number of attention heads in the GATv2 layer of the GraphRanker"
    )
    parser.add_argument("--edge_embed_dim", type=int,
        help="size of output of the GATv2 layer of GraphPranker"
    )
    parser.add_argument("--dropout_rate", type=float,
        help="the dropout probability of a layer in the GATv2 layer of the GraphRanker"
    )
    parser.add_argument("--debug", type=bool,
        help="prints debug statements"
    )
    return parser


def read_embeddings(user_embeddings_uri:str, movie_embeddings_uri:str, batch_size:int=1024) -> jnp.ndarray:
    """
    read the user and movie embeddings and concatentate them: [row of zeros, user embeddings, movie embeddings].  the
    row of zeros is needed because user_ids start with 1.
    :param user_embeddings_uri:
    :param movie_embeddings_uri:
    :param batch_size:
    :return: a tuple of dictionary of original user_id to new user_id, a dictionary of original movie id to new movie id, and the
    concatenated embeddings.
    """
    user_emb = _read_embeddings(user_embeddings_uri, batch_size=batch_size)
    movie_emb = _read_embeddings(movie_embeddings_uri, batch_size=batch_size)
    zero_row = jnp.zeros((1, user_emb.shape[1]))
    emb = jnp.concatenate([zero_row, user_emb, movie_emb])
    return emb

    
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
        raise e
    finally:
        if reader is not None:
            reader.close()
    return jnp.array(embeddings)

def read_user_negatives(negatives_uri:Union[str, List[str]], batch_size:int=1048) -> Dict[int, Set[int]]:
    """
    read the array_record at negatives_uri into a dictionary with key user_id and values
    being the set of movies that the user was recommended, but did not like.
    :param negatives_uri: the uri for the exact hard negatives array_record
    :param batch_size: batch size to use in reading the array_record.
    :return: a dictionary with key user_id and values
    being the set of movies that the user was recommended, but did not like.
    """
    lookup: Dict[int, Set[int]] = {}
    if isinstance(negatives_uri, str):
        negatives_uri = [negatives_uri]
    for negative_uri in negatives_uri:
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(negative_uri)
            n_records = reader.num_records()
            for i in range(0, n_records, batch_size):
                stop = min(i + batch_size, n_records)
                batch_bytes = reader.read([x for x in range(i, stop)])
                batch = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  # list of tuples user_id, movie_ids
                for record in batch:
                    if record[0] not in lookup:
                        lookup[record[0]] = set()
                    lookup[record[0]].update(record[1])
        except Exception as e:
            raise e
        finally:
            if reader is not None:
                reader.close()
    return lookup

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
            raise e
        finally:
            if reader is not None:
                reader.close()
    print(f'rewrite lookup size = {len(lookup)}')
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
