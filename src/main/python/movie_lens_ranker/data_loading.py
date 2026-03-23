from collections import defaultdict
from typing import Tuple, Dict, List, Set

import array_record
import grain.python as grain
import jax.numpy as jnp
from array_record.python import array_record_module
import msgpack

# each kubeflow worker will load these into memory
# or this will be changed to a distributable loading

def read_embeddings(embeddings_uri:str, batch_size:int=1024) -> Dict[int, List]:
    lookup = {}
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(embeddings_uri)
        n_records = reader.num_records()
        for i in range(0, n_records, batch_size):
            stop = min(i + batch_size, n_records)
            batch_bytes = reader.read([x for x in range(i, stop)])
            batch = [msgpack.unpackb(b, use_list=True) for b in
                batch_bytes]  # list of [int, [list]] of id and embedding
            for record in batch:
                lookup[record[0]] = record[1]
    except Exception as e:
        raise e
    finally:
        if reader is not None:
            reader.close()
    return lookup

def read_user_exact_negatives(exact_hard_negatives_uri:str, batch_size:int=1048) -> Dict[int, Set[int]]:
    """
    read the array_record at exact_hard_negatives_uri into a dictionary with key user_id and values
    being the set of movies that the user was recommended, but did not like.
    :param exact_hard_negatives_uri: the uri for the exact hard negatives array_record
    :param batch_size: batch size to use in reading the array_record.
    :return: a dictionary with key user_id and values
    being the set of movies that the user was recommended, but did not like.
    """
    return _read_user_recommendations_array_record(exact_hard_negatives_uri, batch_size)

def _read_user_recommendations_array_record(user_uri:str, batch_size:int=1048) -> Dict[int, Set[int]]:
    """
    read the array_record at user_uri into a dictionary with key user_id and values
    being the set of movie_ids.
    :param user_uri: the uri for the array_record of user recommendations
    :param batch_size: batch size to use in reading the array_record.
    :return: a dictionary with key user_id and values
    being the set of movies.
    """
    lookup:Dict[int, Set[int]] = {}
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(user_uri)
        n_records = reader.num_records()
        for i in range(0, n_records, batch_size):
            stop = min(i + batch_size, n_records)
            batch_bytes = reader.read([x for x in range(i, stop)])
            batch = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  # list of tuples user_id, movie_ids
            for record in batch:
                lookup[record[0]] = set(record[1])
    except Exception as e:
        raise e
    finally:
        if reader is not None:
            reader.close()
    return lookup

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
            batch = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  #list of ids
            movie_ids.extend(batch)
    except Exception as e:
        raise e
    finally:
        if reader is not None:
            reader.close()
    return movie_ids

def read_user_unseen_recommendations(user_unseen_recommendations_uri:str, batch_size:int=1048) -> Dict[int, Set[int]]:
    """
    read the array_record at user_unseen_recommendations_uri into a dictionary with key user_id and values
    being the set of movies that the user was recommended, but did not like.
    :param user_unseen_recommendations_uri: the uri for the array_record of user recommendations that exclude their seen movies
    :param batch_size: batch size to use in reading the array_record.
    :return: a dictionary with key user_id and values
    being the set of movies that the user was recommended and hasn't seen.
    """
    return _read_user_recommendations_array_record(user_unseen_recommendations_uri, batch_size)


def build_history_lookup(ratings_uri: str,
        batch_size: int = 1024) -> Dict[int, Tuple[List, List, List]]:
    """
    Scans the training ratings once to build an O(1) user lookup.
    Arguments:
        ratings_uri: uri to ratings array_record holding tuples of user_id, movie_id, rating, timestamp
        batch_size: size of batch to use when reading.  does not affect returned data structure size
    returns: defaultdict of { user_id: {ts, movie_id, rating} } in which ts, movie_id
    and rating values are numpy arrays sorted by timestamp.
    """
    
    lookup = defaultdict(
        lambda: {"ts": [], "movie_id": [], "rating": []})
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(ratings_uri)
        n_records = reader.num_records()
        for i in range(0, n_records, batch_size):
            stop = min(i + batch_size, n_records)
            batch_bytes = reader.read([x for x in range(i, stop)])
            batch = [msgpack.unpackb(b, use_list=False) for b in
                batch_bytes]  # list of tuples
            for record in batch:
                u = record[0]
                lookup[u]["ts"].append(record[3])
                lookup[u]["movie_id"].append(record[1])
                lookup[u]["rating"].append(record[2])
    except Exception as e:
        raise e
    finally:
        if reader is not None:
            reader.close()
    print(f'rewrite lookup size = {len(lookup)}')
    lookup2 = {}
    for u in lookup:
        # sort all lists by timestamp
        ts = lookup[u]["ts"]
        idx = sorted(range(len(ts)), key=lambda i: ts[i])
        m = lookup[u]["movie_id"]
        r = lookup[u]["rating"]
        lookup2[u] = (
            [ts[i] for i in idx], [m[i] for i in idx],
            [r[i] for i in idx])
    
    return lookup2
