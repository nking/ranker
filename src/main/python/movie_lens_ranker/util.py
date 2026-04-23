from collections import defaultdict
from typing import Tuple, Dict, List, Union, Set
import jax.numpy as jnp
from array_record.python import array_record_module
import msgpack
import numpy as np

# each kubeflow worker will load these into memory
# or this will be changed to a distributable loading

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
