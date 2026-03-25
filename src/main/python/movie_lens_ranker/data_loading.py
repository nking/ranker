from collections import defaultdict
from typing import Tuple, Dict, List, Set
import jax.numpy as jnp
from array_record.python import array_record_module
import msgpack

# each kubeflow worker will load these into memory
# or this will be changed to a distributable loading

'''
We renumber the movie ids in order to have sequential numbering without gaps that starts after
the number of unique users.  This allows the GraphRanker to concatenate embeddings for all users
followed by embeddings for all movies for more efficient use of data structures.

(1) given user_emb_uri, read user embeddings and returns a list of the embeddings ordered by
id and a user_id_fwd dict.
note that user_ids are numbered 1 through n_users so downstream has to use numbering
of 0 through n_users-1. The offset is always -1, but will use a dictionary to future
proof the mapping and to be consistent with the movie_id mapping.
(2) given n_users, and movie_uri, read movie embeddings. make a dictionary of movie_id_fwd_dict
and return the embeddings and dictionary.
(3)  make wrappers to transform the ids as they are read from the other files

'''

def read_embeddings(user_embeddings_uri:str, movie_embeddings_uri:str, batch_size:int=1024) -> Tuple[Dict[int, int], Dict[int, int], jnp.ndarray]:
    """
    read the user and movie embeddings and return dictionaries for the new user_ids and movie_ids to use with the
    concatenated embeddings.
    :param user_embeddings_uri:
    :param movie_embeddings_uri:
    :param batch_size:
    :return: a tuple of dictionary of original user_id to new user_id, a dictionary of original movie id to new movie id, and the
    concatenated embeddings.
    """
    user_fwd_dict, user_emb = _read_embeddings(user_embeddings_uri, batch_size=batch_size, offset=0)
    movie_fwd_dict, movie_emb = _read_embeddings(movie_embeddings_uri, batch_size=batch_size, offset=len(user_fwd_dict))
    emb = jnp.concatenate([user_emb, movie_emb])
    return user_fwd_dict, movie_fwd_dict, emb

def _read_embeddings(embeddings_uri:str, batch_size:int=1024, offset:int=0) -> Tuple[Dict[int, int], jnp.ndarray]:
    """
    given the embedding uri and an offset to start numbering from, return a tuple of a dictionary
    of key=read movie_id, value = new movie_id for concatenated dataset, and the embeddings
    as a 2D jnp array.
    :param embeddings_uri:
    :param batch_size:
    :param offset:
    :return:
    """
    id_fwd_dict = {}
    embeddings = []
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
                id_fwd_dict[record[0]] = offset + len(id_fwd_dict)
                embeddings.append(record[1])
    except Exception as e:
        raise e
    finally:
        if reader is not None:
            reader.close()
    return id_fwd_dict, jnp.array(embeddings)

def read_user_exact_negatives(exact_hard_negatives_uri:str, user_id_fwd_dict:Dict[int,int],
        movie_id_fwd_dict:Dict[int, int], batch_size:int=1048) -> Dict[int, Set[int]]:
    """
    read the array_record at exact_hard_negatives_uri into a dictionary with key user_id and values
    being the set of movies that the user was recommended, but did not like.
    :param movie_id_fwd_dict: dictionary to get new movie_id given original movie_id
    :param user_id_fwd_dict: dictionary to get new user_id given original user_id
    :param exact_hard_negatives_uri: the uri for the exact hard negatives array_record
    :param batch_size: batch size to use in reading the array_record.
    :return: a dictionary with key user_id and values
    being the set of movies that the user was recommended, but did not like.
    """
    return _read_user_recommendations_array_record(exact_hard_negatives_uri, user_id_fwd_dict, movie_id_fwd_dict, batch_size=batch_size)

def _read_user_recommendations_array_record(user_uri:str, user_id_fwd_dict:Dict[int,int],
        movie_id_fwd_dict:Dict[int, int], batch_size:int=1048) -> Dict[int, Set[int]]:
    """
    read the array_record at user_uri into a dictionary with key user_id and values
    being the set of movie_ids.
    :param user_uri: the uri for the array_record of user recommendations
     :param movie_id_fwd_dict: dictionary to get new movie_id given original movie_id
    :param user_id_fwd_dict: dictionary to get new user_id given original user_id
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
                user_id = user_id_fwd_dict[record[0]]
                movie_ids = {movie_id_fwd_dict[m] for m in record[1]}
                lookup[user_id] = movie_ids
    except Exception as e:
        raise e
    finally:
        if reader is not None:
            reader.close()
    return lookup

def read_movies_array_record(movies_uri:str, movie_id_fwd_dict:Dict[int, int], batch_size:int=1048) -> List[int]:
    """
    read the array_record at movies_uri into a list of movie ids
    :param movies_uri: the uri for the array_record of movie ids
     :param movie_id_fwd_dict: dictionary to get new movie_id given original movie_id
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
            movie_ids.extend([movie_id_fwd_dict[m] for m in batch])
    except Exception as e:
        raise e
    finally:
        if reader is not None:
            reader.close()
    return movie_ids

def read_user_unseen_recommendations(user_unseen_recommendations_uri:str, user_id_fwd_dict:Dict[int,int],
        movie_id_fwd_dict:Dict[int, int], batch_size:int=1048) -> Dict[int, Set[int]]:
    """
    read the array_record at user_unseen_recommendations_uri into a dictionary with key user_id and values
    being the set of movies that the user was recommended, but did not like.
    :param user_unseen_recommendations_uri: the uri for the array_record of user recommendations that exclude their seen movies
     :param movie_id_fwd_dict: dictionary to get new movie_id given original movie_id
    :param user_id_fwd_dict: dictionary to get new user_id given original user_id
    :param batch_size: batch size to use in reading the array_record.
    :return: a dictionary with key user_id and values
    being the set of movies that the user was recommended and hasn't seen.
    """
    return _read_user_recommendations_array_record(user_unseen_recommendations_uri, user_id_fwd_dict, movie_id_fwd_dict, batch_size=batch_size)


def build_history_lookup(ratings_uri: str, user_id_fwd_dict:Dict[int,int],
        movie_id_fwd_dict:Dict[int, int], batch_size: int = 1024) -> Tuple[Dict[int, Tuple[List, List, List]], int]:
    """
    Scans the training ratings once to build an O(1) user lookup.
    Arguments:
        ratings_uri: uri to ratings array_record holding tuples of user_id, movie_id, rating, timestamp
        movie_id_fwd_dict: dictionary to get new movie_id given original movie_id
        user_id_fwd_dict: dictionary to get new user_id given original user_id
        batch_size: size of batch to use when reading.  does not affect returned data structure size
    returns: tuple of dictionary of { user_id: {ts, movie_id, rating} } in which ts, movie_id
    and rating values are numpy arrays sorted by timestamp, and returns the maximum number of movies seen by any user
    in this dataset.
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
    max_history = 0
    lookup2 = {}
    for u in lookup:
        # sort all lists by timestamp
        ts = lookup[u]["ts"]
        idx = sorted(range(len(ts)), key=lambda ii: ts[ii])
        m = lookup[u]["movie_id"]
        r = lookup[u]["rating"]
        max_history = max(max_history, len(m))
        lookup2[user_id_fwd_dict[u]] = (
            [ts[ii] for ii in idx], [movie_id_fwd_dict[m[ii]] for ii in idx],
            [r[ii] for ii in idx])
    
    return lookup2, max_history
