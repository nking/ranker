from typing import Dict, Tuple, Union, List, Set
import numpy as np
import grain.python as pgrain
import msgpack
from array_record.python import array_record_module

#TODO: refactor to use np.memmap if needed

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
            stop = min(i + batch_size, n_records - 1)
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
            stop = min(i + batch_size, n_records - 1)
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

class HardNegativeSamplingTransform(pgrain.MapTransform):
    """
    class to map a user's local history to the same local history enriched with negative sampling
    as "candidate_ids" and "labels"
    """
    def __init__(self, history_lookup: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        all_movie_ids:List[int], exact_negatives_dict:Dict[int, Set[int]],
        unseen_recommendations:Dict[int, Set[int]], num_candidates=20,
        seed:int = 0):
        """
        initialize a CandidateSamplingTransform object
        :param history_lookup:  Dict[user_id:int, Tuple(arrays of ts, movie_id, rating)]
        :param all_movie_ids: list of all movie_ids
        :param exact_negatives_dict: dictionary with key=user_id, value = set of exact negative movie ids
        :param unseen_recommendations: dictionary with key=user_id, value = set of unseen recommended movie ids
        :param num_candidates: total number of candidates to create from 1 postive and mulitple negatives
        :param seed: seed for random number generator
        """
        self.history_lookup = history_lookup
        self.exact_negatives_dict = exact_negatives_dict
        self.all_movie_ids = all_movie_ids #to o use in approx hard negatives
        self.unseen_recommendations = unseen_recommendations # to use in approx hard negatives
        self.num_candidates = num_candidates
        self.rng = np.random.default_rng(seed)

    def map(self, record:Dict[str, Union[int, np.ndarray]]) -> Dict[str, Union[int, np.ndarray]]:
        """
        given the current user history, add a hard negative mining list as "column_ids" and "labels"
        :param record: dictionary containing
            'user_id':int
            'movie_id':int,
            'rating': int,
            'timestamp': int,
            "history_movie_ids": np.ndarray,
            "history_ratings": np.ndarray,
            "history_length": int
        :return: dictionary containing
            'user_id':int
            'movie_id':int,
            'rating': int,
            'timestamp': int,
            "history_movie_ids": np.ndarray,
            "history_ratings": np.ndarray,
            "history_length": int
            "candidate_ids": np.ndarray,
            "labels": np.ndarray
        """
        user_id = record["user_id"]
        pos_id = record["movie_id"]
        
        # Get Hard Negatives (from Retrieval model)
        hard_negs = self.exact_negatives_dict.get(user_id, [])
        hard_negs = [m for m in hard_negs if m != pos_id]
        
        n_approx = self.num_candidates//2
        n_hard = self.num_candidates - 1 - n_approx
        if len(hard_negs) < n_hard:
            n_hard = len(hard_negs)
            n_approx = self.num_candidates - 1 - n_hard
        elif len(hard_negs) > n_hard:
            hard_negs = self.rng.choice(hard_negs, size=n_hard, replace=False)

        #choose approx negatives from "all movies - pos_id - has_seen - was recommended"
        subtr = set(pos_id)
        if self.history_lookup.get(user_id):
            #tuple (timestamps, movie_ids, ratings)
            subtr.add(self.history_lookup.get(user_id)[1])
        if self.unseen_recommendations.get(user_id):
            subtr.add(self.unseen_recommendations.get(user_id))
        approx_negs = self.rng.choice(self.all_movie_ids, size=n_approx, replace=False)
        
        candidate_ids = np.array([pos_id] + hard_negs + approx_negs, dtype=np.int32)
        
        # Create Labels (1.0 for the first one, 0.0 for the rest)
        labels = np.zeros((self.num_candidates,), dtype=np.float32)
        labels[0] = 1.0
        
        # Shuffle the candidates and labels together!
        # (Otherwise the model learns "index 0 is always the winner")
        p = np.random.permutation(self.num_candidates)
        
        return {
            **record,
            "candidate_ids": candidate_ids[p],
            "labels": labels[p]
        }