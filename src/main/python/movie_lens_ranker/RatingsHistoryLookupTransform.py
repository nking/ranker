from typing import Tuple, Dict, List, Union
import grain.python as pgrain
from collections import defaultdict
import numpy as np
import msgpack
from array_record.python import array_record_module

def build_history_lookup(ratings_uri:str,
        batch_size: int = 1024) -> Dict[int, Tuple[List, List, List]]:
    """
    Scans the training ratings once to build an O(1) user lookup.
    Arguments:
        ratings_uri: uri to ratings array_record holding tuples of user_id, movie_id, rating, timestamp
        batch_size: size of batch to use when reading.  does not affect returned data structure size
    returns: defaultdict of { user_id: {ts, movie_id, rating} } in which ts, movie_id
    and rating values are numpy arrays sorted by timestamp.
    """
    
    lookup = defaultdict(lambda: {"ts": [], "movie_id": [], "rating": []})
    reader = None
    try:
        reader = array_record_module.ArrayRecordReader(ratings_uri)
        n_records = reader.num_records()
        for i in range(0, n_records, batch_size):
            stop = min(i + batch_size, n_records - 1)
            batch_bytes = reader.read([x for x in range(i, stop)])
            batch = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  # list of tuples
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
        idx = sorted(range(len(ts)), key = lambda i: ts[i])
        m = lookup[u]["movie_id"]
        r = lookup[u]["rating"]
        lookup2[u] = (
            [ts[i] for i in idx], [m[i] for i in idx], [r[i] for i in idx])
        
    return lookup2

class RatingsHistoryLookupTransform(pgrain.MapTransform):
    def __init__(self, history_lookup: Dict[int, Tuple[list, list, list]], max_history: int = 20):
        """
        history_lookup: the results of method build_history_lookup
        max_history: Fixed size for the history window (crucial for JAX).
        """
        self.history_lookup = history_lookup
        self.max_history = max_history
    
    def map(self, batch: Tuple[int, int, int, int]) -> List[Dict[str, Union[int, List]]]:
        """
        map the input train record dictionary to a dictionary containing it and padded history entries
        :param batch: a list, that is batch, of tuples containing the user_id, movie_id, rating, and timestamp
        :return: a list of dictionaries containing
             'user_id':int
            'movie_id':int,
            'rating': int,
            'timestamp': int,
            "history_movie_ids": np.ndarray,
            "history_ratings": np.ndarray,
            "history_length": int
        """
        results = []
        for record in batch:
        
            user_id = record[0]
            current_ts = record[3]
            
            # O(1) Lookup: Get this user's full history arrays
            # If user not found, we use empty arrays
            user_ts, user_movies, user_ratings = self.history_lookup.get( user_id, ([], [], []))
            
            # Temporal Safety: Find index where time < current_ts
            # np.searchsorted finds the insertion point to maintain order
            idx = int(np.searchsorted(user_ts, current_ts, side='left'))
            
            valid_movies_history = user_movies[:idx]
            valid_ratings_history = user_ratings[:idx]
            
            # 'max_history' most recent movies
            recent_movies_history = valid_movies_history[-self.max_history:]
            recent_ratings_history = valid_ratings_history[-self.max_history:]
            
            n_hist = len(recent_movies_history)
            # JAX-Required Padding.  return a fixed shape (e.g., 20) or JAX will crash/recompile
            if n_hist < self.max_history:
                n_padding = self.max_history - n_hist
                recent_movies_history.extend([-1] * n_padding)
                recent_ratings_history.extend([-1] * n_padding)
            
            # Return updated record with the "Context" attached
            results.append({
                'user_id': record[0],
                'movie_id': record[1],
                'rating': record[2],
                'timestamp': record[3],
                "history_movie_ids": recent_movies_history,
                "history_ratings": recent_ratings_history,
                "history_length": n_hist
            })
        return results