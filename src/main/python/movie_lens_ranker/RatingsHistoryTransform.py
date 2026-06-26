from typing import Tuple, Dict, List, Union
import grain.python as pgrain
import numpy as np
import os
import sys
from array_record.python import array_record_module

from movie_lens_ranker.UserHistory import UserHistory

class RatingsHistoryLookupTransform(pgrain.MapTransform):
    def __init__(self, history_lookup: UserHistory, max_history: int = 20):
        """
        history_lookup: the results of method build_history_lookup
        max_history: Fixed size for the history window (crucial for JAX).  This can be made as large as
        the longest history among all users to keep all ratings and pad to max_history.
        """
        # This will print from every single worker process.
        # If you don't see this in your logs, the process is dying before it even hits this code.
        print(f"DEBUG: Worker PID {os.getpid()} - PYTHONPATH: {sys.path}", flush=True)
        print(f"DEBUG: Worker PID {os.getpid()} - app_rootfs present: {'app_rootfs' in str(sys.path)}",
            flush=True)
        self.history_lookup = history_lookup
        self.max_history = max_history
        self.pad_value = -1
    
    def map(self, batch: List[Tuple[int, int, int, int]]) -> Dict[
        str, np.ndarray]:
        """
        map the input train record dictionary to a dictionary containing it and padded history entries
        :param batch: a list, that is batch, of tuples containing the user_id, movie_id, rating, and timestamp
        :return: a dictionary containing numpy arrays where the arrays are 1D of length batch_size
            'user_id', shape(batch_size, )
            'movie_id',
            'rating',
            'timestamp',
            "history_movie_ids", shape (batch_size, max_history)
            "history_ratings",
            "history_length"
        """
        user_ids = []
        movie_ids = []
        ratings = []
        timestamps = []
        for record in batch:
            u_id, m_id, r, ts = record[0], record[1], record[2], record[3]
            user_ids.append(u_id)
            movie_ids.append(m_id)
            ratings.append(r)
            timestamps.append(ts)
            
        user_ids = np.array(user_ids, dtype=np.int32)
        movie_ids = np.array(movie_ids, dtype=np.int32)
        ratings = np.array(ratings, dtype=np.float32)
        timestamps = np.array(timestamps, dtype=np.int64)
        
        #history_n=movies is shape(len(user_ids, self.max_history) with any empty values being
        history_movies, history_ratings = self.history_lookup.get_history_before_timestamp(
            user_ids, timestamps, self.max_history)
        
        #history_* shapes are (batch_size, max_history)
        history_lengths = np.sum(history_movies != -1, axis=1)
        
        # Convert everything to a single dictionary of NumPy arrays
        return {
            'user_id': user_ids,
            'movie_id': movie_ids,
            'rating': ratings,
            'timestamp': timestamps,
            # (Batch, Max_History)
            'history_movie_ids': history_movies,
            # (Batch, Max_History)
            'history_ratings': history_ratings,
            # (Batch, Max_History)
            'history_length': history_lengths
        }