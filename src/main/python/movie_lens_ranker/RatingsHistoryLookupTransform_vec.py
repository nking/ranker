from typing import Tuple, Dict, List, Union
import grain.python as pgrain
import numpy as np
from array_record.python import array_record_module

class RatingsHistoryLookupTransform(pgrain.MapTransform):
    def __init__(self, history_lookup: Dict[int, Tuple[list, list, list]],
            max_history: int = 20):
        """
        history_lookup: the results of method build_history_lookup
        max_history: Fixed size for the history window (crucial for JAX).
        """
        self.history_lookup = history_lookup
        self.max_history = max_history
    
    def map(self, batch: List[Tuple[int, int, int, int]]) -> Dict[
        str, np.ndarray]:
        """
        map the input train record dictionary to a dictionary containing it and padded history entries
        :param batch: a list, that is batch, of tuples containing the user_id, movie_id, rating, and timestamp
        :return: a dictionary containing numpy arrays
             'user_id'.
            'movie_id',
            'rating',
            'timestamp',
            "history_movie_ids",
            "history_ratings",
            "history_length"
        """
        user_ids = []
        movie_ids = []
        ratings = []
        timestamps = []
        history_movies = []
        history_ratings = []
        history_lengths = []
        
        for record in batch:
            u_id, m_id, r, ts = record[0], record[1], record[2], record[3]
            
            u_ts, u_movies, u_ratings = self.history_lookup.get(u_id,
                ([], [], []))
            
            # best done per-record due to different array sizes
            idx = np.searchsorted(u_ts, ts, side='left')
            
            # Slice and Pad
            # We slice first to avoid copying the whole history
            h_movies = u_movies[max(0, idx - self.max_history):idx]
            h_ratings = u_ratings[max(0, idx - self.max_history):idx]
            n_hist = len(h_movies)
            
            # Fixed-width padding for JAX stability
            padded_movies = np.full((self.max_history,), -1,
                dtype=np.int32)
            padded_ratings = np.full((self.max_history,), -1,
                dtype=np.float32)
            
            padded_movies[:n_hist] = h_movies
            padded_ratings[:n_hist] = h_ratings
            
            user_ids.append(u_id)
            movie_ids.append(m_id)
            ratings.append(r)
            timestamps.append(ts)
            history_movies.append(padded_movies)
            history_ratings.append(padded_ratings)
            history_lengths.append(n_hist)
        
        # Convert everything to a single dictionary of NumPy arrays
        return {
            'user_id': np.array(user_ids, dtype=np.int32),
            'movie_id': np.array(movie_ids, dtype=np.int32),
            'rating': np.array(ratings, dtype=np.float32),
            'timestamp': np.array(timestamps, dtype=np.int32),
            'history_movie_ids': np.array(history_movies),
            # (Batch, Max_History)
            'history_ratings': np.array(history_ratings),
            # (Batch, Max_History)
            'history_length': np.array(history_lengths, dtype=np.int32)
        }