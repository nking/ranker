from typing import Union, List, Tuple

import numpy as np
from array_record.python import array_record_module
import msgpack

from movie_lens_ranker.data_loading import build_history_lookup


class UserHistory (object):
    def __init__(self, ratings_uri_list: Union[str, List[str]], fixed_size:int = 2048, pad_value:int=-1):
        #each user's the movie_ids, ratings and timestamps is already sorted by timestamp
        self.user_ids, self.movie_ids, self.ratings, self.timestamps = self._load_history(ratings_uri_list, fixed_size, pad_value)
        self.fixed_size = fixed_size
        self.pad_value = pad_value
        
    def _load_history(self, ratings_uri_list: Union[str, List[str]],
            fixed_size:int = 2048, pad_value:int=-1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        #buildnumpy vectors, making padded lists of length fixed_history_length for movies, ratings, and timestamps
        lookup, max_history = build_history_lookup(ratings_uri_list)
        print(f'max_history found = {max_history}.  fixed_size={fixed_size}')
        
        n_users = len(lookup)
        
        #NOTE: results are sorted by timestamp
        user_ids = []
        movie_ids = np.full((n_users, fixed_size), pad_value)
        ratings = np.full((n_users, fixed_size), pad_value)
        timestamps = np.full((n_users, fixed_size), pad_value)
        
        for i, user_id in enumerate(lookup.keys()):
            user_ids.append(user_id)
            #these are ordered by timestamp already:
            user_ts, user_movies, user_ratings = lookup[user_id]
            
            length = min(len(user_ts), fixed_size)
            
            timestamps[i][:length] = user_ts[:length]
            movie_ids[i][:length] = user_movies[:length]
            ratings[i][:length] = user_ratings[:length]
        
        #sort by user_ids to enable np.searchsorted later
        user_ids = np.array(user_ids, dtype=np.int64)
        sort_indices = np.argsort(user_ids)
        user_ids = user_ids[sort_indices]
        movie_ids = movie_ids[sort_indices]
        ratings = ratings[sort_indices]
        timestamps = timestamps[sort_indices]
        
        return user_ids, movie_ids, ratings, timestamps
    
    def get_movieids_before_timestamp(self, user_id: np.ndarray, timestamp: int, max_hist:int, pad_value:int=-1) -> np.ndarray:
        """
        given array of user_ids, return max_hist of movies < timestamp, padded by pad_value when not enough history
        :param user_id: input array of shape (None,), e.g. np.array([2,4])
        :param timestamp: timestamp representing current time.  any user rated movies with timestamps < timestamp are returned
           up to max_hist in length.
        :param max_hist: number of user rated movies to return
        :param pad_value: the number to use to fill a user's history when they have fewer than max_hist for the time frame.
        :return: user rated movies < timestamp, limited to max_hist number of movies.  shape of return is ( len(user_id), max_hist)
        """
        #transform user_ids into user_idxs.  can use searchsorted because already sorted by user_ids
        user_idx = np.searchsorted(self.user_ids, user_id)
        n_user_selected = len(user_idx)
        
        sub_timestamps = self.timestamps[user_idx]  # Shape: (num_selected, n_movies)
        sub_movie_ids = self.movie_ids[user_idx]  # Shape: (num_selected, n_movies)
        
        #can use comparison partition because timestamps are already sorted
        mask = sub_timestamps < timestamp
        occurrence_count = np.cumsum(mask, axis=1)
        final_mask = mask & (occurrence_count <= max_hist)
        row_coords, col_coords = np.where(final_mask)
        new_col_coords = occurrence_count[final_mask] - 1
        
        ret_movie_ids = np.full((n_user_selected, max_hist), pad_value, dtype=sub_movie_ids.dtype)
        ret_movie_ids[row_coords, new_col_coords] = sub_movie_ids[final_mask]
        
        return ret_movie_ids
    
    def get_history_before_timestamp(self, user_id: np.ndarray,
            timestamp: int, max_hist: int, pad_value: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        given array of user_ids, return max_hist of movies < timestamp, padded by pad_value when not enough history.  also returns
        the ratings for those movies, also padded by -1 for missing values.
        :param user_id: input array of shape (None,), e.g. np.array([2,4])
        :param timestamp: timestamp representing current time.  any user rated movies with timestamps < timestamp are returned
           up to max_hist in length.
        :param max_hist: number of user rated movies to return
        :param pad_value: the number to use to fill a user's history when they have fewer than max_hist for the time frame.
        :return: user rated movies < timestamp, limited to max_hist number of movies and ratings for those movies.
        shape of each of the returned np.ndarrays is ( len(user_id), max_hist)
        """
        # transform user_ids into user_idxs
        user_idx = np.searchsorted(self.user_ids, user_id)
        n_user_selected = len(user_idx)
        
        sub_timestamps = self.timestamps[user_idx]  # Shape: (num_selected, n_movies)
        sub_movie_ids = self.movie_ids[user_idx]  # Shape: (num_selected, n_movies)
        sub_ratings = self.ratings[user_idx]  # Shape: (num_selected, n_movies)
        
        mask = sub_timestamps < timestamp
        occurrence_count = np.cumsum(mask, axis=1)
        final_mask = mask & (occurrence_count <= max_hist)
        row_coords, col_coords = np.where(final_mask)
        new_col_coords = occurrence_count[final_mask] - 1
        
        ret_movie_ids = np.full((n_user_selected, max_hist), pad_value,
            dtype=sub_movie_ids.dtype)
        ret_movie_ids[row_coords, new_col_coords] = sub_movie_ids[final_mask]
        
        ret_ratings = np.full((n_user_selected, max_hist), pad_value,
            dtype=sub_ratings.dtype)
        ret_ratings[row_coords, new_col_coords] = sub_ratings[final_mask]
        
        return ret_movie_ids, ret_ratings