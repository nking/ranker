from typing import Tuple

import numpy as np
from array_record.python import array_record_module
import msgpack

class Negatives (object):
    """
    class to read files of user hard and natural negatives and supply vectorized get methods
    """
    def __init__(self, negatives_uri: str, fixed_size:int = 2048, pad_value:int=-1):
        #each user's the movie_ids, ratings and timestamps is already sorted by timestamp
        self.user_ids, self.movie_ids = self._load_negatives(negatives_uri, pad_value, fixed_size)
        self.fixed_size = fixed_size
        self.pad_value = pad_value
        
    def _load_negatives(self, negatives_uri:str,
            pad_value:int, fixed_size:int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(negatives_uri)
            n_records = reader.num_records()
            user_ids = []
            movie_ids = np.full((n_records, fixed_size), pad_value, dtype=np.int32)
            batch_bytes = reader.read([x for x in range(n_records)])
            batch = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  # list of tuples user_id, movie_ids
            for i, record in enumerate(batch):
                user_ids.append(record[0])
                length = min(len(record[1]), fixed_size)
                movie_ids[i][:length] = record[1][:length]
        except Exception as e:
            raise e
        finally:
            if reader is not None:
                reader.close()
                
        return np.array(user_ids, dtype=np.int32), movie_ids
    
    def get_negatives(self, user_id: np.ndarray, length:int, seed:int=0) -> np.ndarray:
        """
        given array of user_ids, return max_hist of negatives, padded by pad_value when not enough negatives.
        :param user_id: input array of shape (None,), e.g. np.array([2,4])
        :param length: number of user rated movies to return for each user
        :return: user's negative movie ids, limited to max_hist number of movies.  shape of return is ( len(user_id), max_hist)
        """
        #transform user_ids into user_idxs.  can use searchsorted because already sorted by user_ids
        user_idx = np.searchsorted(self.user_ids, user_id)
        n_user_selected = len(user_idx)
        # shape (n_user_selected, self.fixed_size)
        sub_movie_ids = self.movie_ids[user_idx]  # Shape: (num_selected, fixed_size)
        
        rng = np.random.default_rng(seed)
        
        noise = rng.random(sub_movie_ids.shape)
        # Penalize the indices where movie_id is -1
        # We set their noise to -1.0 (since rng.random is 0.0 to 1.0)
        noise[sub_movie_ids == self.pad_value] = -1.0
        #BUT if need the noise to be deterministic for each user:
        #noise = np.array([np.random.default_rng(seed + uid).random(self.fixed_size) for uid in user_idx])
        
        # Get the indices that would sort the noise DESCENDING
        # This puts the highest random values (valid movies) first
        # Shape: (n_user_selected, fixed_size)
        shuffled_idx = np.argsort(noise, axis=1)[:, ::-1]
        
        # Take only the first 'length' indices
        # Shape: (n_user_selected, length)
        top_k_idx = shuffled_idx[:, :length]
        
        # Use Fancy Indexing to extract the movie IDs
        row_grid = np.arange(n_user_selected)[:, np.newaxis]
        sampled_movie_ids = sub_movie_ids[row_grid, top_k_idx]
        
        return sampled_movie_ids