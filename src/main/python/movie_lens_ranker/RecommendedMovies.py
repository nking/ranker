from typing import Union

import numpy as np
from array_record.python import array_record_module
import msgpack

class RecommendedMovies (object):
    def __init__(self, movie_rec_file_path:str, movie_rec_ts_file_path:str):
        self.movies :np.ndarray = self._read(movie_rec_file_path)
        self.timestamps : np.ndarray = self._read(movie_rec_ts_file_path)
        
    def _read(self, file_path:str):
        ids = []
        results = []
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(file_path)
            count = reader.num_records()
            batch_bytes = reader.read([x for x in range(0, count)])
            records = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]
            for record in records:
                ids.append(record[0])
                results.append(record[1])
        finally:
            if reader is not None:
                reader.close()
        return np.array([val for _, val in sorted(zip(ids, results))], dtype=np.int64)
    
    def get_unseen_movies(self, user_id: np.ndarray, timestamp: int, top_k:int=200) -> np.ndarray:
        """
        given array of user_ids, return top_k recommended movies for user that they haven't seen before time=timestamp
        :param user_id: input array of shape (None,), e.g. np.array([2,4])
        :param timestamp: timestamp representing current time.  any recommendations with timestamps > timestamp are yet unseen.
        :param top_k: number of top unseen recommendations to return
        :return: top k of movie recommendations unseen by user_id.  shape returned is (len(user_id_, top_k)
        """
        user_idx = user_id - 1
        mask = self.timestamps[user_idx] > timestamp
        sort_indices = np.argsort(~mask, axis=1, kind='stable')
        #equiv of row-wise tf.gather:
        sel_movies = np.take_along_axis(self.movies[user_idx], sort_indices, axis=1)
        
        return sel_movies[:, :top_k]
    
    def get_unseen_movies_scalar(self, user_id: int, timestamp: int, top_k: int = 200) -> np.ndarray:
        """
        given scalar user_id, return top_k recommended movies for user that they haven't seen before time=timestamp
        :param user_id: scalar input user_id
        :param timestamp: timestamp representing current time.  any recommendations with timestamps > timestamp are yet unseen.
        :param top_k: number of top unseen recommendations to return
        :return: top k of movie recommendations unseen by user_id.  shape returned is (top_k,)
        """
        user_idx = user_id - 1
        mask = self.timestamps[user_idx] > timestamp
        sort_indices = np.argsort(~mask, axis=0, kind='stable')
        # equiv of row-wise tf.gather:
        sel_movies = np.take_along_axis(self.movies[user_idx], sort_indices, axis=0)
        
        return sel_movies[:top_k]