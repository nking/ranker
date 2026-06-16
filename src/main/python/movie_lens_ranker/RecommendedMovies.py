from typing import Tuple

import numpy as np
from array_record.python import array_record_module
import msgpack

class RecommendedMovies (object):
    def __init__(self, num_users:int, movie_rec_file_uri:str, movie_rec_ts_file_uri:str):
        """
        read in the recommended movies.  assumes that each list of recommended movies has the same length, and
        that a user_id netry in movie_rec_file_path is complemented by a user_id entry in
        movie_rec_ts_file_path.
        :param num_users:  number of users in the entire user catalog
        :param movie_rec_file_uri: file of movies recommended for each user in format such that each row is
        usr_id, [movie_ids].
        :param movie_rec_ts_file_uri: file of timestamps of movies recommended for each user in format such that each row is
        user_id, [timestamps] and the row for a user_id in this file is complementary to the row of same user_id in
        movie_rec_file_path and both lists of movie_ids and timestamps are ordered to represent the same movies.
        notr that recommended movies which have not been seen should have a timestamp larger than feasibly in the
        watch history, like ts_2050 = 2524608000 used by the retrieval project.
        """
        self.pad_value = -1
        self.user_ids, self.movies, self.row_length = self._read(num_users, file_uri=movie_rec_file_uri, pad_value=self.pad_value)
        _ , self.timestamps, _ = self._read(num_users, movie_rec_ts_file_uri,  pad_value=self.pad_value, dtype2=np.int64)
        self.num_users = num_users
        
        
    def _read(self, num_users:int, file_uri:str, pad_value:int, dtype2:type=np.int32) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        read the recommended movies or recommended movies' timestamps file.   note that the array_records
        have been written such that the movies decrease in score and such that the
        timestamps are for the movie_ids at same index in orther file.
        :param num_users: total number of users in the user catalog.  this is probably the same as
        the length of the file at file_uri
        :param file_uri:
        :param dtype2:
        :return:
        """
        reader = None
        try:
            reader = array_record_module.ArrayRecordReader(file_uri)
            
            # peek at length of record[1]
            row0_list = [msgpack.unpackb(b, use_list=False) for b in reader.read([0])]
            len_arr = len(row0_list[0][1])
            
            user_ids = np.array([i for i in range(0, num_users + 1)], dtype=np.int32)
            #the padding value doesn't matter because items gets completely filled with real values
            items = np.full((num_users + 1, len_arr), fill_value=pad_value, dtype=dtype2)
            
            count = reader.num_records()
            batch_bytes = reader.read([x for x in range(0, count)])
            records = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]
            for record in records:
                user_idx = record[0]
                items[user_idx, :] = record[1]
        finally:
            if reader is not None:
                reader.close()
        return user_ids, items, len_arr
    
    def get_unseen_movies(self, user_id: np.ndarray, timestamp: np.ndarray, top_k:int=200) -> np.ndarray:
        """
        given array of user_ids, return top_k recommended movies for user that they haven't seen before time=timestamp.
        the unseen movies have been moved to front of array and retain their respective original order which is by decreasing similarity score.
        Note that if top_k is > the file's
        :param user_id: input array of shape (None,), e.g. np.array([2,4])
        :param timestamp: timestamp representing current time.  any recommendations with timestamps > timestamp are yet unseen.
        :param top_k: number of top unseen recommendations to return
        :return: top k of movie recommendations unseen by user_id.  shape returned is (len(user_id_, top_k)
        """
        if top_k > self.row_length:
            raise ValueError(f"top_k must be smaller than the number of recommendations per suer = {self.row_length}")
        
        user_idx = self.user_ids[user_id]
        
        mask = self.timestamps[user_idx] > timestamp[:, np.newaxis]
        sort_indices = np.argsort(~mask, axis=1, kind='stable')
        #equiv of row-wise tf.gather:
        sel_movies = np.take_along_axis(self.movies[user_idx], sort_indices, axis=1)
        
        #the unseen movies have been moved to front of array and retain their respective original order which is by decreasing similarity score.
        return sel_movies[:, :top_k]