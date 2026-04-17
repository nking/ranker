import os.path
import unittest
import time

import numpy as np
from array_record.python import array_record_module
from movie_lens_ranker.RecommendedMovies import RecommendedMovies

from helper import *
from movie_lens_ranker.BatchSampler import BatchSampler
from movie_lens_ranker.RandomAccessArrayRecordDataSource import *
from movie_lens_ranker.UserHistory_vec import UserHistory
from movie_lens_ranker.data_loading import *
import grain

from movie_lens_ranker.data_loading import _read_embeddings

class TestUserHistory_vec(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_User_history(self):
        
        ratings_train_uri, ratings_val_uri, ratings_test_uri \
            = get_train_val_test_liked_uris(use_small=True)
        
        ratings_uri_list = [ratings_train_uri, ratings_val_uri]
        
        '''
        some expected examples:
            6040::6888::4::956703932
            6040::6630::5::956703954
            6040::8356::4::956703954
            6040::6684::5::956704257
            ...
            6039::9368::4::956705581
            6039::6940::4::956705636
        
        choose  timestamps < 956705600
        '''
        uh = UserHistory(ratings_uri_list = ratings_uri_list, fixed_size=2048, pad_value=-1)
        
        user_ids = np.array([6040, 6039])
        
        ts = 956705600
        fixed_length = 200
        pad_value = -1
        
        movie_hist = uh.get_movieids_before_timestamp(user_ids, timestamp=ts, max_hist=fixed_length, pad_value=pad_value)
        
        self.assertTrue((len(user_ids), fixed_length) == movie_hist.shape)
        
        #not an indep test, but use dictionary to check timestamps:
        lookup, max_history = build_history_lookup(ratings_uri_list)
        
        for user_id, movies in zip(user_ids, movie_hist):
            self.assertEqual(fixed_length, len(movies))
            count = np.sum(movies != -1)
            
            user_ts, user_movies, user_ratings = lookup[user_id]
            end_idx = np.searchsorted(user_ts, ts)
            self.assertEqual(end_idx, count)
            
            if user_id == 6039:
                test_idx = np.searchsorted(user_ts, 956705636)
                test_movie = user_movies[test_idx]#6940
                self.assertEqual(6940, test_movie)
                self.assertTrue(test_movie not in movies)
            
            user_movies = user_movies[:end_idx]
            self.assertEqual(len(user_movies), count)
        
        movie_hist2, ratings_hist2 = uh.get_history_before_timestamp(user_ids, timestamp=ts,
            max_hist=fixed_length, pad_value=pad_value)
        
        np.testing.assert_array_equal(movie_hist2, movie_hist)
        count = np.sum(movie_hist != -1, axis=1)
        count2 = np.sum(movie_hist2 != -1, axis=1)
        np.testing.assert_array_equal(count, count2)
        count3 = np.sum(ratings_hist2 != -1, axis=1)
        np.testing.assert_array_equal(count, count3)

if __name__ == '__main__':
    unittest.main()
