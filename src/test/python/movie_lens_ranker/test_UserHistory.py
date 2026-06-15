import unittest

import numpy as np
from array_record.python import array_record_module

from helper import *

from movie_lens_ranker.util import build_history_lookup
from movie_lens_ranker.UserHistory import UserHistory

class TestUserHistory_vec(unittest.TestCase):
    def setUp(self):
        
        ratings_uri_dict = get_train_val_test_liked_uris(use_small=True)
        
        self.ratings_train_liked_uri = ratings_uri_dict["train_liked"]
        self.ratings_val_liked_uri = ratings_uri_dict["val_liked"]
        self.ratings_test_liked_uri = ratings_uri_dict["test_liked"]
        
        self.ratings_train_3_uri = ratings_uri_dict["train_3"]
        self.ratings_val_3_uri = ratings_uri_dict["val_3"]
        self.ratings_test_3_uri = ratings_uri_dict["test_3"]
        
        self.ratings_train_disliked_uri = ratings_uri_dict["train_disliked"]
        self.ratings_val_disliked_uri = ratings_uri_dict["val_disliked"]
        self.ratings_test_disliked_uri = ratings_uri_dict["test_disliked"]
    
    def test_user_history(self):
        
        #the full history:
        ratings_uri_list = [self.ratings_train_liked_uri, self.ratings_train_3_uri,
            self.ratings_train_disliked_uri]
        
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
        uh = UserHistory(ratings_uri_list = ratings_uri_list, fixed_size=2048)
        
        user_ids = np.array([6040, 6039])
        
        # ==== test scalar timestamps ========
        ts = 956705600
        fixed_length = 200
        
        movie_hist = uh.get_movieids_before_timestamp(user_ids, timestamp=ts, max_hist=fixed_length)
        
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
            max_hist=fixed_length)
        
        np.testing.assert_array_equal(movie_hist2, movie_hist)
        count = np.sum(movie_hist != -1, axis=1)
        count2 = np.sum(movie_hist2 != -1, axis=1)
        np.testing.assert_array_equal(count, count2)
        count3 = np.sum(ratings_hist2 != -1, axis=1)
        np.testing.assert_array_equal(count, count3)
        
        import operator
        #======= test array timestamps ======
        inp_timestamps = np.array([956704000, 956705600])
        movie_hist10 = uh.get_movieids_before_timestamp(user_ids, timestamp=inp_timestamps, max_hist=fixed_length)
        count10 = np.sum(movie_hist10 != -1, axis=1)
        np.testing.assert_array_compare(operator.le, count10, count,
            err_msg="count10 has values greater than count!")
        for i, c in enumerate(count10):
            np.testing.assert_array_equal(movie_hist[i][:c], movie_hist10[i][:c])
        
        movie_hist11, ratings_hist11 = uh.get_history_before_timestamp(user_ids,
            timestamp=inp_timestamps, max_hist=fixed_length)
        np.testing.assert_array_equal(movie_hist10, movie_hist11)
        count11 = np.sum(ratings_hist11 != -1, axis=1)
        np.testing.assert_array_equal(count10, count11)

if __name__ == '__main__':
    unittest.main()
