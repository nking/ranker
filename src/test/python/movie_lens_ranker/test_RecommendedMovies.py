import os.path
import unittest
import time

import numpy as np
from array_record.python import array_record_module
from movie_lens_ranker.RecommendedMovies import RecommendedMovies

from helper import *
from movie_lens_ranker.BatchSampler import BatchSampler
from movie_lens_ranker.RandomAccessArrayRecordDataSource import *
from movie_lens_ranker.data_loading import *
import grain

from movie_lens_ranker.data_loading import _read_embeddings

class TestRecommendedMovies(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_RecommendedMovies(self):
        
        test_res_dir = os.path.join(get_project_dir(), "src/test/resources/data")
        
        recommended_movies_getter = RecommendedMovies(movie_rec_file_path=
            os.path.join(test_res_dir, "recommended_movies.array_record"),
            movie_rec_ts_file_path=
            os.path.join(test_res_dir, "recommended_movies_timestamps.array_record"))
        
        ts = 978133414 #first timestamp from test dataset
        top_k = 20
        
        user_id = 2
        movies = recommended_movies_getter.get_unseen_movies_scalar(user_id, timestamp=ts, top_k=top_k)
        self.assertEqual(np.shape(movies), (top_k,))
        
        user_id = np.array([2, 4])
        movies = recommended_movies_getter.get_unseen_movies(user_id, timestamp=ts, top_k=top_k)
        self.assertEqual(np.shape(movies), (len(user_id), top_k))
        
        #demonstrating what to do if have inputs of this form:
        user_id = np.array([[2], [4]])
        if user_id.ndim > 1:
            user_id = user_id.squeeze()
        movies = recommended_movies_getter.get_unseen_movies(user_id, timestamp=ts, top_k=top_k)
        self.assertEqual(np.shape(movies), (len(user_id), top_k))
        
if __name__ == '__main__':
    unittest.main()
