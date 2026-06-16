import os.path
import unittest
import time

import numpy as np
from array_record.python import array_record_module
from movie_lens_ranker.RecommendedMovies import RecommendedMovies

from helper import *
from movie_lens_ranker.data_loading import *

from movie_lens_ranker.util import _read_embeddings, read_embeddings


class TestRecommendedMovies(unittest.TestCase):
    def setUp(self):
        ratings_uri_dict = get_train_val_test_liked_uris(use_small=True)
        
        self.ratings_train_liked_uri = ratings_uri_dict["train_liked"]
        self.ratings_val_liked_uri = ratings_uri_dict["val_liked"]
        #self.ratings_test_liked_uri = ratings_uri_dict["test_liked"]
        
        self.ratings_train_3_uri = ratings_uri_dict["train_3"]
        self.ratings_val_3_uri = ratings_uri_dict["val_3"]
        #self.ratings_test_3_uri = ratings_uri_dict["test_3"]
        
        self.ratings_train_disliked_uri = ratings_uri_dict["train_disliked"]
        self.ratings_val_disliked_uri = ratings_uri_dict["val_disliked"]
        #self.ratings_test_disliked_uri = ratings_uri_dict["test_disliked"]
        
        # user recommendations with each user history subtracted already:
        # (user id, (movie_ids))
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/recommended_movies.array_record")
        
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        
        self.user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb-00000-of-00001.array_record")
        
        self.movie_ids_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")
        
        # (user_id, int array of movie_ids as a tuple) is full catalog for each user, no history subtracted
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record")
        self.recommendations_ts_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/recommended_movies_timestamps.array_record")
        
        # (movie_id, title, genres)
        self.movies_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")
        
        self.embeddings, self.num_users = read_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
        
        self.recommended_movies_getter = RecommendedMovies(
            num_users=self.num_users,
            movie_rec_file_uri=self.recommendations_uri,
            movie_rec_ts_file_uri=self.recommendations_ts_uri)
    
    def test_RecommendedMovies(self):
        
        ts = 978133414 #first timestamp from test dataset
        top_k = 20
        
        user_id = np.array([2, 4])
        timestamps = np.array([978133414, 978133414])
        movies = self.recommended_movies_getter.get_unseen_movies(user_id, timestamp=timestamps, top_k=top_k)
        self.assertEqual(np.shape(movies), (len(user_id), top_k))
        
        #demonstrating what to do if have inputs of this form:
        user_id = np.array([[2], [4]])
        if user_id.ndim > 1:
            user_id = user_id.squeeze()
        movies = self.recommended_movies_getter.get_unseen_movies(user_id, timestamp=timestamps, top_k=top_k)
        self.assertEqual(np.shape(movies), (len(user_id), top_k))
        
if __name__ == '__main__':
    unittest.main()
