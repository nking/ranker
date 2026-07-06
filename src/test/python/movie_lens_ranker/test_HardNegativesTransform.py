import os.path
import unittest

import numpy as np
from array_record.python import array_record_module
from helper import *

from movie_lens_ranker.RatingsHistoryTransform import RatingsHistoryLookupTransform
from movie_lens_ranker.UserHistory import UserHistory
from movie_lens_ranker.data_loading import *
from movie_lens_ranker.util import get_num_users_movies


class TestRanker(unittest.TestCase):
    
    def setUp(self):
        
        ratings_uri_dict = get_train_val_test_liked_uris(data_size=DataSize.TINY)
        
        self.ratings_train_liked_uri = ratings_uri_dict["train_liked"]
        self.ratings_val_liked_uri = ratings_uri_dict["val_liked"]
        self.ratings_test_liked_uri = ratings_uri_dict["test_liked"]
        
        self.ratings_train_3_uri = ratings_uri_dict["train_3"]
        self.ratings_val_3_uri = ratings_uri_dict["val_3"]
        self.ratings_test_3_uri = ratings_uri_dict["test_3"]
        
        self.ratings_train_disliked_uri = ratings_uri_dict["train_disliked"]
        self.ratings_val_disliked_uri = ratings_uri_dict["val_disliked"]
        self.ratings_test_disliked_uri = ratings_uri_dict["test_disliked"]
        
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

        self.num_users, self.num_movies, self.emb_len = get_num_users_movies(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
        )
        self.embeddings = read_user_movie_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
        
        self.recommended_movies_getter = RecommendedMovies(
            num_users=self.num_users, movie_rec_file_uri=self.recommendations_uri,
            movie_rec_ts_file_uri=self.recommendations_ts_uri)
        
    def test_HardNegativesTransform(self):
        batch_size = 1024
        max_history = 20
        num_candidates = 20
        
        watch_history = UserHistory(ratings_uri_list=[self.ratings_train_liked_uri, self.ratings_train_3_uri,
            self.ratings_train_disliked_uri], max_history=max_history)
        
        disliked_history = UserHistory(ratings_uri_list=[self.ratings_train_disliked_uri], max_history=max_history)

        all_movie_ids: List[int] = read_movies_array_record(self.movie_ids_uri, batch_size=batch_size)
       
        batch = [(1875, 1101, 4, 975768800), (635, 2068, 4, 975768823),
            (635, 2357, 4, 975768823)]
        
        transform1 = RatingsHistoryLookupTransform(history_lookup=watch_history, max_history=max_history)
        
        result1:Dict[str, np.ndarray] = transform1.map(batch)
        
        results_dict = {}
        
        for i, seed in enumerate((0, 0, 123)):
            
            rng = np.random.default_rng(seed)

            transform2 = HardNegativeSamplingTransform(
                history_lookup=watch_history,
                history_lookup_disliked=disliked_history,
                all_movie_ids= all_movie_ids,
                recommendations=self.recommended_movies_getter,
                num_candidates = num_candidates)
                
            result2:Dict[str, np.ndarray] = transform2.random_map(result1, rng=rng)
            
            self.assertTrue(isinstance(result2, dict))
            
            results_dict[i] = result2
            
            expected_keys = {"user_id", "movie_id", "rating", "timestamp",
                "history_movie_ids", "history_ratings", "history_length",
                "candidate_ids", "labels"}
            for expected_key in expected_keys:
                self.assertTrue(expected_key in result2.keys())
                self.assertTrue(isinstance(result2[expected_key], np.ndarray))
            
            for i, user_id in enumerate(result2["user_id"]):
                self.assertEqual(batch[i][0], user_id)
        
        dict0 = results_dict[0]
        dict1 = results_dict[1]
        dict2 = results_dict[2]
        
        a = dict0['candidate_ids']
        b = dict1['candidate_ids']
        np.testing.assert_array_equal(a, b, strict=True)
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            dict1['candidate_ids'],
            dict2['candidate_ids']
        )
        
    if __name__ == '__main__':
        unittest.main()
