import os.path
import unittest
from array_record.python import array_record_module
from helper import *

from movie_lens_ranker.RatingsHistoryTransform import RatingsHistoryLookupTransform
from movie_lens_ranker.data_loading import *
from movie_lens_ranker.util import get_num_users_movies


class TestRanker(unittest.TestCase):
    def setUp(self):
        
        ratings_uri_dict = get_train_val_test_liked_uris(data_size=DataSize.SMALL)
        
        self.ratings_train_liked_uri = ratings_uri_dict["train_liked"]
        self.ratings_val_liked_uri = ratings_uri_dict["val_liked"]
        self.ratings_test_liked_uri = ratings_uri_dict["test_liked"]
        
        self.ratings_train_3_uri = ratings_uri_dict["train_3"]
        self.ratings_val_3_uri = ratings_uri_dict["val_3"]
        self.ratings_test_3_uri = ratings_uri_dict["test_3"]
        
        self.ratings_train_disliked_uri = ratings_uri_dict["train_disliked"]
        self.ratings_val_disliked_uri = ratings_uri_dict["val_disliked"]
        self.ratings_test_disliked_uri = ratings_uri_dict["test_disliked"]
        
        # the approximate hard negatives are the samples drawn from unwatched movies
        # the negatives uri has for each user, the list of negatives prioritized by:
        #    the "elite" hard negatives are the intersection of the natural hard negatives with the recommended movies,
        #    the natural hard negatives are the ones which user rated 1 or 2
        #  (user_id, tuple of negative movie_ids)
        self.negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record")
        
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        
        self.user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb-00000-of-00001.array_record")
        
        self.movie_ids_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")

        self.num_users, self.num_movies, self.emb_len = get_num_users_movies(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
        )
        self.embeddings = read_user_movie_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)


    def test_RatingsHistoryTransform(self):
        batch_size = 1024
        max_history = 2000 #a number large enough to test that padding works
        
        watch_history = UserHistory(
            ratings_uri_list=[self.ratings_train_liked_uri,
                self.ratings_train_3_uri,
                self.ratings_train_disliked_uri], max_history=max_history)
        
        transform1 = RatingsHistoryLookupTransform(
            history_lookup=watch_history, max_history=max_history)
        
        '''
        some ratings in partition 1
        1875::1101::4::975768800
        635::2068::4::975768823
        635::2357::4::975768823
        '''
        
        batch = [(1875,1101,4,975768800), (635, 2068, 4, 975768823), (635, 2357, 4, 975768823)]
        batch_size = len(batch)
        result = transform1.map(batch)
        self.assertIsNotNone(result)
        self.assertEqual(batch_size, len(result['history_length']))
        
        self.assertEqual(batch_size, len(result['history_movie_ids']))
        self.assertEqual(batch_size, len(result['history_ratings']))
        self.assertEqual(max_history, len(result['history_movie_ids'][0]))
        self.assertEqual(max_history, len(result['history_ratings'][0]))
        
        self.assertEqual(batch_size, len(result['movie_id']))
        self.assertEqual(batch_size, len(result['user_id']))
        self.assertEqual(batch_size, len(result['rating']))
        self.assertEqual(batch_size, len(result['timestamp']))
       
    if __name__ == '__main__':
        unittest.main()
