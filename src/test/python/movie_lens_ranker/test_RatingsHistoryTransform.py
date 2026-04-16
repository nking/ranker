import os.path
import unittest
from array_record.python import array_record_module
from helper import *

from movie_lens_ranker.RatingsHistoryLookupTransform import *
from movie_lens_ranker.data_loading import *

class TestRanker(unittest.TestCase):
    def setUp(self):
        
        self.ratings_train_uri, self.ratings_val_uri, self.ratings_test_uri \
            = get_train_val_test_liked_uris(use_small=True)
        
        # user recommendations with each user history subtacted already:
        # (user id, (movie_ids))
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/recommended_movies.array_record")
        
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
        
        self.unseen_recommendations_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record")
        
        self.embeddings = read_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
        
    def test_RatingsHistoryTransform(self):
        batch_size = 1024
        max_history = 2000 #a number large enough to test that padding works
        history_dict, max_history__ = build_history_lookup(self.ratings_train_uri,
            batch_size=batch_size)
        
        transform = RatingsHistoryLookupTransform(history_lookup=history_dict,
            max_history=max_history)
            
        '''
        some ratings in partition 1
        1875::1101::4::975768800
        635::2068::4::975768823
        635::2357::4::975768823
        '''
        
        batch = [(1875,1101,4,975768800), (635, 2068, 4, 975768823), (635, 2357, 4, 975768823)]
        result:List[Dict[str, Union[int, List]]] = transform.map(batch)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(batch))
        for entry in result:
            self.assertTrue(isinstance(entry['user_id'], int))
            self.assertTrue(isinstance(entry['movie_id'], int))
            self.assertTrue(isinstance(entry['rating'], int))
            self.assertTrue(isinstance(entry['timestamp'], int))
            self.assertTrue(isinstance(entry['history_length'], int))
            self.assertTrue(isinstance(entry['history_movie_ids'], list))
            self.assertTrue(
                isinstance(entry['history_ratings'], list))
            self.assertEqual(max_history, len(entry['history_movie_ids']))
            self.assertEqual(max_history, len(entry['history_ratings']))
            self.assertEqual(-1, entry['history_movie_ids'][-1])
            self.assertEqual(-1, entry['history_ratings'][-1])
    
    if __name__ == '__main__':
        unittest.main()
