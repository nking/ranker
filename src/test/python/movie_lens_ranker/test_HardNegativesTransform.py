import os.path
import unittest
from array_record.python import array_record_module
from helper import *
from movie_lens_ranker.HardNegativeSamplingTransform import *

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
        
        test_res_dir = os.path.join(get_project_dir(), "src/test/resources/data")
        self.recommended_movies_getter = RecommendedMovies(movie_rec_file_path=
            os.path.join(test_res_dir, "recommended_movies.array_record"),
            movie_rec_ts_file_path=
            os.path.join(test_res_dir, "recommended_movies_timestamps.array_record"))
        
        self.embeddings = read_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
    
    def test_HardNegativesTransform(self):
        batch_size = 1024
        max_history = 20
        num_candidates = 20
        
        history_dict, max_history__ = build_history_lookup(self.ratings_train_uri,
            batch_size=batch_size)
        all_movie_ids: List[int] = read_movies_array_record(
            self.movie_ids_uri, batch_size=batch_size)
        exact_negatives_dict: Dict[
            int, Set[int]] = read_user_negatives(
            self.negatives_uri,
            batch_size)
        
        batch = [(1875, 1101, 4, 975768800), (635, 2068, 4, 975768823),
            (635, 2357, 4, 975768823)]
        
        transform1 = RatingsHistoryLookupTransform(
            history_dict,
            max_history=max_history)
            
        result1:List[Dict[str, Union[int, List]]] = transform1.map(batch)
        
        
        transform2 = HardNegativeSamplingTransform(
            history_lookup=history_dict,
            all_movie_ids= all_movie_ids,
            exact_negatives_dict = exact_negatives_dict,
            recommendations=self.recommended_movies_getter, num_candidates = num_candidates,
            seed= 0)
        
        result2:List[Dict[str, Union[int, List[int], np.ndarray]]] = transform2.map(result1)
        self.assertTrue(isinstance(result2, list))
        expected_keys = {"user_id", "movie_id", "rating", "timestamp",
            "history_movie_ids", "history_ratings", "history_length",
            "candidate_ids", "labels"}
        for entry in result2:
            self.assertTrue(isinstance(entry, dict))
            keys = entry.keys()
            self.assertTrue(keys == expected_keys)
            self.assertTrue(isinstance(entry["user_id"], int))
            self.assertTrue(isinstance(entry["movie_id"], int))
            self.assertTrue(isinstance(entry["rating"], int))
            self.assertTrue(isinstance(entry["timestamp"], int))
            self.assertTrue(isinstance(entry["history_movie_ids"], list))
            self.assertTrue(isinstance(entry["history_ratings"], list))
            self.assertTrue(isinstance(entry["history_length"], int))
            self.assertTrue(isinstance(entry["candidate_ids"], np.ndarray))
            self.assertTrue(isinstance(entry["labels"], np.ndarray))

    if __name__ == '__main__':
        unittest.main()
