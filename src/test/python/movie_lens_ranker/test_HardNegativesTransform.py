import os.path
import unittest
from array_record.python import array_record_module
from helper import *
from movie_lens_ranker.HardNegativeSamplingTransform import *

from movie_lens_ranker.RatingsHistoryLookupTransform import *

class TestRanker(unittest.TestCase):
    def setUp(self):
        
        self.ratings_train_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_1.array_record")
        
        self.ratings_test_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_2.array_record")
        
        self.exact_hard_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/user_recommendations_disliked_in_train.array_record")
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/user_recommendations_without_train_val.array_record")
        
        self.ratings_train_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_1.array_record")
        
        self.ratings_test_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_2.array_record")
        
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/movie_embeddings.array_record")
        
        self.user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/user_embeddings.array_record")
        
        self.movie_ids_uri = os.path.join(get_project_dir(),
            "src/test/resources/movie_ids.array_record")
        
        self.unseen_recommendations_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/user_recommendations_without_train_val.array_record")
        
    def test_read_arra_records(self):
        batch_size = 1024
        
        all_movie_ids: List[int] = read_movies_array_record(self.movie_ids_uri, batch_size=batch_size)
        self.assertEqual(len(all_movie_ids), 3883)
        self.assertTrue(isinstance(all_movie_ids, list))
        self.assertTrue(isinstance(all_movie_ids[0], int))
        
        exact_negatives_dict: Dict[int, Set[int]] = read_user_exact_negatives(self.exact_hard_negatives_uri,
            batch_size)
        self.assertTrue(isinstance(exact_negatives_dict, dict))
        min_user_id = min(exact_negatives_dict.keys())
        entry = exact_negatives_dict[min_user_id]
        self.assertTrue(isinstance(entry, set))
        self.assertTrue(isinstance(next(iter(entry)), int))
        
        unseen_recommendations: Dict[int, Set[int]] = read_user_unseen_recommendations(self.unseen_recommendations_uri, batch_size=batch_size)
        self.assertTrue(isinstance(unseen_recommendations, dict))
        min_user_id = min(unseen_recommendations.keys())
        entry = unseen_recommendations[min_user_id]
        self.assertTrue(isinstance(entry, set))
        self.assertTrue(isinstance(next(iter(entry)), int))
        
    def test_HardNegativesTransform(self):
        batch_size = 1024
        max_history = 20
        num_candidates = 20
        
        history_dict : Dict[int, Tuple[List, List, List]] = build_history_lookup(self.ratings_train_uri,
            batch_size=batch_size)
        all_movie_ids: List[int] = read_movies_array_record(
            self.movie_ids_uri, batch_size=batch_size)
        exact_negatives_dict: Dict[
            int, Set[int]] = read_user_exact_negatives(
            self.exact_hard_negatives_uri,
            batch_size)
        unseen_recommendations: Dict[
            int, Set[int]] = read_user_unseen_recommendations(
            self.unseen_recommendations_uri, batch_size=batch_size)
        
        batch = [(1875, 1101, 4, 975768800), (635, 2068, 4, 975768823),
            (635, 2357, 4, 975768823)]
        
        transform1 = RatingsHistoryLookupTransform(history_lookup=history_dict,
            max_history=max_history)
            
        result1:List[Dict[str, Union[int, List]]] = transform1.map(batch)
        
        
        transform2 = HardNegativeSamplingTransform(
            history_lookup=history_dict,
            all_movie_ids= all_movie_ids,
            exact_negatives_dict = exact_negatives_dict,
            unseen_recommendations = unseen_recommendations, num_candidates = num_candidates,
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
