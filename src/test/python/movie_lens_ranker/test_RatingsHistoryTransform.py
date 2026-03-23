import os.path
import unittest
from array_record.python import array_record_module
from helper import *

from movie_lens_ranker.RatingsHistoryLookupTransform import *
from movie_lens_ranker.data_loading import *

class TestRanker(unittest.TestCase):
    def setUp(self):
        
        self.ratings_train_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_1.array_record")
        
        self.ratings_test_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_2.array_record")
        
    def test_RatingsHistoryTransform(self):
        batch_size = 1024
        max_history = 2000 #a number large enough to test that padding works
        history_dict : Dict[int, Tuple[List, List, List]] = build_history_lookup(self.ratings_train_uri,
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
