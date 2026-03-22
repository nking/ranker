import collections
import os.path
import unittest
from typing import Tuple, Union, Dict, Sequence, TypeVar, Generic, \
    Iterator

from array_record.python import array_record_module

from helper import *

from movie_lens_ranker.RatingsHistoryLookupTransform import *

class TestRanker(unittest.TestCase):
    def setUp(self):
        
        self.ratings_train_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_1.array_record")
        
        self.ratings_test_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_2.array_record")
        
    def test_build_history_lookup(self):
        batch_size = 1024
        #expecting Dict[int, Tuple[list, list, list]]
        history_dict = build_history_lookup(self.ratings_train_uri,
            batch_size=batch_size)
        self.assertTrue(isinstance(history_dict, dict))
        n_hist = len(history_dict) #number of users who rated movies in train dataset
        self.assertTrue(n_hist > 0 and n_hist < 6040)
        min_user_id = min(history_dict.keys())
        entry_tuples = history_dict[min_user_id]
        self.assertEqual(3, len(entry_tuples))
        for i in range(3):
            self.assertTrue(isinstance(entry_tuples[i], list))
            self.assertTrue(len(entry_tuples[i]) > 0)
            self.assertTrue(isinstance(entry_tuples[i][0], int))
        
    if __name__ == '__main__':
        unittest.main()
