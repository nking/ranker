import os.path
import unittest
import numpy as np
from helper import *
from movie_lens_ranker.Negatives_vec import Negatives


class TestNegatives(unittest.TestCase):
    def setUp(self):
        self.negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/train_negatives.array_record")
    
    def test_Negatives(self):
        
        negatives = Negatives(self.negatives_uri, fixed_size=256)
        
        '''
        some test data:
        6040::7726::2::956704081
        6039::9393::2::956705323
        '''
        inp_user_id = np.array([6040, 6039], dtype=np.int32)
        length = 100
        movie_ids = negatives.get_negatives(user_id=inp_user_id, length=length, seed=0)
        
        self.assertEqual((len(inp_user_id), length), movie_ids.shape)
        for user_id, movies in zip(inp_user_id, movie_ids):
            if user_id == 6039:
                self.assertTrue(9393 in movies)
            else:
                self.assertTrue(7726 in movies)
        
if __name__ == '__main__':
    unittest.main()
