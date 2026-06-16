import time
from unittest import TestCase
import numpy as np
from movie_lens_ranker.util_numba import *
from movie_lens_ranker.util_numba import _extract_shuffle_and_append

class NumbaOpsTest(TestCase):
    
    def test_row_wise_intersect(self):
        pad_value = -1
        a = np.array([[row * 4 + col for col in range(4)] for row in range(2)])
        b = np.roll(a, shift=1, axis=1)
        for row in range(b.shape[0]):
            b[row][-1] = pad_value
        c = row_wise_intersect(a, b, pad_value)
        
        expected = np.array([[0,1,3, -1], [4,5,7, -1]])
        
        np.testing.assert_array_equal(c, expected, strict=True)
    
    def row_wise_subtract(self):
        #def row_wise_subtract(arr1:np.ndarray, arr2:np.ndarray, pad_value:int) -> np.ndarray:
        pad_value = -1
        a = np.array([[row * 4 + col for col in range(4)] for row in range(2)])
        b = np.roll(a, shift=1, axis=1)
        for row in range(b.shape[0]):
            b[row][-1] = pad_value
        c = row_wise_sortedset_subtract(a, b, pad_value)
        
        expected = np.array([[3, -1, -1, -1], [6, -1, -1, -1]])
        
        np.testing.assert_array_equal(c, expected, strict=True)
    
    def simultaneous_shuffle(self):
        seed = 0
        a = np.array([[row * 4 + col for col in range(4)] for row in range(2)])
        b = np.roll(a, shift=1, axis=1)
        a_orig = a.copy()
        b_orig = b.copy()
        simultaneous_shuffle(a, b, seed)
        
        #asserrt that they aren't equal
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            a,
            a_orig
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            b,
            b_orig
        )
        for row in range(a.shape[0]):
            for col in range(a.shape[1]):
                col2 = a[row].tolist().index(a_orig[row][col])
                self.assertEqual(b[row][col], b_orig[row][col2])
                
    def test_generate_type_4_negatives(self):
        seed = 0
        pad_value=-1
        num_movies = 20
        n_negs = 10
        all_movie_ids = np.asarray([i for i in range(num_movies)])
        movie_histories = [[i for i in range(num_movies) if (i&1)==0]]
        movie_histories[0].extend([-1,-1])
        movie_histories = np.asarray(movie_histories)
        
        expected_pool_to_draw_from = row_wise_sortedset_subtract(
            np.broadcast_to(all_movie_ids, (movie_histories.shape[0], len(all_movie_ids))),
            movie_histories, pad_value)
        
        are_odd = (expected_pool_to_draw_from[0] % 2 == 1)
        np.testing.assert_array_equal(are_odd, True)
        
        type_4_neg = generate_type_4_negatives(all_movie_ids, movie_histories, n_negs=n_negs,
            pad_value=pad_value, seed=seed)
        
        self.assertEqual((1, n_negs), type_4_neg.shape)
        
        are_odd = (type_4_neg[0] % 2 == 1)
        np.testing.assert_array_equal(are_odd, True)
        
        #movie catalog - watch history
        #assert no movie_histories are in final selection and no repeats
        hist_set = set(movie_histories[0].tolist())
        type_4_set = set(type_4_neg[0].tolist())
        
        self.assertEqual(len(type_4_set), type_4_neg.shape[1])
        
        self.assertEqual(0, len(type_4_set.intersection(hist_set)))

    def test_build_negative_pool_numba(self):
        #build_negative_pool_numba(arr1:np.ndarray, arr2:np.ndarray, arr3:np.ndarray,
        #arr4:np.ndarray, target1:int, target2:int, target3:int, num_negatives:int,
        #pad_value:int) -> np.ndarray:
        arr1=np.array([[1, 2, -1],
            [-1, -1, -1]])
        arr2 = np.array([[3, 4, -1, -1],
            [9, -1, -1, -1]])
        arr3 = np.array([[-1, -1, -1, -1],
            [10, -1, -1, -1]])
        arr4 = np.array([[5, 6, 7, 8],
            [20, 21, 22, 23]])
        target1 = 1
        target2 = 1
        target3 = 1
        num_negatives = 4
        pad_value = -1
        seed = int(time.time())
        
        negatives = build_negative_pool_numba(arr1=arr1, arr2=arr2, arr3=arr3,
            arr4=arr4, target1=target1, target2=target2, target3=target3,
            num_negatives=num_negatives, pad_value=pad_value, seed=seed)
        
        #np.array([[1 or 2,  3 or 4,  5or6or7or8, 5or6or7or8],
        #         [9, 10, 21or22or23or34, 21or22or23or34]])"
        self.assertTrue(negatives[0][0] in set(arr1[0, 0:1].tolist()))
        self.assertTrue(negatives[0][1] in set(arr2[0, 0:1].tolist()))
        self.assertTrue(negatives[0][2] in set(arr4[0].tolist()))
        
        self.assertTrue(negatives[1][0] == arr2[1][0])
        self.assertTrue(negatives[1][1] == arr3[1][0])
        self.assertTrue(negatives[1][2] in set(arr4[1].tolist()))
        
    def test_extract_shuffle_and_append(self):
        pad_value = -1
        seed = int(time.time())
        np.random.seed(seed)
        
        num_negatives = 7
        
        source_row = np.array([1, 2, 3, 4, 5])
        
        max_take = 3
        
        #write to pool_row
        pool_row = np.full((num_negatives), -1, dtype=source_row.dtype)
        
        buff1 = np.empty(len(source_row), dtype=source_row.dtype)
        
        write_to_idx = 0
        
        write_to_idx = _extract_shuffle_and_append(source_row=source_row,
            max_take=max_take, pool_row=pool_row, write_idx=write_to_idx,
            num_negatives=num_negatives, pad_value=pad_value, buffer_row=buff1,
            seed = None)
        
        self.assertEqual(write_to_idx, max_take)
        cond0 = set(source_row.tolist())
        for i in range (write_to_idx):
            self.assertTrue(pool_row[i] in cond0)
        np.testing.assert_array_equal(pool_row[write_to_idx:], -1)
        
        pool_row2 = np.full((num_negatives), -1, dtype=source_row.dtype)
        max_take2 = num_negatives
        write_to_idx2 = _extract_shuffle_and_append(source_row=source_row,
            max_take=max_take2, pool_row=pool_row2, write_idx=write_to_idx,
            num_negatives=num_negatives, pad_value=pad_value, buffer_row=buff1,
            seed=None)
        #[-1, -1, -1, _, _, _, _]
        self.assertEqual(write_to_idx2, max_take2)
        np.testing.assert_array_equal(pool_row2[0:write_to_idx], -1)
        cond0 = set(source_row.tolist())
        for i in range(write_to_idx, num_negatives):
            self.assertTrue(pool_row2[i] in cond0)
        