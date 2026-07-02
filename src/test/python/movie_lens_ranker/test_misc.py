import logging
import warnings
from typing import Dict, Tuple

import jax
import numpy as np

from movie_lens_ranker.train import create_fake_jagged_batch, pad_graph_tuple_batch
from movie_lens_ranker.util import calc_number_jax_graph_components
from movie_lens_ranker.util_np import optimized_batch_and_pad

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*'\.value' access is now deprecated\..*"
)

import unittest
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)], # Route to stdout to avoid red text in PyCharm
    force=True # CRITICAL: Overrides any logging setups initialized by jax, grain, or mlflow
)

class TestMisc(unittest.TestCase):

    def setUp(self):
        pass

    def _dictionaries_are_same(self, d0:Dict[str, np.ndarray], d1:Dict[str, np.ndarray]):
        self.assertEqual(len(d0), len(d1))
        self.assertEqual(d0.keys(), d1.keys())
        for key in d0.keys():
            np.testing.assert_array_equal(d0[key], d1[key])

    def _tuples_are_same(self, t0:Tuple[str], t1:Tuple[str]):
        self.assertEqual(len(t0), len(t1))
        for i, v in enumerate(t0):
            self.assertEqual(v, t1[i])


    def test_optimized_batch_and_pad(self):
        
        batch_size = 3
        max_history = 4
        num_candidates = 5
        
        user_id_range = (1, 10)
        movie_id_range = (1, 10)
        
        jax_graph_comp_dict = calc_number_jax_graph_components(
            batch_size=batch_size, max_history=max_history,
            num_candidates=num_candidates,
            n_local_devices=jax.local_device_count())
        
        fake_batch = create_fake_jagged_batch(batch_size=batch_size,
            max_history=max_history,
            num_candidates=num_candidates, user_id_range=user_id_range,
            movie_id_range=movie_id_range)

        #import pprint
        #pprint.pprint(f"fake_jagged_batch=\n{fake_batch}")
        print(f"fake_jagged_batch=\n{fake_batch}")


        n_local_devices = jax.local_device_count()
        
        padded_super_graph_0 = pad_graph_tuple_batch(fake_batch,
            jax_graph_comp_dict)
            
        padded_super_graph_1, _ = optimized_batch_and_pad(
            batch=fake_batch,
            max_nodes=jax_graph_comp_dict['max_nodes'],
            max_edges=jax_graph_comp_dict['max_edges'],
            max_graphs=jax_graph_comp_dict['max_graphs'],
        )

        print(f"padded_super_graph_1=\n{padded_super_graph_1}")


        ## compare the graphs
        self._dictionaries_are_same(padded_super_graph_1.edges, padded_super_graph_0.edges)
        np.testing.assert_array_equal(padded_super_graph_1.n_edge, padded_super_graph_0.n_edge)
        np.testing.assert_array_equal(padded_super_graph_1.n_node, padded_super_graph_0.n_node)
        self._tuples_are_same(padded_super_graph_1._fields, padded_super_graph_0._fields)
        self.assertIsNone(padded_super_graph_0.globals)
        self.assertIsNone(padded_super_graph_1.globals)

        np.testing.assert_array_equal(padded_super_graph_1.receivers, padded_super_graph_0.receivers)
        np.testing.assert_array_equal(padded_super_graph_1.senders, padded_super_graph_0.senders)


if __name__ == '__main__':
    unittest.main()
