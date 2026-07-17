import unittest

import jax.distributed
import os
from array_record.python import array_record_module
from dotenv import dotenv_values

from movie_lens_ranker.train import *

from helper import *

class TestGraph(unittest.TestCase):

    def setUp(self):

        # === these are so that grain dataloader can read data from fake gcs server running in docker ====

        ratings_uri_dict = get_train_val_test_liked_uris(data_size=DataSize.TINY, use_gcs_uri=True)

        self.ratings_train_liked_uri = ratings_uri_dict["train_liked"]
        self.ratings_val_liked_uri = ratings_uri_dict["val_liked"]
        self.ratings_test_liked_uri = ratings_uri_dict["test_liked"]

        self.ratings_train_3_uri = ratings_uri_dict["train_3"]
        self.ratings_val_3_uri = ratings_uri_dict["val_3"]
        self.ratings_test_3_uri = ratings_uri_dict["test_3"]

        self.ratings_train_disliked_uri = ratings_uri_dict["train_disliked"]
        self.ratings_val_disliked_uri = ratings_uri_dict["val_disliked"]
        self.ratings_test_disliked_uri = ratings_uri_dict["test_disliked"]

        # (movie_id, float array of embed_dim as a tuple)
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
                                                 "src/test/resources/data/movie_emb-00000-of-00001.array_record")

        # (user_id, float array of embed_dim as a tuple)
        self.user_embeddings_uri = os.path.join(get_project_dir(),
                                                "src/test/resources/data/user_emb-00000-of-00001.array_record")

        # (user_id, int array of movie_ids as a tuple) is full catalog for each user, no history subtracted
        self.recommendations_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record")
        self.recommendations_ts_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies_timestamps.array_record")

        # (movie_id, title, genres)
        self.movies_uri = os.path.join(get_project_dir(),
                                       "src/test/resources/data/movies-00000-of-00001.array_record")


    def test_optimized_batch_and_pad(self):

        batch_size = 3
        max_history = 4
        num_candidates = 5

        user_id_range = (1, 6040)
        movie_id_range = (6041, 6041 + 3883)

        jax_graph_comp_dict = calc_number_jax_graph_components(
            batch_size=batch_size, max_history=max_history,
            num_candidates=num_candidates,
            n_local_devices=jax.local_device_count())

        fake_batch = create_fake_jagged_batch(batch_size=batch_size,
                                              max_history=max_history,
                                              num_candidates=num_candidates, user_id_range=user_id_range,
                                              movie_id_range=movie_id_range,
                                              user_embeddings_uri=self.user_embeddings_uri,
                                              movie_embeddings_uri=self.movie_embeddings_uri)

        #import pprint
        #pprint.pprint(f"fake_jagged_batch=\n{fake_batch}")
        #print(f"fake_jagged_batch=\n{fake_batch}")

        n_local_devices = jax.local_device_count()

        padded_super_graph_0 = pad_graph_tuple_batch(fake_batch,
                                                     jax_graph_comp_dict)

        padded_super_graph_1, _ = optimized_batch_and_pad(
            batch=fake_batch,
            max_nodes=jax_graph_comp_dict['max_nodes'],
            max_edges=jax_graph_comp_dict['max_edges'],
            max_graphs=jax_graph_comp_dict['max_graphs'],
        )

        #print(f"padded_super_graph_1=\n{padded_super_graph_1}")


        ## compare the graphs
        self._dictionaries_are_same(padded_super_graph_1.edges, padded_super_graph_0.edges)
        np.testing.assert_array_equal(padded_super_graph_1.n_edge, padded_super_graph_0.n_edge)
        np.testing.assert_array_equal(padded_super_graph_1.n_node, padded_super_graph_0.n_node)
        self._tuples_are_same(padded_super_graph_1._fields, padded_super_graph_0._fields)
        self.assertIsNone(padded_super_graph_0.globals)
        self.assertIsNone(padded_super_graph_1.globals)

        np.testing.assert_array_equal(padded_super_graph_1.receivers, padded_super_graph_0.receivers)
        np.testing.assert_array_equal(padded_super_graph_1.senders, padded_super_graph_0.senders)

        np.set_printoptions(threshold=np.inf)
        print(f'rating={np.array2string(padded_super_graph_1.edges["rating"], separator=', ')}')
        print(f'n_edge={np.array2string(padded_super_graph_1.n_edge, separator=', ')}')
        print(f'n_node={np.array2string(padded_super_graph_1.n_node, separator=', ')}')
        for key in padded_super_graph_1.nodes.keys():
            print(f'{key}={np.array2string(padded_super_graph_1.nodes[key], separator=', ')}')
        print(f'receivers={np.array2string(padded_super_graph_1.receivers, separator=', ')}')
        print(f'senders={np.array2string(padded_super_graph_1.senders, separator=', ')}')


    def _dictionaries_are_same(self, d0:Dict[str, np.ndarray], d1:Dict[str, np.ndarray]):
            self.assertEqual(len(d0), len(d1))
            self.assertEqual(d0.keys(), d1.keys())
            for key in d0.keys():
                np.testing.assert_array_equal(d0[key], d1[key])

    def _tuples_are_same(self, t0:Tuple[str], t1:Tuple[str]):
        self.assertEqual(len(t0), len(t1))
        for i, v in enumerate(t0):
            self.assertEqual(v, t1[i])


if __name__ == '__main__':
    unittest.main()
