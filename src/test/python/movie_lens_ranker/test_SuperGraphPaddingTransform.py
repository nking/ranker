import os.path
import unittest

import jax
import jraph
import numpy as np
from array_record.python import array_record_module
from helper import *

from movie_lens_ranker.RatingsHistoryTransform import *
from movie_lens_ranker.UserHistory import UserHistory
from movie_lens_ranker.data_loading import *
from movie_lens_ranker.util import read_embeddings, \
    calc_number_jax_graph_components


class TestSuperGraphPadding(unittest.TestCase):
    
    def setUp(self):
        
        ratings_uri_dict = get_train_val_test_liked_uris(data_size=DataSize.TINY)
        
        self.ratings_train_liked_uri = ratings_uri_dict["train_liked"]
        self.ratings_val_liked_uri = ratings_uri_dict["val_liked"]
        self.ratings_test_liked_uri = ratings_uri_dict["test_liked"]
        
        self.ratings_train_3_uri = ratings_uri_dict["train_3"]
        self.ratings_val_3_uri = ratings_uri_dict["val_3"]
        self.ratings_test_3_uri = ratings_uri_dict["test_3"]
        
        self.ratings_train_disliked_uri = ratings_uri_dict["train_disliked"]
        self.ratings_val_disliked_uri = ratings_uri_dict["val_disliked"]
        self.ratings_test_disliked_uri = ratings_uri_dict["test_disliked"]
        
        # user recommendations with each user history subtracted already:
        # (user id, (movie_ids))
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/recommended_movies.array_record")
        
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        
        self.user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb-00000-of-00001.array_record")
        
        self.movie_ids_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")
        
        # (user_id, int array of movie_ids as a tuple) is full catalog for each user, no history subtracted
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record")
        self.recommendations_ts_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/recommended_movies_timestamps.array_record")
        
        # (movie_id, title, genres)
        self.movies_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")
        
        self.embeddings, self.num_users = read_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
        
        self.recommended_movies_getter = RecommendedMovies(
            num_users=self.num_users, movie_rec_file_uri=self.recommendations_uri,
            movie_rec_ts_file_uri=self.recommendations_ts_uri)
        
    def test_transform(self):
        batch_size = 1024
        max_history = 20
        num_candidates = 20
        
        jax_graph_comp_dict = calc_number_jax_graph_components(
            batch_size, max_history, num_candidates,
            n_local_devices=len(jax.local_devices()))
        
        watch_history = UserHistory(ratings_uri_list=[self.ratings_train_liked_uri, self.ratings_train_3_uri,
            self.ratings_train_disliked_uri], max_history=max_history)
        
        disliked_history = UserHistory(ratings_uri_list=[self.ratings_train_disliked_uri], max_history=max_history)

        all_movie_ids: List[int] = read_movies_array_record(self.movie_ids_uri, batch_size=batch_size)
       
        batch = [(1875, 1101, 4, 975768800), (635, 2068, 4, 975768823),
            (635, 2357, 4, 975768823)]
        
        transform1 = RatingsHistoryLookupTransform(
            history_lookup=watch_history, max_history=max_history)
        
        result1: Dict[str, np.ndarray] = transform1.map(batch)
        results = []
        
        for i, seed in enumerate((0, 0, 123)):
            
            rng = np.random.default_rng(seed)

            transform2 = HardNegativeSamplingTransform(
                history_lookup=watch_history,
                history_lookup_disliked=disliked_history,
                all_movie_ids= all_movie_ids,
                recommendations=self.recommended_movies_getter,
                num_candidates = num_candidates)
            
            transform3 = SparseLocalSubgraphTransform()
            
            transform4 = SuperGraphPaddingTransform(
                batch_size=batch_size, max_history=max_history,
                num_candidates=num_candidates,
                n_local_devices=len(jax.local_devices()), )
            
            result2:Dict[str, np.ndarray] = transform2.random_map(result1, rng=rng)
            result3:List[jraph.GraphsTuple] = transform3.map(result2)
            result4:jraph.GraphsTuple = transform4.map(result3)
            
            results.append(result4)
            
            self.assertTrue(len(result4.edges['rating']), jax_graph_comp_dict['max_edges'])
            self.assertTrue(len(result4.receivers), jax_graph_comp_dict['max_edges'])
            self.assertTrue(len(result4.senders), jax_graph_comp_dict['max_edges'])
            self.assertTrue(len(result4.n_edge), jax_graph_comp_dict['max_graphs'])
            self.assertTrue(len(result4.n_node), jax_graph_comp_dict['max_graphs'])
            for key in result4.nodes.keys():
                self.assertTrue(len(result4.nodes[key]), jax_graph_comp_dict['max_nodes'])
            
            """
            n_node=n_node_padded,
                n_edge=n_edge_padded,
                nodes=nodes_padded,
                edges=edges_padded,
                globals=globals_padded,
                senders=senders_padded,
                receivers=receivers_padded
            """
        
        #results[0] and results[1] should be same
        np.testing.assert_array_equal(results[0].edges['rating'], results[1].edges['rating'], strict=True)
        np.testing.assert_array_equal(results[0].receivers, results[1].receivers, strict=True)
        np.testing.assert_array_equal(results[0].senders, results[1].senders, strict=True)
        np.testing.assert_array_equal(results[0].n_edge, results[1].n_edge, strict=True)
        np.testing.assert_array_equal(results[0].n_node, results[1].n_node, strict=True)
        for key in results[0].nodes.keys():
            a = results[0].nodes[key]
            b = results[1].nodes[key]
            np.testing.assert_array_equal(a, b, strict=True)
        
        a = results[0]
        b = results[2]
        
        # results[0] and results[2] should be different for nodes['ids'] and nodes['label']
        np.testing.assert_array_equal(a.edges['rating'], b.edges['rating'], strict=True)
        
        np.testing.assert_array_equal(a.receivers,
                b.receivers, strict=True)
        np.testing.assert_array_equal(a.senders,
                b.senders, strict=True
        )
        np.testing.assert_array_equal( a.n_edge, b.n_edge,
                strict=True
        )
        np.testing.assert_array_equal(a.n_node, b.n_node,
                strict=True
        )
        for key in a.nodes.keys():
            aa = a.nodes[key]
            bb = b.nodes[key]
            if key == "ids" or key == "label":
                np.testing.assert_raises(
                    AssertionError,
                    np.testing.assert_array_equal,aa, bb, strict=True
                )
            else:
                np.testing.assert_array_equal(aa, bb, strict=True)
        
    if __name__ == '__main__':
        unittest.main()
