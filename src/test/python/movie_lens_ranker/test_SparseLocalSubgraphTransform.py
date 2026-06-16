import os.path
import unittest
from array_record.python import array_record_module
from helper import *

from movie_lens_ranker.SparseLocalSubgraphTransform import *
from movie_lens_ranker.data_loading import *
from movie_lens_ranker.util import read_embeddings

class TestSparseLocalSubgraphTransform(unittest.TestCase):
    def setUp(self):
       
        
        ratings_uri_dict = get_train_val_test_liked_uris(data_size=DataSize.SMALL)
        
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
        

    def test_SparseLocalSubgraphTransform(self):
        batch_size = 1024
        max_history = 20
        num_candidates = 20
        
        #max_history__ is 1849
        watch_history = UserHistory(
            ratings_uri_list=[self.ratings_train_liked_uri,
                self.ratings_train_3_uri,
                self.ratings_train_disliked_uri], fixed_size=2048)
        
        disliked_history = UserHistory(
            ratings_uri_list=[self.ratings_train_disliked_uri],
            fixed_size=2048)
        
        all_movie_ids: List[int] = read_movies_array_record(self.movie_ids_uri,
            batch_size=batch_size)
      
        batch = [(1875, 1101, 4, 975768800), (635, 2068, 4, 975768823),
            (635, 2357, 4, 975768823)]
        
        transform1 = RatingsHistoryLookupTransform(
            history_lookup=watch_history, max_history=max_history)
        
        result1:Dict[str, np.ndarray] = transform1.map(batch)
        
        transform2 = HardNegativeSamplingTransform(
            history_lookup=watch_history,
            history_lookup_disliked=disliked_history,
            all_movie_ids=all_movie_ids,
            recommendations=self.recommended_movies_getter,
            num_candidates=num_candidates)
        
        rng = np.random.default_rng(seed=0)
        
        result2: Dict[str, np.ndarray] = transform2.random_map(result1, rng=rng)
        
        transform3 = SparseLocalSubgraphTransform()
        
        result3: List[jraph.GraphsTuple] = transform3.map(result2)
        
        self.assertTrue(len(result3), len(batch))
        for graph in result3:
            nodes = graph.nodes
            edges = graph.edges
            senders = graph.senders
            receivers = graph.receivers
            n_node = graph.n_node
            n_edge = graph.n_edge
            globals = graph.globals #None
            
            len1 = len(nodes["ids"])
            self.assertTrue(len1 > num_candidates)
            self.assertEqual(len1, len(nodes["label"]))
            self.assertEqual(len1, len(nodes["type"]))
            self.assertEqual(len1, len(nodes["candidate_mask"]))
            
            len2 = len(edges["rating"])
            self.assertTrue(len2 > num_candidates)
            self.assertEqual(len2, len(senders))
            self.assertEqual(len2, len(receivers))
            
            self.assertIsNotNone(n_node)
            self.assertIsNotNone(n_edge)

        #editing assert contents
        '''
        jraph.GraphsTuple(
                nodes={
                    "ids": jnp.concatenate([
                        [record["user_id"]],
                        record["history_movie_ids"][:n_real_history],
                        record["candidate_ids"]
                    ]),
                    "label": node_labels,
                    "type": jnp.array(
                        [0] + [1] * n_real_history + [2] * n_candidates),
                    # 0=User, 1=Hist, 2=Cand
                },
                edges={"rating": jnp.array(edge_features)},
                senders=jnp.array(senders),
                receivers=jnp.array(receivers),
                n_node=jnp.array([total_nodes]),
                n_edge=jnp.array([len(edge_features)]),
                globals=None
            ))
        '''
        

    if __name__ == '__main__':
        unittest.main()
