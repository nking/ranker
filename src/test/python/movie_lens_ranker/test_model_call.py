import json
import os

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import unittest

from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.train import eval_step, score_and_shape_results, create_dummy_super_padded_graph
from movie_lens_ranker.util_plots import get_project_dir


class TestModelMethods(unittest.TestCase):

    def build_mock_graph_batch(self, num_candidates: int, embed_dim: int) -> jraph.GraphsTuple:
        """
        Creates a mock padded supergraph consisting of:
        - Graph 1: Targets a 'Head' movie (tier 0) at candidate index 2
        - Graph 2: Targets a 'Torso' movie (tier 1) at candidate index 1
        - Graph 3: Targets a 'Tail' movie (tier 2) at candidate index 0 (within top_k=2)
        - Graph 4: 0 Nodes (Dummy padding graph)
        """
        batch_size = 4
        max_history = 2
        '''
        user_id_range=(1,4)
        movie_id_range=(6041, 6060)
        user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb-00000-of-00001.array_record")
        movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        #movie_embeddings_uri =  self.transform_to_gs_uri(movie_embeddings_uri),
        #user_embeddings_uri = self.transform_to_gs_uri(user_embeddings_uri),

        fake0 = create_dummy_super_padded_graph(batch_size, max_history, num_candidates,
                user_id_range, movie_id_range, user_embeddings_uri, movie_embeddings_uri)
        '''
        # --- Graph 1: 1 User (type 1), 2 History (type 2), K Candidates (type 3) ---
        g1_n_hist = 2
        g1_nodes = 1 + g1_n_hist + num_candidates
        g1_edges = g1_n_hist + num_candidates

        g1_types = [1] + [2]*g1_n_hist + [3]*num_candidates
        # user(0), history(0,0), candidates(0, 0, 1, 0 -> target at index 2)
        g1_labels = [0] + [0]*g1_n_hist + [0,0,1,0]
        g1_ids    = [1] + [88, 77] + [100, 101, 10, 102]
        #ratings are edges, so no user entry.  has real_history=4 or 5, candidates are 0
        g1_ratings = [4, 5] + [0]*num_candidates
        g1_senders = [1, 2] + [0]*num_candidates
        g1_receivers = [0]*g1_n_hist + [1+g1_n_hist+i for i in range(num_candidates)]

        # --- Graph 2: 1 User (type 1), 1 History (type 2), K Candidates (type 3) ---
        g2_n_hist = 1
        g2_nodes = 1 + g2_n_hist + num_candidates
        g2_edges = g2_n_hist + num_candidates

        g2_types = [1] + [2]*g2_n_hist + [3]*num_candidates
        # user(0), history(0), candidates(0, 1, 0, 0 -> target at index 1)
        g2_labels = [0] + [0]*g2_n_hist + [0,1,0,0]
        g2_ids = [2] + [87] + [200, 20, 201, 202] # ID 20 is at index 1
        g2_ratings = [5] + [0]*num_candidates
        g2_senders = [1] + [0]*num_candidates
        g2_receivers = [0]*g2_n_hist + [1+g2_n_hist+i for i in range(num_candidates)]

        # --- Graph 3: 1 User (type 1), 1 History (type 2), K Candidates (type 3) ---
        g3_n_hist = 1
        g3_nodes = 1 + g3_n_hist + num_candidates
        g3_edges = g3_n_hist + num_candidates

        g3_types = [1] + [2]*g3_n_hist + [3]*num_candidates
        # user(0), history(0), candidates(1, 0, 0, 0 -> target at index 0, within top_k=2)
        g3_labels = [0] + [0]*g3_n_hist + [1, 0,0,0]
        g3_ids = [3] + [86] + [30, 300, 301, 302]
        g3_ratings = [4] + [0]*num_candidates
        g3_senders = [1] + [0]*num_candidates
        g3_receivers = [0]*g3_n_hist + [1+g3_n_hist+i for i in range(num_candidates)]

        # --- Graph 4: no Dummy Padding Graph ---
        g4_n_hist = 0
        g4_nodes = num_candidates
        g4_edges = num_candidates
        g4_types = [0]*num_candidates
        g4_labels = [0]*num_candidates
        g4_ids =  [0]*num_candidates
        g4_ratings = [0]*num_candidates
        g4_senders = [0]*num_candidates
        g4_receivers = [1+i for i in range(num_candidates)]

        total_nodes = g1_nodes + g2_nodes + g3_nodes + g4_nodes
        total_edges = g1_edges + g2_edges + g3_edges + g4_edges

        embeddings = jax.random.normal(jax.random.PRNGKey(0), (total_nodes, embed_dim))

        fake = jraph.GraphsTuple(
            n_node=jnp.array([g1_nodes, g2_nodes, g3_nodes, g4_nodes, 0], dtype=jnp.int32),
            n_edge=jnp.array([g1_edges, g2_edges, g3_edges, g4_edges, 0], dtype=jnp.int32),
            nodes={
                "ids": jnp.array(g1_ids + g2_ids + g3_ids + g4_ids, dtype=jnp.int32),
                "label": jnp.array(g1_labels + g2_labels + g3_labels + g4_labels, dtype=jnp.int32),
                "type": jnp.array(g1_types + g2_types + g3_types + g4_types, dtype=jnp.int32),
                "candidate_mask": jnp.array(g1_types + g2_types + g3_types + g4_types) == 3,
                "embeddings": embeddings
            },
            edges={"rating": jnp.array(g1_ratings + g2_ratings + g3_ratings + g4_ratings, dtype=jnp.int32)},
            senders=jnp.array(g1_senders + g2_senders + g3_senders + g4_senders, dtype=jnp.int32),
            receivers=jnp.array(g1_receivers + g2_receivers + g3_receivers + g4_receivers, dtype=jnp.int32),
            globals=None
        )
        return fake

    def test_score_and_shape_results(self):
        NUM_CANDIDATES = 4
        EMBED_DIM = 16

        model = GraphRanker(
            emb_in_dim=EMBED_DIM,
            num_candidates=NUM_CANDIDATES,
            hidden_features=32,
            num_layers=1,
            out_features=16,
            heads=2
        )

        mock_graph = self.build_mock_graph_batch(NUM_CANDIDATES, EMBED_DIM)
        scores_2d, labels_2d, main_mask, cand_ids_2d = score_and_shape_results(model, mock_graph)

        num_total_graphs = mock_graph.n_node.shape[0]
        expected_shape = (num_total_graphs, NUM_CANDIDATES)

        assert scores_2d.shape == expected_shape
        assert labels_2d.shape == expected_shape
        assert main_mask.shape == expected_shape
        assert cand_ids_2d.shape == expected_shape

        # Verify that the last dummy graph row is completely masked out
        assert not jnp.any(main_mask[-1]), "Dummy padding graph was not correctly masked out!"
        print(f"✅ test_score_and_shape_results passed.")

    def test_graph_ranker_call(self):
        NUM_CANDIDATES = 4
        EMBED_DIM = 16

        model = GraphRanker(
            emb_in_dim=EMBED_DIM,
            num_candidates=NUM_CANDIDATES,
            hidden_features=32,
            num_layers=1,
            out_features=16,
            heads=2
        )

        mock_graph = self.build_mock_graph_batch(NUM_CANDIDATES, EMBED_DIM)
        scores = model(mock_graph)
        n_graphs = len(mock_graph.n_node)
        expected_shape = (n_graphs * NUM_CANDIDATES,)
        assert scores.shape == expected_shape, "Output shape mismatch!"
        assert not jnp.isnan(scores).any(), "Model produced NaNs!"

        print("✅ Model call executes perfectly and returns strictly compiled static shapes.")

    def test_eval_step(self):
        NUM_CANDIDATES = 4
        EMBED_DIM = 16
        TOP_K = 4
        MOVIE_OFFSET = 0

        model = GraphRanker(
            emb_in_dim=EMBED_DIM,
            num_candidates=NUM_CANDIDATES,
            hidden_features=32,
            num_layers=1,
            out_features=16,
            heads=2
        )

        mock_graph = self.build_mock_graph_batch(NUM_CANDIDATES, EMBED_DIM)

        # Build lookup map assigning explicit tiers:
        # ID 10 -> Tier 0 (Head)
        # ID 20 -> Tier 1 (Torso)
        # ID 30 -> Tier 2 (Tail)
        max_id = int(jnp.max(mock_graph.nodes["ids"])) + 1
        movie_tiers = np.zeros((max_id,), dtype=np.int32)
        movie_tiers[10] = 0
        movie_tiers[20] = 1
        movie_tiers[30] = 2

        with jax.disable_jit():
            scores_2d, labels_2d, main_mask, cand_ids_2d = score_and_shape_results(model, mock_graph)
            print("\n--- EAGER MODE DEBUGGING ---")
            print("Candidate IDs 2D:\n", cand_ids_2d)
            print("Labels 2D:\n", labels_2d)
            print("Main Mask:\n", main_mask)

            # Check mapped tiers for each candidate
            resolved_tiers = movie_tiers[cand_ids_2d]
            print("Resolved Tiers per Candidate:\n", resolved_tiers)

            metrics = eval_step(model, mock_graph, movie_tiers, MOVIE_OFFSET, top_k=TOP_K)

        assert "loss" in metrics
        assert f"ndcg_{TOP_K}" in metrics
        assert f"ndcg_head_{TOP_K}" in metrics
        assert f"ndcg_torso_{TOP_K}" in metrics
        assert f"ndcg_tail_{TOP_K}" in metrics
        assert "composite_ndcg_" + str(TOP_K) in metrics

        for key, value in metrics.items():
            assert not jnp.isnan(value).any(), f"Metric '{key}' contains NaNs!"

        print(f"metrics={metrics}")

        self.assertTrue(metrics[f'ndcg_tail_{TOP_K}'] > 0)
        self.assertTrue(metrics[f'ndcg_torso_{TOP_K}'] > 0)
        self.assertTrue(metrics[f'ndcg_head_{TOP_K}'] > 0)
        print("✅ test_eval_step passed (with all tiers represented).")


if __name__ == "__main__":
    unittest.main()