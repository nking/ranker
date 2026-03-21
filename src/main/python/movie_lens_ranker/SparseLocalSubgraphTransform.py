from typing import Dict, Tuple, Union, List, Set

import jraph
import jax.numpy as jnp
import grain.python as pgrain
import numpy as np

class SparseLocalSubgraphTransform(pgrain.MapTransform):
    
    def __init__(self):
        pass
    
    def map(self, record:Dict[str, Union[int, np.ndarray]]) -> jraph.GraphsTuple:
        """
        create a local subgraph for the record in which all indexes are relative to just these local
        variables, not the global embedding indices.
        The returned Graph Topology:
        Node 0: The User.
        Nodes 1 to H: History Movies (H = max_history).
        Nodes H+1 to H+C: Candidate Movies (C = num_candidates)
        Edges are the user ratings for the movie.
        :param record : dictionary containing
            'user_id':int
            'movie_id':int,
            'rating': int,
            'timestamp': int,
            "history_movie_ids": np.ndarray,
            "history_ratings": np.ndarray,
            "history_length": int
            "candidate_ids": np.ndarray,
            "labels": np.ndarray
        :return: a sparsely populated jraph.GraphsTuple representation of the local subgraph for
        the train user_id.  Note that this is not the padded version to give to the model being trained.
        """
        
        # NOTE: method returns a sparse GraphsTuple, ignoring the padded variables in record, because
        # the resulting datastructure is not seen by the GPU.
        
        n_real_history = record["history_length"]
        n_candidates = len(record["candidate_ids"])
        #max_hist = len(record["history_movie_ids"])
        
        # Define Node Counts
        total_nodes = 1 + n_real_history + n_candidates
        
        # Define Senders and Receivers (Edges)
        # Strategy: Star Graph. User connects TO History and Candidates.
        # Edge: User (0) -> History (1..H)
        # then
        # Edge: User (0) -> Candidates (H+1..H+C)
        senders = [0] * (n_real_history + n_candidates)
        receivers = [i+1 for i in range(n_real_history + n_candidates)]
        edge_features = [record["history_ratings"][i] for i in range(n_real_history)]
        edge_features.extend([0] * n_candidates)
        
        # 3. Construct the GraphsTuple
        return jraph.GraphsTuple(
            nodes={
                "ids": np.concatenate([
                    [record["user_id"]],
                    record["history_movie_ids"][:n_real_history],
                    record["candidate_ids"]
                ]),
                "type": np.array(
                    [0] + [1] * n_real_history + [2] * n_candidates)
                # 0=User, 1=Hist, 2=Cand
            },
            edges={"rating": jnp.array(edge_features)},
            senders=jnp.array(senders),
            receivers=jnp.array(receivers),
            n_node=jnp.array([total_nodes]),
            n_edge=jnp.array([len(edge_features)]),
            globals=None
        )