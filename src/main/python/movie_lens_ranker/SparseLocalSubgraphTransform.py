from typing import Dict, Union, List

import jraph
import jax.numpy as jnp
import grain.python as pgrain
import numpy as np

class SparseLocalSubgraphTransform(pgrain.MapTransform):
    
    def __init__(self):
        pass
    
    def map(self, batch:List[Dict[str, Union[int, np.ndarray]]]) -> List[jraph.GraphsTuple]:
        """
        create a local subgraph for the record in which all indexes are relative to just these local
        variables, not the global embedding indices.
        The returned Graph Topology:
        Node 0: The User.
        Nodes 1 to H: History Movies (H = max_history).
        Nodes H+1 to H+C: Candidate Movies (C = num_candidates)
        Edges are the user ratings for the movie.
        :param batch : list of dictionaries containing
            'user_id':int
            'movie_id':int,
            'rating': int,
            'timestamp': int,
            "history_movie_ids": np.ndarray,
            "history_ratings": np.ndarray,
            "history_length": int
            "candidate_ids": np.ndarray,
            "labels": np.ndarray
        :return: a list of a sparsely populated jraph.GraphsTuple representation of the local subgraph for
        the train user_id.  Note that this is not the padded version to give to the model being trained.
        """
        
        # NOTE: method returns a sparse GraphsTuple, ignoring the padded variables in record, because
        # the resulting datastructure is not seen by the GPU.
        results = []
        for record in batch:
            n_real_history = record["history_length"]
            n_candidates = len(record["candidate_ids"])
            total_nodes = 1 + n_real_history + n_candidates
            
            # Define Senders and Receivers (Edges)
            # Strategy: Star Graph. User connects TO History and Candidates.
            # Edge: User (0) -> History (1..H)
            # then
            # Edge: User (0) -> Candidates (H+1..H+C)
            senders = []
            receivers = []
            edge_features = []
            
            # History -> User (Inward)
            # Senders: [1, 2, ... H], Receiver: [0, 0, ... 0]
            for i in range(n_real_history):
                senders.append(i + 1)  # History nodes
                receivers.append(0)  # User node
                edge_features.append(record["history_ratings"][i])
            
            # User -> Candidates (Outward)
            # Sender: [0, 0, ... 0], Receivers: [H+1, ... H+C]
            for i in range(n_candidates):
                senders.append(0)  # User node
                receivers.append(1 + n_real_history + i)  # Candidate nodes
                edge_features.append(0)  # No rating for candidates
            
            #TODO: if have pre-processed ratings to be scaled to 0 to 1,
            # then here in the n_candidates loop, edge_features.append(-1) instead of 0
            
            node_ids = jnp.concatenate([
                jnp.array([record["user_id"]], dtype=int),
                jnp.array(record["history_movie_ids"][:n_real_history], dtype=int),
                jnp.array(record["candidate_ids"], dtype=int)
            ], dtype=int)
            
            node_labels = jnp.concatenate([
                jnp.array([0.0]),  # User (Type 0)
                jnp.zeros(n_real_history),  # History (Type 1)
                record["labels"],#numpy array
                # Candidates (Type 2) - should be size n_candidates
            ], dtype=float)
            
            # Node attributes must match total_nodes
            assert len(node_ids) == len(node_labels) == total_nodes
            # Edge attributes must match the number of connections
            assert len(senders) == len(receivers) == len(edge_features)
            
            # Create a mask for the nodes
            # User (False), History (False), Candidates (True)
            # This identifies which nodes are candidates in the local graph
            node_is_candidate = jnp.array(
                [False] + [False] * n_real_history + [True] * n_candidates)
            
            #  Construct the GraphsTuple
            results.append(jraph.GraphsTuple(
                nodes={
                    "ids": node_ids,
                    "label": node_labels,
                    "type": jnp.array(
                        [0] + [1] * n_real_history + [2] * n_candidates),
                    "candidate_mask": node_is_candidate
                },
                edges={"rating": jnp.array(edge_features)},
                senders=jnp.array(senders),
                receivers=jnp.array(receivers),
                n_node=jnp.array([total_nodes]),
                n_edge=jnp.array([len(senders)]),
                globals=None
            ))
        return results