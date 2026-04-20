from typing import Dict, List

import jraph
import grain.python as pgrain
import numpy as np

class SparseLocalSubgraphTransform(pgrain.MapTransform):
    
    def __init__(self):
        pass
    
    def map(self, batch:Dict[str, np.ndarray]) -> List[jraph.GraphsTuple]:
        """
        create a local subgraph for the record in which all indexes are relative to just these local
        variables, not the global embedding indices.
        The returned Graph Topology:
        Node 0: The User.
        Nodes 1 to H: History Movies (H = max_history).
        Nodes H+1 to H+C: Candidate Movies (C = num_candidates)
        Edges are the user ratings for the movie.
        :param batch : dictionary containing ndarrays with follong keys
            'user_id',
            'movie_id',
            'rating',
            'timestamp',
            "history_movie_ids",
            "history_ratings",
            "history_length",
            "candidate_ids",
            "labels"
        :return: a list of a sparsely populated jraph.GraphsTuple representation of the local subgraph for
        the train user_id.  Note that this is not the padded version to give to the model being trained.
        """
        
        #form the list of jraph.GraphsTuple
        
        #note that the max value possible in "history_length" is the max_history given to the UserHistory getter
        #   in the RatingsHistoryLookupTransform
        
        # NOTE: method returns a sparse GraphsTuple, ignoring the padded variables in record, because
        # the resulting datastructure is not seen by the GPU.
        results = []
        for i in range(len(batch['user_id'])):
            n_real_history = batch["history_length"][i]
            n_candidates = len(batch["candidate_ids"][i])
            total_nodes = 1 + n_real_history + n_candidates
            
            # Define Senders and Receivers (Edges)
            # Strategy: Star Graph. User connects TO History and Candidates.
            # Edge: User (0) -> History (1..H)
            # then
            # Edge: User (0) -> Candidates (H+1..H+C)
            #these are all length:  n_real_history + n_candidates
            
            edge_shape = (n_real_history + n_candidates,)
            senders = np.full(shape=edge_shape, fill_value=-1)
            receivers = np.full(shape=edge_shape, fill_value=-1)
            edge_features = np.full(shape=edge_shape, fill_value=-1)
            
            # History -> User (Inward)
            # Senders: [1, 2, ... H], Receiver: [0, 0, ... 0]
            senders[:n_real_history] = np.arange(1, n_real_history + 1)
            receivers[:n_real_history] = np.zeros((n_real_history,))
            edge_features[:n_real_history] = batch["history_ratings"][i][:n_real_history]
            
            
            # User -> Candidates (Outward)
            # Sender: [0, 0, ... 0], Receivers: [H+1, ... H+C]
            senders[n_real_history:n_real_history+n_candidates] = np.zeros((n_candidates,))
            receivers[n_real_history:n_real_history+n_candidates] = np.arange(1+n_real_history, 1+n_real_history+n_candidates)
            edge_features[n_real_history:n_real_history+n_candidates] = np.zeros((n_candidates,))
            
            #NOTE: if have pre-processed ratings to be scaled to 0 to 1,
            # then here in the n_candidates loop, edge_features.append(-1) instead of 1 to 5
            # currently, the datasets are ratings from 1
            
            node_ids = np.concatenate([
                np.array([batch["user_id"][i]], dtype=int),
                np.array(batch["history_movie_ids"][i][:n_real_history], dtype=int),
                np.array(batch["candidate_ids"][i], dtype=int)
            ], dtype=int)
            
            node_labels = np.concatenate([
                np.array([0.0]),  # User (Type 0)
                np.zeros(n_real_history),  # History (Type 1)
                batch["labels"][i],#numpy array
                # Candidates (Type 2) - should be size n_candidates
            ], dtype=float)
            
            # Node attributes must match total_nodes
            assert len(node_ids) == len(node_labels) == total_nodes
            # Edge attributes must match the number of connections
            assert len(senders) == len(receivers) == len(edge_features)
            
            # Create a mask for the nodes
            # User (False), History (False), Candidates (True)
            # This identifies which nodes are candidates in the local graph
            node_is_candidate = np.array(
                [False] + [False] * n_real_history + [True] * n_candidates)
            
            #using NUMPY arrays because we are still on the CPU in grain dataloader
            
            #  Construct the GraphsTuple
            results.append(jraph.GraphsTuple(
                # modes arrays are length 1 + n_real_history + self.num_candidates
                nodes={
                    "ids": node_ids,
                    "label": node_labels,
                    "type": np.array(
                        [0] + [1] * n_real_history + [2] * n_candidates),
                    "candidate_mask": node_is_candidate
                },
                # edges, senders, receivers are length  n_real_history + self.num_candidates
                edges={"rating": edge_features},
                senders=senders,
                receivers=receivers,
                n_node=np.array([total_nodes]),
                n_edge=np.array([len(senders)]),
                globals=None
            ))
        return results