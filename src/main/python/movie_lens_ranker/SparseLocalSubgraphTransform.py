from typing import Dict, List

import jraph
import grain.python as pgrain
import numpy as np

from movie_lens_ranker.util_numba import build_graph_arrays

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
        :param batch : dictionary of np.ndarrays:
            'user_id' has shape (batch_size,)
            'movie_id'  has shape (batch_size,)
            'rating'  has shape (batch_size,)
            'timestamp'  has shape (batch_size,)
            "history_movie_ids"  has shape (batch_size, max_history)
            "history_ratings"  has shape (batch_size, max_history)
            "history_length"  has shape (batch_size,)
            "candidate_ids"  has shape (batch_size, num_candidates)
            "labels"  has shape (batch_size, num_candidates)
                Note that candidate_ids is guaranteed to not have padding values, they're all real movie_ids'.
                labels are all 0.0 with exception of being 1.0 at the index where candidate_ids has the target positive movie_id.
        :return: a list of a sparsely populated jraph.GraphsTuple representation of the local subgraph for
        the train user_id.  Note that this is not the padded version to give to the model being trained.
        The node array lengths are = 1 + n_real_history + n_candidates.
        The edge array lengths are = n_real_history + n_candidates.
        """
        
        #form the list of jraph.GraphsTuple
        
        #note that the max value possible in "history_length" is the max_history given to the UserHistory getter
        #   in the RatingsHistoryLookupTransform
        
        # NOTE: method returns a sparse GraphsTuple, ignoring the padded variables in record, because
        # the resulting datastructure is not seen by the GPU.
        results = []
        batch_size = len(batch['user_id'])
        
        for i in range(batch_size):
            u_id = batch["user_id"][i]
            n_hist = batch["history_length"][i]
            c_ids = batch["candidate_ids"][i]
            h_rats = batch["history_ratings"][i]
            h_m_ids = batch["history_movie_ids"][i]
            lbls = batch["labels"][i]
            
            # Run the Numba kernel
            (senders, receivers, edge_features, node_ids,
                node_labels, node_types, candidate_mask,
                total_nodes, total_edges) = build_graph_arrays(u_id, n_hist, c_ids, h_rats, h_m_ids, lbls)
            
            # Construct the GraphsTuple (Python side)
            results.append(jraph.GraphsTuple(
                nodes={
                    "ids": node_ids,
                    "label": node_labels,
                    "type": node_types,
                    "candidate_mask": candidate_mask
                },
                edges={"rating": edge_features},
                senders=senders,
                receivers=receivers,
                n_node=np.array([total_nodes]),
                n_edge=np.array([total_edges]),
                globals=None
            ))
        
        return results