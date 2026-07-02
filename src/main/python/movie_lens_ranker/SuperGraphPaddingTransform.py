from typing import Dict, List

import jraph
import grain.python as pgrain
import numpy as np

from movie_lens_ranker.util import calc_number_jax_graph_components
from movie_lens_ranker.util_np import optimized_batch_and_pad
from movie_lens_ranker.util_numba import build_graph_arrays

class SuperGraphPaddingTransform(pgrain.MapTransform):
    
    def __init__(self, batch_size:int, max_history:int, num_candidates:int,
        n_local_devices:int):
        self.jax_graph_comp_dict = calc_number_jax_graph_components(
            batch_size, max_history, num_candidates,
            n_local_devices=n_local_devices)
    
    def map(self, batch:List[jraph.GraphsTuple]) -> jraph.GraphsTuple:
        """
       given a list of jraph tuples in which one graph may have different length arrays than
        another graph, forms a padded super graph.
       :param batch: list of jraph.GraphsTuple(
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
            ))  where he node array lengths are = 1 + n_real_history + n_candidates, and
                the edge array lengths are = n_real_history + n_candidates.
        
        :returns: a padded super graph
            jraph.GraphsTuple(
                n_node=n_node_padded,
                n_edge=n_edge_padded,
                nodes=nodes_padded,
                edges=edges_padded,
                globals=globals_padded,
                senders=senders_padded,
                receivers=receivers_padded
            )
            where dimensions are max_nodes, max_edges, and max_graphs
        """
        padded_super_graph, _ = optimized_batch_and_pad(
            batch=batch,
            max_nodes=self.jax_graph_comp_dict['max_nodes'],
            max_edges=self.jax_graph_comp_dict['max_edges'],
            max_graphs=self.jax_graph_comp_dict['max_graphs'],
        )
        return padded_super_graph