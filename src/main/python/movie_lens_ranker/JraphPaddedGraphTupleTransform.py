from typing import List

import grain.python as pgrain
import jraph

from movie_lens_ranker.util import calc_number_jax_graph_components


class JraphPaddedGraphTupleTransform(pgrain.MapTransform):
    def __init__(self,  batch_size: int, max_history: int, num_candidates: int):
        
        self.jax_graph_comp_dict = calc_number_jax_graph_components(batch_size,
            max_history, num_candidates)
    
    def map(self, batch: List[jraph.GraphsTuple]) -> jraph.GraphsTuple:
        
        super_graph = jraph.batch(batch)
        # n_graph is usually batch_size + 1 (the padding graph)
        padded_super_graph = jraph.pad_with_graphs(
            super_graph,
            n_node=self.jax_graph_comp_dict['max_nodes'],
            n_edge=self.jax_graph_comp_dict['max_edges'],
            n_graph=self.jax_graph_comp_dict['max_graphs'],
        )
        return padded_super_graph