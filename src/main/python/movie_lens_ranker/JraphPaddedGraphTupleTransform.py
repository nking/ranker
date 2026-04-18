from typing import List, Dict

#NOTE: this runs on the CPU in the grain dataloader before anything
# is handed off to the GPU.
# *** the file loading the dataloader should use:
#    os.environ["JAX_PLATFORMS"] = "cpu"
# to prevent jax from tryin to use the GPU

import grain.python as pgrain
import jraph

class JraphPaddedGraphTupleTransform(pgrain.MapTransform):
    
    def __init__(self,  batch_size: int, max_history: int, num_candidates: int):
        
        self.jax_graph_comp_dict = self.calc_number_jax_graph_components(batch_size,
            max_history, num_candidates)
    
    def calc_number_jax_graph_components(self, batch_size: int, max_history: int,
            num_candidates: int) -> Dict[str, int]:
        
        # 40->50, #123->200, #1234->2000, #12345->20000
        def next_64(x) -> int:
            return 64 * (1 + int(x // 64))
        
        max_nodes = next_64(batch_size * (1 + max_history + num_candidates))
        max_edges = next_64(batch_size * (max_history + num_candidates))
        max_graphs = batch_size + 1
        return {'max_nodes': max_nodes, 'max_edges': max_edges,
            'max_graphs': max_graphs}
 
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