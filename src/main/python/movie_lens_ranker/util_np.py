from typing import List, Tuple, Sequence

import numpy as np
import jraph
import jax.tree_util as tree

def optimized_batch_and_pad(batch: Sequence[jraph.GraphsTuple], max_nodes: int, max_edges: int,
        max_graphs: int) -> Tuple[jraph.GraphsTuple, int]:
    """
    Highly vectorized, single-pass replacement for jraph.batch followed by jraph.pad_with_graphs.
    Eliminates redundant memory allocations and Python loops.
    :param max_edges:
    :param max_graphs:
    :param max_nodes:
    :param batch: batch: list of jraph.GraphsTuple(
                nodes={
                    "ids": node_ids,
                    "label": node_labels,
                    "type": node_types,
                    "candidate_mask": candidate_mask,
                    "embeddings" : embeddings for node_ids
                },
                edges={"rating": edge_features},
                senders=senders,
                receivers=receivers,
                n_node=np.array([total_nodes]),
                n_edge=np.array([total_edges]),
                globals=None
            ))  where he node array lengths are = 1 + n_real_history + n_candidates, and
                the edge array lengths are = n_real_history + n_candidates.
    :return the padded super graph, the number of graphs in the input where the padded super graph is
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
    
    graphs = batch

    # Fast, single-pass concatenations of graph metadata
    n_node_arr = np.concatenate([g.n_node for g in graphs])
    n_edge_arr = np.concatenate([g.n_edge for g in graphs])
    
    total_nodes = np.sum(n_node_arr)
    total_edges = np.sum(n_edge_arr)
    total_graphs = len(graphs)
    
    # Calculate padding requirements
    pad_n_node = int(max_nodes - total_nodes)
    pad_n_edge = int(max_edges - total_edges)
    pad_n_graph = int(max_graphs - total_graphs)
    
    if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
        raise RuntimeError(
            f"Graph too large for padding. difference: "
            f"n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}"
        )
    
    # Create padded n_node and n_edge arrays
    # (1 dummy graph for padding, followed by empty graphs)
    pad_n_empty_graph = pad_n_graph - 1
    
    n_node_padded = np.concatenate([
        n_node_arr,
        np.array([pad_n_node], dtype=np.int32),
        np.zeros(pad_n_empty_graph, dtype=np.int32)
    ])
    
    n_edge_padded = np.concatenate([
        n_edge_arr,
        np.array([pad_n_edge], dtype=np.int32),
        np.zeros(pad_n_empty_graph, dtype=np.int32)
    ])
    
    # Vectorized Sender/Receiver offset calculation
    # Calculate offsets using the sum of nodes PER graph tuple to cleanly handle
    # GraphTuples that might already contain multiple graphs implicitly.
    nodes_per_tuple = np.array([np.sum(g.n_node) for g in graphs], dtype=np.int32)
    
    offsets = np.cumsum(np.concatenate([np.array([0], dtype=np.int32), nodes_per_tuple[:-1]]))
    
    edge_counts = [int(np.sum(g.n_edge)) for g in graphs]
    repeated_offsets = np.repeat(offsets, edge_counts)
    
    # Concatenate real edges, apply vector offset, then append padding edges
    # The padding edges MUST point to the first node of the padding graph (index `total_nodes`),
    # instead of index 0. If they point to 0, they connect to real data and corrupt node 0!
    senders_padded = np.concatenate([
        np.concatenate([g.senders for g in graphs]) + repeated_offsets,
        np.full(pad_n_edge, total_nodes, dtype=np.int32)
    ])
    receivers_padded = np.concatenate([
        np.concatenate([g.receivers for g in graphs]) + repeated_offsets,
        np.full(pad_n_edge, total_nodes, dtype=np.int32)
    ])
    
    # Fast Tree Map for feature padding (nodes, edges, globals)
    def pad_features(pad_size, *nests):
        batched_feats = np.concatenate(nests, axis=0)
        padding = np.zeros((pad_size,) + batched_feats.shape[1:], dtype=batched_feats.dtype)
        return np.concatenate([batched_feats, padding], axis=0)
    
    nodes_padded = tree.tree_map(lambda *args: pad_features(pad_n_node, *args),
        *[g.nodes for g in graphs])
    edges_padded = tree.tree_map(lambda *args: pad_features(pad_n_edge, *args),
        *[g.edges for g in graphs])
    globals_padded = tree.tree_map(
        lambda *args: pad_features(pad_n_graph, *args),
        *[g.globals for g in graphs])
    
    return jraph.GraphsTuple(
        n_node=n_node_padded,
        n_edge=n_edge_padded,
        nodes=nodes_padded,
        edges=edges_padded,
        globals=globals_padded,
        senders=senders_padded,
        receivers=receivers_padded
    ), total_graphs

def shuffle_and_slice(arr:np.ndarray, pad_value:int=-1, max_take=None):
    """
    Randomly shuffles valid elements to the front of each row.
    runtime complexity is O(n1*n2*log(n2)) where n1 = arr.shape[0] and n2 = arr.shape[1]
    :param arr: 2D array
    :param pad_value: values represeting "empty"
    :param max_take: the maximum number elements to take from each row.
    :return: a matrix of shape(n1, max_take) that might have pad_value at largest indicies in the
    rows if not enough non-pad values were available.
    """
    
    # Create a random weight matrix
    rand_weights = np.random.rand(*arr.shape)
    
    # Force -1s to the end by giving them an artificially high weight
    rand_weights[arr == pad_value] = 2.0
    
    # Sort indices along the rows
    sort_idx = np.argsort(rand_weights, axis=1)
    
    # Gather the elements in their new random order
    shuffled = np.take_along_axis(arr, sort_idx, axis=1)
    
    # Slice off the maximum requested
    if max_take is not None:
        return shuffled[:, :max_take]
    
    return shuffled

def push_invalid_right(arr:np.ndarray, pad_value:int=-1):
    """
    Pushes all pad_values to the far right side of the array
    :param arr: 1 2D array
    :param pad_value: the value to shoft to ends of rows if found.
    :return: matrix in which the elements in each row have been shifted to the end of the
     array such that the pad_values are at highest indices. The order of non-pad_values is maintained.
    """
    valid_mask = (arr != -1)
    
    # ~valid_mask makes valid items 0 (False) and invalid items 1 (True).
    # argsort puts 0s before 1s. kind='stable' preserves the order of the valid items.
    sort_idx = np.argsort(~valid_mask, axis=1, kind='stable')
    return np.take_along_axis(arr, sort_idx, axis=1)