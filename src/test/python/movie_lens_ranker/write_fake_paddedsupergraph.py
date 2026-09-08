import json
from typing import Tuple

from array_record.python import array_record_module

#pip install safetensors==0.8.0
import numpy as np
from safetensors.numpy import save_file, load_file

import argparse

from movie_lens_ranker.train import create_fake_jagged_batch
from movie_lens_ranker.util import calc_number_jax_graph_components
from movie_lens_ranker.util_np import optimized_batch_and_pad


def write_fake_graph_safetensor(
        output_path: str,
        user_embeddings_uri: str,
        movie_embeddings_uri: str,
        max_history:int=4,
        batch_size:int=2,
        num_candidates:int=5,
        user_id_range:Tuple[int, int]=(1, 6040),
        movie_id_range:Tuple[int, int]=(6041, 6041 + 3883),
        jax_n_local_devices:int=1
    ):
    """
    make a graph to test that rust graph building is the same
    """

    fake_batch = create_fake_jagged_batch(batch_size=batch_size,
                                          max_history=max_history,
                                          num_candidates=num_candidates,
                                          user_id_range=user_id_range,
                                          movie_id_range=movie_id_range,
                                          movie_embeddings_uri = movie_embeddings_uri,
                                          user_embeddings_uri = user_embeddings_uri)

    jax_graph_comp_dict = calc_number_jax_graph_components(
        batch_size=batch_size, max_history=max_history,
        num_candidates=num_candidates, 
        n_local_devices=jax_n_local_devices)

    fake_padded_graph, _ = optimized_batch_and_pad(
        batch=fake_batch,
        max_nodes=jax_graph_comp_dict['max_nodes'],
        max_edges=jax_graph_comp_dict['max_edges'],
        max_graphs=jax_graph_comp_dict['max_graphs'],
    )

    tensors = {
        "n_node": np.asarray(fake_padded_graph.n_node, dtype=np.int32),
        "n_edge": np.asarray(fake_padded_graph.n_edge, dtype=np.int32),
        "senders": np.asarray(fake_padded_graph.senders, dtype=np.int32),
        "receivers": np.asarray(fake_padded_graph.receivers, dtype=np.int32),
        "edge_features": np.asarray(fake_padded_graph.edges['rating'], dtype=np.int32),
        "node_ids": np.asarray(fake_padded_graph.nodes["ids"], dtype=np.int32),
        "node_label": np.asarray(fake_padded_graph.nodes["label"], dtype=np.int32),
        "node_type": np.asarray(fake_padded_graph.nodes["type"], dtype=np.int32),
        "candidate_mask": np.asarray(fake_padded_graph.nodes["candidate_mask"], dtype=np.bool_),
        "embeddings": np.asarray(fake_padded_graph.nodes["embeddings"], dtype=np.float32)
    }

    save_file(tensors, output_path)

    print(f'wrote to {output_path}')

def assert_can_read(output_path: str):
    # Load all saved tensors back into a dictionary of numpy arrays
    tensors = load_file(output_path)

    expected_keys = [
        "n_node",
        "n_edge",
        "senders",
        "receivers",
        "edge_features",
        "node_ids",
        "node_label",
        "node_type",
        "candidate_mask",
        "embeddings",
    ]

    for key in expected_keys:
        assert key in tensors, f"Key '{key}' missing from loaded safetensors file."
        array = tensors[key]
        assert isinstance(array, np.ndarray), f"Expected np.ndarray for key '{key}'"
        print(f"Verified {key}: shape={array.shape}, dtype={array.dtype}")

    print(f"\nSuccessfully verified all tensors in {output_path}", flush=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="inference_inputs")
    parser.add_argument(
        '--output_path',
        help="path to write the output file to",
        type=str,
    )
    parser.add_argument(
        '--user_id_range',
        help="string encoded list of candidate ids.  e.g. json.dumps((1, 6040))",
        type=str,
    )
    parser.add_argument(
        '--movie_id_range',
        help="string encoded list of candidate ids.  e.g. json.dumps((6041, 6041 + 3883))",
        type=str,
    )
    parser.add_argument(
        '--max_history',
        help="maximum user history to include in final graph",
        type=int,
    )
    parser.add_argument(
        '--num_candidates',
        help="number of candidate movie_ids to include in final graph",
        type=int,
    )
    parser.add_argument(
        '--batch_size',
        help="number of user graphs to put into a super padded graph",
        type=int,
    )
    parser.add_argument(
        '--user_embeddings_uri',
        help="path to the user embeddings array_record",
        type=str,
    )
    parser.add_argument(
        '--movie_embeddings_uri',
        help="path to the movie embeddings array_record",
        type=str,
    )
    parser.add_argument(
        '--jax_n_local_devices',
        help="number of local devices that the graph will be partitioned over",
        type=int,
        default=1
    )

    args, _ = parser.parse_known_args()
    args_dict = vars(args)

    user_id_range = json.loads(args_dict['user_id_range'])
    movie_id_range = json.loads(args_dict['movie_id_range'])

    write_fake_graph_safetensor(
        output_path=args_dict['output_path'],
        user_embeddings_uri=args_dict['user_embeddings_uri'],
        movie_embeddings_uri=args_dict['movie_embeddings_uri'],
        max_history=args_dict['max_history'],
        batch_size=args_dict['batch_size'],
        num_candidates=args_dict['num_candidates'],
        user_id_range=user_id_range,
        movie_id_range=movie_id_range,
        jax_n_local_devices=args_dict['jax_n_local_devices']
    )

    assert_can_read(args_dict['output_path'])
