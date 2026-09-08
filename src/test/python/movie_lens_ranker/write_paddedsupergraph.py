import json

from array_record.python import array_record_module

import jraph

#pip install safetensors==0.8.0
import numpy as np
from safetensors.numpy import save_file, load_file

from movie_lens_ranker.CandidateIdTransform import CandidateIdTransform
from movie_lens_ranker.RatingsHistoryTransform import RatingsHistoryLookupTransform
from movie_lens_ranker.SparseLocalSubgraphTransform import SparseLocalSubgraphTransform
from movie_lens_ranker.SuperGraphPaddingTransform import SuperGraphPaddingTransform
from movie_lens_ranker.UserHistory import UserHistory
from movie_lens_ranker.util import read_user_movie_embeddings

import argparse

def test_inference_inputs(
        output_path: str,
        in_batch:list,
        ratings_uri:str,
        candidate_ids,
        user_embeddings_uri: str,
        movie_embeddings_uri: str,
        max_history:int=4,
        batch_size:int=2,
        num_candidates:int=5,
        jax_n_local_devices:int=1):
    """
    making a graph for inference to compare to rust code graph
    ratings_uri:
        e.g.
            ratings_uri_dict = get_train_val_test_liked_uris(DataSize.TINY, use_gcs_uri=False)
            ratings_uri = ratings_uri_dict['train_liked']
    candidate_ids:
        e.g.
            candidate_ids = np.array([
                [6610, 6252, 9083, 6564, 6584],
                [9477, 6941, 6948, 8475, 6356]
            ])
    """

    print(f'batch={in_batch}, num_candidates={num_candidates}', flush=True)

    #lengths are num_candidates - 1 each.  leaving room for the positive, target in the batch
    if len(candidate_ids) > num_candidates:
        candidate_ids = [candidate_ids[i : i + num_candidates] for i in range(0, len(candidate_ids), num_candidates)]
    candidate_ids = np.array(candidate_ids)

    user_movie_embeddings = read_user_movie_embeddings(
        user_embeddings_uri=user_embeddings_uri,
        movie_embeddings_uri=movie_embeddings_uri)

    embed_len = user_movie_embeddings.shape[1]

    user_history = UserHistory(ratings_uri_list=[ratings_uri], max_history=2048)

    tr1 = RatingsHistoryLookupTransform(
        history_lookup=user_history,
        max_history=max_history)
    tr2 = CandidateIdTransform(num_candidates=num_candidates)
    tr3 = SparseLocalSubgraphTransform(user_movie_embeddings=user_movie_embeddings)
    tr4 = SuperGraphPaddingTransform(batch_size=batch_size,
                                     max_history=max_history, num_candidates=num_candidates,
                                     n_local_devices=jax_n_local_devices)

    r1 = tr1.map(in_batch)
    r2 = tr2.map(r1, candidate_ids)
    r3 = tr3.map(r2)
    r4 : jraph.GraphsTuple = tr4.map(r3)

    tensors = {
        "n_node": np.asarray(r4.n_node, dtype=np.int32),
        "n_edge": np.asarray(r4.n_edge, dtype=np.int32),
        "senders": np.asarray(r4.senders, dtype=np.int32),
        "receivers": np.asarray(r4.receivers, dtype=np.int32),
        "edge_features": np.asarray(r4.edges['rating'], dtype=np.int32),
        "node_ids": np.asarray(r4.nodes["ids"], dtype=np.int32),
        "node_label": np.asarray(r4.nodes["label"], dtype=np.int32),
        "node_type": np.asarray(r4.nodes["type"], dtype=np.int32),
        "candidate_mask": np.asarray(r4.nodes["candidate_mask"], dtype=np.bool_),
        "embeddings": np.asarray(r4.nodes["embeddings"], dtype=np.float32)
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
        '--in_batch',
        help="the string serialzied the batch as rows of [user_id, movie_id, rating, timestamp]",
        type=str,
    )
    parser.add_argument(
        '--ratings_uri',
        help="array_record file of rows of user_id, movie_id, rating, timestamp",
        type=str,
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
        '--candidate_ids',
        help="string encoded list of candidate ids.  e.g. json.dumps([1,2])",
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
        '--jax_n_local_devices',
        help="number of local devices that the graph will be partitioned over",
        type=int,
        default=1
    )

    args, _ = parser.parse_known_args()
    args_dict = vars(args)

    candidate_ids = json.loads(args_dict['candidate_ids'])

    in_batch = json.loads(args_dict['in_batch'])
    print(f'in_batch={in_batch}')

    test_inference_inputs(
        output_path=args_dict['output_path'],
        ratings_uri=args_dict['ratings_uri'],
        in_batch=in_batch,
        candidate_ids=candidate_ids,
        user_embeddings_uri=args_dict['user_embeddings_uri'],
        movie_embeddings_uri=args_dict['movie_embeddings_uri'],
        max_history=args_dict['max_history'],
        batch_size=args_dict['batch_size'],
        num_candidates=args_dict['num_candidates'],
        jax_n_local_devices=args_dict['jax_n_local_devices'])

    assert_can_read(args_dict['output_path'])
