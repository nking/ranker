import json
import os
import jax
import jraph
from flax import nnx
from orbax import export
from orbax.export import ServingConfig
from sqlalchemy.engine import create

from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.util import calc_number_jax_graph_components

import tensorflow as tf

def create_serving_signature(max_nodes:int, max_edges:int, max_graphs:int, embed_len:int, signature_name:str) -> ServingConfig :
    serving_config = export.ServingConfig(
        signature_key=signature_name,
        input_signature=[
            {
                # Nodes attributes
                "node_candidate_mask": tf.TensorSpec(shape=(max_nodes,), dtype=tf.bool, name="node_candidate_mask"),
                "node_ids": tf.TensorSpec(shape=(max_nodes,), dtype=tf.int32, name="node_ids"),
                "node_label": tf.TensorSpec(shape=(max_nodes,), dtype=tf.int32, name="node_label"),
                "node_type": tf.TensorSpec(shape=(max_nodes,), dtype=tf.int32, name="node_type"),

                "node_embeddings" : tf.TensorSpec(shape=(max_nodes, embed_len), dtype=tf.float32, name="node_embeddings"),

                # Edges attributes
                "edge_features": tf.TensorSpec(shape=(max_edges,), dtype=tf.int32, name="edge_features"),

                # Core Graph Topology
                "receivers": tf.TensorSpec(shape=(max_edges,), dtype=tf.int32,  name="receivers"),
                "senders": tf.TensorSpec(shape=(max_edges,), dtype=tf.int32,  name="senders"),

                # Metadata
                "n_node": tf.TensorSpec(shape=(max_graphs,), dtype=tf.int32, name="n_node"),
                "n_edge": tf.TensorSpec(shape=(max_graphs,), dtype=tf.int32,  name="n_edge"),
            }
        ])
    return serving_config

def save_metadata(output_file_uri:str, batch_size:int, max_history:int, num_candidates:int,
                  max_nodes:int, max_edges:int, max_graphs:int, embed_len:int, signature_name:str):
    metadata = {
        "signature_name" : signature_name,
        "batch_size": batch_size,
        "max_history": max_history,
        "num_candidates": num_candidates,
        "max_nodes" : max_nodes,
        "max_edges" : max_edges,
        "max_graphs" : max_graphs,
        "embed_len" : embed_len
    }
    with open(output_file_uri, "w") as f:
        json.dump(metadata, f)

def export_models(trained_model: GraphRanker, batch_size:int, max_history:int,
    num_candidates:int, embed_len:int,  output_savedmodel_dir_uri:str, n_local_devices:int=1):
    """
    export the model to TF SavedModel format along with a method to apply the model on the data.
    makes an export with a signature for  single inference mode and a batch inference mode.
    :param embed_len:
    :param output_savedmodel_dir_uri: uri for the directory to save the model to.  Note that the
        version number should already be included in the uri as the last part of the path.
           e.g.  /absolute/path/to/your/model_export_dir/1
    :exception
    :param trained_model:
    :param batch_size: batch_size used for model training
    :param max_history: max_history used for model training
    :param num_candidates: num_candidates used for model training
    :return:
    """

    jax_graph_comp_dict_batch = calc_number_jax_graph_components(batch_size,
        max_history, num_candidates, n_local_devices=n_local_devices)

    jax_graph_comp_dict_single = calc_number_jax_graph_components(1,
        max_history, num_candidates, n_local_devices=n_local_devices)

    print(f'jax_graph_comp_dict_single={jax_graph_comp_dict_single}', flush=True)
    print(f'jax_graph_comp_dict_batch={jax_graph_comp_dict_batch}', flush=True)

    single_serving_config = create_serving_signature(
        max_nodes=jax_graph_comp_dict_single['max_nodes'],
        max_edges=jax_graph_comp_dict_single['max_edges'],
        max_graphs=jax_graph_comp_dict_single['max_graphs'],
        embed_len=embed_len, signature_name="serving_default")

    batch_serving_config = create_serving_signature(
        max_nodes=jax_graph_comp_dict_batch['max_nodes'],
        max_edges=jax_graph_comp_dict_batch['max_edges'],
        max_graphs=jax_graph_comp_dict_batch['max_graphs'],
        embed_len=embed_len, signature_name="serving_batch")

    # Split the NNX model into architecture (graphdef), weights (params), and everything else (rest).
    # The '...' catches the key<fry> RNG states so they don't cause a non-exhaustive filter error.
    graphdef, params, rest = nnx.split(trained_model, nnx.Param, ...)

    # Define the pure apply function inside the scope so it has access to `graphdef`
    def pure_apply_fn(params, inputs):
        # Reconstruct the model using the static blueprint + the weights
        model = nnx.merge(graphdef, params, rest)
        model.eval()

        graph_batch = jraph.GraphsTuple(
            nodes={
                'candidate_mask': inputs["node_candidate_mask"],
                'ids': inputs["node_ids"],
                'label': inputs["node_label"],
                'type': inputs["node_type"],
                "embeddings" : inputs["node_embeddings"]
            },
            edges={'rating': inputs["edge_features"]},
            receivers=inputs["receivers"],
            senders=inputs["senders"],
            globals=None,
            n_node=inputs["n_node"],
            n_edge=inputs["n_edge"]
        )

        return model(graph_batch)[:num_candidates]

    # Pass *only* the params to JaxModule, not the whole trained_model
    jax_module = export.JaxModule(
        params=params,
        apply_fn=pure_apply_fn,
        trainable=False
    )

    export_manager = export.ExportManager(jax_module, [single_serving_config, batch_serving_config])
    export_manager.save(output_savedmodel_dir_uri)

    save_metadata(output_file_uri=os.path.join(output_savedmodel_dir_uri, "metadata_single.json"),
        batch_size=1,  max_history=max_history, num_candidates=num_candidates,
        max_nodes=jax_graph_comp_dict_single['max_nodes'],
        max_edges=jax_graph_comp_dict_single['max_edges'],
        max_graphs=jax_graph_comp_dict_single['max_graphs'],
        embed_len=embed_len, signature_name="serving_default")

    save_metadata(output_file_uri=os.path.join(output_savedmodel_dir_uri, "metadata_batch.json"),
        batch_size=batch_size, max_history=max_history, num_candidates=num_candidates,
        max_nodes=jax_graph_comp_dict_batch['max_nodes'],
        max_edges=jax_graph_comp_dict_batch['max_edges'],
        max_graphs=jax_graph_comp_dict_batch['max_graphs'],
        embed_len=embed_len, signature_name="serving_batch")

    print(f"saved model and metadata to {output_savedmodel_dir_uri}")


