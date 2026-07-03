import json
import os
import jax
import jraph
from flax import nnx
from orbax import export
from orbax.export import ServingConfig

from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.util import calc_number_jax_graph_components

import tensorflow as tf

def get_serving_signature(MAX_NODES:int, MAX_EDGES:int, MAX_GRAPHS:int, signature_name:str) -> ServingConfig :
    serving_config = export.ServingConfig(
        signature_key=signature_name,
        input_signature=[
            {
                # Nodes attributes
                "node_candidate_mask": tf.TensorSpec(shape=(MAX_NODES,),
                    dtype=tf.bool, name="node_candidate_mask"),
                "node_ids": tf.TensorSpec(shape=(MAX_NODES,), dtype=tf.int32,
                                          name="node_ids"),
                "node_label": tf.TensorSpec(shape=(MAX_NODES,),
                                            dtype=tf.int32, name="node_label"),
                "node_type": tf.TensorSpec(shape=(MAX_NODES,), dtype=tf.int32,
                                           name="node_type"),

                # Edges attributes
                "edge_features": tf.TensorSpec(shape=(MAX_EDGES,),
                                             dtype=tf.int32, name="edge_rating"),

                # Core Graph Topology
                "receivers": tf.TensorSpec(shape=(MAX_EDGES,), dtype=tf.int32,
                                           name="receivers"),
                "senders": tf.TensorSpec(shape=(MAX_EDGES,), dtype=tf.int32,
                                         name="senders"),

                # Metadata
                "n_node": tf.TensorSpec(shape=(MAX_GRAPHS,), dtype=tf.int32,
                                        name="n_node"),
                "n_edge": tf.TensorSpec(shape=(MAX_GRAPHS,), dtype=tf.int32,
                                        name="n_edge"),
            }
        ])
    return serving_config

def pure_apply_fn(model : GraphRanker, inputs):
    # This recombines your architecture blueprint with the weights dynamically in RAM

    # Reconstruct your exact library GraphsTuple inside the pure function boundaries
    graph_batch = jraph.GraphsTuple(
        nodes={
            'candidate_mask': inputs["node_candidate_mask"],
            'ids': inputs["node_ids"],
            'label': inputs["node_label"],
            'type': inputs["node_type"]
        },
        edges={'rating': inputs["edge_features"]},
        receivers=inputs["receivers"],
        senders=inputs["senders"],
        globals=None,
        n_node=inputs["n_node"],
        n_edge=inputs["n_edge"]
    )

    # Call your GraphRanker's __call__ method natively
    return model(graph_batch)

def save_metadata(output_file_uri:str, batch_size:int, max_history:int, num_candidates:int,
                  MAX_NODES:int, MAX_EDGES:int, MAX_GRAPHS:int, signature_name:str):
    metadata = {
        "signature_name" : signature_name,
        "batch_size": batch_size,
        "max_history": max_history,
        "num_candidates": num_candidates,
        "MAX_NODES" : MAX_NODES,
        "MAX_EDGES" : MAX_EDGES,
        "MAX_GRAPHS" : MAX_GRAPHS
    }
    with open(output_file_uri, "w") as f:
        json.dump(metadata, f)

def export_models(trained_model: GraphRanker, batch_size:int, max_history:int,
                 num_candidates:int, output_savedmodel_dir_uri:str):
    """
    export the model to TF SavedModel format along with a method to apply the model on the data.
    makes an export with a signature for  single inference mode and a batch inference mode.
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

    jax_graph_comp_dict = calc_number_jax_graph_components(batch_size,
        max_history, num_candidates, n_local_devices=jax.local_device_count())

    MAX_NODES = jax_graph_comp_dict['max_nodes']
    MAX_EDGES = jax_graph_comp_dict['max_edges']
    MAX_GRAPHS = 1  # Usually 1 if doing single-request real-time ranking

    single_serving_config = get_serving_signature(MAX_NODES, MAX_EDGES, MAX_GRAPHS, signature_name="serving_default")

    batch_serving_config = get_serving_signature(MAX_NODES, MAX_EDGES, jax_graph_comp_dict['max_graphs'],
                                                 signature_name=f"serving_batch_{batch_size}")

    jax_module = export.JaxModule(
        params=trained_model,
        apply_fn=pure_apply_fn,
        trainable=False
    )

    export_manager = export.ExportManager(jax_module, [single_serving_config, batch_serving_config])
    export_manager.save(output_savedmodel_dir_uri)

    save_metadata(output_file_uri=os.path.join(output_savedmodel_dir_uri, "metadata_single.json"),
        batch_size=1, max_history=max_history, num_candidates=num_candidates, MAX_NODES=MAX_NODES,
        MAX_EDGES=MAX_EDGES, MAX_GRAPHS=MAX_GRAPHS, signature_name="serving_default")

    save_metadata(output_file_uri=os.path.join(output_savedmodel_dir_uri, "metadata_batch.json"),
                  batch_size=batch_size, max_history=max_history, num_candidates=num_candidates, MAX_NODES=MAX_NODES,
                  MAX_EDGES=MAX_EDGES, MAX_GRAPHS=jax_graph_comp_dict['max_graphs'],
                  signature_name="serving_batch")

    print(f"saved model and metadata to {output_savedmodel_dir_uri}")


