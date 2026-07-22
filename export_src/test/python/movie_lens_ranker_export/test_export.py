import unittest
from tensorflow import saved_model as tf_saved_model
from helper import *
from movie_lens_ranker.train import create_dummy_super_padded_graph
from movie_lens_ranker.util import calc_number_jax_graph_components
from movie_lens_ranker_export.export import export_models, make_jax_module, create_serving_signature
from movie_lens_ranker_export.restore_from_orbax import restore_model_from_checkpoint
from orbax.export.validate import ValidationManager, ValidationReportOption

class ExportTest(unittest.TestCase):

    def test_export(self):

        checkpoint_uri = os.path.join(get_project_dir(),
                "src/test/resources/checkpoint-bucket/best/kaggle-hpo/train_0/")
        checkpoint_uri = os.path.abspath(checkpoint_uri)
        savedmodel_dir = os.path.join(get_bin_dir(), "savedmodels", "1")
        try:
            os.rmdir(savedmodel_dir)
        except Exception:
            pass
        os.makedirs(savedmodel_dir, exist_ok=True)

        #tmp_mounts is the directory holding the directories backing the dbs
        embeddings_dir = os.path.join(get_project_dir(), "tmp_mounts/fake-gcs-server/")

        restore_dict = restore_model_from_checkpoint(checkpoint_uri=checkpoint_uri,
            replace_embeddings_prefixes=['gs://', '/tmp/gcs_data'], replace_embeddings_prefixes_with=embeddings_dir)

        batch_size = 256

        user_id_range = (1, restore_dict['config']['num_users'])
        movie_id_range = (restore_dict['config']['num_users'] + 1, restore_dict['config']['num_users'] + restore_dict['config']['num_movies'])

        movie_embeddings_uri : str = restore_dict['config']['movie_embeddings_uri']
        user_embeddings_uri : str = restore_dict['config']['user_embeddings_uri']

        fake_single = create_dummy_super_padded_graph(batch_size=1,
                                              max_history=restore_dict['config']['max_history'],
                                              num_candidates=restore_dict['config']['num_candidates'],
                                              user_id_range=user_id_range,
                                              movie_id_range=movie_id_range,
                                              movie_embeddings_uri = movie_embeddings_uri,
                                              user_embeddings_uri = user_embeddings_uri)

        fake_batch = create_dummy_super_padded_graph(batch_size=batch_size,
                                              max_history=restore_dict['config']['max_history'],
                                              num_candidates=restore_dict['config']['num_candidates'],
                                          user_id_range=user_id_range,
                                          movie_id_range=movie_id_range,
                                          movie_embeddings_uri = movie_embeddings_uri,
                                          user_embeddings_uri = user_embeddings_uri)

        num_candidates = restore_dict['config']['num_candidates']

        export_models(
            restore_dict['model'],
            batch_size=batch_size,
            max_history = restore_dict['config']['max_history'],
            num_candidates=restore_dict['config']['num_candidates'],
            embed_len=restore_dict['config']['embed_len'],
            num_catalog_users=restore_dict['config']['num_users'],
            num_catalog_movies=restore_dict['config']['num_movies'],
            output_savedmodel_dir_uri = savedmodel_dir)


        #load the savedmodel to score some fake data
        loaded_saved_model = tf_saved_model.load(savedmodel_dir)

        single_inference_sig = loaded_saved_model.signatures["serving_default"]

        batch_inference_sig = loaded_saved_model.signatures["serving_batch"]

        response = single_inference_sig(
            node_candidate_mask = fake_single.nodes["candidate_mask"],
            node_ids = fake_single.nodes["ids"],
            node_label = fake_single.nodes["label"],
            node_type = fake_single.nodes["type"],
            node_embeddings = fake_single.nodes["embeddings"],
            edge_features = fake_single.edges["rating"],
            receivers = fake_single.receivers,
            senders = fake_single.senders,
            n_node = fake_single.n_node,
            n_edge = fake_single.n_edge,
        )
        predictions_single = response['outputs']
        print(f'predictions_single={predictions_single}', flush=True)
        print(type(predictions_single), predictions_single.dtype)
        self.assertEqual(num_candidates, predictions_single.shape[0])

        predictions_batch = batch_inference_sig(
            node_candidate_mask = fake_batch.nodes["candidate_mask"],
            node_ids = fake_batch.nodes["ids"],
            node_label = fake_batch.nodes["label"],
            node_type = fake_batch.nodes["type"],
            node_embeddings = fake_batch.nodes["embeddings"],
            edge_features = fake_batch.edges["rating"],
            receivers = fake_batch.receivers,
            senders = fake_batch.senders,
            n_node = fake_batch.n_node,
            n_edge = fake_batch.n_edge,
        )
        predictions_batch = response['outputs']
        print(f'predictions_batch={predictions_batch}', flush=True)
        self.assertEqual(num_candidates, predictions_batch.shape[0])


        jax_module = make_jax_module(restore_dict['model'],  restore_dict['config']['num_candidates'])

        jax_graph_comp_dict_single = calc_number_jax_graph_components(1,
                restore_dict['config']['max_history'], restore_dict['config']['num_candidates'], n_local_devices=1)

        single_serving_config = create_serving_signature(
            max_nodes=jax_graph_comp_dict_single['max_nodes'],
            max_edges=jax_graph_comp_dict_single['max_edges'],
            max_graphs=jax_graph_comp_dict_single['max_graphs'],
            embed_len=restore_dict['config']['embed_len'],
            signature_name="serving_default")

        single_inputs = {"node_candidate_mask": fake_single.nodes["candidate_mask"],
            "node_ids" : fake_single.nodes["ids"],
            "node_label" : fake_single.nodes["label"],
            "node_type" : fake_single.nodes["type"],
            "node_embeddings" : fake_single.nodes["embeddings"],
            "edge_features" : fake_single.edges["rating"],
            "receivers" : fake_single.receivers,
            "senders" :  fake_single.senders,
            "n_node" :  fake_single.n_node,
            "n_edge" :  fake_single.n_edge,}

        import numpy as np
        np.set_printoptions(threshold=np.inf)
        print(f'"instances": [\n{single_inputs}\n]\n')

        validation_inputs = {
            "serving_default": [single_inputs]
        }

        validation_mgr = ValidationManager(jax_module, [single_serving_config], validation_inputs)

        report_options = ValidationReportOption(
            floating_atol=1e-5,  # Absolute tolerance
            floating_rtol=1e-5   # Relative tolerance
        )

        validation_reports = validation_mgr.validate(loaded_saved_model, report_option=report_options)

        # `validation_reports` is a python dict and the key is TF SavedModel serving_key.
        for key in validation_reports:
            # Users can also save the converted json to file.
            print(validation_reports[key].to_json(indent=2))
            assert(validation_reports[key].status.name == 'Pass')

        single_inputs = {"node_candidate_mask":fake_single.nodes["candidate_mask"].tolist(),
                         "node_ids" :fake_single.nodes["ids"].tolist(),
                         "node_label" :fake_single.nodes["label"].tolist(),
                         "node_type" :fake_single.nodes["type"].tolist(),
                         "node_embeddings" :fake_single.nodes["embeddings"].tolist(),
                         "edge_features" :fake_single.edges["rating"].tolist(),
                         "receivers" :fake_single.receivers.tolist(),
                         "senders" : fake_single.senders.tolist(),
                         "n_node" : fake_single.n_node.tolist(),
                         "n_edge" : fake_single.n_edge.tolist(),
        }

        import json
        payload = {
            "inputs": single_inputs
        }

        # Write the perfectly formatted JSON to a file
        with open(os.path.join(get_bin_dir(), "test_request.json"), "w") as f:
            json.dump(payload, f)
        #then use:
        #curl -X POST http://172.17.0.1:8511/v1/models/graph-ranker:predict -H "Content-Type: application/json" -d @bin/test_request.json

if __name__ == '__main__':
    unittest.main()





