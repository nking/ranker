import unittest
import os

import shutil

from tensorflow import saved_model as tf_saved_model
from helper import *
from movie_lens_ranker.train import create_fake_jagged_batch, create_dummy_super_padded_graph
from movie_lens_ranker_export.export import export_models
from movie_lens_ranker_export.restore_from_orbax import restore_model_from_checkpoint


class ExportTest(unittest.TestCase):

    def test_export(self):

        checkpoint_uri = os.path.join(get_project_dir(), "export_src/test/resources/orbax_checkpoint/")
        savedmodel_dir = os.path.join(get_bin_dir(), "savedmodels", "1")
        try:
            os.rmdir(savedmodel_dir)
        except Exception:
            pass
        os.makedirs(savedmodel_dir, exist_ok=True)

        #tmp_mounts is the directory holding the directories backing the dbs
        embeddings_dir = os.path.join(get_project_dir(), "tmp_mounts/fake-gcs-server/")

        restore_dict = restore_model_from_checkpoint(checkpoint_uri=checkpoint_uri, replace_embeddings_gs_uri=embeddings_dir)

        batch_size = 256

        export_models(
            restore_dict['model'],
            batch_size=batch_size,
            max_history = restore_dict['config']['max_history'],
            num_candidates=restore_dict['config']['num_candidates'],
            embed_len=restore_dict['config']['embed_len'],
            output_savedmodel_dir_uri = savedmodel_dir)

        user_id_range = (1, restore_dict['config']['num_users'])
        movie_id_range = (restore_dict['config']['num_users'] + 1, restore_dict['config']['num_users'] + restore_dict['config']['num_movies'])

        fake_single = create_dummy_super_padded_graph(batch_size=1,
                                              max_history=restore_dict['config']['max_history'],
                                              num_candidates=restore_dict['config']['num_candidates'],
                                              user_id_range=user_id_range,
                                              movie_id_range=movie_id_range,
                                              movie_embeddings_uri = restore_dict['config']['movie_embeddings_uri'],
                                              user_embeddings_uri = restore_dict['config']['user_embeddings_uri'])

        fake_batch = create_dummy_super_padded_graph(batch_size=batch_size,
                                              max_history=restore_dict['config']['max_history'],
                                              num_candidates=restore_dict['config']['num_candidates'],
                                          user_id_range=user_id_range,
                                          movie_id_range=movie_id_range,
                                          movie_embeddings_uri = restore_dict['config']['movie_embeddings_uri'],
                                          user_embeddings_uri = restore_dict['config']['user_embeddings_uri'])

        num_candidates = restore_dict['config']['num_candidates']

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
        predictions_single = response['output_0']
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
        predictions_batch = response['output_0']
        print(f'predictions_batch={predictions_batch}', flush=True)
        self.assertEqual(num_candidates, predictions_batch.shape[0])

if __name__ == '__main__':
    unittest.main()





