from unittest import TestCase
import os

import shutil

from helper import *
from movie_lens_ranker_export.export import export_models
from movie_lens_ranker_export.restore_from_orbax import restore_model_from_checkpoint


class ExportTest(TestCase):

    def test_export(self):

        checkpoint_uri = os.path.join(get_project_dir(), "export_src/test/resources/orbax_checkpoint/")
        out_dir = os.path.join(get_bin_dir(), "savedmodels", "1")
        try:
            os.rmdir(out_dir)
        except Exception:
            pass
        os.makedirs(out_dir, exist_ok=True)

        #tmp_mounts is the directory holding the directories backing the dbs
        embeddings_dir = os.path.join(get_project_dir(), "tmp_mounts/fake-gcs-server/")

        restore_dict = restore_model_from_checkpoint(checkpoint_uri=checkpoint_uri, replace_embeddings_gs_uri=embeddings_dir)

        export_models(
            restore_dict['model'],
            batch_size=256,
            max_history = restore_dict['config']['max_history'],
            num_candidates=restore_dict['config']['num_candidates'],
            embed_len=restore_dict['config']['embed_len'],
            output_savedmodel_dir_uri = out_dir)
