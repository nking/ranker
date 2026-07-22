#pip install -q plotly kaleido
import fsspec
import json
import os

import glob
import unittest

from urllib.parse import urlparse

from helper import get_project_dir, get_bin_dir
from movie_lens_ranker.util_plots import plot_metrics_dict


class PlotTrainingTest(unittest.TestCase):

    def test_plot(self):

        in_path = os.path.join(get_project_dir(), "src/test/resources/train_val_metrics.json")

        output_dir = os.path.join(get_bin_dir(), "training_metrics_pngs")

        with fsspec.open(in_path, mode='r') as f:
            content = f.read()
            metrics_dict = json.loads(content)

        plot_metrics_dict(metrics_dict, out_dir=output_dir)

        count = 0
        for file_path in glob.glob(f'{output_dir}/*'):
            parsed_url = urlparse(file_path)
            os.path.exists(parsed_url.path)
            count += 1

        self.assertEqual(4, count)

