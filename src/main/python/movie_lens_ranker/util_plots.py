import glob
import os
from typing import Dict

import fsspec
import json
import plotly.graph_objects as go
from urllib.parse import urlparse
from pathlib import Path

from mlflow.tracking import MlflowClient

def get_project_dir() -> str:
  cwd = os.getcwd()
  head = cwd
  proj_dir = ""
  while head and head != os.sep:
    head, tail = os.path.split(head)
    if tail:  # Add only if not an empty string (e.g., from root or multiple separators)
      if tail == "ranker":
        proj_dir = os.path.join(head, tail)
        break
  return proj_dir

def get_bin_dir() -> str:
  return os.path.join(get_project_dir(), "bin")

def _read_mlflow_metrics(metrics_dir):
    metrics_dict = {}
    for file_path in glob.glob(f'{metrics_dir}/*'):
        parsed_url = urlparse(file_path)
        metric_name = Path(parsed_url.path).name
        metrics_dict[metric_name] = {'x':[], 'y':[]}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                ts, value, epoch = line.strip().split()
                metrics_dict[metric_name]['x'].append(float(epoch))
                metrics_dict[metric_name]['y'].append(float(value))
    return metrics_dict

def get_mlflow_metrics_by_exp_name(mlflow_tracking_uri: str,
        experiment_name: str) -> Dict[str, Dict]:
    dict_of_dicts = {}
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    # the first in runs is the latest
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    for run in runs:
        run_id = run.info.run_id
        metrics_dict = {}
        for key in ("loss", "ndcg_20", "recall_20", "mrr_20"):
            for key_t in (f"train_{key}", f"val_{key}"):
                metrics_dict[key_t] = {'x': [], 'y': []}
                m_dict = client.get_metric_history(run_id, key=key_t)
                for m in m_dict:
                    metrics_dict[key_t]['x'].append(int(m.step))
                    metrics_dict[key_t]['y'].append(float(m.value))
        dict_of_dicts[run_id] = metrics_dict
    return dict_of_dicts

def plot_metrics_dict(metrics_dict: dict, out_dir: str):

    os.makedirs(out_dir, exist_ok=True)

    for key in ["loss", 'ndcg_20', 'recall_20', 'mrr_20']:
        # Extract Polars Series directly out of the dictionary
        epochs = metrics_dict[f'train_{key}']['x']
        train_vals = metrics_dict[f'train_{key}']['y']
        val_vals = metrics_dict[f'val_{key}']['y']

        # Build the figure with zero memory overhead
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_vals, mode='lines', name='train'))
        fig.add_trace(go.Scatter(x=epochs, y=val_vals, mode='lines', name='val'))

        fig.update_layout(
            title={
                'text': f"{key.upper()} over Epochs",
                'font': {'size': 22}
            },
            xaxis={
                'title': 'epoch',
                'title_font': {'size': 18},
                'tickfont': {'size': 14}
            },
            yaxis={
                'title': key,
                'title_font': {'size': 18},
                'tickfont': {'size': 14}
            },
            legend={
                'font': {'size': 16}
            },
            template="plotly_white",
            width=800,
            height=500
        )

        # Save to PNG
        out_path = os.path.join(out_dir, f"{key}.png")
        fig.write_image(out_path, scale=2) # scale=2 doubles resolution (~200 PPI)

        #render in notebook:
        #fig.show(renderer="iframe")
def plot_metrics(json_path: str, out_dir:str):
    try:

        with fsspec.open(json_path, mode='r') as f:
            content = f.read()
            metrics_dict = json.loads(content)

        plot_metrics_dict(metrics_dict=metrics_dict, out_dir=out_dir)

    except Exception as ex:
        print(f'Error: {ex}')