import glob
import os
from typing import Dict

import polars as pl
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

def plot_mlflow_metrics(metrics_dict:dict):
   
    for key in ["loss", 'ndcg_20', 'recall_20', 'mrr_20']:
        df = pl.DataFrame({
            'epoch': metrics_dict[f'train_{key}']['x'],
            'train': metrics_dict[f'train_{key}']['y'],
            'val': metrics_dict[f'val_{key}']['y'],
        })
        
        df_long = df.unpivot(index="epoch", on=["train", "val"])
        df_long = df_long.rename({"value": key})
        chart = df_long.plot.line(
            x="epoch",
            y=key,
            color="variable"  # This creates the legend automatically
        ).encode(tooltip=["epoch", "variable", key] )
        chart.save(os.path.join(get_bin_dir(), f"{key}.png"), ppi=200)
        
    