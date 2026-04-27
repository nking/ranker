import glob
import os

import polars as pl
from urllib.parse import urlparse
from pathlib import Path

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
        
def plot_mlflow_metrics(metrics_dir:str):
    metrics_dict = _read_mlflow_metrics(metrics_dir)
   
    for key in ["loss", 'ndcg_20', 'recall_20', 'mrr_20']:
        df = pl.DataFrame({
            'epoch': metrics_dict[f'train_{key}']['x'],
            'train': metrics_dict[f'train_{key}']['y'],
            'val': metrics_dict[f'val_{key}']['y'],
        })
        
        df_long = df.unpivot(index="epoch", on=["train", "val"])
        print(f'df_long: {df_long}')
        chart = df_long.plot.line(
            x="epoch",
            y="value",
            color="variable"  # This creates the legend automatically
        ).encode(tooltip=["epoch", "variable", "value"] )
        chart.save(os.path.join(get_bin_dir(), f"{key}.png"), ppi=200)
        