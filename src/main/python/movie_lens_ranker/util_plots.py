import glob
import os
from typing import Dict, Any

import fsspec
import json
import plotly.graph_objects as go
from urllib.parse import urlparse
from pathlib import Path
from plotly.subplots import make_subplots

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
        experiment_name: str, run_name:str) -> Dict[str, Dict]:
    dict_of_dicts = {}
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    # the first in runs is the latest
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
    )
    for run in runs:
        run_id = run.info.run_id
        metrics_dict = {}
        m_run = client.get_run(run_id)
        all_metric_keys = list(m_run.data.metrics.keys())
        metric_keys = set()
        is_test = False
        for k in all_metric_keys:
            v :str = k.removeprefix("train_")
            v :str = v.removeprefix("val_")
            if v.startswith("test"):
                is_test = True
            metric_keys.add(v)
        if is_test:
            for key in metric_keys:
                metrics_dict[key] = {'x': [], 'y': []}
                m_dict = client.get_metric_history(run_id, key=key)
                for m in m_dict:
                    metrics_dict[key]['x'].append(int(m.step))
                    metrics_dict[key]['y'].append(float(m.value))
        else:
            for key in metric_keys:
                for key_t in (f"train_{key}", f"val_{key}"):
                    metrics_dict[key_t] = {'x': [], 'y': []}
                    m_dict = client.get_metric_history(run_id, key=key_t)
                    for m in m_dict:
                        metrics_dict[key_t]['x'].append(int(m.step))
                        metrics_dict[key_t]['y'].append(float(m.value))
        dict_of_dicts[run_id] = metrics_dict
    return dict_of_dicts

def smooth_curve(scalars: list, weight: float = 0.6) -> list:
    """
    Computes the Exponential Moving Average (EMA) to smooth the curve,
    matching TensorBoard's default smoothing algorithm.
    """
    if not scalars:
        return []

    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_metrics_dict(metrics_dict: dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    keys = set()
    for k in metrics_dict:
        v = k.removeprefix("train_")
        v = v.removeprefix("val_")
        keys.add(v)

    # TensorBoard's default line colors
    c_train = "rgb(60, 81, 107)"       # Slate blue/grey for train
    c_train_faint = "rgba(60, 81, 107, 0.25)"
    c_val = "rgb(0, 188, 212)"         # Cyan for validation
    c_val_faint = "rgba(0, 188, 212, 0.25)"

    for key in keys:
        epochs = metrics_dict[f'train_{key}']['x']
        if epochs is None or len(epochs)==0 or f'val_{key}' not in metrics_dict:
            continue
        train_raw = metrics_dict[f'train_{key}']['y']
        val_raw = metrics_dict[f'val_{key}']['y']

        # Apply TensorBoard's 0.6 default smoothing weight
        train_smooth = smooth_curve(train_raw, weight=0.6)
        val_smooth = smooth_curve(val_raw, weight=0.6)

        # Build figure with a Line Chart (top) and Table (bottom)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            specs=[[{"type": "xy"}],
                   [{"type": "table"}]]
        )

        # 1. Train Raw (Faint)
        fig.add_trace(go.Scatter(
            x=epochs, y=train_raw, mode='lines',
            line=dict(color=c_train_faint, width=1.5),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

        # 2. Train Smoothed (Solid)
        fig.add_trace(go.Scatter(
            x=epochs, y=train_smooth, mode='lines',
            line=dict(color=c_train, width=2),
            name='train'
        ), row=1, col=1)

        # 3. Val Raw (Faint)
        fig.add_trace(go.Scatter(
            x=epochs, y=val_raw, mode='lines',
            line=dict(color=c_val_faint, width=1.5),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

        # 4. Val Smoothed (Solid)
        fig.add_trace(go.Scatter(
            x=epochs, y=val_smooth, mode='lines',
            line=dict(color=c_val, width=2),
            name='validation'
        ), row=1, col=1)

        # Extract last step calculations for the table
        step_val = epochs[-1]
        t_sm_last = round(train_smooth[-1], 4)
        t_raw_last = round(train_raw[-1], 4)
        v_sm_last = round(val_smooth[-1], 4)
        v_raw_last = round(val_raw[-1], 4)

        # 5. Bottom Table
        fig.add_trace(go.Table(
            header=dict(
                values=['<b>Run</b>', '<b>Smoothed</b>', '<b>Value</b>', '<b>Step</b>', '<b>Relative</b>'],
                align='left',
                font=dict(size=12, color='black'),
                fill_color='white',
                line_color='white'
            ),
            cells=dict(
                values=[
                    ['&#9679; train', '&#9679; validation'], # &#9679; is a solid HTML circle
                    [t_sm_last, v_sm_last],
                    [t_raw_last, v_raw_last],
                    [step_val, step_val],
                    ['-', '-']  # Replaces time missing from JSON
                ],
                align='left',
                font=dict(
                    color=[
                        [c_train, c_val],      # Match run colors
                        ['black', 'black'],
                        ['black', 'black'],
                        ['black', 'black'],
                        ['black', 'black']
                    ],
                    size=12
                ),
                fill_color='white',
                line_color='white',
                height=25
            )
        ), row=2, col=1)

        # Format Layout
        fig.update_layout(
            title=dict(text=f"epoch_{key}", font=dict(size=16)),
            template="plotly_white",
            showlegend=False, # Replaced by the bottom table
            margin=dict(l=40, r=40, t=50, b=10),
            width=500,
            height=500
        )

        # Add TensorBoard-like grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5', zeroline=False, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5', zeroline=False, row=1, col=1)

        out_path = os.path.join(out_dir, f"{key}.png")

        # 500px * 1.5 scale = 750 pixels (Exactly 2.5 inches printed at 300 PPI)
        fig.write_image(out_path, scale=1.5)
        print(f"wrote to {out_dir}")

def plot_metrics(json_path: str, out_dir:str):
    try:

        with fsspec.open(json_path, mode='r') as f:
            content = f.read()
            metrics_dict = json.loads(content)

        plot_metrics_dict(metrics_dict=metrics_dict, out_dir=out_dir)

    except Exception as ex:
        print(f'Error: {ex}')