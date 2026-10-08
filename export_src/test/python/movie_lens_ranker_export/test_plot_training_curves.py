#pip install -q plotly kaleido
import re
from typing import List, Tuple

import json
import os

import unittest

#install tensorflow==2.20.0
#isntall tensorboard==2.20.0

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util
import matplotlib.pyplot as plt

import numpy as np

from helper import get_project_dir, get_bin_dir

class PlotTrainingTest(unittest.TestCase):

    def test_plot(self):

        #logdir = os.path.join(get_project_dir(), "src/test/resources/train_val_test_metrics")
        #saved_model_dir = os.path.join(get_bin_dir(), "src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/1")
        #hyperparams_dict = self.get_hyperparams_dict_from_saved_model(saved_model_dir)

        logdir = os.path.join(get_project_dir(), "TMP22/hpo-results-bucket/kaggle-tune-train-test/kaggle-hpo/")
        hyperparams_path = os.path.join(get_project_dir(), "TMP22/hpo-results-bucket/kaggle-tune-train-test/kaggle-hpo/tune/hparams.json")
        hyperparams_dict = self.get_hyperparams_dict(hyperparams_path)

        outdir = os.path.join(get_bin_dir(), "training_metrics_pngs")

        num_catalog_movies = hyperparams_dict.get("num_movies", 3883)
        top_k = hyperparams_dict.get("top_k", 20)
        batch_size = hyperparams_dict["batch_size"]
        temperature = hyperparams_dict["temperature"]
        num_candidates = hyperparams_dict["num_candidates"]

        random_ndcg = self.calc_random_ndcg(k=top_k, movie_catalog_size=num_catalog_movies)
        random_recall = self.calc_random_recall(k=top_k, movie_catalog_size=num_catalog_movies)
        random_precision = self.calc_random_precision(k=top_k, movie_catalog_size=num_catalog_movies)
        random_mrr = self.calc_random_mrr(k=top_k, movie_catalog_size=num_catalog_movies)
        random_loss = self.calc_random_softmax_loss(num_candidates)


        train_dir = os.path.join(logdir, "train")
        val_dir = os.path.join(logdir, "train")
        test_dir = os.path.join(logdir, "test")
        #irred_err_dict = {} # self.get_irred_error_dict(saved_model_dir)
        metrics = self.list_metrics(train_dir)
        print(f'metrics: {metrics}', flush=True)

        for metric in metrics:
            if metric.find('final_') > -1 or metric.find("time") > -1:
                continue
            outfile = os.path.join(outdir, f"{metric}.png")
            random_metric = None
            if metric.find("recall") > -1:
                random_metric = random_recall
            elif metric.find("ndcg") > -1:
                random_metric = random_ndcg
            elif metric.find("hit_rate") > -1:
                #for 1 relevant ground_truth item:
                random_metric = random_ndcg
            elif metric.find("mrr") > -1:
                #for 1 relevant ground_truth item:
                random_metric = random_mrr
            elif metric.find("precision") > -1:
                #for 1 relevant ground_truth item:
                random_metric = random_precision
            elif metric.find("loss") > -1:
                random_metric = random_loss
            generate_tensorboard_chart(train_dir, val_dir, test_dir, None, random_metric,
                                       scalar_name=metric, output_path=outfile)

        for metric in ['logit_min', 'logit_mean', 'logit_max']:
            outfile = os.path.join(outdir, f"{metric}.png")
            export_scalars_to_png(train_dir, outfile, metric)

        print(f'wrote pngs to {outdir}')

    def calc_random_softmax_loss(self, num_candidates:int):
        return np.log(num_candidates)

    def calc_random_ndcg(self, k: int, movie_catalog_size: int) -> float:
        s = np.sum([1 / np.log2(r + 1) for r in range(1, k + 1)])
        return (s / movie_catalog_size).item()

    def calc_random_recall(self, k: int, movie_catalog_size: int) -> float:
        return k / movie_catalog_size

    def calc_random_precision(self, k: int, movie_catalog_size: int) -> float:
        return 1 / movie_catalog_size

    def calc_random_mrr(self, k: int, movie_catalog_size: int) -> float:
        #assuming 1 relevant item
        s = np.sum([1/r for r in range(1, k + 1)])
        return (s / movie_catalog_size).item()

    def find_irred_dict(self, irred_err_dict, metric_name):
        #NOTE: this isn't currently in the ranker project
        #irred_err_dict keys: {'composite_ndcg_k', 'hit_rate', 'mean_loss', 'mrr_k',
        # 'ndcg_k', 'ndcg_head_k', 'ndcg_tail_k','ndcg_torso_k', 'recall_k'}
        # metric_name usually starts with epoch_
        metric_name = metric_name.replace("epoch_", "")
        return irred_err_dict.get(metric_name, None)

    def get_hyperparams_dict_from_saved_model(self, saved_model_dir:str):
        file_path = os.path.join(saved_model_dir, "assets.extra", "training_hyperparameters.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)

    def get_hyperparams_dict(self, file_path:str):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)

    def list_metrics(self, log_dir) -> List[str]:

        file_path = os.path.join(log_dir, "metrics.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                m_dict = json.load(f)

        out = []
        for key in m_dict:
            k = re.sub(r'^train_', "", key)
            k = re.sub(r'^val_', "", k)
            k = re.sub(r'^test_', "", k)
            out.append(k)
        return out


def generate_tensorboard_chart(train_dir, val_dir, test_dir, irred_err_dict,
                               random_metric, scalar_name, output_path, smoothing_weight=0.6):
    """Generates a single PNG chart (~3.5x4 inches) with overplotted train/val curves,

    faded raw lines, smoothed lines, a random baseline line, and a data summary
    table.
    """
    name = re.sub(r'^', "train_", scalar_name)
    train_steps, train_values = read_metrics(train_dir, name)

    name = re.sub(r'^', "val_", scalar_name)
    val_steps, val_values = read_metrics(val_dir, name)

    name = re.sub(r'^', "test_", scalar_name)
    test_steps, test_values = read_metrics(test_dir, name, is_test=True)
    test_steps = [len(train_steps)]

    if train_steps is None or val_steps is None or test_steps is None:
        print("Could not load data for runs. Exiting.")
        return

    is_loss = scalar_name.find("loss") > -1

    train_smoothed = exponential_smoothing(train_values, smoothing_weight)
    val_smoothed = exponential_smoothing(val_values, smoothing_weight)
    test_smoothed = exponential_smoothing(test_values, smoothing_weight)

    # --- Dynamic Bounds Logic ---
    bound_val = None
    bound_moe = None
    bound_label = "irred error" if is_loss else "ceiling"

    if irred_err_dict is not None:
        if is_loss:
            bound_val = irred_err_dict.get("irred_error")
        else:
            bound_val = irred_err_dict.get("ceiling")

        bound_moe = irred_err_dict.get("margin_of_error_on_irred_err")

    # Extract latest test stats
    latest_test_step = test_steps[-1]
    latest_test_value = test_values[-1]
    latest_test_smoothed = test_smoothed[-1]

    fig = plt.figure(figsize=(2.5, 2.5), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.2], hspace=0.25)

    # Plot Axis
    ax = fig.add_subplot(gs[0])

    # High-contrast color scheme
    train_color = "#1f77b4"  # Blue
    train_color_faded = "#aec7e8"  # Faded Blue
    val_color = "#ff7f0e"  # Orange
    val_color_faded = "#ffbb78"  # Faded Orange
    test_color = "#d62728"  # Red
    bound_color = "#2ca02c"  # Green
    random_color = "#000000"  # Black for random baseline

    # Plot raw (faded) and smoothed (solid) lines
    ax.plot(
        train_steps,
        train_values,
        color=train_color_faded,
        alpha=0.7,
        linewidth=0.6,
        linestyle="-",
    )
    ax.plot(
        val_steps,
        val_values,
        color=val_color_faded,
        alpha=0.7,
        linewidth=0.6,
        linestyle="-",
    )
    ax.plot(
        train_steps,
        train_smoothed,
        color=train_color,
        linewidth=1.2,
        linestyle="-",
    )
    ax.plot(
        val_steps, val_smoothed, color=val_color, linewidth=1.2, linestyle="-"
    )

    # Plot Test Point
    ax.plot(
        latest_test_step,
        latest_test_value,
        marker="o",
        markersize=4,
        color=test_color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=5,
    )

    # Plot Irreducible Error / Ceiling
    if bound_val is not None:
        ax.axhline(
            y=bound_val,
            color=bound_color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )
        if bound_moe is not None:
            ax.axhspan(
                bound_val - bound_moe,
                bound_val + bound_moe,
                color=bound_color,
                alpha=0.15,
                zorder=1,
                )

    # --- Plot Horizontal Black Dotted Line for Random Metric ---
    if random_metric is not None:
        ax.axhline(
            y=random_metric,
            color=random_color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
        )

    # Typography & layout scaling
    ax.set_title(
        scalar_name, loc="left", fontsize=7.5, fontweight="bold", pad=4
    )
    ax.tick_params(axis="x", labelsize=5.5, pad=1)
    ax.tick_params(axis="y", labelsize=5.5, pad=1)

    # Grid guidelines
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="-",
        linewidth=1.0,
        alpha=0.75,
        color="#cbd5e1",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Y-limits check (includes random_metric so line isn't clipped)
    all_values = (
            train_values
            + val_values
            + train_smoothed
            + val_smoothed
            + [latest_test_value]
    )
    #if bound_val is not None:
    #    all_values.append(bound_val)
    #if random_metric is not None:
    #    all_values.append(random_metric)

    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_padding = max((y_max - y_min) * 0.08, 0.01)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Table Axis
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    latest_train_step = train_steps[-1]
    latest_train_value = train_values[-1]
    latest_train_smoothed = train_smoothed[-1]

    latest_val_step = val_steps[-1]
    latest_val_value = val_values[-1]
    latest_val_smoothed = val_smoothed[-1]

    # Format bounds for table
    if bound_val is not None:
        bound_val_str = f"{bound_val:.4f}"
        if bound_moe is not None:
            bound_val_str += f" ±{bound_moe:.4f}"
    else:
        bound_val_str = "-"

    # Build Table Content & Labels dynamically
    # Build Table Content & Labels dynamically
    cell_text = [
        [
            f"{latest_train_smoothed:.4f}",
            f"{latest_train_value:.4f}",
            f"{latest_train_step}",
        ],
        [
            f"{latest_val_smoothed:.4f}",
            f"{latest_val_value:.4f}",
            f"{latest_val_step}",
        ],
        [
            f"{latest_test_smoothed:.4f}",
            f"{latest_test_value:.4f}",
            f"{latest_test_step}",
        ],
        # Fixed: Removed the 4th element to match the 3 col_labels
        ["-", bound_val_str, "-"],
    ]
    row_labels = ["train", "validation", "test", bound_label]

    # Append random row if estimate exists
    if random_metric is not None:
        # Fixed: Removed the 4th element here as well
        cell_text.append(["-", f"{random_metric:.4f}", "-"])
        row_labels.append("random")

    col_labels = ["Smoothed", "Value", "Step"]

    the_table = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(5.0)

    color_map = {
        "train": train_color,
        "validation": val_color,
        "test": test_color,
        bound_label: bound_color,
        "random": random_color,
    }

    for i, row_label in enumerate(row_labels):
        cell = the_table[i + 1, -1]
        cell.set_text_props(ha="left", weight="medium")
        color = color_map.get(row_label, "#000000")
        cell.get_text().set_text(f"● {row_label}")
        cell.get_text().set_color(color)

    # Style Table Header and Cells
    for (i, j), cell in the_table.get_celld().items():
        cell.set_edgecolor("#f3f4f6")
        if i == 0:
            cell.set_text_props(weight="bold", color="#4b5563")
            cell.set_facecolor("#f9fafb")
        else:
            cell.set_facecolor("#ffffff")

    # Save output chart
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Output Dictionary
    out_dict = {
        "latest_train_step": latest_train_step,
        "latest_train_value": latest_train_value,
        "latest_train_smoothed": latest_train_smoothed,
        "latest_val_step": latest_val_step,
        "latest_val_value": latest_val_value,
        "latest_val_smoothed": latest_val_smoothed,
        "latest_test_step": latest_test_step,
        "latest_test_value": latest_test_value,
        "latest_test_smoothed": latest_test_smoothed,
        "random_metric": random_metric,
    }

    if is_loss:
        out_dict["irreducible_error"] = bound_val
    else:
        out_dict["ceiling"] = bound_val

    output_file_path = output_path.replace(".png", ".json")
    with open(output_file_path, "w") as f:
        json.dump(out_dict, f, indent=4)
        # print(f"Successfully saved chart with high-contrast grids to {output_path}")

def exponential_smoothing(scalars, weight=0.6):
    """
    Applies exponential smoothing to a list of scalar values.
    """
    smoothed = []
    if not scalars:
        return smoothed
    last = scalars[0]  # First value is the initial smoothed value
    smoothed.append(last)
    for current in scalars[1:]:
        # EMA = (1 - weight) * previous_EMA + weight * current_value
        # Note: TensorBoard's default weight is roughly 0.6
        smoothed_val = (1 - weight) * last + weight * current
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def export_scalars_to_png(log_dir, output_png_path, scalar_name='loss'):
    # OPTIMIZATION: Prevent memory bloat and truncation
    # '0' means "keep all" for scalars/tensors (default truncates at 10k steps)
    # '1' is the minimum allowed, preventing massive RAM usage for heavy assets
    size_guidance = {
        'scalars': 0,
        'tensors': 0,
        'images': 1,
        'audio': 1,
        'histograms': 1,
    }

    # Initialize accumulator and load data
    ea = event_accumulator.EventAccumulator(log_dir,
                                            size_guidance=size_guidance)
    ea.Reload()

    steps, values = [], []
    available_tags = ea.Tags()
    scalar_keys = available_tags.get('scalars', [])
    tensor_keys = available_tags.get('tensors', [])

    # Extract data (Accounts for differences between TF1 and TF2 logging)
    if scalar_name in scalar_keys:
        events = ea.Scalars(scalar_name)
        steps = [e.step for e in events]
        values = [e.value for e in events]
    elif scalar_name in tensor_keys:
        # TensorFlow 2.x often logs scalars as rank-0 tensors
        events = ea.Tensors(scalar_name)
        steps = [e.step for e in events]
        values = [float(tensor_util.make_ndarray(e.tensor_proto)) for e in
                  events]
    else:
        print(f"Metric '{scalar_name}' not found.")
        print(f"Available scalars: {scalar_keys}")
        print(f"Available tensors: {tensor_keys}")
        return

        # Plot using standard matplotlib
    plt.figure(figsize=(2.5, 2.5))
    plt.plot(steps, values, label=scalar_name, color='#ff7043', linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel(scalar_name.capitalize())
    plt.title(f'{scalar_name.capitalize()} over Time')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save as PNG
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved chart to {output_png_path}")

    # Plot using standard matplotlib
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, label=scalar_name, color='#ff7043', linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel(scalar_name.capitalize())
    plt.title(f'{scalar_name.capitalize()} over Time')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save as PNG
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()

def read_metrics(log_dir, scalar_name, is_test:bool=False) -> Tuple[List[int], List[float]]:
    """
    Extracts steps, values for a given scalar from TensorBoard logs.
    """
    file_path = os.path.join(log_dir, "metrics.json")
    with open(file_path, "r") as f:
        m_dict = json.load(f)

        values_dict = m_dict[scalar_name]

        if is_test:
            return None, [values_dict]
        return values_dict['x'], values_dict['y']
