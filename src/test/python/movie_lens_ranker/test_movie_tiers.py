import glob
import os.path

import time
from unittest import TestCase
import numpy as np
from helper import *
import polars as pl

def print_ascii_hist(df: pl.DataFrame, col: str, title: str, width: int = 40):
    # Group by tier, count occurrences, and sort by tier value
    counts = df.group_by(col).len().sort(col)
    max_count = counts["len"].max() or 1

    print(f"\n{'=' * 10} {title} {'=' * 10}")
    for val, count in counts.iter_rows():
        label = f"{val}" if val is not None else "Null"
        bar_len = int((count / max_count) * width)
        bar = "█" * bar_len
        print(f"{label:>5} | {bar:<{width}} {count}")

class NumbaOpsTest(TestCase):

    def test_tiers(self):
        in_path = os.path.join(get_project_dir(), "src/test/resources/data/movie_tiers.json")
        movie_tiers_df = pl.read_ndjson(in_path)

        train_df = pl.read_parquet(glob.glob(os.path.join(get_project_dir(),
            "src/test/resources/data/ratings_train_liked*parquet")))

        val_df = pl.read_parquet(
            glob.glob(os.path.join(get_project_dir(), "src/test/resources/data/ratings_val_liked*parquet")))

        train_df = train_df.join(movie_tiers_df, on="movie_id", how="left")
        val_df = val_df.join(movie_tiers_df, on="movie_id", how="left")

        print_ascii_hist(train_df, "tier", "Train Tier Distribution")
        print_ascii_hist(val_df, "tier", "Val Tier Distribution")
