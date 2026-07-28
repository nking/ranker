#pip install polars
import glob
import os
from typing import Tuple, List

import polars as pl
import unittest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from urllib.parse import urlparse

from helper import get_project_dir, get_bin_dir
from movie_lens_ranker.util_plots import plot_metrics_dict

def get_single_genre_polarized_users(
        train_val_files: list[str],
        test_positives_pattern: str,
        movies_path: str,
        threshold: float = 0.75,
        min_history: int = 20
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Finds users whose positive training history is predominantly (> threshold)
    concentrated in a single genre across all specified categories,
    and ensures they have evaluateable ground truth in the test set.
    """
    movies = pl.scan_parquet(movies_path)
    positives = pl.scan_parquet(train_val_files)
    test = pl.scan_parquet(test_positives_pattern, glob=True)

    genres_list = (
        movies.lazy()
        .with_columns(
            pl.col("genres").str.replace_all("Children's", "Children")
        )
        .select(
            pl.col("genres").str.split("|").explode().alias("genre")
        )
        .select("genre")
        .unique()
        .sort("genre")
        .collect()
        .get_column("genre")
        .to_list()
    )
    genres_list = [g for g in genres_list if g and g != "(no genres listed)"]

    df = positives.join(movies, on="movie_id")

    # Dynamically build expressions to count occurrences of each genre
    genre_aggregations = [
        pl.col("genres").str.contains(genre).sum().alias(f"count_{genre}")
        for genre in genres_list
    ]

    #  Aggregate total positive ratings and per-genre counts per user
    user_profiles = df.group_by("user_id").agg([
        pl.len().alias("total_pos_ratings"),
        *genre_aggregations
    ])

    # Calculate ratios and find the maximum concentration for each user
    ratio_expressions = [
        (pl.col(f"count_{genre}") / pl.col("total_pos_ratings")).alias(f"ratio_{genre}")
        for genre in genres_list
    ]

    ratio_cols = [f"ratio_{genre}" for genre in genres_list]

    user_profiles = user_profiles.with_columns(ratio_expressions).with_columns([
        pl.max_horizontal(ratio_cols).alias("max_genre_ratio"),
        pl.max_horizontal([f"count_{g}" for g in genres_list]).alias("max_genre_count")
    ])

    # Filter for users who meet the single-genre dominance threshold
    polarized_users = user_profiles.filter(
        (pl.col("total_pos_ratings") >= min_history) &
        (pl.col("max_genre_ratio") >= threshold)
    )

    # Ensure they exist in the test set with evaluatable ground truth
    test_positives = (
        test.filter(pl.col("rating") > 3)
        .select("user_id")
        .unique()
    )

    final_test_users = polarized_users.join(test_positives, on="user_id", how="inner")

    return (final_test_users.collect(), genres_list)

def plot_polarized_users_plotly(polarized_df: pl.DataFrame, genres_list: list[str], save_path: str = None):
    """
    Analyzes and visualizes polarized users using Polars and Plotly.
    Supports rendering and static image export via Kaleido.
    """
    if polarized_df.is_empty():
        print("No polarized users found to plot.")
        return

    ratio_cols = [f"ratio_{g}" for g in genres_list]

    # 1. Efficiently unpivot the DataFrame to find the max ratio and genre per user
    dominant_df = (
        polarized_df
        .unpivot(
            index=["user_id", "total_pos_ratings"],
            on=ratio_cols,
            variable_name="genre_col",
            value_name="dominant_ratio"
        )
        .with_columns(
            pl.col("genre_col").str.replace("ratio_", "").alias("dominant_genre")
        )
        .sort("dominant_ratio", descending=True)
        .unique(subset=["user_id"], keep="first")
    )

    # 2. Aggregate user counts per dominant genre
    counts_df = (
        dominant_df
        .group_by("dominant_genre")
        .agg(pl.len().alias("user_count"))
        .sort("user_count", descending=False)
    )

    # Convert to Pandas only for Plotly box plot handling convenience
    dominant_pd = dominant_df.to_pandas()
    counts_pd = counts_df.to_pandas()

    # 3. Create 1x2 Subplots layout
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "User Count per Dominant Genre",
            "Genre Concentration Purity (Ratio Distribution)"
        ),
        horizontal_spacing=0.15
    )

    # Panel 1: Horizontal Bar Chart of User Counts
    fig.add_trace(
        go.Bar(
            x=counts_pd["user_count"],
            y=counts_pd["dominant_genre"],
            orientation="h",
            marker_color="#2b5c8f",
            name="User Count"
        ),
        row=1, col=1
    )

    # Panel 2: Box Plot of Ratios per Dominant Genre
    for genre in counts_pd["dominant_genre"]:
        genre_subset = dominant_pd[dominant_pd["dominant_genre"] == genre]
        fig.add_trace(
            go.Box(
                x=genre_subset["dominant_ratio"],
                name=genre,
                boxpoints=False,
                showlegend=False,
                marker_color="#2b5c8f"
            ),
            row=1, col=2
        )

    # 4. Layout Optimization
    fig.update_layout(
        height=650,
        width=1250,
        template="plotly_white",
        title_text="<b>Polarized User Distribution & Purity Analysis</b>",
        title_font_size=18,
        showlegend=False
    )

    fig.update_xaxes(title_text="User Count", row=1, col=1)
    fig.update_yaxes(title_text="Dominant Genre", row=1, col=1)
    fig.update_xaxes(title_text="Ratio of Positive Ratings", row=1, col=2)
    fig.update_yaxes(title_text="Dominant Genre", row=1, col=2)

    # 5. Render or Export via Kaleido
    if save_path:
        fig.write_image(save_path, engine="kaleido")
        print(f"Plot saved successfully to {save_path}")

    fig.show()

class PolarizedUsersTest(unittest.TestCase):

    def test_find_polarized_users(self):

        base_dir = os.path.join(get_project_dir(), "src/test/resources/data/")
        train_val_files = list(
                glob.glob(base_dir + "ratings_train_*.parquet") +
                glob.glob(base_dir + "ratings_val_*.parquet")
        )
        test_liked_path = os.path.join(get_project_dir(), "src/test/resources/data/ratings_test_liked*parquet")
        movies_path = os.path.join(get_project_dir(), "src/test/resources/data/movies.parquet")

        df, genres_list = get_single_genre_polarized_users(train_val_files=train_val_files,
            test_positives_pattern=test_liked_path, movies_path=movies_path)

        out_path = os.path.join(get_bin_dir(), "polarized_users.csv")
        df.write_csv(out_path)
        print(f"wrote to {out_path}")

        out_path = os.path.join(get_bin_dir(), "polarized_users.png")

        plot_polarized_users_plotly(df, genres_list, out_path)
        print(f"wrote to {out_path}")

