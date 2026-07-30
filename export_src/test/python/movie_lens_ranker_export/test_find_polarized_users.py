#pip install polars
import glob
import os

import polars as pl
import unittest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helper import get_project_dir, get_bin_dir
from movie_lens_ranker_export.export_util import get_single_genre_polarized_users

def plot_polarized_users_plotly(polarized_df: pl.DataFrame, genres_list: list[str],
                                save_path: str = None, prefix: str = "pos_", intersection:bool=False):
    """
    Analyzes and visualizes polarized users using Polars and Plotly.
    Supports rendering and static image export via Kaleido.
    """
    if polarized_df.is_empty():
        print("No polarized users found to plot.")
        return

    print(f'columns={polarized_df.columns}')

    ratio_cols = [f"{prefix}ratio_{g}" for g in genres_list]
    total_ratings_col = f"{prefix}total_ratings"

    # 1. Efficiently unpivot the DataFrame to find the max ratio and genre per user
    dominant_df = (
        polarized_df
        .unpivot(
            index=["user_id", total_ratings_col],
            on=ratio_cols,
            variable_name="genre_col",
            value_name="dominant_ratio"
        )
        .with_columns(
            # Strip the dynamic prefix to get just the clean genre name
            pl.col("genre_col").str.replace(f"{prefix}ratio_", "").alias("dominant_genre")
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
    if prefix == "pos_":
        if intersection:
            title_type = "Intersection Positive"
        else:
            title_type = "Positive"
    else:
        if intersection:
            title_type = "Intersection Negative"
        else:
            title_type = "Negative"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"User Count per Dominant {title_type} Genre",
            f"Genre Concentration Purity ({title_type} Ratio Distribution)"
        ),
        horizontal_spacing=0.15
    )

    # Panel 1: Horizontal Bar Chart of User Counts
    fig.add_trace(
        go.Bar(
            x=counts_pd["user_count"],
            y=counts_pd["dominant_genre"],
            orientation="h",
            marker_color="#2b5c8f" if prefix == "pos_" else "#d9534f", # Red for negatives
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
                marker_color="#2b5c8f" if prefix == "pos_" else "#d9534f"
            ),
            row=1, col=2
        )

    # 4. Layout Optimization
    fig.update_layout(
        height=650,
        width=1250,
        template="plotly_white",
        title_text=f"<b>Polarized User Distribution & Purity Analysis ({title_type})</b>",
        title_font_size=18,
        showlegend=False
    )

    fig.update_xaxes(title_text="User Count", row=1, col=1)
    fig.update_yaxes(title_text="Dominant Genre", row=1, col=1)
    fig.update_xaxes(title_text=f"Ratio of {title_type} Ratings", row=1, col=2)
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

        pos_df, neg_df, joined_df, genres_list = get_single_genre_polarized_users(train_val_files=train_val_files,
                test_positives_pattern=test_liked_path, movies_path=movies_path)

        out_path = os.path.join(get_bin_dir(), "polarized_users_pos.csv")
        pos_df.write_csv(out_path)
        print(f"wrote to {out_path}")
        out_path = os.path.join(get_bin_dir(), "polarized_users_pos.png")
        plot_polarized_users_plotly(pos_df, genres_list, out_path)
        print(f"wrote to {out_path}")

        out_path = os.path.join(get_bin_dir(), "polarized_users_neg.csv")
        neg_df.write_csv(out_path)
        print(f"wrote to {out_path}")
        out_path = os.path.join(get_bin_dir(), "polarized_users_neg.png")
        plot_polarized_users_plotly(neg_df, genres_list, out_path, prefix="neg_")
        print(f"wrote to {out_path}")

        # ---- intersection plots ---
        out_path = os.path.join(get_bin_dir(), "polarized_users_intersect_pos.csv")
        joined_df.write_csv(out_path)
        print(f"wrote to {out_path}")
        out_path = os.path.join(get_bin_dir(), "polarized_users_intersect_pos.png")
        plot_polarized_users_plotly(joined_df, genres_list, out_path, prefix="pos_", intersection=True)
        print(f"wrote to {out_path}")

        out_path = os.path.join(get_bin_dir(), "polarized_users_intersect_neg.csv")
        joined_df.write_csv(out_path)
        print(f"wrote to {out_path}")
        out_path = os.path.join(get_bin_dir(), "polarized_users_intersect_neg.png")
        plot_polarized_users_plotly(joined_df, genres_list, out_path, prefix="neg_", intersection=True)
        print(f"wrote to {out_path}")

