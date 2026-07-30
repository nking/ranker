from typing import Tuple, List
import polars as pl

def get_single_genre_polarized_users(
        train_val_files: list[str],
        test_positives_pattern: str,
        movies_path: str,
        threshold: float = 0.75,
        min_history: int = 20
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str]]:
    """
    Finds highly polarized users and returns three cohorts.
    The positive cohort includes a condensed summary of disliked genres
    relative to their dominant positive genre count.
    """
    movies = pl.scan_parquet(movies_path)
    train_val = pl.scan_parquet(train_val_files)
    test_positives = pl.scan_parquet(test_positives_pattern, glob=True)

    # 1. Extract unified genre list
    genres_list = (
        movies.lazy()
        .with_columns(pl.col("genres").str.replace_all("Children's", "Children"))
        .select(pl.col("genres").str.split("|").explode().alias("genre"))
        .select("genre")
        .unique()
        .sort("genre")
        .collect()
        .get_column("genre")
        .to_list()
    )
    genres_list = [g for g in genres_list if g and g != "(no genres listed)"]

    # 2. Prepare base tables
    train_val = train_val.join(movies, on="movie_id")
    train_val_positives = train_val.filter(pl.col("rating") > 3)
    train_val_negatives = train_val.filter(pl.col("rating") < 3)

    test_users_with_ground_truth = (
        test_positives.filter(pl.col("rating") > 3)
        .select("user_id")
        .unique()
    )

    # =========================================================================
    # BASE AGGREGATIONS
    # =========================================================================
    pos_aggs = [pl.len().alias("pos_total_ratings")] + [
        pl.col("genres").str.contains(g).sum().alias(f"pos_count_{g}") for g in genres_list
    ]
    pos_profiles = train_val_positives.group_by("user_id").agg(pos_aggs)

    neg_aggs = [pl.len().alias("neg_total_ratings")] + [
        pl.col("genres").str.contains(g).sum().alias(f"neg_count_{g}") for g in genres_list
    ]
    # We build neg_profiles up front so we can reuse the counts for pos_cohort
    neg_profiles = train_val_negatives.group_by("user_id").agg(neg_aggs)


    # =========================================================================
    # COHORT 1: Positive Dominant Users (with Dislike Summary)
    # =========================================================================
    pos_ratios = [
        (pl.col(f"pos_count_{g}") / pl.col("pos_total_ratings")).alias(f"pos_ratio_{g}")
        for g in genres_list
    ]

    pos_profiles = pos_profiles.with_columns(pos_ratios).with_columns(
        pl.max_horizontal([f"pos_ratio_{g}" for g in genres_list]).alias("pos_max_ratio")
    )

    pos_dominant_exprs = [
        pl.when(pl.col(f"pos_ratio_{g}") >= threshold).then(pl.lit(g)).otherwise(None)
        for g in genres_list
    ]

    # Build the base positive cohort
    pos_cohort = pos_profiles.filter(
        (pl.col("pos_total_ratings") >= min_history) &
        (pl.col("pos_max_ratio") >= threshold)
    ).with_columns(
        pl.concat_list(pos_dominant_exprs).list.drop_nulls().list.join(", ").alias("pos_dominant_genre")
    ).join(
        test_users_with_ground_truth, on="user_id", how="inner"
    )

    # Left-join negative counts to build the dislike summary
    pos_cohort = pos_cohort.join(
        neg_profiles.select(["user_id"] + [f"neg_count_{g}" for g in genres_list]),
        on="user_id",
        how="left"
    ).with_columns(
        [pl.col(f"neg_count_{g}").fill_null(0) for g in genres_list]
    )

    # Denominator: The exact count of positive ratings for their dominant genre
    pos_dominant_count = pl.col("pos_total_ratings") * pl.col("pos_max_ratio")

    # Format string like: "Action (0.45)" only if they actually disliked the genre
    dislike_exprs = [
        pl.when(pl.col(f"neg_count_{g}") > 0)
        .then(
            pl.format(
                "{} ({})",
                pl.lit(g),
                (pl.col(f"neg_count_{g}") / pos_dominant_count).round(2)
            )
        )
        .otherwise(None)
        for g in genres_list
    ]

    pos_cohort = pos_cohort.with_columns(
        pl.concat_list(dislike_exprs)
        .list.drop_nulls()
        .list.join(", ")
        .alias("disliked_genres_summary")
    ).drop(
        # Clean up the raw neg_count columns so they don't clutter the DataFrame
        # or cause collisions in the later Both Cohort inner join.
        [f"neg_count_{g}" for g in genres_list]
    )


    # =========================================================================
    # COHORT 2: Negative Dominant Users
    # =========================================================================
    neg_ratios = [
        (pl.col(f"neg_count_{g}") / pl.col("neg_total_ratings")).alias(f"neg_ratio_{g}")
        for g in genres_list
    ]

    neg_profiles = neg_profiles.with_columns(neg_ratios).with_columns(
        pl.max_horizontal([f"neg_ratio_{g}" for g in genres_list]).alias("neg_max_ratio")
    )

    neg_dominant_exprs = [
        pl.when(pl.col(f"neg_ratio_{g}") >= threshold).then(pl.lit(g)).otherwise(None)
        for g in genres_list
    ]

    neg_cohort = neg_profiles.filter(
        (pl.col("neg_total_ratings") >= min_history) &
        (pl.col("neg_max_ratio") >= threshold)
    ).with_columns(
        pl.concat_list(neg_dominant_exprs).list.drop_nulls().list.join(", ").alias("neg_dominant_genre")
    ).join(test_users_with_ground_truth, on="user_id", how="inner")


    # =========================================================================
    # COHORT 3: Bi-Polarized Users (Inner Join)
    # =========================================================================
    both_cohort = pos_cohort.join(neg_cohort, on="user_id", how="inner")

    return pos_cohort.collect(), neg_cohort.collect(), both_cohort.collect(), genres_list