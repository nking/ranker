from typing import Tuple, List


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

    return final_test_users.collect(), genres_list
