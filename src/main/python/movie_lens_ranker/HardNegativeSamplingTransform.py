from typing import Dict, List, Union
import numpy as np
import grain.python as pgrain
from array_record.python import array_record_module

from movie_lens_ranker.RecommendedMovies import RecommendedMovies
from movie_lens_ranker.UserHistory import UserHistory
from movie_lens_ranker.util_numba import (
    build_negative_pool,
    generate_type_4_negatives,
    row_wise_intersect,
    row_wise_sortedset_subtract,
    simultaneous_shuffle,
)

class HardNegativeSamplingTransform(pgrain.RandomMapTransform):
    """
    Maps a batch of user queries to `"candidate_ids"` and `"labels"` of shape `(batch_size, num_candidates)`.
    """

    def __init__(
        self,
        history_lookup: UserHistory,
        history_lookup_disliked: UserHistory,
        all_movie_ids: Union[List[int], np.ndarray],
        recommendations: RecommendedMovies,
        num_candidates: int = 20,
        strategy: str = "mixed_hard",
    ):
        self.history_lookup = history_lookup
        self.history_lookup_disliked = history_lookup_disliked
        self.all_movie_ids = np.asarray(all_movie_ids, dtype=np.int32)
        self.recommendations = recommendations
        self.num_candidates = num_candidates
        self.strategy = strategy

        n_negs = max(1, num_candidates - 1)
        self.num_1_negatives = round(n_negs * 0.15)
        self.num_2_negatives = round(n_negs * 0.55)
        self.num_3_negatives = round(n_negs * 0.15)

        self.pad_value = self.history_lookup.pad_value
        if self.history_lookup_disliked.pad_value != self.pad_value:
            raise ValueError(
                "history_lookup_disliked.pad_value must match history_lookup.pad_value"
            )
        if self.recommendations.pad_value != self.pad_value:
            raise ValueError(
                "recommendations.pad_value must match history_lookup.pad_value"
            )

    def sample_candidates(
        self,
        batch: Dict[str, np.ndarray],
        rng: np.random.Generator,
        shuffle_candidates: bool = True,
    ) -> Dict[str, np.ndarray]:
        user_ids = np.asarray(batch["user_id"], dtype=np.int32)
        positive_ids = np.asarray(batch["movie_id"], dtype=np.int32)
        timestamps = np.asarray(batch["timestamp"], dtype=np.int64)

        n_users = user_ids.shape[0]
        n_negs = self.num_candidates - 1

        movie_histories = self.history_lookup.get_movieids_before_timestamp(
            user_id=user_ids,
            timestamp=timestamps,
            max_hist=getattr(
                self.history_lookup, "max_observed_history",
                self.history_lookup.max_history
            ),
        )
        
        movie_histories_disliked = (
            self.history_lookup_disliked.get_movieids_before_timestamp(
                user_id=user_ids,
                timestamp=timestamps,
                max_hist=getattr(
                    self.history_lookup_disliked,
                    "max_observed_history",
                    self.history_lookup_disliked.max_history,
                ),
            )
        )

        # Prepend positive_ids at column 0 so pad_values remain at the end of each row
        watched_and_positive = np.hstack(
            [positive_ids[:, np.newaxis], movie_histories]
        )

        seed1 = int(rng.integers(0, 2**31 - 1))
        type_4_neg = generate_type_4_negatives(
            self.all_movie_ids,
            watched_and_positive,
            n_negs=n_negs,
            pad_value=self.pad_value,
            seed=seed1,
        )

        seed2 = int(rng.integers(0, 2**31 - 1))

        if self.strategy == "worst_recommended":
            worst_unseen = (
                self.recommendations.get_lowest_scoring_unseen_movies(
                    user_id=user_ids,
                    timestamp=timestamps,
                    worst_unseen_k=min(
                        n_negs * 2, self.recommendations.row_length
                    ),
                )
            )
            worst_filtered = row_wise_sortedset_subtract(
                worst_unseen, watched_and_positive, pad_value=self.pad_value
            )
            empty_pool = np.full((n_users, 1), self.pad_value, dtype=np.int32)
            negatives = build_negative_pool(
                arr1=worst_filtered,
                arr2=empty_pool,
                arr3=empty_pool,
                arr4=type_4_neg,
                target1=n_negs,
                target2=0,
                target3=0,
                num_negatives=n_negs,
                pad_value=self.pad_value,
                seed=seed2,
            )
        else:
            all_recommendations = (
                self.recommendations.get_all_recommended_movies(
                    user_id=user_ids
                )
            )
            unseen_recommendations = self.recommendations.get_unseen_movies(
                user_id=user_ids,
                timestamp=timestamps,
                latest_unseen_k=min(
                    max(n_negs * 2, 20), self.recommendations.row_length
                ),
            )

            # Type 1: "hard negatives" = recommended intersect user's disliked (< timestamp)
            type_1_neg = row_wise_intersect(
                all_recommendations,
                movie_histories_disliked,
                pad_value=self.pad_value,
            )

            # Type 2: "implicit hard negatives" = unseen recommended - (watch history U positive)
            type_2_neg = row_wise_sortedset_subtract(
                unseen_recommendations,
                watched_and_positive,
                pad_value=self.pad_value,
            )

            # Type 3: "out of distr negatives" = disliked - all recommended
            type_3_neg = row_wise_sortedset_subtract(
                movie_histories_disliked,
                all_recommendations,
                pad_value=self.pad_value,
            )

            negatives = build_negative_pool(
                arr1=type_1_neg,
                arr2=type_2_neg,
                arr3=type_3_neg,
                arr4=type_4_neg,
                target1=self.num_1_negatives,
                target2=self.num_2_negatives,
                target3=self.num_3_negatives,
                num_negatives=n_negs,
                pad_value=self.pad_value,
                seed=seed2,
            )

        candidate_ids = np.hstack([positive_ids[:, np.newaxis], negatives])

        if (candidate_ids == self.pad_value).any():
            mask = candidate_ids == self.pad_value
            candidate_ids[mask] = rng.choice(
                self.all_movie_ids, size=int(np.sum(mask))
            )

        labels = np.zeros((n_users, self.num_candidates), dtype=np.int32)
        labels[:, 0] = 1

        if shuffle_candidates:
            seed3 = int(rng.integers(0, 2**31 - 1))
            simultaneous_shuffle(candidate_ids, labels, seed=seed3)

        return {
            **batch,
            "candidate_ids": candidate_ids,
            "labels": labels,
        }

    def random_map(
        self, batch: Dict[str, np.ndarray], rng: np.random.Generator
    ) -> Dict[str, np.ndarray]:
        return self.sample_candidates(batch, rng)