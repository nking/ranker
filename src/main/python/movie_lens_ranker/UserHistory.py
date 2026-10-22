from typing import Union, List, Tuple, Optional

import numpy as np
from array_record.python import array_record_module

from movie_lens_ranker.util import build_history_lookup
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class UserHistory(object):
    """
    Stores per-user interaction histories (movie_ids, ratings, timestamps) in dense 2D arrays
    of shape `(n_users + 1, max_history)` where outer row index `i` corresponds directly
    to `user_id = i`.

    - Row `0` (`user_id == 0`) is a permanent OOV / cold-start sentinel row filled with
      `self.pad_value` (-1) and `self.ts_pad_value` (2524608000).
    - Any request with `user_id <= 0` or `user_id > self.n_users` is remapped to index `0`,
      automatically yielding an empty (padded) history in O(1) time.
    """

    def __init__(
        self,
        ratings_uri_list: Union[str, List[str]],
        max_history: int,
        n_users: Optional[int] = None,
    ):
        self.pad_value = -1
        self.ts_pad_value = 2524608000  # year 2050
        self.max_history = max_history
        self.n_users = n_users
        # Outer row index is `user_id` directly (0..=self.n_users), with row 0 as padded sentinel.
        self.movie_ids, self.ratings, self.timestamps = self._load_history(
            ratings_uri_list, n_users=n_users
        )

    def _load_history(
        self,
        ratings_uri_list: Union[str, List[str]],
        n_users: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds dense 2D NumPy arrays of shape `(self.n_users + 1, self.max_history)` for
        movie_ids, ratings, and timestamps indexed directly by `user_id`.
        Row 0 and any `user_id` absent from `ratings_uri_list` remain populated with
        `self.pad_value` (-1) and `self.ts_pad_value`.
        """
        lookup, max_history = build_history_lookup(ratings_uri_list)
        logging.info(
            f"max of users' histories = {max_history}. self.max_history={self.max_history}"
        )

        max_uid = int(max(lookup.keys(), default=0))
        total_users = max(max_uid, int(n_users) if n_users is not None else 0)
        self.n_users = total_users

        # Allocate +1 row so index 0 is the padded sentinel row and 1..=total_users map 1:1 to user_id
        n_rows = total_users + 1
        movie_ids = np.full(
            (n_rows, self.max_history), self.pad_value, dtype=np.int32
        )
        ratings = np.full(
            (n_rows, self.max_history), self.pad_value, dtype=np.int32
        )
        timestamps = np.full(
            (n_rows, self.max_history), self.ts_pad_value, dtype=np.int64
        )

        for user_id, (user_ts, user_movies, user_ratings) in lookup.items():
            uid = int(user_id)
            if uid <= 0:
                raise ValueError(
                    f"Expected 1-based user_id >= 1 (index 0 is reserved for padded sentinel), found user_id={uid}"
                )
            length = min(len(user_ts), self.max_history)
            if length > 0:
                # Store the most recent `length` interactions at the beginning of row `uid`
                timestamps[uid, :length] = user_ts[-length:]
                movie_ids[uid, :length] = user_movies[-length:]
                ratings[uid, :length] = user_ratings[-length:]

        return movie_ids, ratings, timestamps

    def _resolve_user_indices(
        self, user_id: Union[int, List[int], np.ndarray]
    ) -> np.ndarray:
        """
        Maps valid catalog `user_id`s in `1..=self.n_users` directly to their own index,
        and remaps `user_id <= 0` or `user_id > self.n_users` to index `0` (the padded sentinel row).
        """
        user_id_arr = np.asarray(user_id, dtype=np.int64).reshape(-1)
        return np.where(
            (user_id_arr >= 1) & (user_id_arr <= self.n_users), user_id_arr, 0
        )

    def get_movieids_before_timestamp(
        self,
        user_id: np.ndarray,
        timestamp: Union[int, np.ndarray],
        max_hist: int,
    ) -> np.ndarray:
        """
        Given an array of user_ids, returns up to `max_hist` movie_ids with
        timestamps strictly `< timestamp`, padded with `self.pad_value` (-1).
        Out-of-range user_ids (`<= 0` or `> self.n_users`) are remapped to index `0`.

        :param user_id: 1D array of user_ids, e.g. np.array([2, 4])
        :param timestamp: scalar timestamp or 1D array of timestamps per user
        :param max_hist: maximum number of rated movies to return per user
        :return: array of shape `(len(user_id), max_hist)` padded with `self.pad_value` (-1)
        """
        user_idx = self._resolve_user_indices(user_id)
        n_user_selected = len(user_idx)

        ret_movie_ids = np.full(
            (n_user_selected, max_hist), self.pad_value, dtype=self.movie_ids.dtype
        )
        if n_user_selected == 0 or max_hist <= 0:
            return ret_movie_ids

        sub_timestamps = self.timestamps[user_idx]  # Shape: (n_user_selected, max_history)
        sub_movie_ids = self.movie_ids[user_idx]    # Shape: (n_user_selected, max_history)

        if np.ndim(timestamp) == 0:
            mask = (sub_timestamps < timestamp) & (sub_timestamps < self.ts_pad_value)
        else:
            ts_col = np.asarray(timestamp, dtype=np.int64).reshape(-1, 1)
            mask = (sub_timestamps < ts_col) & (sub_timestamps < self.ts_pad_value)

        occurrence_count = np.cumsum(mask, axis=1)
        final_mask = mask & (occurrence_count <= max_hist)
        row_coords, _ = np.where(final_mask)
        new_col_coords = occurrence_count[final_mask] - 1

        ret_movie_ids[row_coords, new_col_coords] = sub_movie_ids[final_mask]

        return ret_movie_ids

    def get_history_before_timestamp(
        self,
        user_id: np.ndarray,
        timestamp: Union[int, np.ndarray],
        max_hist: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given an array of user_ids, returns `(ret_movie_ids, ret_ratings)` for up to
        `max_hist` interactions strictly `< timestamp`, padded with `self.pad_value` (-1).
        Out-of-range user_ids (`<= 0` or `> self.n_users`) are remapped to index `0`.

        :param user_id: 1D array of user_ids, e.g. np.array([2, 4])
        :param timestamp: scalar timestamp or 1D array of timestamps per user
        :param max_hist: maximum number of rated movies to return per user
        :return: tuple `(ret_movie_ids, ret_ratings)` each of shape `(len(user_id), max_hist)`
        """
        user_idx = self._resolve_user_indices(user_id)
        n_user_selected = len(user_idx)

        ret_movie_ids = np.full(
            (n_user_selected, max_hist), self.pad_value, dtype=self.movie_ids.dtype
        )
        ret_ratings = np.full(
            (n_user_selected, max_hist), self.pad_value, dtype=self.ratings.dtype
        )
        if n_user_selected == 0 or max_hist <= 0:
            return ret_movie_ids, ret_ratings

        sub_timestamps = self.timestamps[user_idx]  # Shape: (n_user_selected, max_history)
        sub_movie_ids = self.movie_ids[user_idx]    # Shape: (n_user_selected, max_history)
        sub_ratings = self.ratings[user_idx]        # Shape: (n_user_selected, max_history)

        if np.ndim(timestamp) == 0:
            mask = (sub_timestamps < timestamp) & (sub_timestamps < self.ts_pad_value)
        else:
            ts_col = np.asarray(timestamp, dtype=np.int64).reshape(-1, 1)
            mask = (sub_timestamps < ts_col) & (sub_timestamps < self.ts_pad_value)

        occurrence_count = np.cumsum(mask, axis=1)
        final_mask = mask & (occurrence_count <= max_hist)
        row_coords, _ = np.where(final_mask)
        new_col_coords = occurrence_count[final_mask] - 1

        ret_movie_ids[row_coords, new_col_coords] = sub_movie_ids[final_mask]
        ret_ratings[row_coords, new_col_coords] = sub_ratings[final_mask]

        return ret_movie_ids, ret_ratings
