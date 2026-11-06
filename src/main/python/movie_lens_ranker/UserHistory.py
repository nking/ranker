from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

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
    """

    def __init__(
        self,
        ratings_uri_list: Union[str, List[str]],
        max_history: int,
        n_users: Optional[int] = None,
        #min_rating: Optional[int] = None,
        #max_rating: Optional[int] = None,
    ):
        self.pad_value = -1
        self.ts_pad_value = 2524608000  # year 2050
        self.max_history = max_history
        self.n_users = n_users
        #self.min_rating = min_rating
        #self.max_rating = max_rating
        #implicit indices are the user_ids
        #parallel arrays are sorted by timestamps and the empty slots are filled by the ts_pad_valu or pad_value and are at the ends of the arrays
        self.movie_ids, self.ratings, self.timestamps = self._load_history(
            ratings_uri_list, n_users=n_users
        )

    def _load_history(
        self,
        ratings_uri_list: Union[str, List[str]],
        n_users: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lookup, max_observed_history = build_history_lookup(ratings_uri_list)
        logging.info(
            f"Max of users' histories = {max_observed_history}. self.max_history={self.max_history}"
        )
        self.max_observed_history = max(int(self.max_history),
            int(max_observed_history))
        
        max_uid = int(max(lookup.keys(), default=0))
        total_users = max(max_uid, int(n_users) if n_users is not None else 0)
        self.n_users = total_users
        
        n_rows = total_users + 1
        
        movie_ids = np.full((n_rows, self.max_observed_history),
            self.pad_value, dtype=np.int32)
        ratings = np.full((n_rows, self.max_observed_history), self.pad_value,
            dtype=np.int32)
        timestamps = np.full((n_rows, self.max_observed_history),
            self.ts_pad_value, dtype=np.int64)
        
        for user_id, (user_ts, user_movies, user_ratings) in lookup.items():
            uid = int(user_id)
            length = len(user_ts)
            if length > 0:
                timestamps[uid, :length] = user_ts
                movie_ids[uid, :length] = user_movies
                ratings[uid, :length] = user_ratings

        return movie_ids, ratings, timestamps

    def _resolve_user_indices(
        self, user_id: Union[int, List[int], np.ndarray]
    ) -> np.ndarray:
        user_id_arr = np.asarray(user_id, dtype=np.int64).reshape(-1)
        return np.where(
            (user_id_arr >= 1) & (user_id_arr <= self.n_users), user_id_arr, 0
        )

    def get_history_with_timestamps_before_timestamp(
        self,
        user_id: np.ndarray,
        timestamp: Union[int, np.ndarray],
        max_hist: int,
        most_recent_first: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns `(ret_movie_ids, ret_ratings, ret_timestamps)` for the `max_hist` MOST RECENT
        interactions strictly `< timestamp`, padded at the right with `pad_value` (-1) and `ts_pad_value`.
        :param user_id:
        :param timestamp:
        :param max_hist:
        :param most_recent_first: for True and False, the max_hist most recent history is extracted and the rights die
        of the array is filled with pad_caly or ts_pad_value.  For the default True, the max_hist entries are
        sorted in descending order which is newest to oldest and for most_recent_first the sort order is
        ascending which is oldest to newest.
        :return:
        """
        user_idx = self._resolve_user_indices(user_id)
        n_user_selected = len(user_idx)

        ret_movie_ids = np.full(
            (n_user_selected, max_hist), self.pad_value, dtype=self.movie_ids.dtype
        )
        ret_ratings = np.full(
            (n_user_selected, max_hist), self.pad_value, dtype=self.ratings.dtype
        )
        ret_timestamps = np.full(
            (n_user_selected, max_hist),
            self.ts_pad_value,
            dtype=self.timestamps.dtype,
        )
        if n_user_selected == 0 or max_hist <= 0:
            return ret_movie_ids, ret_ratings, ret_timestamps

        sub_timestamps = self.timestamps[user_idx]
        sub_movie_ids = self.movie_ids[user_idx]
        sub_ratings = self.ratings[user_idx]

        #TODO: consider != self.ts_pad_value instead of < ts.pad_value
        if np.ndim(timestamp) == 0:
            mask = (sub_timestamps < timestamp) & (sub_timestamps < self.ts_pad_value)
        else:
            ts_col = np.asarray(timestamp, dtype=np.int64).reshape(-1, 1)
            mask = (sub_timestamps < ts_col) & (sub_timestamps < self.ts_pad_value)

        # Select the MOST RECENT `max_hist` valid interactions per user
        total_valid = np.sum(mask, axis=1, keepdims=True)
        occ = np.cumsum(mask, axis=1)
        recent_mask = mask & (occ > (total_valid - max_hist))

        recent_occ = np.cumsum(recent_mask, axis=1)
        recent_count = np.minimum(total_valid.squeeze(1), max_hist)

        row_coords, _ = np.where(recent_mask)
        if most_recent_first:
            new_col_coords = recent_count[row_coords] - recent_occ[recent_mask]
        else:
            new_col_coords = recent_occ[recent_mask] - 1

        ret_movie_ids[row_coords, new_col_coords] = sub_movie_ids[recent_mask]
        ret_ratings[row_coords, new_col_coords] = sub_ratings[recent_mask]
        ret_timestamps[row_coords, new_col_coords] = sub_timestamps[recent_mask]

        return ret_movie_ids, ret_ratings, ret_timestamps

    def get_history_before_timestamp(
        self,
        user_id: np.ndarray,
        timestamp: Union[int, np.ndarray],
        max_hist: int,
        most_recent_first: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        
        :param user_id:
        :param timestamp:
        :param max_hist:
        :param most_recent_first: for True and False, the max_hist most recent history is extracted and the rights die
        of the array is filled with pad_caly or ts_pad_value.  For the default True, the max_hist entries are
        sorted in descending order which is newest to oldest and for most_recent_first the sort order is
        ascending which is oldest to newest.
        the oldest history is extracted
        :return:
        """
        ret_movie_ids, ret_ratings, _ = (
            self.get_history_with_timestamps_before_timestamp(
                user_id=user_id,
                timestamp=timestamp,
                max_hist=max_hist,
                most_recent_first=most_recent_first,
            )
        )
        return ret_movie_ids, ret_ratings

    def get_movieids_before_timestamp(
        self,
        user_id: np.ndarray,
        timestamp: Union[int, np.ndarray],
        max_hist: int,
        most_recent_first: bool = True,
    ) -> np.ndarray:
        """
        
        :param user_id:
        :param timestamp:
        :param max_hist:
        :param most_recent_first: for True and False, the max_hist most recent history is extracted and the rights die
        of the array is filled with pad_caly or ts_pad_value.  For the default True, the max_hist entries are
        sorted in descending order which is newest to oldest and for most_recent_first the sort order is
        ascending which is oldest to newest.
        the oldest history is extracted
        :return:
        """
        ret_movie_ids, _, _ = self.get_history_with_timestamps_before_timestamp(
            user_id=user_id,
            timestamp=timestamp,
            max_hist=max_hist,
            most_recent_first=most_recent_first,
        )
        return ret_movie_ids