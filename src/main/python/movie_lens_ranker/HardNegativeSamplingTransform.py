from typing import Dict, List
import numpy as np
import grain.python as pgrain
from array_record.python import array_record_module

from movie_lens_ranker.Negatives_vec import Negatives
from movie_lens_ranker.RecommendedMovies import RecommendedMovies

from movie_lens_ranker.UserHistory import UserHistory

class HardNegativeSamplingTransform(pgrain.MapTransform):
    """
    class to map a user's local history to the same local history enriched with negative sampling
    as "candidate_ids" and "labels"
    """
    def __init__(self, history_lookup: UserHistory, all_movie_ids:List[int], negatives:Negatives,
        recommendations:RecommendedMovies, num_candidates=20, top_k:int=200,
        seed:int = 0):
        """
        initialize a CandidateSamplingTransform object
        :param history_lookup:  Dict[user_id:int, Tuple(arrays of ts, movie_id, rating)]
        :param all_movie_ids: list of all movie_ids
        :param negatives: instance holding user negatives
        :param recommendations: class to retrieve unseen move recommendations for batch of users
        :param num_candidates: total number of candidates to create from 1 postive and mulitple negatives
        :param seed: seed for random number generator
        """
        self.history_lookup = history_lookup
        self.negatives = negatives
        self.all_movie_ids = np.asarray(all_movie_ids) #to o use in approx hard negatives
        self.num_candidates = num_candidates
        self.n_approx = self.num_candidates // 2
        self.n_hard = self.num_candidates - 1 - self.n_approx
        self.top_k = top_k
        self.recommendations = recommendations
        self.seed = seed

    def map(self, batch:Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        given the current user history, add a hard negative mining list as "column_ids" and "labels"
        :param batch: a dictionary containing np.ndarrays
            'user_id' of length batch_size,
            'movie_id' of length batch_size,
            'rating' of length batch_size,
            'timestamp' of length batch_size,
            "history_movie_ids" of shape (batch_size, max_history)
            "history_ratings" of shape (batch_size, max_history)
            "history_length" of length batch_size,
        :return: dictionary of np.ndarrays:
            'user_id',
            'movie_id',
            'rating',
            'timestamp',
            "history_movie_ids",
            "history_ratings",
            "history_length",
            "candidate_ids",
            "labels"
        """
        
        # we want to form the list of positive and negatives for ranking and their labels as 1 and 0 respectively.
        # for each row in the batch:
        #   candidate_ids = the positive movie rated + self.num_candidates - 1 negatives.
        #       half of the negatives are from the hard negatives if possible, and the other
        #       half or more if needed, are randomly sampled from the full movie catalog excluding
        #       the user's history and recommendations for them.
        #       note that timestamps are used in forming these lists.
        #   labels = an array of length self.num_candidates where the first is a 1 and the rest are 0s.
        # the candidate_ids and labels are similarly shuffled to prevent the model from memorizing
        # that the first candidate is correct.
        
        #print(f'HNST batch:{batch}')
        n_approx = self.num_candidates // 2
        n_hard = self.num_candidates - 1 - n_approx
        
        #hard negatives and natural negatives:
        hard_negative_movie_ids = self.negatives.get_negatives(user_id=batch['user_id'], length=n_hard, seed=self.seed)
        
        #these are to be excluded from a random selection of self.all_movie_ids to choose approx_negatives
        movie_histories_long = self.history_lookup.get_movieids_before_timestamp(
            user_id=batch['user_id'], timestamp=batch['timestamp'],
            max_hist=self.history_lookup.fixed_size, pad_value=-1)
        movies_recommended_unseen = self.recommendations.get_unseen_movies(user_id=batch['user_id'],
            timestamp=batch['timestamp'], top_k=self.top_k)
        exclude = np.hstack([movie_histories_long, movies_recommended_unseen, batch['movie_id'][:,np.newaxis]])
        
        n_users = movie_histories_long.shape[0]
        
        #final candidate_ids are [pos_id] + hard_negs[:n_hard] + approx_negs[:n_approx]
        # shape (n_users, self.num_candidates)
        # but there will be missing hard_negs for some users
        # so we generate a base layer of shape (n_users, self.num_candidates) filled with approx_negatives,
        # then fill in the smallest indices with hard_negatives where they exist.
        
        n_movies = len(self.all_movie_ids)
        # Pick twice as many or more to handle collisions
        n_approx_candidates = (self.num_candidates - 1) * 2
        
        # Draw candidates for all users at once
        # We draw indices 0 to n_movies-1
        rng = np.random.default_rng(self.seed)
        approx_candidate_indices = rng.integers(0, n_movies,  size=(n_users, n_approx_candidates))
        approx_candidate_movie_ids = self.all_movie_ids[approx_candidate_indices]
        
        # Vectorized Collision Check
        # We compare: (n_users, n_candidates, 1) == (n_users, 1, n_forbidden)
        # This creates a boolean mask of shape (n_users, n_candidates, n_forbidden)
        # Then we check if ANY forbidden match exists for each candidate
        is_forbidden = np.any(approx_candidate_movie_ids[:, :, np.newaxis] == exclude[:, np.newaxis, :], axis=2)
        
        # Filter and Ranking Trick
        # Assign random noise to all candidates, but penalize forbidden ones
        noise = rng.random(np.shape(approx_candidate_movie_ids))
        noise[is_forbidden] = -1.0
        
        # Sort to bring valid candidates (high noise) to the front
        shuffled_idx = np.argsort(noise, axis=1)[:, ::-1]
        
        # 5. Extract and Truncate
        row_grid = np.arange(n_users)[:, np.newaxis]
        total_negatives = approx_candidate_movie_ids[row_grid, shuffled_idx[:, :self.num_candidates - 1]]
        
        hard_mask = (hard_negative_movie_ids != -1)
        # Overwrite the first n_hard columns of our random matrix
        # with the hard negatives where they are valid
        total_negatives[:, :n_hard][hard_mask] = hard_negative_movie_ids[hard_mask]
        
        valid_mask = noise[row_grid, shuffled_idx[:, :self.num_candidates - 1]] != -1.0
        # Only invalidate slots that weren't part of the hard_mask overwrite
        total_negatives[~valid_mask & ~np.pad(hard_mask, ((0, 0), (0, n_approx)))] = -1
        
        candidate_ids = np.hstack([batch['movie_id'][:,np.newaxis], total_negatives])
        labels = np.hstack([np.ones((n_users,1), dtype=np.float32), np.zeros((n_users, self.num_candidates-1), dtype=np.float32)])
        
        #shuffle the labels and candidate_ids order, together
        perm_idx = np.argsort(rng.random((n_users, self.num_candidates)), axis=1)
        row_grid = np.arange(n_users)[:, np.newaxis]
        
        return {
            **batch,
            "candidate_ids": candidate_ids[row_grid, perm_idx],
            "labels": labels[row_grid, perm_idx],
        }
        