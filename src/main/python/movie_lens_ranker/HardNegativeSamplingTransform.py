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
        self.n_approx = self.num_candidates // 2
        self.n_hard = self.num_candidates - 1 - self.n_approx

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
    
        n_users = batch['user_id'].shape[0]
        
        rng = np.random.default_rng(self.seed)
        
        # (1): n_hard number of hard negatives and natural negatives
        hard_negative_movie_ids = self.negatives.get_negatives(user_id=batch['user_id'],
            length=self.n_hard, seed=self.seed)
        # where there are not enough hard negatives, supplement them
        # with the unseen recommended.
        counts = np.sum(hard_negative_movie_ids != -1, axis=1, keepdims=True)
        indices = np.arange(self.n_hard)  # (n_hard,)
        is_padding = indices >= counts
        fetch_indices = np.maximum(0, indices - counts)
        #Calculate the 'Fetch Indices' for the recommendations
        # For a row with 3 valid negs, we want to fetch rec indices 0, 1, 2...
        # for columns 3, 4, 5...
        # fetch_indices = [0, 0, 0, 0, 1, 2, 3...] (clamped to 0 for non-padding)
        movies_recommended_unseen = self.recommendations.get_unseen_movies(
            user_id=batch['user_id'], timestamp=batch['timestamp'], top_k=self.n_hard)
        filler_values = np.take_along_axis(movies_recommended_unseen, fetch_indices, axis=1)
        hard_negatives_pool = np.where(is_padding, filler_values, hard_negative_movie_ids)
        
        # (2) n_approx negatives are chosen randomly from the full movie catalog minus the exclude list:
        
        #these are to be excluded from a random selection of self.all_movie_ids to choose approx_negatives
        movie_histories_long = self.history_lookup.get_movieids_before_timestamp(
            user_id=batch['user_id'], timestamp=batch['timestamp'],
            max_hist=self.history_lookup.fixed_size)
        
        #if user has watched and rated the entire catalog, and if UserData limits are same as
        # entire catalog length, then there is a possibility that exclude is the entire catalog.
        #  in the movie-lens 1m dataset, the largest user history is 2000 something, which is
        #  roughly half the movies.dat catalog.
        #  the result below is that noise is all -1, so the shuffle of indexes is ineffective
        #  and a set of n_draw from approx_candidates gets used for this special case.
        exclude = np.hstack([movie_histories_long, batch['movie_id'][:,np.newaxis]])
        
        #final candidate_ids are [pos_id] + hard_negs[:n_hard] + approx_negs[:n_approx]
        # shape (n_users, self.num_candidates)
        # but there will be missing hard_negs for some users
        # so we generate a base layer of shape (n_users, self.num_candidates) filled with approx_negatives,
        # then fill in the smallest indices with hard_negatives where they exist.
        
        n_draw = self.n_approx * 3
        approx_indices = rng.integers(0, len(self.all_movie_ids), size=(n_users, n_draw))
        approx_candidates = self.all_movie_ids[approx_indices]
        is_forbidden = np.any(approx_candidates[:, :, np.newaxis] == exclude[:, np.newaxis, :], axis=2)
        
        # Use a high-noise sort to push forbidden items (noise -1) to the back
        noise = rng.random(approx_candidates.shape)
        noise[is_forbidden] = -1.0
        shuffled_idx = np.argsort(noise, axis=1)[:, ::-1]
        
        # Take ONLY the first n_approx valid ones
        row_grid = np.arange(n_users)[:, np.newaxis]
        approx_negatives_pool = approx_candidates[row_grid, shuffled_idx[:, :self.n_approx]]
        
        #no pad_value in these, so no need to look for later:
        candidate_ids = np.hstack([
            batch['movie_id'][:, np.newaxis],
            hard_negatives_pool,
            approx_negatives_pool
        ])
        
        # In the extreme case a user has seen everything, fill -1 with a random movie
        if (candidate_ids == -1).any():
            mask = (candidate_ids == -1)
            candidate_ids[mask] = rng.choice(self.all_movie_ids, size=np.sum(mask))

        # 5. LABELS AND SHUFFLE
        labels = np.zeros((n_users, self.num_candidates), dtype=np.float32)
        labels[:, 0] = 1.0 # The positive is at index 0 before shuffle
        
        perm_idx = np.argsort(rng.random((n_users, self.num_candidates)), axis=1)
        final_candidates = candidate_ids[row_grid, perm_idx]
        final_labels = labels[row_grid, perm_idx]
        
        return {**batch, "candidate_ids": final_candidates,
            "labels": final_labels}