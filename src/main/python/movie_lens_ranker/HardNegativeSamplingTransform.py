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
        Note that candidate_ids is guaranteed to not have padding values, they're all real movie_ids
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
        n_negs = self.num_candidates - 1  # Total negatives needed per user
        rng = np.random.default_rng(self.seed)
        
        # PREPARE EXCLUSIONS
        # Get history to ensure approx negatives are actually "unseen"
        movie_histories = self.history_lookup.get_movieids_before_timestamp(
            user_id=batch['user_id'], timestamp=batch['timestamp'],
            max_hist=self.history_lookup.fixed_size)
        exclude = np.hstack([movie_histories, batch['movie_id'][:, np.newaxis]])
        
        # CREATE THE "BASE" APPROXIMATE POOL
        # We draw a surplus to ensure we can fill all n_negs slots after filtering
        n_draw = n_negs * 3
        approx_indices = rng.integers(0, len(self.all_movie_ids), size=(n_users, n_draw))
        approx_candidates = self.all_movie_ids[approx_indices]
        
        # Collision Check: (n_users, n_draw, 1) == (n_users, 1, n_forbidden)
        is_forbidden = np.any(
            approx_candidates[:, :, np.newaxis] == exclude[:, np.newaxis, :], axis=2)
        
        # Sort by noise, pushing forbidden items (noise = -1) to the back
        noise = rng.random(approx_candidates.shape)
        noise[is_forbidden] = -1.0
        shuffled_idx = np.argsort(noise, axis=1)[:, ::-1]
        
        # Initialize the negatives_pool with ONLY valid approximate negatives
        row_grid = np.arange(n_users)[:, np.newaxis]
        negatives_pool = approx_candidates[row_grid, shuffled_idx[:, :n_negs]]
        
        # OVERWRITE WITH HARD NEGATIVES
        # Fetch hard negatives
        hard_negs = self.negatives.get_negatives(user_id=batch['user_id'], length=self.n_hard, seed=self.seed)
        
        # Vectorized overwrite: Only replace the approx negative if the hard negative is valid (!= -1)
        # We only look at the first self.n_hard slots of our pool
        is_valid_hard = (hard_negs != -1)
        negatives_pool[:, :self.n_hard] = np.where(is_valid_hard, hard_negs, negatives_pool[:, :self.n_hard])
        
        # 4. FINAL ASSEMBLY
        # Stack: [Positive] + [Negatives Pool (Hard + Approx)]
        candidate_ids = np.hstack([
            batch['movie_id'][:, np.newaxis],
            negatives_pool
        ])
        
        # ULTIMATE SAFETY VALVE
        # In the nearly impossible case a user saw the entire catalog,
        # replace any remaining -1s with a truly random draw
        if (candidate_ids == -1).any():
            mask = (candidate_ids == -1)
            candidate_ids[mask] = rng.choice(self.all_movie_ids, size=np.sum(mask))
        
        # LABELS AND SHUFFLE
        labels = np.zeros((n_users, self.num_candidates), dtype=np.float32)
        labels[:, 0] = 1.0  # Positive is at index 0
        
        # Shuffle labels and candidates together
        perm_idx = np.argsort(rng.random((n_users, self.num_candidates)), axis=1)
        final_candidates = candidate_ids[row_grid, perm_idx]
        final_labels = labels[row_grid, perm_idx]
        
        return {
            **batch,
            "candidate_ids": final_candidates,
            "labels": final_labels
        }