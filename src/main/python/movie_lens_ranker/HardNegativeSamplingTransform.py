from typing import Dict, List
import numpy as np
import grain.python as pgrain
from array_record.python import array_record_module

from movie_lens_ranker.RecommendedMovies import RecommendedMovies

from movie_lens_ranker.UserHistory import UserHistory
from movie_lens_ranker.util_numba import row_wise_intersect, \
    row_wise_sortedset_subtract, generate_type_4_negatives, build_negative_pool_numba, \
    simultaneous_shuffle

class HardNegativeSamplingTransform(pgrain.RandomMapTransform):
    """
    class to map a user's local history to the same local history enriched with negative sampling
    as "candidate_ids" and "labels"
    """
    def __init__(self, history_lookup : UserHistory, history_lookup_disliked : UserHistory,
            all_movie_ids:List[int],
            recommendations:RecommendedMovies, num_candidates=20):
        """
        initialize a CandidateSamplingTransform object.
        The negative lists dynamically created in map are composed from 4 types of negatives:
        1) "hard negatives" = recommended intersection with user's disliked.
        These are "False positives".
        2) "implicit hard negatives" = recommended minus users watch history
        3) "out of distr negatives" = disliked - recommended.
        4) "easy negatives" = movie catalog - watch history
        
        For example, given:
            movie catalog is A,B,C,D,E,F,G,H,I,J
            watched = A,B,C,D,G
            disliked = A,B,C,D
            recommended = B,D,F,G,H
        we have for the 4 types:
            1) intersect({B,D,F,G,H}, {A,B,C,D}) = B,D
            2) subtract({B,D,F,G,H}, {A,B,C,D,G}) = F,H
            3) subtract({A,B,C,D}, {B,D,F,G,H}) = A,C
            4) subtract({A,B,C,D,E,F,G,H,I,J}, {A,B,C,D,G}) = E,F,H,I,J
 
        :param history_lookup:  Dict[user_id:int, Tuple(arrays of ts, movie_id, rating)]
        :param history_lookup_disliked:  Dict[user_id:int, Tuple(arrays of ts, movie_id, rating)] for ratings > 3
        :param all_movie_ids: list of all movie_ids
        :param recommendations: class to retrieve unseen movie recommendations for batch of users
        :param num_candidates: total number of candidates to create from 1 positive and multiple negatives
        :param seed: seed for random number generator
        """
        self.history_lookup = history_lookup
        self.history_lookup_disliked = history_lookup_disliked
        self.all_movie_ids = np.asarray(all_movie_ids)
        self.recommendations = recommendations
        self.num_candidates = num_candidates
        
        self.num_1_negatives = round(num_candidates * 0.15)
        self.num_2_negatives = round(num_candidates * 0.55)
        self.num_3_negatives = round(num_candidates * 0.15)
        #self.num_4_negatives = self.num_candidates - (self.num_1_negatives + self.num_2_negatives + self.num_3_negatives)
        
        self.n_approx = self.num_candidates // 2
        self.n_hard = self.num_candidates - 1 - self.n_approx
        
        self.pad_value = self.history_lookup.pad_value
        if self.history_lookup.pad_value != self.pad_value:
            raise ValueError("history_lookup.pad_value must be equal to self.history_lookup.pad_value")
        if self.recommendations.pad_value != self.pad_value:
            raise ValueError("recommendations.pad_value must be equal to self.history_lookup.pad_value")
  
    def random_map(self, batch:Dict[str, np.ndarray], rng:np.random.Generator) -> Dict[str, np.ndarray]:
        """
        given the current user history, add a negative mining list as "column_ids" and "labels"
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
        #       see the code comments for the composition of negatives.
        #       note that timestamps are used in forming these lists.
        #   labels = an array of length self.num_candidates where the first is a 1 and the rest are 0s.
        # the candidate_ids and labels are similarly shuffled to prevent the model from memorizing
        # that the first candidate is correct.
        
        #print(f'HNST batch:{batch}')
    
        n_users = batch['user_id'].shape[0]
        n_negs = self.num_candidates - 1  # Total negatives needed per user
        
        #empty values are self.pad_value which is -1
        movie_histories = self.history_lookup.get_movieids_before_timestamp(
            user_id=batch['user_id'], timestamp=batch['timestamp'],
            max_hist=self.history_lookup.fixed_size)
        
        # empty values are -1
        movie_histories_disliked = self.history_lookup_disliked.get_movieids_before_timestamp(
            user_id=batch['user_id'], timestamp=batch['timestamp'],
            max_hist=self.history_lookup_disliked.fixed_size)
        
        movie_recommendations = self.recommendations.get_unseen_movies(
            user_id=batch['user_id'], timestamp=batch['timestamp'], top_k=n_negs)
        
        # Type 1: "hard negatives" = recommended intersection with user's disliked.
        # highest scoring are at beginning of array. empty values of pad_value are at end of array.
        type_1_neg = row_wise_intersect(movie_histories, movie_histories_disliked, pad_value=self.pad_value)
        
        #Type 2: "implicit hard negatives" = recommended - watch history
        type_2_neg = row_wise_sortedset_subtract(movie_recommendations, movie_histories, pad_value=self.pad_value)
        
        #Type 3: "out of distr negatives" = disliked - recommended
        type_3_neg = row_wise_sortedset_subtract(movie_histories_disliked, movie_recommendations, pad_value=self.pad_value)
        
        #Type 4: "easy negatives" = movie catalog - watch history
        type_4_neg = generate_type_4_negatives(self.all_movie_ids, movie_histories, n_negs=n_negs,
            pad_value=self.pad_value, seed=int(rng.integers(0, 2 ** 31 - 1)))
        
        negatives = build_negative_pool_numba(arr1=type_1_neg, arr2=type_2_neg, arr3=type_3_neg,
                arr4=type_4_neg, target1 = self.num_1_negatives, target2 = self.num_2_negatives,
                target3=self.num_3_negatives, num_negatives=n_negs, pad_value=self.pad_value,
                seed=int(rng.integers(0, 2 ** 31 - 1)))
            
        # Stack: [Positive] + [Negatives Pool]
        candidate_ids = np.hstack([
            batch['movie_id'][:, np.newaxis],
            negatives
        ])
        
        # ULTIMATE SAFETY VALVE
        # In the nearly impossible case a user saw the entire catalog,
        # replace any remaining -1s with a truly random draw
        if (candidate_ids == self.pad_value).any():
            mask = (candidate_ids == self.pad_value)
            candidate_ids[mask] = rng.choice(self.all_movie_ids, size=np.sum(mask))
        
        # LABELS
        labels = np.zeros((n_users, self.num_candidates), dtype=np.float32)
        labels[:, 0] = 1.0  # Positive is at index 0
        
        #shuffle so the model doesn't learn that first label is always right
        simultaneous_shuffle(candidate_ids, labels, seed=int(rng.integers(0, 2 ** 31 - 1)))
        
        return {
            **batch,
            "candidate_ids": candidate_ids,
            "labels": labels
        }
    