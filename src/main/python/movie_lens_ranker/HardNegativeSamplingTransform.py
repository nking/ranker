from typing import Dict, Tuple, Union, List, Set
import numpy as np
import grain.python as pgrain
import msgpack
from array_record.python import array_record_module

#TODO: refactor to use np.memmap if needed


class HardNegativeSamplingTransform(pgrain.MapTransform):
    """
    class to map a user's local history to the same local history enriched with negative sampling
    as "candidate_ids" and "labels"
    """
    def __init__(self, history_lookup: Dict[int, Tuple[list, list, list]],
        all_movie_ids:List[int], exact_negatives_dict:Dict[int, Set[int]],
        unseen_recommendations:Dict[int, Set[int]], num_candidates=20,
        seed:int = 0):
        """
        initialize a CandidateSamplingTransform object
        :param history_lookup:  Dict[user_id:int, Tuple(arrays of ts, movie_id, rating)]
        :param all_movie_ids: list of all movie_ids
        :param exact_negatives_dict: dictionary with key=user_id, value = set of exact negative movie ids
        :param unseen_recommendations: dictionary with key=user_id, value = set of unseen recommended movie ids
        :param num_candidates: total number of candidates to create from 1 postive and mulitple negatives
        :param seed: seed for random number generator
        """
        self.history_lookup = history_lookup
        self.exact_negatives_dict = exact_negatives_dict
        self.all_movie_ids = all_movie_ids #to o use in approx hard negatives
        self.unseen_recommendations = unseen_recommendations # to use in approx hard negatives
        self.num_candidates = num_candidates
        self.seed = seed

    def map(self, batch:List[Dict[str, Union[int, List[int]]]]) -> List[Dict[str, Union[int, List[int], np.ndarray]]]:
        """
        given the current user history, add a hard negative mining list as "column_ids" and "labels"
        :param batch: list of dictionaries containing
            'user_id':int
            'movie_id':int,
            'rating': int,
            'timestamp': int,
            "history_movie_ids": list,
            "history_ratings": list,
            "history_length": int
        :return: list of dictionaries containing
            'user_id':int
            'movie_id':int,
            'rating': int,
            'timestamp': int,
            "history_movie_ids": list,
            "history_ratings": list,
            "history_length": int
            "candidate_ids": np.ndarray,
            "labels": np.ndarray
        """
        results = []
        for record in batch:
            user_id = record["user_id"]
            pos_id = record["movie_id"]
            
            per_record_rng = np.random.default_rng(self.seed + user_id)
            
            # Get Hard Negatives (from Retrieval model)
            hard_negs = self.exact_negatives_dict.get(user_id, [])
            hard_negs = [m for m in hard_negs if m != pos_id]
            
            n_approx = self.num_candidates//2
            n_hard = self.num_candidates - 1 - n_approx
            if len(hard_negs) < n_hard:
                n_hard = len(hard_negs)
                n_approx = self.num_candidates - 1 - n_hard
            elif len(hard_negs) > n_hard:
                hard_negs = per_record_rng.choice(hard_negs, size=n_hard, replace=False).tolist()
    
            #choose approx negatives from "all movies - pos_id - has_seen - was recommended"
            subtr = {pos_id}
            if self.history_lookup.get(user_id):
                #tuple (timestamps, movie_ids, ratings)
                subtr.update(self.history_lookup.get(user_id)[1])
            if self.unseen_recommendations.get(user_id):
                subtr.update(self.unseen_recommendations.get(user_id))
            
            approx_negs = per_record_rng.choice(self.all_movie_ids, size=n_approx + len(subtr), replace=False)
            approx_negs = [int(x) for x in approx_negs if x not in subtr]
            
            candidate_ids = np.array([pos_id] + hard_negs + approx_negs[:n_approx], dtype=np.int32)
            
            # Create Labels (1.0 for the first one, 0.0 for the rest)
            labels = np.zeros((self.num_candidates,), dtype=np.float32)
            labels[0] = 1.0
            
            # Shuffle the candidates and labels together!
            # (Otherwise the model learns "index 0 is always the winner")
            p = np.random.permutation(self.num_candidates)
            
            results.append({
                **record,
                "candidate_ids": candidate_ids[p],
                "labels": labels[p]
            })
        return results