from typing import Dict, List
import numpy as np
from array_record.python import array_record_module

class CandidateIdTransform():
    """
    class to map a user's local history to the same local history enriched with
    "candidate_ids" and "labels".  This is for use in test to create graphs to compare to
    the rust graphs for inference.
    """
    def __init__(self,  num_candidates=20):
        self.num_candidates = num_candidates

    def map(self, batch:Dict[str, np.ndarray], candidate_ids: np.ndarray) -> Dict[str, np.ndarray]:
        if candidate_ids.shape[1] != self.num_candidates:
            raise ValueError(f"expecting that the number of candidates == {self.num_candidates}, but instead got {candidate_ids.shape[1]}")
        if candidate_ids.shape[0] != batch['movie_id'].shape[0]:
            raise ValueError(f"expecting number of rows in candidate_ids and batch['movie_id'] to be equal")

        labels = np.ones((batch['user_id'].shape[0], self.num_candidates), dtype=np.int32)

        return {
            **batch,
            "candidate_ids": candidate_ids,
            "labels": labels
        }

    