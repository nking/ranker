from typing import Optional

import grain
import numpy as np


class BatchSampler(grain.samplers.SequentialSampler):
    """
    a sequential BatchSampler which facilitates batch sequential reads of
    the random access array_record for use with DataLoader.
    """
    def __init__(self, num_records: int, batch_size: int, shuffle:bool=False, seed:Optional[int]=None,
            shard_options: grain.python.ShardOptions = grain.python.NoSharding()):
        super().__init__(num_records, shard_options, seed)
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            if seed is None:
                seed = 0
            self.rng = np.random.default_rng(seed)
    
    # override
    def __repr__(self) -> str:
        return (
            f"BatchSampler(num_records={self._num_records}, "
            f"batch_size={self.batch_size}, shuffle={self.shuffle}, "
            f"shard_options={self._shard_options!r})"
        )
    
    # override
    def __getitem__(self, index: int) -> grain.python.RecordMetadata:
        if index < 0 or index >= self._max_index:
            raise IndexError(
                f"RecordMetadata object index is out of bounds; Got index {index},"
                f" allowed indices should be in [0, {self._max_index}]"
            )
        end = min(index + self.batch_size, self._max_index)
        indices = [x for x in range(index, end)]
        if self.shuffle:
            for i in range(len(indices)-1, 0, -1):
                j = self.rng.integers(0, i) #inclusive so is [0,i]
                if i != j:
                    indices[i], indices[j] = indices[j], indices[i]
        return grain.python.RecordMetadata(index=[index, end],
            record_key=indices, rng=None)