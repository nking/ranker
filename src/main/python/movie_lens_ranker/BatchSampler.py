from typing import Optional

import grain
import numpy as np

class BatchSampler(grain.samplers.IndexSampler):
    def __init__(self, num_records: int, num_epochs:int, batch_size: int,
            shuffle:bool=False, seed:int=0,
            shard_options: grain.sharding.ShardOptions = grain.sharding.NoSharding()):
        super().__init__(num_records=num_records, shard_options=shard_options,
            shuffle=shuffle, seed=seed, num_epochs=num_epochs)
        self.batch_size = batch_size
        self.num_batches = max(1, self._num_records // batch_size)
        
    # override
    def __repr__(self) -> str:
        return (
            f"BatchSampler(num_records={self._num_records}, "
            f"batch_size={self.batch_size}, shuffle={self.shuffle}, "
            f"shard_options={self._shard_options!r})"
        )
    
    # override
    def __getitem__(self, index: int) -> grain.RecordMetadata:
        epoch = index // self.num_batches
        batch_in_epoch = index % self.num_batches
        #ensure check-pointing succeeds:
        pair_rng = np.random.default_rng(self._seed + epoch + batch_in_epoch)
        
        epoch_rng = np.random.default_rng(self._seed + epoch)
        permutation = epoch_rng.permutation(self.num_batches)
        shuffled_block_idx = permutation[batch_in_epoch]
        
        start = shuffled_block_idx * self.batch_size
        stop = min(start + self.batch_size, self._num_records)
        indices = np.arange(start, stop)
        
        if self._shuffle:
            pair_rng.shuffle(indices)
            
        return grain.RecordMetadata(
            index=index,
            record_key=indices.tolist(),
            rng=None # We handle RNG manually for reproducibility
        )