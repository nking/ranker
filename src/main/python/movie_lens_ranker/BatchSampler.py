import grain
import numpy as np

class BatchSampler(grain.samplers.IndexSampler):
    def __init__(self, num_records: int, num_epochs:int, batch_size: int,
            shuffle:bool=False, seed:int=0,
            shard_options: grain.sharding.ShardOptions = grain.sharding.NoSharding()):
        self.num_batches_per_epoch = max(1, num_records // batch_size)
        self.total_records = num_records
        self.batch_size = batch_size
        super().__init__(num_records=self.num_batches_per_epoch, shard_options=shard_options,
            shuffle=shuffle, seed=seed, num_epochs=num_epochs)
        
    # override
    def __repr__(self) -> str:
        return (
            f"BatchSampler(num_records={self.total_records}, num_epochs={self._num_epochs},"
            f"batch_size={self.batch_size}, shuffle={self._shuffle}, seed={self._seed}"
            f"shard_options={self._shard_options!r})"
        )
    
    def __getitem__(self, index: int) -> grain.RecordMetadata:
        # Because you initialized super() with self.num_batches_per_epoch,
        # base_meta.record_key is ALREADY your deterministically shuffled block index.
        base_meta = super().__getitem__(index)
        shuffled_block_idx = base_meta.record_key
        
        # Calculate the item boundaries for this specific batch block
        start = shuffled_block_idx * self.batch_size
        stop = min(start + self.batch_size, self.total_records)
        indices = np.arange(start, stop)
        
        if self._shuffle:
            # We don't need to spawn. We just consume one state advancement
            # of this unique Generator.
            base_meta.rng.shuffle(indices)
            
            # 4. Return the record, passing the advanced RNG down the pipeline
        return grain.RecordMetadata(
            index=base_meta.index,
            record_key=indices.tolist(),
            rng=base_meta.rng  # <--- Safe, deterministic, and pipeline-ready
        )
    
    # override
    def __getitem_previous__(self, index: int) -> grain.RecordMetadata:
        epoch = index // self.num_batches_per_epoch
        batch_in_epoch = index % self.num_batches_per_epoch
        
        if self._shuffle:
            # Deterministically find the shuffled block index for this epoch
            # Note: For massive datasets, consider a more efficient
            # stateless shuffle than np.random.permutation
            epoch_rng = np.random.default_rng(self._seed + epoch)
            shuffled_block_idx = epoch_rng.permutation(self.num_batches_per_epoch)[batch_in_epoch]
        else:
            shuffled_block_idx = batch_in_epoch
        
        start = shuffled_block_idx * self.batch_size
        stop = min(start + self.batch_size, self.total_records)
        indices = np.arange(start, stop)
        
        # 3. Optional: Shuffle the items WITHIN the batch
        if self._shuffle:
            # Use the UNIQUE global index to ensure no collisions
            intra_batch_rng = np.random.default_rng(self._seed + index)
            intra_batch_rng.shuffle(indices)
        
        return grain.RecordMetadata(
            index=index,
            record_key=indices.tolist(),
            rng=None # We handle RNG manually for reproducibility
        )