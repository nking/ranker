import os
#=== these are so that grain dataloader can read data from fake gcs server running in docker ====
os.environ["STORAGE_EMULATOR_HOST"] = "http://127.0.0.1:4443"
os.environ["GOOGLE_CLOUD_PROJECT"] = "local-dev"
os.environ["GOOGLE_AUTH_EXTERNAL_ACCOUNT_TOKEN_PROHIBIT"] = "true"

# ==== these in addtion to above, are for orbax to read and write to fake_gcs_Server running in docker ====
# For the C++ GCS client (crucial for performance-heavy libs)
os.environ["CLOUD_STORAGE_EMULATOR_HOST"] = "http://127.0.0.1:4443"
# For TensorStore (Orbax uses this for sharded JAX arrays)
# Some versions of the C++ lib look for this specifically
os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = "http://127.0.0.1:4443"
# Force the library to use HTTP instead of HTTPS
os.environ["STORAGE_EMULATOR_HOST_HTTP"] = "true"
# 4. Disable authentication checks that cause the 'wait'
os.environ["NO_GCE_CHECK"] = "true"
os.environ["GCS_LAMBDA_TOKEN"] = "none"

import os.path
import unittest
import time

from array_record.python import array_record_module
from helper import *
from movie_lens_ranker.util import *
from movie_lens_ranker.util import _read_embeddings
from movie_lens_ranker.RandomAccessArrayRecordDataSource import *
from movie_lens_ranker.data_loading import *
import grain

class TestBatchSampler(unittest.TestCase):
    def setUp(self):
        self.ratings_test_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/ratings_test_liked.array_record")

    def _get_record_0(self):
        reader = array_record_module.ArrayRecordReader(self.ratings_test_uri)
        
        batch_bytes = reader.read([0])
        data = [msgpack.unpackb(b, use_list=False) for b in batch_bytes]  # list of tuples of 4 integers
        reader.close()
        return data
    
    def test_BatchSampler(self):
       
        seed = 0
        batch_size = 100
        num_epochs = 1
        
        datasource = RandomAccessArrayRecordDataSource(self.ratings_test_uri)
        num_records = datasource.__len__()
        print(f'num_records={num_records}', flush=True)
        
        n_batches = num_records // batch_size
        
        #shard_opts = grain.sharding.ShardOptions(shard_index=0, shard_count=1)
        shard_opts = grain.sharding.ShardByJaxProcess()
        
        ra_sampler = BatchSampler(num_records=datasource.__len__(),
            batch_size=batch_size, shuffle=True, seed=seed, num_epochs=num_epochs,
            shard_options=shard_opts)
        
        dataloader0 = grain.DataLoader(
            data_source=datasource,
            sampler=ra_sampler,
            operations=[
                # grain.python.Batch(batch_size=batch_size),  #do not use the batch transform, instead use batch in sampling
                # grain.python.BatchShuffle(seed=42, buffer_size=10000),
            ],
            # worker_count=1,
            worker_buffer_size=0,
            shard_options=shard_opts,
            # read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=0)
        )
        
        count = 0
        for i, batch in enumerate(dataloader0):  # batch is a tuple of lists of the 4 datums: ([user_ids],[movie_ids],[ratings],[timestamps])
            count += 1
            if i == 0:
                first_row_file = self._get_record_0()[0] #[(24, 6505, 4, 978133414)]
                first_row_file = " ".join(map(str, first_row_file))
                first_row_batch = batch[0]
                first_row_batch = " ".join(map(str, first_row_batch))
                print(f'first_row in file={first_row_file}', flush=True)
                print(f'first row in batch={first_row_batch}', flush=True)
                self.assertTrue(first_row_file != first_row_batch)
        self.assertEqual(count, n_batches)
   
if __name__ == '__main__':
    unittest.main()
