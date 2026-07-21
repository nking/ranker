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

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        # (user_id, (tuple of negative movie_ids))
        self.negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/ratings_train_disliked.array_record")
        
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record")
        
        self.ratings_train_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/ratings_train_liked.array_record")
        
        self.ratings_train_uri_tiny = os.path.join(get_project_dir(),
            "src/test/resources/data/small/ratings_train_liked.array_record")
        
        self.ratings_test_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/ratings_test_liked.array_record")
        
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        
        self.user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb-00000-of-00001.array_record")
        
        self.movie_ids_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")

    def test_read_embeddings(self):
        
        if fake_gcs_server_is_alive():
            gs_uri = "gs://data/movie_emb-00000-of-00001.array_record"
            emb = _read_embeddings(gs_uri, batch_size=1024)
            self.assertTrue(emb is not None)
            self.assertEqual(3883, len(emb))
            self.assertTrue(isinstance(emb, jnp.ndarray))
        
        emb = _read_embeddings(self.user_embeddings_uri, batch_size=1024)
        self.assertTrue(emb is not None)
        self.assertEqual(6040, len(emb))
        self.assertTrue(isinstance(emb, jnp.ndarray))
        
        emb = _read_embeddings(self.movie_embeddings_uri,
            batch_size=1024)
        self.assertTrue(emb is not None)
        self.assertEqual(3883, len(emb))
        self.assertTrue(isinstance(emb, jnp.ndarray))
        
        embeddings = read_user_movie_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
        self.assertTrue(isinstance(embeddings, jnp.ndarray))
        self.assertEqual(len(embeddings), 6040 + 3883 + 1)
    
    def test_read_ratings_array_records(self):
        """
        comparing the speed of iterating over the 800,000 items of ratings_part_1 using different
        libraries and data structures.
        
        this shows that array_record_module.ArrayRecordReader is best
        
        type                                         avg record read time
        ---------------------------------------      ----------------------
        array_record_module.ArrayRecordReader        batched: 0.0002 sec
                                                     sequentially: 0.0035 sec
        
        grain.sources.ArrayRecordDataSource          batched: ?? needs write of array_record to use group_size:batch_size???
                                                     batched: 0.008
                                                     sequentially:  0.008 sec
                                                     
        grain.python.DataLoader                      batched:  through transform:
                                                               0.008 sec
                                                     single:  0.000132 sec
        
        RandomAccessArrayRecordDataSource            batched:  0.00005 to 0.00009 sec
                                                     sequential:  0.019508 sec
                                                     
        DataLoader using BatchSampler
        and RandomAccessArrayRecordDataSource        batched:  0.0002 sec

        :return:
        """
        # ====   how to use all of that ======
        max_history = 20
        num_candidates = 20
        batch_size = 64
        worker_count = 0  # max(1, os.cpu_count() - 1)
        
        # this fast return shows it is indexed
        reader = array_record_module.ArrayRecordReader(self.ratings_train_uri)
        # print(f"Total records: {reader.num_records()}")
        
        count = 0
        start_time = time.perf_counter()
        batch_bytes = reader.read([x for x in range(0,
            batch_size)])  # a single list of encodings, each being a list of 4 integers
        data = [msgpack.unpackb(b, use_list=False) for b in
            batch_bytes]  # list of tuples of 4 integers
        stop_time = time.perf_counter()
        print(
            f'avg time to read ArrayRecordReader item (batched) = {(stop_time - start_time) / batch_size:.6f} sec')
        start_time = time.perf_counter()
        for ii in range(batch_size):
            count += 1
            b = reader.read([ii])
            msgpack.unpackb(b[0], use_list=False)  # is a tuple of 4 integers
        stop_time = time.perf_counter()
        print(
            f'avg time to read ArrayRecordReader item (sequentially) = {(stop_time - start_time) / batch_size:.6f} sec')
        # this is 0.000183 sec which is 50 times faster than using grain.python.DataLoader
        reader.close()
        # ----------------------------------
        ratings_train_data_source = grain.sources.ArrayRecordDataSource(
            self.ratings_train_uri
        )
        # print(f"Number of records: {len(ratings_train_data_source)}")
        # print(ratings_train_data_source[100])
        # print(msgpack.unpackb(ratings_train_data_source[10000], raw=False))
        # {'user_id': 4887, 'movie_id': 377, 'rating': 4, 'timestamp': 962739544,
        dataset = grain.MapDataset.source(ratings_train_data_source)
        dataset = dataset.map(lambda record: msgpack.unpackb(record, raw=False,
            use_list=False)).batch(batch_size)
        count = 0
        start_time = time.perf_counter()
        for batch in dataset:
            count += 1
            break
        stop_time = time.perf_counter()
        print(
            f'avg time to read {batch_size} grain.MapDataset.source item (batched) = {(stop_time - start_time) / batch_size:.6f} sec')
        dataset = grain.MapDataset.source(ratings_train_data_source)
        dataset = dataset.map(
            lambda record: msgpack.unpackb(record, raw=False, use_list=False))
        count = 0
        start_time = time.perf_counter()
        for item in dataset:
            count += 1
            if count == batch_size:
                break
        stop_time = time.perf_counter()
        print(
            f'avg time to read {batch_size} grain.MapDataset.source item (sequentially) = {(stop_time - start_time) / batch_size:.6f} sec')
        
        # ----------------
        datasource = RandomAccessArrayRecordDataSource[MovieRating](
            self.ratings_train_uri)
        count = 0
        start_time = time.perf_counter()
        for item in datasource:  # item is a tuple
            count += 1
            if count == batch_size:
                break
        stop_time = time.perf_counter()
        print(
            f'avg time to read a RandomAccessArrayRecordDataSource item (sequentially) = {(stop_time - start_time) / batch_size:.6f} sec')
        
        count = 0
        start_time = time.perf_counter()
        batch = datasource.__getitems__(
            [x for x in range(100, 100 + batch_size)])  # is a list of tuples
        stop_time = time.perf_counter()
        print(
            f'avg time to read a RandomAccessArrayRecordDataSource item (batched) = {(stop_time - start_time) / batch_size:.6f} sec')
        # -------------------------------------------
        
        # batch_size=2
        num_epochs = 1
        # 8 records
        # datasource = RandomAccessArrayRecordDataSource(self.ratings_train_uri_tiny)
        
        datasource = RandomAccessArrayRecordDataSource(self.ratings_train_uri)
        
        shard_opts = grain.sharding.ShardOptions(shard_index=0, shard_count=1)
        
        ra_sampler = BatchSampler(num_records=datasource.__len__(),
            batch_size=batch_size, shuffle=True, seed=0, num_epochs=num_epochs,
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
        c = 10
        start_time = time.perf_counter()
        for batch in dataloader0:  # batch is a tuple of lists of the 4 datums: ([user_ids],[movie_ids],[ratings],[timestamps])
            count += 1
            if count == c:
                break
        stop_time = time.perf_counter()
        print(
            f'avg time to read a dataloader item (batched) = {(stop_time - start_time) / (count * batch_size):.6f} sec')
    
    def test_build_history_lookup(self):

        batch_size = 1024
        # expecting Dict[int, Tuple[list, list, list]]
        history_dict, max_history = build_history_lookup(
            self.ratings_train_uri, batch_size=batch_size)
        self.assertTrue(isinstance(history_dict, dict))
        n_hist = len(
            history_dict)  # number of users who rated movies in train dataset
        self.assertTrue(n_hist > 0 and n_hist < 6040)
        min_user_id = min(history_dict.keys())
        entry_tuples = history_dict[min_user_id]
        self.assertEqual(3, len(entry_tuples))
        for i in range(3):
            self.assertTrue(isinstance(entry_tuples[i], list))
            self.assertTrue(len(entry_tuples[i]) > 0)
            self.assertTrue(isinstance(entry_tuples[i][0], int))
    
    def test_read_array_records2(self):
        batch_size = 1024
        
        user_movie_embeddings = read_user_movie_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
        self.assertIsNotNone(user_movie_embeddings)
        self.assertTrue(len(user_movie_embeddings) > 100)
        
        all_movie_ids: List[int] = read_movies_array_record(
            self.movie_ids_uri, batch_size=batch_size)
        self.assertEqual(len(all_movie_ids), 3883)
        self.assertTrue(isinstance(all_movie_ids, list))
        self.assertTrue(isinstance(all_movie_ids[0], int))
        
        recommendations: np.ndarray = read_recommendations(self.recommendations_uri, batch_size=batch_size)
        self.assertTrue(isinstance(recommendations, np.ndarray))
        min_user_id = 1
        entry = recommendations[min_user_id - 1]
        self.assertTrue(isinstance(entry, np.ndarray))
        self.assertTrue(isinstance(entry[0], np.int64))

if __name__ == '__main__':
    unittest.main()
