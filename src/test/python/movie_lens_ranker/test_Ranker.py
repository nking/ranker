import collections
import os.path
import unittest
from typing import Tuple, Union, Dict, Sequence, TypeVar, Generic, \
    Iterator

import grain
import grain.python as pgrain
import msgpack
import numpy as np
from collections import defaultdict
import time

from array_record.python import array_record_module
import jraph
import jax
import jax.numpy as jnp
from helper import *
from movie_lens_ranker.BatchSampler import BatchSampler
from movie_lens_ranker.RandomAccessArrayRecordDataSource import *
from movie_lens_ranker.RatingsHistoryLookupTransform import *
from movie_lens_ranker.HardNegativeSamplingTransform import *
from movie_lens_ranker.SparseLocalSubgraphTransform import \
    SparseLocalSubgraphTransform


class TestRanker(unittest.TestCase):
    def setUp(self):
        # each item is {'user_id':int, 'retrieved_ids':List[int]}
        self.exact_hard_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/user_recommendations_disliked_in_train.array_record")
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/user_recommendations_without_train_val.array_record")
        
        self.ratings_train_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_1.array_record")
        
        self.ratings_test_uri = os.path.join(get_project_dir(),
            "src/test/resources/ratings_part_2.array_record")
        
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/movie_embeddings.array_record")
        
        self.user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/user_embeddings.array_record")
        
        self.movie_ids_uri = os.path.join(get_project_dir(),
            "src/test/resources/movie_ids.array_record")
        
        self.unseen_recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/user_recommendations_without_train_val.array_record")
        
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
        worker_count = 0#max(1, os.cpu_count() - 1)
        
        #this fast return shows it is indexed
        reader = array_record_module.ArrayRecordReader(self.ratings_train_uri)
        # print(f"Total records: {reader.num_records()}")
        
        count = 0
        start_time = time.perf_counter()
        batch_bytes = reader.read([x for x in range(0, batch_size)]) # a single list of encodings, each being a list of 4 integers
        data = [msgpack.unpackb(b, use_list=False) for b in batch_bytes] # list of tuples of 4 integers
        stop_time = time.perf_counter()
        print( f'avg time to read ArrayRecordReader item (batched) = {(stop_time - start_time)/batch_size:.6f} sec')
        start_time = time.perf_counter()
        for ii in range(batch_size):
            count += 1
            b = reader.read([ii])
            msgpack.unpackb(b[0], use_list=False) #is a tuple of 4 integers
        stop_time = time.perf_counter()
        print(f'avg time to read ArrayRecordReader item (sequentially) = {(stop_time - start_time)/batch_size:.6f} sec')
        #this is 0.000183 sec which is 50 times faster than using grain.python.DataLoader
        reader.close()
        # ----------------------------------
        ratings_train_data_source = grain.sources.ArrayRecordDataSource(
            self.ratings_train_uri
        )
        #print(f"Number of records: {len(ratings_train_data_source)}")
        #print(ratings_train_data_source[100])
        #print(msgpack.unpackb(ratings_train_data_source[10000], raw=False))
        # {'user_id': 4887, 'movie_id': 377, 'rating': 4, 'timestamp': 962739544,
        dataset = grain.MapDataset.source(ratings_train_data_source)
        dataset = dataset.map( lambda record: msgpack.unpackb(record, raw=False, use_list=False)).batch(batch_size)
        count = 0
        start_time = time.perf_counter()
        for batch in dataset:
            count += 1
            break
        stop_time = time.perf_counter()
        print(f'avg time to read {batch_size} source item (batched) = {(stop_time - start_time) / batch_size:.6f} sec')
        dataset = grain.MapDataset.source(ratings_train_data_source)
        dataset = dataset.map(lambda record: msgpack.unpackb(record, raw=False, use_list=False))
        count = 0
        start_time = time.perf_counter()
        for item in dataset:
            count += 1
            if count == batch_size:
                break
        stop_time = time.perf_counter()
        print(f'avg time to read {batch_size} source item (sequentially) = {(stop_time - start_time)/batch_size:.6f} sec')
        
        # ----------------
        datasource = RandomAccessArrayRecordDataSource[MovieRating](self.ratings_train_uri)
        count = 0
        start_time = time.perf_counter()
        for item in datasource: #item is a tuple
            count += 1
            if count == batch_size:
                break
        stop_time = time.perf_counter()
        print(f'avg time to read a RandomAccessArrayRecordDataSource item (sequentially) = {(stop_time - start_time) / batch_size:.6f} sec')
        
        count = 0
        start_time = time.perf_counter()
        batch = datasource.__getitems__([x for x in range(100, 100 + batch_size)]) # is a list of tuples
        stop_time = time.perf_counter()
        print(f'avg time to read a RandomAccessArrayRecordDataSource item (batched) = {(stop_time - start_time) / batch_size:.6f} sec')
        
        # -------------------------------------------
        
        datasource = RandomAccessArrayRecordDataSource(self.ratings_train_uri)
        shard_opts = grain.python.ShardOptions(shard_index=0,shard_count=1)
        
        ra_sampler = BatchSampler(num_records=datasource.__len__(),
            batch_size=batch_size, shuffle=True, shard_options=shard_opts)
       
        dataloader0 = grain.python.DataLoader(
            data_source=datasource,
            sampler=ra_sampler,
            operations=[
                #grain.python.Batch(batch_size=batch_size),  #do not use the batch transform, instead use batch in sampling
                #grain.python.BatchShuffle(seed=42, buffer_size=10000),
            ],
            #worker_count=1,
            #worker_buffer_size=batch_size,
            shard_options=shard_opts,
            #read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=0)
        )
        
        count = 0
        c = 10
        start_time = time.perf_counter()
        for batch in dataloader0:  #batch is a tuple of lists of the 4 datums: ([user_ids],[movie_ids],[ratings],[timestamps])
            count += 1
            if count == c:
                break
        stop_time = time.perf_counter()
        print(f'avg time to read a dataloader item (batched) = {(stop_time - start_time)/(c*batch_size):.6f} sec')
        
    def test_load_ratings(self):
        
        def get_jax_dataset(grain_dataloader:grain.python.DataLoader, max_nodes:int, max_edges:int, max_graphs:int):
            """
            A generator that yields padded, JAX-ready SuperGraphs.
            """
            for batch in grain_dataloader:
                # 'batch' is a list of GraphsTuples from Grain
                # Convert list of graphs -> 1 big SuperGraph
                super_graph = jraph.batch(batch)
                
                # Pad to static shapes to prevent XLA recompilation
                # n_node: total nodes across all users in batch
                # n_edge: total edges across all users in batch
                # n_graph: total number of subgraphs (usually batch_size + 1 for padding graph)
                padded_graph = jraph.pad_with_graphs(
                    super_graph,
                    n_node=max_nodes,
                    n_edge=max_edges,
                    n_graph=max_graphs
                )
                
                #  Move to GPU (Device)
                yield jax.tree_util.tree_map(jnp.array, padded_graph)

        def calc_number_jax_graph_components(batch_size:int, max_history:int, num_candidates:int):
            def round_up_10(x)-> int:
                p = jnp.floor(jnp.log10(1234))
                x = jnp.round(x / jnp.pow(10, p)) + 1
                return x * jnp.power(10, p)
            max_nodes = round_up_10(batch_size * (1 + max_history + num_candidates))
            max_edges = round_up_10(batch_size * (max_history + num_candidates))
            max_graphs = batch_size  + 1
            return {'max_nodes': max_nodes, 'max_edges': max_edges, 'max_graphs': max_graphs}
        
        # ====   how to use all of that ======
        
        max_history = 20
        num_candidates = 20
        batch_size = 1024
        worker_count = 0#max(1, os.cpu_count() - 1)
        
        #each worker will have its own copy of these:
        history_dict = build_history_lookup(self.ratings_train_uri, batch_size=batch_size)
        user_exact_negatives = read_user_exact_negatives(self.exact_hard_negatives_uri, batch_size=batch_size)
        all_movie_ids = read_movies_array_record(self.movie_ids_uri, batch_size=batch_size)
        unseen_recommendations = read_user_unseen_recommendations(self.unseen_recommendations_uri, batch_size=batch_size)
        
        datasource = RandomAccessArrayRecordDataSource(self.ratings_train_uri)
        shard_opts = grain.python.ShardOptions(shard_index=0,shard_count=1)
        ra_sampler = BatchSampler(num_records=datasource.__len__(),
            batch_size=batch_size, shuffle=True, shard_options=shard_opts)
        
        #NOTE that history_dict, etc are passed by reference to the MapTransforms
        train_dataloader = grain.python.DataLoader(
            data_source=datasource,
            sampler=ra_sampler,
            operations=[
                #enrich the train records with local subgraphs:
                RatingsHistoryLookupTransform(history_lookup=history_dict, max_history=max_history),
                HardNegativeSamplingTransform(history_lookup=history_dict, all_movie_ids=all_movie_ids,
                    exact_negatives_dict=user_exact_negatives,
                    unseen_recommendations=unseen_recommendations, num_candidates=num_candidates),
                SparseLocalSubgraphTransform(),
            ],
            worker_count=worker_count,
            shard_options=shard_opts
        )
        
        train_batch = next(iter(train_dataloader))
        
        jax_size_dict = calc_number_jax_graph_components(batch_size, max_history, num_candidates)
        
        jax_train_ds = get_jax_dataset(train_dataloader,
            max_nodes=jax_size_dict['max_nodes'],
            max_edges=jax_size_dict['max_edges'],
            max_graphs=jax_size_dict['max_graphs']
        )
        print(f'have jax dataset')
        
        
    def test_load_embeddings(self):
        movie_emb_data_source = grain.sources.ArrayRecordDataSource(self.movie_embeddings_uri)
        user_emb_data_source = grain.sources.ArrayRecordDataSource(self.user_embeddings_uri)

    def test_make_hard_negatives(self):
        exact_hard_neg = grain.sources.ArrayRecordDataSource(self.exact_hard_negatives_uri)
        rec_neg = grain.sources.ArrayRecordDataSource(self.recommendations_uri)
        
        
    if __name__ == '__main__':
        unittest.main()
