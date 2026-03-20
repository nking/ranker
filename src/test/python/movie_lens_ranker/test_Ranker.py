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
        dataset = dataset.map( lambda record: msgpack.unpackb(record, raw=False)).batch(batch_size)
        count = 0
        start_time = time.perf_counter()
        for batch in dataset:
            count += 1
            break
        stop_time = time.perf_counter()
        print(f'avg time to read {batch_size} source item (batched) = {(stop_time - start_time) / batch_size:.6f} sec')
        dataset = grain.MapDataset.source(ratings_train_data_source)
        dataset = dataset.map(lambda record: msgpack.unpackb(record, raw=False))
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
            batch_size=batch_size, shuffle=False, shard_options=shard_opts)
       
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
        c = 1
        start_time = time.perf_counter()
        for batch in dataloader0:  #batch is a tuple of lists of the 4 datums: ([user_ids],[movie_ids],[ratings],[timestamps])
            count += 1
            if count == c:
                break
        stop_time = time.perf_counter()
        print(f'avg time to read a dataloader item (batched) = {(stop_time - start_time)/(c*batch_size):.6f} sec')
        
    def test_load_ratings(self):
        
        def build_history_lookup(ratings_ds:Union[grain.MapDataset, grain.python.DataLoader]) -> defaultdict[Tuple[np.array, np.array, np.array]]:
            """
            Scans the training ratings once to build an O(1) user lookup.
            Arguments:
                ratings_ds: either a grain datasource or grain dataset or raw data, iterable.
            returns: defaultdict of { user_id: (ts, movie_id, rating) }
            """
           
            lookup = defaultdict(lambda: {"ts": [], "movie_id": [], "rating": []})
            
            if isinstance(ratings_ds, grain.MapDataset):
                for record in ratings_ds:
                    u = record[0]
                    lookup[u]["movie_id"].append(record[1])
                    lookup[u]["rating"].append(record[2])
                    lookup[u]["ts"].append(record[3])
            elif isinstance(ratings_ds, grain.python.DataLoader):
                for record in ratings_ds:
                    for i in range(len(record[0])):
                        u = record[0][i].item()
                        lookup[u]["ts"].append(record[3][i].item())
                        lookup[u]["movie_id"].append(record[1][i].item())
                        lookup[u]["rating"].append(record[2][i].item())
            else:
                raise TypeError(f"type of ratings _ds isn't supported {type(ratings_ds)}")
                
            for u in lookup:
                #sort all lists by timestamp
                ts = np.array(lookup[u]["ts"], dtype=np.int64)
                idx = np.array(ts).argsort()
                lookup[u] = (ts[idx], np.array(lookup[u]["movie_id"], dtype=np.int32)[idx],
                    np.array(lookup[u]["rating"], dtype=np.int32)[idx])
                
            return lookup
        
        def build_history_lookup_for_reader(reader: array_record_module.ArrayRecordReader,
            batch_size:int=64) -> defaultdict[Tuple[np.array, np.array, np.array]]:
            """
            Scans the training ratings once to build an O(1) user lookup.
            Arguments:
                ratings_ds: either a grain datasource or grain dataset or raw data, iterable.
            returns: defaultdict of { user_id: (ts, movie_id, rating) }
            """
            
            lookup = defaultdict(
                lambda: {"ts": [], "movie_id": [], "rating": []})
            
            n_records = reader.num_records()
            for i in range(0, n_records, batch_size):
                stop = min(i + batch_size, n_records - 1)
                batch_bytes = reader.read([x for x in range(i, stop)])
                batch = [msgpack.unpackb(b) for b in batch_bytes] #list of dictionaries
                for record in batch:
                    u = record[0]
                    lookup[u]["ts"].append(record[3])
                    lookup[u]["movie_id"].append(record[1])
                    lookup[u]["rating"].append(record[2])
            for u in lookup:
                # sort all lists by timestamp
                ts = np.array(lookup[u]["ts"], dtype=np.int64)
                idx = np.array(ts).argsort()
                lookup[u] = (ts[idx],
                    np.array(lookup[u]["movie_id"], dtype=np.int32)[idx],
                    np.array(lookup[u]["rating"], dtype=np.int32)[idx])
            
            return lookup
        
        class RatingsHistoryLookupTransform(pgrain.MapTransform):
            def __init__(self, history_lookup:defaultdict[Tuple[np.array, np.array, np.array]], max_history:int=20):
                """
                history_lookup: a dictionary with key:'user_id:int and value is a tuple
                   of 2 arrays, the first being np array of sorted timestamps of movies rated
                   by the user (train examples arelady subtracted), and the second array being
                   an np array of the movies correcsponding to the timestamps).
                max_history: Fixed size for the history window (crucial for JAX).
                """
                self.history_lookup = history_lookup
                self.max_history = max_history
            
            def map(self, record):
                """
                map the input train record dictionary to a dictionary containing it and padded history entries
                :param record: a dictionary with keys "user_id", "movie_id", "rating", "timestamp"
                :return: a dictionary containing a copy of the input record and new entries for keys
                "history_movie_ids", "history_ratings", and "history_length" where the movie_ids and ratings
                have been padding with -1 if needed to reach length of self.max_history
                """
                user_id = record["user_id"]
                current_ts = record["timestamp"]
                
                # O(1) Lookup: Get this user's full history arrays
                # If user not found, we use empty arrays
                user_ts, user_movies, user_ratings = self.history_lookup.get(
                    user_id, (np.array([], dtype=np.int64),
                        np.array([], dtype=np.int32), np.array([], dtype=np.int32))
                )
                
                # Temporal Safety: Find index where time < current_ts
                # np.searchsorted finds the insertion point to maintain order
                idx = np.searchsorted(user_ts, current_ts, side='left')
                
                valid_movies_history = user_movies[:idx]
                valid_ratings_history = user_ratings[:idx]
                
                n_hist = len(valid_movies_history)

                # 'max_history' most recent movies
                recent_movies_history = valid_movies_history[-self.max_history:]
                recent_ratings_history = valid_ratings_history[-self.max_history:]

                # 5. JAX-Required Padding
                # We MUST return a fixed shape (e.g., 20) or JAX will crash/recompile
                padded_movies_history = np.full((self.max_history,), -1,  dtype=np.int32)
                padded_movies_history[:len(recent_movies_history)] = recent_movies_history
                padded_ratings_history = np.full((self.max_history,), -1,  dtype=np.int32)
                padded_ratings_history[:len(recent_ratings_history)] = recent_ratings_history

                # Return updated record with the "Context" attached
                return {
                    **record,
                    "history_movie_ids": padded_movies_history,
                    "history_ratings_ids": padded_ratings_history,
                    "history_length": n_hist
                }
            
        def create_subgraph(record, num_candidates=20) -> jraph.GraphsTuple:
            """
            create a local subgraph for the record in which all indexes are relative to just these local
            variable, not the global embedding indices.
            The returned Graph Topology:
            Node 0: The User.
            Nodes 1 to H: History Movies (H = max_history).
            Nodes H+1 to H+C: Candidate Movies (C = num_candidates)
            Edges are the user ratings for the movie.
            :param record : the results dictionary of using the RatingsHistoryLookupTransform map to create the subgraph
            for a train dataset item of user_id, movie_id, rating, timestamp.  The other keys
            are "history_movie_ids", "history_ratings", and "history_length" where "history_length" represents
            the number of the entries in "history_movie_ids" and "history_ratings" that are not padded (padded
            values are -1)
            :param num_candidates: the sum of the number of Positive (1 movie) + Hard Negatives (e.g., 5 movies)
              + Approximate Negatives (e.g., 14 movies).  must be less than or equal to the top_k retrieval
              recommendations.  this parameter is given to Rax for loss calculation later.
            :return: a sparsely populated jraph.GraphsTuple representation of the local subgraph for
            the train user_id.  Note that this is not the padded version to give to the model being trained.
            """
            
            #NOTE: method returns a sparse GraphsTuple, ignoring the padded variables in record, because
            # the resulting datastructure is not seen by the GPU.
            
            max_hist = len(record["history_movie_ids"])
            
            # Define Node Counts
            total_nodes = 1 + max_hist + num_candidates
            
            # Define Senders and Receivers (Edges)
            # Strategy: Star Graph. User connects TO History and Candidates.
            senders = []
            receivers = []
            edge_features = []  # We can put ratings here
            
            # Edge: User (0) -> History (1..H)
            for i in range(max_hist):
                movie_id = record["history_movie_ids"][i]
                if movie_id != -1:  # Only connect real movies, not padding
                    senders.append(0)
                    receivers.append(i + 1)
                    edge_features.append(record["history_ratings_ids"][i])
            
            # Edge: User (0) -> Candidates (H+1..H+C)
            for i in range(num_candidates):
                senders.append(0)
                receivers.append(1 + max_hist + i)
                edge_features.append(0)  # Candidates don't have ratings yet
            
            # 3. Construct the GraphsTuple
            return jraph.GraphsTuple(
                nodes={
                    "ids": np.concatenate([
                        [record["user_id"]],
                        record["history_movie_ids"],
                        record["candidate_ids"]
                    ]),
                    "type": np.array(
                        [0] + [1] * max_hist + [2] * num_candidates)
                    # 0=User, 1=Hist, 2=Cand
                },
                edges={"rating": np.array(edge_features)},
                senders=np.array(senders),
                receivers=np.array(receivers),
                n_node=np.array([total_nodes]),
                n_edge=np.array([len(edge_features)]),
                globals=None
            )
        
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
        
        class DeserializeSource(pgrain.MapTransform):
            def map(self, r):
                return msgpack.unpackb(r, raw=False)
            
        class DeserializeBatchSource(pgrain.MapTransform):
            def map(self, r):
                return [msgpack.unpackb(b, raw=False) for b in r]
            
        # ====   how to use all of that ======
        
        max_history = 20
        num_candidates = 20
        batch_size = 64
        worker_count = 0#max(1, os.cpu_count() - 1)
        
        datasource = RandomAccessArrayRecordDataSource(self.ratings_train_uri)
        shard_opts = grain.python.ShardOptions(shard_index=0, shard_count=1)
        
        dataloader0 = grain.python.DataLoader(
            data_source=datasource,
            sampler=BatchSampler(num_records=datasource.__len__(),
                batch_size=batch_size, shard_options=shard_opts),
            operations=[
                # grain.python.Batch(batch_size=batch_size),  #do not use the batch transform, instead use batch in sampling
                # grain.python.BatchShuffle(seed=42, buffer_size=10000),
            ],
            # worker_count=1,
            # worker_buffer_size=batch_size,
            shard_options=shard_opts,
            # read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=0)
        )
        
        history_dict = build_history_lookup(dataloader0)
        
        def training_generator(reader, batch_size):
            """Yield batches for training."""
            n_records = reader.num_records()
            for i in range(0, n_records + 1, batch_size):
                stop = np.min(i + batch_size, n_records)
                batch_bytes = reader.read([i, stop])
                batch = [msgpack.unpackb(b) for b in batch_bytes] #each batch is list of dictionaries
                for record in batch:
                    u = record['user_id']
                    ts = record['timestamp']
                    m = record['movie_id']
                    r = record['rating']
                    """
                    apply the equivalent of these to the data:
                    RatingsHistoryLookupTransform(history_dict, max_history=max_history),
                    grain.python.Map(lambda record: create_subgraph(record, num_candidates=num_candidates)),
                    grain.python.Batch(batch_size=batch_size)
                    """
            #for batch in mnist_dataset["train"].iter(batch_size):
            #    x, y = batch["image"], batch["label"]
            #    yield x, y
                
        print(f'have history_dict')
        
        print(f'have dataloader')
        jax_size_dict = calc_number_jax_graph_components(batch_size, max_history, num_candidates)
        # Initialize your Grain-based generator
        jax_train_ds = get_jax_dataset(
            dataloader0,
            max_nodes=jax_size_dict['max_nodes'],
            max_edges=jax_size_dict['max_edges'],
            max_graphs=jax_size_dict['max_graphs']
        )
        print(f'have jax dataset')
        # The Training Loop
        for step_graph in jax_train_ds:
            print(step_graph)
            break
            # Everything in 'step_graph' is now a jnp.array on the GPU
            #params, opt_state, loss = train_step(params, opt_state, step_graph)
            #if step % 100 == 0:
            #    print(f"Step {step}, Loss: {loss}")
            
    def test_load_embeddings(self):
        movie_emb_data_source = grain.sources.ArrayRecordDataSource(self.movie_embeddings_uri)
        user_emb_data_source = grain.sources.ArrayRecordDataSource(self.user_embeddings_uri)

    def test_make_hard_negatives(self):
        exact_hard_neg = grain.sources.ArrayRecordDataSource(self.exact_hard_negatives_uri)
        rec_neg = grain.sources.ArrayRecordDataSource(self.recommendations_uri)
        
        
    if __name__ == '__main__':
        unittest.main()
