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
from movie_lens_ranker.data_loading import *

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
        
    def test_load_ratings(self):
        
       
        # ====   how to use all of that ======
        
        max_history = 20
        num_candidates = 20
        batch_size = 1024
        worker_count = max(1, os.cpu_count() - 1)
        
        #each worker will have its own copy of these:
        history_dict = build_history_lookup(self.ratings_train_uri, batch_size=batch_size)
        user_exact_negatives = read_user_exact_negatives(self.exact_hard_negatives_uri, batch_size=batch_size)
        all_movie_ids = read_movies_array_record(self.movie_ids_uri, batch_size=batch_size)
        unseen_recommendations = read_user_unseen_recommendations(self.unseen_recommendations_uri, batch_size=batch_size)
        
        batch_size = 2
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
            #worker_count=worker_count,
            shard_options=shard_opts
        )
        
        train_batch : List[jraph.GraphsTuple] = next(iter(train_dataloader))
       
        '''
        def train_step(model: nnx.Module, optimizer: nnx.Optimizer,
                batch: jax.tree_util.tree_map[jnp.array, gn_graph.GraphsTuple]
        ):
            # Convert numpy arrays to jax.Array on GPU
            batch_tokens = jnp.array(batch["text"])
            labels = jnp.array(batch["label"], dtype=jnp.int32)
            
            grad_fn = nnx.value_and_grad(compute_losses_and_logits,
                has_aux=True)
            (loss, logits), grads = grad_fn(model, batch_tokens, labels)
            
            optimizer.update(grads)  # In-place updates.
            
            return loss
        
        @nnx.jit
        def eval_step(
                model: nnx.Module, batch: dict[str, jax.Array],
                eval_metrics: nnx.MultiMetric
        ):
            # Convert numpy arrays to jax.Array on GPU
            batch_tokens = jnp.array(batch["text"])
            labels = jnp.array(batch["label"], dtype=jnp.int32)
            loss, logits = compute_losses_and_logits(model,
                batch_tokens, labels)
            
            eval_metrics.update(
                loss=loss,
                logits=logits,
                labels=labels,
            )
        '''

        
    if __name__ == '__main__':
        unittest.main()
