import os.path
import unittest

import jraph
import optax
from array_record.python import array_record_module
from flax import nnx

from helper import *
from movie_lens_ranker.BatchSampler import BatchSampler
from movie_lens_ranker.RandomAccessArrayRecordDataSource import *
from movie_lens_ranker.RatingsHistoryLookupTransform import *
from movie_lens_ranker.HardNegativeSamplingTransform import *
from movie_lens_ranker.SparseLocalSubgraphTransform import \
    SparseLocalSubgraphTransform
from movie_lens_ranker.data_loading import *
from movie_lens_ranker.model import GraphRanker


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
        
        self.unseen_recommendations_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/user_recommendations_without_train_val.array_record")
    
    def test_load_ratings(self):
        max_history = 20
        num_candidates = 20
        batch_size = 1024
        num_epochs = 1
        seed = 1234
        worker_count = max(1, os.cpu_count() - 1)
        
        user_id_fwd_dict, movie_id_fwd_dict, embeddings = read_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
        
        # each worker will have its own copy of these:
        history_dict, max_history__ = build_history_lookup(
            self.ratings_train_uri, user_id_fwd_dict,
            movie_id_fwd_dict, batch_size=batch_size)
        user_exact_negatives = read_user_exact_negatives(
            self.exact_hard_negatives_uri,
            user_id_fwd_dict, movie_id_fwd_dict, batch_size=batch_size)
        all_movie_ids = read_movies_array_record(self.movie_ids_uri,
            movie_id_fwd_dict, batch_size=batch_size)
        unseen_recommendations = read_user_unseen_recommendations(
            self.unseen_recommendations_uri,
            user_id_fwd_dict, movie_id_fwd_dict, batch_size=batch_size)
        
        batch_size = 2
        datasource = RandomAccessArrayRecordDataSource(
            self.ratings_train_uri)
        shard_opts = grain.ShardOptions(shard_index=0,
            shard_count=1)
        ra_sampler = BatchSampler(num_records=datasource.__len__(),
            num_epochs=num_epochs,
            batch_size=batch_size, shuffle=True, seed=seed,
            shard_options=shard_opts)
        
        # NOTE that history_dict, etc are passed by reference to the MapTransforms
        train_dataloader = grain.DataLoader(
            data_source=datasource,
            sampler=ra_sampler,
            operations=[
                # enrich the train records with local subgraphs:
                RatingsHistoryLookupTransform(
                    history_lookup=history_dict,
                    user_id_fwd_dict=user_id_fwd_dict,
                    movie_id_fwd_dict=movie_id_fwd_dict,
                    max_history=max_history),
                HardNegativeSamplingTransform(
                    history_lookup=history_dict,
                    all_movie_ids=all_movie_ids,
                    exact_negatives_dict=user_exact_negatives,
                    unseen_recommendations=unseen_recommendations,
                    num_candidates=num_candidates, seed=seed),
                SparseLocalSubgraphTransform(),
            ],
            # worker_count=worker_count,
            shard_options=shard_opts
        )
        train_batch : List[jraph.GraphsTuple] = next(iter(train_dataloader))
        node_ids = train_batch[0].nodes["ids"]
        is_movie = (train_batch[0].nodes["type"] > 0)
        min_movie_id = jnp.min(node_ids[is_movie])
        if min_movie_id < len(user_id_fwd_dict):
            self.fail(f"🚨 Error: Found a movie with ID {min_movie_id}. "
                  f"It's colliding with User IDs!")
        # train_batch : List[jraph.GraphsTuple] = next(iter(train_dataloader))
        
        out_dim = 64
        hidden_dim = 64  # 2 * embed_in_dim is probably good
        num_layers = 2  # captures the 2-hop neighborhood.  3 tends to oversmooth
        num_heads = 4  # each head sees 64 hidden / 4 heads = 16 dimensional subspace
        dropout_rate = 0.1
        rngs = nnx.Rngs(0)
        
        model = GraphRanker(user_movie_embeds=embeddings, num_candidates=num_candidates,
            hidden_features=hidden_dim, num_layers=num_layers,
            out_features=out_dim, heads=num_heads,
            dropout_rate=dropout_rate, rngs=rngs)
        
        optimizer = nnx.Optimizer(model, optax.adam(1e-3),
            wrt=nnx.Param)
    
    if __name__ == '__main__':
        unittest.main()
