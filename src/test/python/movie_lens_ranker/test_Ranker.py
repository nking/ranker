import os.path
import unittest

import jraph
import optax
from array_record.python import array_record_module
from flax import nnx
from scipy.stats import triang_gen

from helper import *
from movie_lens_ranker.BatchSampler import BatchSampler
from movie_lens_ranker.RandomAccessArrayRecordDataSource import *
from movie_lens_ranker.RatingsHistoryLookupTransform import *
from movie_lens_ranker.HardNegativeSamplingTransform import *
from movie_lens_ranker.SparseLocalSubgraphTransform import \
    SparseLocalSubgraphTransform
from movie_lens_ranker.data_loading import *
from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.train import *

class TestRanker(unittest.TestCase):
    def setUp(self):
        
        # user recommendations with each user history subtacted already:
        # (user id, (movie_ids))
        self.recommendations_uri = os.path.join(get_project_dir(),
            "src/test/resources/recommended_movies.array_record")
        
        #(user_id, movie_id, rating, timestamp)
        self.ratings_train_uri, self.ratings_val_uri, self.ratings_test_uri \
            = get_train_val_test_liked_uris(use_small=True)
        
        # (user_id, movie_id, rating, timestamp)
        self.ratings_train_disliked_uri, self.ratings_val_disliked_uri, self.ratings_test_disliked_uri \
            = get_train_val_test_disliked_uris(use_small=True)
        
        # (movie_id, float array of embed_dim as a tuple)
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        
        # (user_id, float array of embed_dim as a tuple)
        self.user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb-00000-of-00001.array_record")
        
        # (user_id, int array of movie_ids as a tuple)
        self.unseen_recommendations_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record")
        
        # (movie_id, title, genres)
        self.movies_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")
        
        #the approximate hard negatives are the samples drawn from unwatched movies
        # the negatives uri has for each user, the list of negatives prioritized by:
        #    the "elite" hard negatives are the intersection of the natural hard negatives with the recommended movies,
        #    the natural hard negatives are the ones which user rated 1 or 2
        #  (user_id, tuple of negative movie_ids)
        self.negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/user_recommendations_disliked_in_train.array_record")
       
    def test_load_ratings(self):
        max_history = 20
        num_candidates = 20
        batch_size = 1024
        batch_size = 2
        num_epochs = 1
        seed = 1234
        worker_count = max(1, os.cpu_count() - 1)
        
        train_ratings_uri = self.ratings_train_uri
        
        embeddings = read_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
        
        # each worker will have its own copy of these:
        train_history_dict, max_history__ = build_history_lookup(
            train_ratings_uri, batch_size=batch_size)
        
        user_exact_negatives = read_user_exact_negatives(
            self.negatives_uri, batch_size=batch_size)
        
        all_movie_ids = read_movies_array_record(self.movies_uri,
            batch_size=batch_size)
        
        unseen_recommendations = read_user_unseen_recommendations(
            self.unseen_recommendations_uri, batch_size=batch_size)
        
        train_datasource = RandomAccessArrayRecordDataSource(train_ratings_uri)
        shard_opts = grain.sharding.ShardOptions(shard_index=0, shard_count=1)
        
        train_ra_sampler = BatchSampler(num_records=train_datasource.__len__(),
            num_epochs=num_epochs,
            batch_size=batch_size, shuffle=True, seed=seed,
            shard_options=shard_opts)
        
        #TODO: split train into train and val
        
        # NOTE that train_history_dict, etc are passed by reference to the MapTransforms
        train_dataloader = grain.DataLoader(
            data_source=train_datasource,
            sampler=train_ra_sampler,
            operations=[
                # enrich the train records with local subgraphs:
                RatingsHistoryLookupTransform(
                    history_lookup=train_history_dict,
                    max_history=max_history),
                HardNegativeSamplingTransform(
                    history_lookup=train_history_dict,
                    all_movie_ids=all_movie_ids,
                    exact_negatives_dict=user_exact_negatives,
                    unseen_recommendations=unseen_recommendations,
                    num_candidates=num_candidates, seed=seed),
                SparseLocalSubgraphTransform(),
            ],
            # worker_count=worker_count,
            shard_options=shard_opts
        )
        '''
        train_batch : List[jraph.GraphsTuple] = next(iter(train_dataloader))
        node_ids = train_batch[0].nodes["ids"]
        is_movie = (train_batch[0].nodes["type"] > 0)
        min_movie_id = jnp.min(node_ids[is_movie])
        if min_movie_id < len(user_id_fwd_dict):
            self.fail(f"🚨 Error: Found a movie with ID {min_movie_id}. "
                  f"It's colliding with User IDs!")
        # train_batch : List[jraph.GraphsTuple] = next(iter(train_dataloader))
        '''
        
        learning_rate = 1e-3
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
        
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate),
            wrt=nnx.Param)
            
        train_metrics = train_fn(model=model, num_epochs=num_epochs, train_dataloader=train_dataloader,
            optimizer=optimizer, batch_size=batch_size, max_history=max_history,
            num_candidates=num_candidates)
        print(f'train_metrics: {train_metrics}')
        
        #run the eval method temporarily on train data
        eval_metrics = test_fn(model=model, num_epochs=num_epochs, dataloader=val_dataloader,
            optimizer=optimizer, batch_size=batch_size, max_history=max_history,
            num_candidates=num_candidates)
        
    
    if __name__ == '__main__':
        unittest.main()
