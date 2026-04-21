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
from movie_lens_ranker.JraphPaddedGraphTupleTransform import JraphPaddedGraphTupleTransform
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
            = get_train_val_test_liked_uris(use_small=False)
        
        # (user_id, movie_id, rating, timestamp)
        self.ratings_train_disliked_uri, self.ratings_val_disliked_uri, self.ratings_test_disliked_uri \
            = get_train_val_test_disliked_uris(use_small=True)
        
        # (movie_id, float array of embed_dim as a tuple)
        self.movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        
        # (user_id, float array of embed_dim as a tuple)
        self.user_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/user_emb-00000-of-00001.array_record")
        
        # (user_id, int array of movie_ids as a tuple) is full catalog for each user, no history subtracted
        self.recommendations_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies.array_record")
        self.recommendations_ts_uri = os.path.join(
            get_project_dir(),
            "src/test/resources/data/recommended_movies_timestamps.array_record")
        
        # (movie_id, title, genres)
        self.movies_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")
        
        #these are the "elite" hard negatives (=intersection between train_disliked and recommended movies)
        # + train disliked.
        self.train_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/train_negatives.array_record")
        self.val_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/val_negatives.array_record")
        self.test_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/test_negatives.array_record")
        self.train_val_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/train_val_negatives.array_record")
        self.train_val_test_negatives_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/train_val_test_negatives.array_record")
    
    def test_grain_dataloader(self):
        
        #for dataloading, which is always on CPU, use this flag to prevent jax from trying to
        # put jax arrays on GPU before training (which happens in another component)
        os.environ["JAX_PLATFORMS"] = "cpu"
        
        import jax
        #jax.config.update("jax_debug_nans", True)
        #np.set_printoptions(threshold=np.inf)
        
        max_history = 200
        num_candidates = 40
        batch_size = 64
        num_epochs = 120
        seed = 1234
        top_k = 100
        worker_count = max(1, os.cpu_count() - 1)
        shard_opts = grain.sharding.ShardOptions(shard_index=0, shard_count=1)
        
        all_movie_ids: List[int] = read_movies_array_record(self.movies_uri,
            batch_size=batch_size)
        
        #the number per user must be >= half of num_candidates
        recommendations = RecommendedMovies(
            movie_rec_file_path=self.recommendations_uri,
            movie_rec_ts_file_path=self.recommendations_ts_uri)
        
        # each worker will have its own copy of these:
        train_history = UserHistory(ratings_uri_list=self.ratings_train_uri, fixed_size=2048)
        val_history = UserHistory(ratings_uri_list=self.ratings_val_uri, fixed_size=2048)

        train_negatives = Negatives(self.train_negatives_uri, fixed_size=256)
        train_val_negatives = Negatives(self.train_val_negatives_uri, fixed_size=256)
        val_negatives = Negatives(self.val_negatives_uri, fixed_size=256)
       
        train_datasource = RandomAccessArrayRecordDataSource(self.ratings_train_uri)
        val_datasource = RandomAccessArrayRecordDataSource(self.ratings_val_uri)
       
        train_ra_sampler = BatchSampler(num_records=train_datasource.__len__(),
            num_epochs=num_epochs,
            batch_size=batch_size, shuffle=True, seed=seed,
            shard_options=shard_opts)
        val_ra_sampler = BatchSampler(num_records=val_datasource.__len__(),
            num_epochs=num_epochs,
            batch_size=batch_size, shuffle=True, seed=seed,
            shard_options=shard_opts)
        
        # NOTE that train_history_dict, etc are passed by reference to the MapTransforms
        train_dataloader = grain.DataLoader(
            data_source=train_datasource,
            sampler=train_ra_sampler,
            operations=[
                # enrich the train records with local subgraphs:
                RatingsHistoryLookupTransform(
                    history_lookup=train_history,
                    max_history=max_history),
                HardNegativeSamplingTransform(
                    history_lookup=train_history,
                    all_movie_ids=all_movie_ids,
                    negatives=train_negatives,
                    recommendations=recommendations,
                    num_candidates=num_candidates, top_k=top_k, seed=seed),
                SparseLocalSubgraphTransform(),
                JraphPaddedGraphTupleTransform(batch_size=batch_size, max_history=max_history, num_candidates=num_candidates),
            ],
            # worker_count=worker_count,
            shard_options=shard_opts
        )
        
        val_dataloader = grain.DataLoader(
            data_source=val_datasource,
            sampler=val_ra_sampler,
            operations=[
                # enrich the train records with local subgraphs:
                RatingsHistoryLookupTransform(
                    history_lookup=val_history,
                    max_history=max_history),
                HardNegativeSamplingTransform(
                    history_lookup=val_history,
                    all_movie_ids=all_movie_ids,
                    negatives=val_negatives,
                    #negatives=train_val_negatives,
                    recommendations=recommendations,
                    num_candidates=num_candidates, top_k=top_k, seed=seed),
                SparseLocalSubgraphTransform(),
                JraphPaddedGraphTupleTransform(batch_size=batch_size, max_history=max_history, num_candidates=num_candidates),
            ],
            # worker_count=worker_count,
            shard_options=shard_opts
        )
       
        if False:
            train_batch : jraph.GraphsTuple = next(iter(train_dataloader))
            print(f'train_batch={train_batch}')

            val_batch: jraph.GraphsTuple = next(iter(train_dataloader))
            print(f'val_batch={val_batch}')
            return
        
        learning_rate = 5e-4#1e-3
        weight_decay = 1e-4
        out_dim = 32
        hidden_dim = 64  # 2 * embed_in_dim is probably good
        num_layers = 2  # captures the 2-hop neighborhood.  3 tends to oversmooth
        num_heads = 4  # each head sees 64 hidden / 4 heads = 16 dimensional subspace
        dropout_rate = 0.1
        rngs = nnx.Rngs(seed)
        
        embeddings = read_embeddings(
            user_embeddings_uri=self.user_embeddings_uri,
            movie_embeddings_uri=self.movie_embeddings_uri,
            batch_size=1024)
        
        model = GraphRanker(user_movie_embeds=embeddings, num_candidates=num_candidates,
            hidden_features=hidden_dim, num_layers=num_layers,
            out_features=out_dim, heads=num_heads,
            dropout_rate=dropout_rate, rngs=rngs)
        
        optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=weight_decay), wrt=nnx.Param)
        
        train_metrics = train_fn(model=model, num_epochs=num_epochs, train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer, batch_size=batch_size, max_history=max_history,
            num_candidates=num_candidates)
        print(f'train_metrics: {train_metrics}')
        
        if False:
            test_history = UserHistory(ratings_uri_list=self.ratings_val_uri,
                fixed_size=2048)
            test_negatives = Negatives(self.test_negatives_uri, fixed_size=256)
            
            test_datasource = RandomAccessArrayRecordDataSource(
                self.ratings_test_uri)
            test_ra_sampler = BatchSampler(
                num_records=test_datasource.__len__(),
                num_epochs=num_epochs,
                batch_size=batch_size, shuffle=True, seed=seed,
                shard_options=shard_opts)
            
            test_dataloader = grain.DataLoader(
                data_source=test_datasource,
                sampler=test_ra_sampler,
                operations=[
                    # enrich the train records with local subgraphs:
                    RatingsHistoryLookupTransform(
                        history_lookup=test_history,
                        max_history=max_history),
                    HardNegativeSamplingTransform(
                        history_lookup=test_history,
                        all_movie_ids=all_movie_ids,
                        negatives=test_negatives,
                        # negatives=train_val_negatives,
                        recommendations=recommendations,
                        num_candidates=num_candidates, top_k=top_k, seed=seed),
                    SparseLocalSubgraphTransform(),
                    JraphPaddedGraphTupleTransform(batch_size=batch_size,
                        max_history=max_history,
                        num_candidates=num_candidates),
                ],
                # worker_count=worker_count,
                shard_options=shard_opts
            )
            
            eval_metrics = test_fn(model=model, num_epochs=num_epochs, test_dataloader=test_dataloader,
                optimizer=optimizer, batch_size=batch_size, max_history=max_history,
                num_candidates=num_candidates)
        
    if __name__ == '__main__':
        unittest.main()
