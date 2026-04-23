from typing import Tuple, List
from array_record.python import array_record_module
from grain import DataLoader
from grain.sharding import ShardOptions

from movie_lens_ranker.BatchSampler import BatchSampler
from movie_lens_ranker.HardNegativeSamplingTransform import \
    HardNegativeSamplingTransform
from movie_lens_ranker.JraphPaddedGraphTupleTransform import \
    JraphPaddedGraphTupleTransform
from movie_lens_ranker.Negatives_vec import Negatives
from movie_lens_ranker.RandomAccessArrayRecordDataSource import \
    RandomAccessArrayRecordDataSource
from movie_lens_ranker.RatingsHistoryLookupTransform import \
    RatingsHistoryLookupTransform
from movie_lens_ranker.RecommendedMovies import RecommendedMovies
from movie_lens_ranker.SparseLocalSubgraphTransform import \
    SparseLocalSubgraphTransform
from movie_lens_ranker.UserHistory import UserHistory
from movie_lens_ranker.util import read_movies_array_record

def create_train_and_val_dataloaders(total_workers:int, worker_rank:int,
        movies_uri:str, recommendations_uri:str, recommendations_ts_uri:str,
        train_ratings_uri:str, val_ratings_uri:str,
        train_negatives_uri:str, val_negatives_uri:str,
        max_history:int, num_candidates:int,
        num_epochs:int, batch_size:int, seed:int) -> Tuple[DataLoader, DataLoader]:
    
    all_movie_ids: List[int] = read_movies_array_record(movies_uri, batch_size=batch_size)
    
    # the number per user must be >= half of num_candidates
    recommendations = RecommendedMovies(
        movie_rec_file_path=recommendations_uri,
        movie_rec_ts_file_path=recommendations_ts_uri)
    
    train_dataloader = _create_dataloader(
        total_workers=total_workers, worker_rank=worker_rank,
        all_movie_ids=all_movie_ids,
        recommendations=recommendations,
        ratings_uri=train_ratings_uri, negatives_uri=train_negatives_uri,
        max_history=max_history, num_candidates=num_candidates,
        num_epochs=num_epochs, batch_size=batch_size, seed=seed)
    
    val_dataloader = _create_dataloader(
        total_workers=total_workers, worker_rank=worker_rank,
        all_movie_ids=all_movie_ids,
        recommendations=recommendations,
        ratings_uri=val_ratings_uri, negatives_uri=val_negatives_uri,
        max_history=max_history, num_candidates=num_candidates,
        num_epochs=1, batch_size=batch_size, seed=seed)
    
    return train_dataloader, val_dataloader

def create_test_dataloader(total_workers: int, worker_rank: int,
        movies_uri: str, recommendations_uri: str, recommendations_ts_uri: str,
        ratings_uri: str, negatives_uri: str,
        max_history: int, num_candidates: int,
        batch_size: int, seed: int) -> DataLoader:
    
    all_movie_ids: List[int] = read_movies_array_record(movies_uri,
        batch_size=batch_size)
    
    # the number per user must be >= half of num_candidates
    recommendations = RecommendedMovies(
        movie_rec_file_path=recommendations_uri,
        movie_rec_ts_file_path=recommendations_ts_uri)
    
    dataloader = _create_dataloader(
        total_workers=total_workers, worker_rank=worker_rank,
        all_movie_ids=all_movie_ids, recommendations=recommendations,
        ratings_uri=ratings_uri, negatives_uri=negatives_uri,
        max_history=max_history, num_candidates=num_candidates,
        num_epochs=1, batch_size=batch_size, seed=seed)
        
    return dataloader

def _create_dataloader(total_workers: int, worker_rank: int,
        all_movie_ids: List[int], recommendations: RecommendedMovies,
        ratings_uri:str, negatives_uri:str,
        max_history: int, num_candidates: int,
        num_epochs:int, batch_size: int, seed: int) -> DataLoader:
    
    shard_opts = ShardOptions(
        shard_index=worker_rank,
        shard_count=total_workers,
        drop_remainder=True
    )
    
    # each worker will have its own copy of these:
    user_history = UserHistory(ratings_uri_list=ratings_uri, fixed_size=2048)
    
    negatives = Negatives(negatives_uri, fixed_size=256)
    
    datasource = RandomAccessArrayRecordDataSource(ratings_uri)
    
    ra_sampler = BatchSampler(num_records=datasource.__len__(),
        num_epochs=num_epochs,
        batch_size=batch_size, shuffle=True, seed=seed,
        shard_options=shard_opts)
    
    # NOTE that train_history_dict, etc are passed by reference to the MapTransforms
    dataloader = DataLoader(
        data_source=datasource,
        sampler=ra_sampler,
        operations=[
            # enrich the train records with local subgraphs:
            RatingsHistoryLookupTransform(
                history_lookup=user_history,
                max_history=max_history),
            HardNegativeSamplingTransform(
                history_lookup=user_history,
                all_movie_ids=all_movie_ids,
                negatives=negatives,
                recommendations=recommendations,
                num_candidates=num_candidates, seed=seed),
            SparseLocalSubgraphTransform(),
            JraphPaddedGraphTupleTransform(batch_size=batch_size,
                max_history=max_history, num_candidates=num_candidates),
        ],
        # worker_count=worker_count,
        shard_options=shard_opts
    )
    
    return dataloader
