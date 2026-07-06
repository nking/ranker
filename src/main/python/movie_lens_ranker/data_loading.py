
from typing import Tuple, List
import os
import grain.sharding
import jax.numpy as jnp
from array_record.python import array_record_module
from grain import DataLoader

from movie_lens_ranker.BatchSampler import BatchSampler
from movie_lens_ranker.HardNegativeSamplingTransform import \
    HardNegativeSamplingTransform
from movie_lens_ranker.RandomAccessArrayRecordDataSource import \
    RandomAccessArrayRecordDataSource
from movie_lens_ranker.RatingsHistoryTransform import RatingsHistoryLookupTransform
from movie_lens_ranker.RecommendedMovies import RecommendedMovies
from movie_lens_ranker.SparseLocalSubgraphTransform import \
    SparseLocalSubgraphTransform
from movie_lens_ranker.SuperGraphPaddingTransform import \
    SuperGraphPaddingTransform
from movie_lens_ranker.UserHistory import UserHistory
from movie_lens_ranker.util import read_movies_array_record, read_user_movie_embeddings
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def create_train_and_val_dataloaders(
        num_users:int,
        user_embeddings_uri : str,
        movie_embeddings_uri : str,
        movies_uri:str,
        recommendations_uri:str, recommendations_ts_uri:str,
        ratings_train_data_uri:str,
        ratings_train_history_uris:List[str],
        ratings_train_disliked_uris:List[str],
        ratings_val_data_uri:str,
        ratings_val_history_uris:List[str],
        ratings_val_disliked_uris:List[str],
        max_history:int, num_candidates:int,
        num_epochs:int, batch_size:int, seed:int=0) -> Tuple[DataLoader, DataLoader]:
        
    all_movie_ids: List[int] = read_movies_array_record(movies_uri, batch_size=batch_size)

    # the number per user must be >= half of num_candidates
    recommendations = RecommendedMovies(
        num_users=num_users,
        movie_rec_file_uri=recommendations_uri,
        movie_rec_ts_file_uri=recommendations_ts_uri)

    user_movie_embeddings = read_user_movie_embeddings(
        user_embeddings_uri=user_embeddings_uri,
        movie_embeddings_uri=movie_embeddings_uri)
    
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
    
    train_dataloader = _create_dataloader(
        user_movie_embeddings = user_movie_embeddings,
        all_movie_ids=all_movie_ids,
        recommendations=recommendations,
        ratings_data_uri= ratings_train_data_uri,
        ratings_history_uris = ratings_train_history_uris,
        ratings_disliked_uris = ratings_train_disliked_uris,
        max_history=max_history, num_candidates=num_candidates,
        num_epochs=num_epochs, batch_size=batch_size, seed=seed)
    
    val_dataloader = _create_dataloader(
        user_movie_embeddings = user_movie_embeddings,
        all_movie_ids=all_movie_ids,
        recommendations=recommendations,
        ratings_data_uri=ratings_val_data_uri,
        ratings_history_uris=ratings_val_history_uris,
        ratings_disliked_uris=ratings_val_disliked_uris,
        max_history=max_history, num_candidates=num_candidates,
        num_epochs=1, batch_size=batch_size, seed=seed, shuffle=False)
    
    return train_dataloader, val_dataloader

def create_test_dataloader(
        num_users:int,
        user_embeddings_uri : str,
        movie_embeddings_uri : str,
        movies_uri: str,
        recommendations_uri: str,recommendations_ts_uri: str,
        rattings_data_uri: str,
        ratings_history_uris: List[str],
        ratings_disliked_uris: List[str],
        max_history: int, num_candidates: int,
        batch_size: int, seed: int) -> DataLoader:
    """
    create test data loader
    :param num_users:
    :param movies_uri:
    :param recommendations_uri:
    :param recommendations_ts_uri:
    :param rattings_data_uri:
    :param ratings_history_uris:
    :param ratings_disliked_uris:
    :param max_history:
    :param num_candidates:
    :param batch_size:
    :param seed:
    :return:
    """
    
    all_movie_ids: List[int] = read_movies_array_record(movies_uri,
        batch_size=batch_size)
    
    # the number per user must be >= half of num_candidates
    recommendations = RecommendedMovies(
        num_users=num_users,
        movie_rec_file_uri=recommendations_uri,
        movie_rec_ts_file_uri=recommendations_ts_uri)

    user_movie_embeddings = read_user_movie_embeddings(
        user_embeddings_uri=user_embeddings_uri,
        movie_embeddings_uri=movie_embeddings_uri)
    
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
    
    dataloader = _create_dataloader(
        user_movie_embeddings=user_movie_embeddings,
        all_movie_ids=all_movie_ids,
        recommendations=recommendations,
        ratings_data_uri= rattings_data_uri,
        ratings_history_uris = ratings_history_uris,
        ratings_disliked_uris = ratings_disliked_uris,
        max_history=max_history, num_candidates=num_candidates,
        num_epochs=1, batch_size=batch_size, seed=seed, shuffle=False)
        
    return dataloader

def _create_dataloader(
        user_movie_embeddings : jnp.ndarray,
        all_movie_ids: List[int], recommendations: RecommendedMovies,
        ratings_data_uri: str,
        ratings_history_uris: List[str],
        ratings_disliked_uris: List[str],
        max_history: int, num_candidates: int,
        num_epochs:int, batch_size: int, seed: int, shuffle:bool=True) -> DataLoader:
    
    shard_opts = grain.sharding.ShardByJaxProcess()
    logging.info(f'grain shard_opts={shard_opts}')
    
    worker_count = int(os.environ.get("grain_worker_count", 4))
    logging.info(f'grain worker_count={worker_count}')
    
    read_opts = grain.ReadOptions(
        num_threads = int(os.environ.get("grain_read_options_num_threads", 4)),
        prefetch_buffer_size=int(os.environ.get("grain_read_buffer_size", 50)),
    )

    user_history = UserHistory(ratings_uri_list=ratings_history_uris, max_history=max_history)
    
    user_disliked_history = UserHistory(ratings_uri_list=ratings_disliked_uris, max_history=max_history)
    
    datasource = RandomAccessArrayRecordDataSource(ratings_data_uri)
    
    import jax
    num_records = datasource.__len__()
    process_count = jax.process_count()
    if ((num_records // batch_size) // process_count) == 0:
        raise ValueError("batch_size is too small.  num_records={num_records} divided by "
            "batch_size={batch_size} then partitioned over {process_count} processes is 0")
    
    ra_sampler = BatchSampler(num_records=datasource.__len__(),
        num_epochs=num_epochs,
        batch_size=batch_size, shuffle=shuffle, seed=seed,
        shard_options=shard_opts)
    
    original_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
    # Hide GPUs from the next processes to be spawned
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    logging.info("Instantiating Grain DataLoader (hiding GPUs from child workers)...")
    
    # NOTE that train_history_dict, etc. are passed by reference to the MapTransforms
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
                history_lookup_disliked=user_disliked_history,
                all_movie_ids=all_movie_ids,
                recommendations=recommendations,
                num_candidates=num_candidates),
            SparseLocalSubgraphTransform(user_movie_embeddings=user_movie_embeddings),
            SuperGraphPaddingTransform(batch_size=batch_size,
                max_history=max_history, num_candidates=num_candidates,
                n_local_devices=len(jax.local_devices())),
        ],
        worker_count=worker_count,
        shard_options=shard_opts,
        read_options=read_opts,
    )
    
    os.environ["CUDA_VISIBLE_DEVICES"] = original_devices
    logging.info("DataLoader instantiated. Restored parent GPU visibility.")
    
    return dataloader
