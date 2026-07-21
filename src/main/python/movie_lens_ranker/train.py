"""
tune, train, test functions for a multi-host, multi-process Jax AI Stack model
and dataloader using SPMD paradigm.
"""
import time
from functools import partial
from typing import Tuple, Union, Any

import mlflow
import optax
from math import log
import jax
from jax.sharding import PartitionSpec as P
from jax import shard_map, Array

from vizier.service import pyvizier as vz
from vizier.service.clients import Trial
import jraph
import jax.numpy as jnp
from flax import nnx
import rax
import grain
from flax.typing import Array
from grain._src.python.data_loader import DataLoaderIterator

from movie_lens_ranker.SparseLocalSubgraphTransform import *
from movie_lens_ranker.BatchSampler import BatchSampler

import orbax.checkpoint as ocp

from movie_lens_ranker.data_contracts import validate_movies, validate_embedding, validate_movie_recommendations, \
    validate_movie_recommendations_timestamps, validate_ratings
from movie_lens_ranker.data_loading import create_train_and_val_dataloaders, \
    create_test_dataloader
from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.util import \
    stringify_mlflow_params, get_canonical_mlflow_run_name, \
    calc_number_jax_graph_components, get_model_mesh, get_gpu_stats, \
    get_cpu_stats, is_running_on_gpu, model_params_trainable_keys, \
    create_dirs_if_is_filepath, get_num_users_movies, read_user_movie_embeddings

from jax.experimental import multihost_utils

import logging

from movie_lens_ranker.util_np import optimized_batch_and_pad

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#mesh_local = jax.sharding.Mesh(np.array(jax.local_devices()), axis_names=('local_data',))
#data_sharding = jax.sharding.NamedSharding(mesh_local, P('local_data'))

def convert_to_global(arr, mesh, sync:bool=True):
    """
    Universal helper for JAX 0.8.0 multi-host.
    sync=True for Saving (ensures identity).
    sync=False for Restoring (just defines the container).
    Orbax needs all workers to have an identical structural blueprint (metadata)
    of the model, while coordinating their respective memory shards.
    
    different partitions of the data across different workers may have taken
    different branches in the graph and so lazily constructed different parameters.
    whether a rng key gets populated is one correction needed below.
    
    During Save, sync=True: The workers need identical metadata, but independent data shards.
    we wrap the local states into unified NamedSharding global arrays so that worker 0
    can write its data to disk nd worker 1 can write its, ...
    During restore, sync=False: orbax reads a single file structure
    from storage and needs to split those bytes among workers.
    """
    
    if hasattr(arr, 'sharding') and not arr.sharding.is_fully_addressable:
        global_sharding = arr.sharding
        global_shape = arr.shape
        dtype = arr.dtype
    else:
        if not hasattr(arr, 'shape'):
            arr = jnp.asarray(arr)
        global_shape = arr.shape
        dtype = arr.dtype
        global_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    if not sync:
        # Return an abstract metadata mold. This completely avoids device errors
        # and guarantees Orbax returns a true global array spanning [0, 1, 2048, 2049]
        return jax.ShapeDtypeStruct(shape=global_shape, dtype=dtype, sharding=global_sharding)
    
    is_key = jnp.issubdtype(dtype, jax.dtypes.prng_key)
    if is_key:
        arr_to_broadcast = jax.random.key_data(arr)
    else:
        arr_to_broadcast = arr
    
    arr_to_broadcast = multihost_utils.broadcast_one_to_all(
        arr_to_broadcast)
    
    if isinstance(arr_to_broadcast, (np.ndarray, np.generic)):
        arr_to_broadcast = jnp.asarray(arr_to_broadcast)
    
    if is_key:
        arr_to_broadcast = jax.random.wrap_key_data(arr_to_broadcast)
    
    def data_callback(index):
        return arr_to_broadcast[index]
    
    return jax.make_array_from_callback(global_shape, global_sharding, data_callback)

def score_and_shape_results(model: GraphRanker, padded_graph: jraph.GraphsTuple):
    # Forward Pass: returns ONLY candidate scores [num_total_graphs * K]
    all_scores = model(padded_graph) #LinearizeTracer<float32[60]>
    #jax.debug.print("all_scores={all_scores}", all_scores=all_scores, ordered=True)
    num_total_graphs = padded_graph.n_node.shape[0]  # batch_size + 1
    K = model.K  # num_candidates from data loading
    total_candidate_slots = num_total_graphs * K
    # Extract Candidate Data. length is K * num_total_graphs
    cand_indices = jnp.where(
        padded_graph.nodes["type"] == 2,
        size=total_candidate_slots
    )[0]
    #jax.debug.print("cand_indices={cand_indices}", cand_indices=cand_indices, ordered=True)
    # lengths are K * num_total_graphs
    labels_flat = padded_graph.nodes["label"][cand_indices]
    record_mask_flat = padded_graph.nodes["candidate_mask"][cand_indices]
    
    #jax.debug.print("record_mask_flat={record_mask_flat}", record_mask_flat=record_mask_flat, ordered=True)

    # Reshape everything to [Batch, K]
    scores_2d = all_scores.reshape(num_total_graphs, K)
    labels_2d = labels_flat.reshape((num_total_graphs, K))
    record_mask_2d = record_mask_flat.reshape((num_total_graphs, K))
    
    #jax.debug.print("Label sums per row: {x}", x=jnp.sum(labels_2d, axis=1), ordered=True)
    
    # Create Batch Mask (Ignore the last JAX padding graph)
    # real_graph_indices: [0, 1, 2] -> [True, True, False]
    is_real_graph = jnp.arange(num_total_graphs) < (num_total_graphs - 1)
    
    # Broadcast to [3, K]
    batch_mask = jnp.broadcast_to(is_real_graph[:, None],(num_total_graphs, K))
    
    #  Combine Masks
    # Master mask is True only for real candidates in real graphs
    final_mask = record_mask_2d & batch_mask
    
    #jax.debug.print("scores_2d={scores_2d}", scores_2d=scores_2d, ordered=True)
    #jax.debug.print("labels_2d={labels_2d}", labels_2d=labels_2d, ordered=True)
    #jax.debug.print("final_mask={final_mask}", final_mask=final_mask, ordered=True)
    #jax.debug.print("final_mask sums per row: {x}", x=jnp.sum(final_mask, axis=1),  ordered=True)

    return scores_2d, labels_2d, final_mask
    
@nnx.jit
def train_step(model: GraphRanker, padded_graph: jraph.GraphsTuple,
        optimizer: nnx.Optimizer) -> Tuple[float, Dict[str, float]]:
    """
    train step over a batch, where padded_graph contains super graph of the batch
    :param model:
    :param padded_graph:
    :param optimizer:
    :return:
    """
    
    #debug_weight_before = jnp.linalg.norm(model.score_head.kernel.get_value())
    
    def loss_fn(model, padded_graph) -> Array:
        scores_2d, labels_2d, main_mask = score_and_shape_results(model, padded_graph)
        safe_scores = jnp.where(main_mask, scores_2d, -1e9)
        #debug_stats(safe_scores, label="[Scores Statistics]")
        loss = rax.softmax_loss(
            scores=safe_scores,
            labels=labels_2d,
            where=main_mask,
            reduce_fn=jnp.mean  # Let it crash if NaNs happen to enable follow up
        )
        return loss
    
    # the model and optimizer were created with a mesh context, so here in this jax.jit method
    # value_and_grad does the following:
    # in the forward pass, the model is replicated across devices and each device calculates loss for its shard of data.
    # in the backward pass, each device calculates the gradient for its shard of data.
    # then an all gather algorithm sums the loss and divides by number of devices and similarly
    # calculates the mean gradient.
    # then the returned loss and gradients are the same for each device.
    loss, grads = nnx.value_and_grad(loss_fn)(model, padded_graph)
    # each process updates its model with the same values, so the model stays implicitly synchronized.
    optimizer.update(model, grads)
    
    #debug_weight_after = jnp.linalg.norm(model.score_head.kernel.get_value())
    #diff = jnp.abs(debug_weight_before - debug_weight_after)
    ## if > 1E-4, is a significant change
    ## if > 1, exploding gradient or learning rate issue
    #jax.debug.print("Weight Norm: Before={b:.6f}, After={a:.6f}, Delta={d:.8f}",
    #    b=debug_weight_before, a=debug_weight_after, d=diff)
  
    return loss

@nnx.jit
def eval_step(model: GraphRanker, padded_graph: jraph.GraphsTuple, top_k:int) -> dict[str, Array]:
    """
    train step over a batch, where padded_graph contains super graph of the batch
    :param model:
    :param padded_graph:
    :param top_k:
    :return: dictionary of "loss", "mrr", "ndcg", "recall"
    """
    def loss_fn(model, padded_graph) -> Tuple[Array, Dict[str, Array]]:
        scores_2d, labels_2d, main_mask = score_and_shape_results(model, padded_graph)
        safe_scores = jnp.where(main_mask, scores_2d, -1e9)
        # Rax Ranking Loss & Metrics
        # Rax is designed to ignore entries where master_mask is False
        loss = rax.softmax_loss(
            scores=safe_scores,
            labels=labels_2d,
            where=main_mask,
            reduce_fn=jnp.mean
        )
        mrr = rax.mrr_metric(
            safe_scores, labels_2d, where=main_mask, topn=top_k, reduce_fn=jnp.mean)
        ndcg = rax.ndcg_metric(
            safe_scores, labels_2d, where=main_mask, topn=top_k, reduce_fn=jnp.mean)
        recall = rax.recall_metric(
            safe_scores, labels_2d, where=main_mask, topn=top_k, reduce_fn=jnp.mean)
        return loss, {f"mrr": mrr, f"ndcg": ndcg, f"recall": recall}
    
    # has_aux is necessary when loss_fn returns more than scalar loss
    loss, metrics_dict = loss_fn(model, padded_graph)
    metrics_dict['loss'] = loss
    return metrics_dict

def _epoch_validation(model: GraphRanker, val_dataloader_iter: DataLoaderIterator,
        top_k: int, jax_graph_comp_dict:Dict[str, int]) -> Tuple[Dict, int]:
    """
    calc metrics for val dataset. Note, if this method consumes too much memory, use the
    _epoch_validation_chunked instead.   Note that the method uses SPMD paradigm.
    For evaluation on validation metrics, be sure to invoke model.eval() before using this method.
    
    :param model: the GraphRanker model instance
    :param val_dataloader_iter: iterator over the validation dataset
    :param top_k: the @K to be used in metrics NDCG@K, recall@K, MRR@K
    :param jax_graph_comp_dict: dictionary formed from method calc_number_jax_graph_components
    :return: a dictionary of the globally averaged metrics "loss", "mrr", "ndcg", "recall", the number of samples used
    """
    data_mesh = jax.sharding.Mesh(jax.devices(), axis_names=('data',))
    data_pspec = jax.sharding.PartitionSpec(('data'))
    model_pspec = P(('processes', 'local_devices'))
    model_mesh = get_model_mesh()
    global_avg_metrics_batches = {"loss":[], "mrr":[], "ndcg":[], "recall":[]}
    n_samples_tot = 0
    
    # in_specs=P() tells JAX the input is a scalar
    # out_specs=P() (empty) implies the output is a single global (replicated) value
    #  and is sharded across the 'data' axis
    @jax.jit
    @partial(shard_map, mesh=data_mesh, in_specs=P(), out_specs=P())
    def aggregate_metric(scalar_metric):
        return jax.lax.pmean(scalar_metric, axis_name='data')
    
    for padded_super_graph_0 in val_dataloader_iter:
        
        #actually is max_graphs which includes padding:
        n_samples_tot += len(padded_super_graph_0.n_node)
        
        #shard the data to local devices:
        #padded_super_graph = jax.device_put(padded_super_graph_0, data_sharding)
        padded_super_graph = jax.tree_util.tree_map(
            lambda x: multihost_utils.host_local_array_to_global_array(
                x, data_mesh, data_pspec),
            padded_super_graph_0
        )
        
        val_metrics = eval_step(model, padded_super_graph, top_k)
        
        # val_metrics['ndcg'] is now an array of shape (Num_Batches,)
        local_avg_val_metrics = jax.tree.map(jnp.mean, val_metrics)
        #average over all devices. the jax.lax internal to aggregate_metric acts as a barrier
        global_avg_metrics = jax.tree.map(aggregate_metric, local_avg_val_metrics)
        for key in global_avg_metrics:
            global_avg_metrics_batches[key].append(global_avg_metrics[key].item())
            
    #NOTE: at expense of RAM, could instead: stack all the padded graphs (put in a list
    # the mega_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *list_of_padded_graphs))
    # and use one vectorized call val_metrics = vectorized_epoch_eval
    # then the local_avg_val_metrics and global_avg_metrics as above.
    
    out = {key : float(np.average(global_avg_metrics_batches[key], axis=0)) for key in global_avg_metrics_batches}
    
    return out, n_samples_tot

def pad_graph_tuple_batch(graph_tuple_batch: jraph.GraphsTuple, jax_graph_comp_dict:Dict[str, int]) -> jraph.GraphsTuple:
    """
    pad a batch from the grain data loaders for use in sharding over local devices
    :param graph_tuple_batch:
    :param jax_graph_comp_dict: dictionary created from calc_number_jax_graph_components
    :return: the padded graph tuple, ready for harding across local devices
    """
    n_local_devices = jax.local_device_count()
    batch = graph_tuple_batch
    
    batch_size = len(batch)
    
    add_to = n_local_devices - (batch_size % n_local_devices)
    max_graphs = batch_size + n_local_devices + add_to
    
    #logging.info(f"pad_graph_tuple_batch: n_samples={n_samples}")
    
    batch = jraph.batch(batch)
    
    padded_super_graph_0 = jraph.pad_with_graphs(
        batch,
        n_node=jax_graph_comp_dict['max_nodes'],
        n_edge=jax_graph_comp_dict['max_edges'],
        n_graph=max_graphs
    )
    return padded_super_graph_0

def _train_fn(model, train_dataloader: grain.DataLoader,
        val_dataloader: grain.DataLoader,
        optimizer: nnx.Optimizer,
        top_k:int, latest_checkpoint_uri: str, best_checkpoint_uri:str,
        rngs:nnx.Rngs, config_dict:Dict[str, Union[str, int, float]],
        trial: Trial = None, save_checkpoints: bool=False,
        restored_train_dataloader_iter=None, restored_global_step:int=None, validate_checkpoint_restores:bool=False) -> float:
    """
    a shard's portion of the training
    :param model:
    :param train_dataloader:
    :param val_dataloader:
    :param optimizer:
    :param top_k:
    :param latest_checkpoint_uri:
    :param rngs:
    :return:
    """
   
    if not isinstance(train_dataloader._sampler, BatchSampler):
        raise ValueError("train_dataloader sampler must be an instance of BatchSampler")
    if not isinstance(val_dataloader._sampler, BatchSampler):
        raise ValueError("val_dataloader sampler must be an instance of BatchSampler")
        
    logging.info(f'_train_fn config_dict={config_dict}')
    
    start_time = time.perf_counter()
    
    rank = jax.process_index()
    n_local_devices = jax.local_device_count()
    
    #tracked_fn_1 = chex.assert_max_traces(train_step, n=1)
    #tracked_fn_2 = chex.assert_max_traces(eval_step, n=1)
    
    #TOTAL_RECORDS here is the same as train_dataloader._data_source.__len__()
    TRAIN_BATCH_SIZE = train_dataloader._sampler.batch_size
    TOTAL_RECORDS = train_dataloader._sampler.total_records
    STEPS_PER_EPOCH_GLOBAL = train_dataloader._sampler.num_batches_per_epoch  # = 7234
    NUM_TRAIN_SHARDS = train_dataloader._sampler._shard_options.shard_count
    STEPS_PER_EPOCH_LOCAL = STEPS_PER_EPOCH_GLOBAL//NUM_TRAIN_SHARDS
    
    logging.info(f'TRAIN_BATCH_SIZE={TRAIN_BATCH_SIZE}, TOTAL_RECORDS={TOTAL_RECORDS}, NUM_TRAIN_SHARDS={NUM_TRAIN_SHARDS}')
    logging.info(f'STEPS_PER_EPOCH_GLOBAL_TRAIN={STEPS_PER_EPOCH_GLOBAL}')
    logging.info(f'STEPS_PER_EPOCH_LOCAL_TRAIN={STEPS_PER_EPOCH_LOCAL}')
    logging.info(f'NUM_EPOCHS to train={train_dataloader._sampler._num_epochs}')
    
    VAL_BATCH_SIZE = train_dataloader._sampler.batch_size
    TOTAL_RECORDS_VAL = val_dataloader._sampler.total_records
    STEPS_PER_EPOCH_GLOBAL_VAL = val_dataloader._sampler.num_batches_per_epoch # 903
    NUM_VAL_SHARDS = val_dataloader._sampler._shard_options.shard_count
    STEPS_PER_EPOCH_LOCAL_VAL = STEPS_PER_EPOCH_GLOBAL_VAL // NUM_VAL_SHARDS
    
    logging.info(f'VAL_BATCH_SIZE={VAL_BATCH_SIZE}, TOTAL_RECORDS_VAL={TOTAL_RECORDS_VAL}, NUM_VAL_SHARDS={NUM_VAL_SHARDS}')
    logging.info(f'STEPS_PER_EPOCH_GLOBAL_VAL={STEPS_PER_EPOCH_GLOBAL_VAL}')
    logging.info(f'STEPS_PER_EPOCH_LOCAL_VAL={STEPS_PER_EPOCH_LOCAL_VAL}')
    
    if save_checkpoints:
        logging.info(f'worker_rank={rank}: constructing checkpoint managers')
        mngr_latest = ocp.CheckpointManager(latest_checkpoint_uri,
              item_handlers={
                'model': ocp.StandardCheckpointHandler(),
                'opt': ocp.StandardCheckpointHandler(),
                'global_step': ocp.StandardCheckpointHandler(),
                'rngs': ocp.StandardCheckpointHandler(),
                'train_dataloader': grain.checkpoint.CheckpointHandler(),
                'config': ocp.handlers.JsonCheckpointHandler()
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=2)
        )
        mngr_best = ocp.CheckpointManager(best_checkpoint_uri,
            item_handlers={
                'model': ocp.StandardCheckpointHandler(),
                'opt': ocp.StandardCheckpointHandler(),
                'global_step': ocp.StandardCheckpointHandler(),
                'rngs': ocp.StandardCheckpointHandler(),
                'train_dataloader': grain.checkpoint.CheckpointHandler(),
                'config': ocp.handlers.JsonCheckpointHandler()
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1)
        )

    ndcg_text = f'ndcg_{top_k}'
    mrr_text = f'mrr_{top_k}'
    recall_text = f'recall_{top_k}'
    
    #configure for early stopping when ndcg stops changing
    patience = 5
    best_ndcg = -1.0
    epochs_without_improvement = 0
    delay = 10 # min number of epochs to learn.  for large graphs, consider using 20
    
    epoch_avg_train_loss = []
    early_stop_triggered = [False]
    
    if restored_train_dataloader_iter is None:
        train_dataloader_iter = iter(train_dataloader)
        start_batch_idx = 0
    else:
        train_dataloader_iter = restored_train_dataloader_iter
        if restored_global_step is None:
            raise RuntimeError('globalrestored_global_step_step cannot be None if restored_train_dataloader_iter because restore is implicit')
        start_batch_idx = restored_global_step // (TRAIN_BATCH_SIZE * NUM_TRAIN_SHARDS)
    
    #NOTE: cannot improve efficiency for this outer loop because gradient loss needs to
    # be calculated and updated for each iteration.
    
    global_data_pspec = P(('processes', 'local_devices'))
    model_mesh = get_model_mesh()
    
    data_mesh = jax.sharding.Mesh(jax.devices(), axis_names=('data',))
    data_pspec = jax.sharding.PartitionSpec(('data'))
    data_sharding = jax.sharding.NamedSharding(data_mesh, P("data"))
    
    jax_graph_comp_dict = calc_number_jax_graph_components(config_dict['batch_size'],
        config_dict['max_history'], config_dict['num_candidates'], n_local_devices=n_local_devices)
    
    multihost = jax.process_count() > 1
    is_on_gpu = is_running_on_gpu()
    
    use_debug = ("debug" in config_dict and config_dict["debug"])
    
    log_interval = 10
    if (TOTAL_RECORDS // n_local_devices)//TRAIN_BATCH_SIZE > 100:
        log_interval = 100
    
    if is_on_gpu:
        from flax.jax_utils import prefetch_to_device
        device_iterator = prefetch_to_device(train_dataloader_iter, size=2)
    else:
        device_iterator = train_dataloader_iter
    
    import threading
    import queue
    
    def build_sharded_prefetcher(iterator, is_gpu, multihost, data_mesh,
            data_pspec, data_sharding, prefetch_size=2):
        """
        Asynchronously pulls padded batches from Grain, applies NamedSharding,
        and pushes them to device VRAM in a background thread.
        """
        batch_queue = queue.Queue(maxsize=prefetch_size)
        
        def prefetch_worker():
            try:
                for loop_idx, padded_super_graph_0 in enumerate(iterator):
                    # Apply the sharding/transfer logic in the background thread!
                    if is_gpu or multihost:
                        if multihost:
                            padded_super_graph = jax.tree_util.tree_map(
                                lambda x: multihost_utils.host_local_array_to_global_array(
                                    x, data_mesh, data_pspec),
                                padded_super_graph_0
                            )
                        else:
                            padded_super_graph = jax.tree_util.tree_map(
                                lambda x: jax.device_put(x, data_sharding),
                                padded_super_graph_0
                            )
                    else:
                        padded_super_graph = padded_super_graph_0
                    batch_queue.put((loop_idx, padded_super_graph))
            except Exception as e:
                batch_queue.put(e)
            finally:
                batch_queue.put(None)
        # Start the worker thread
        threading.Thread(target=prefetch_worker, daemon=True).start()
        # Yield the batches to the main training loop
        while True:
            item = batch_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
            
    #an atempt to keep the gpu busier:
    device_iterator = build_sharded_prefetcher(
        iterator=train_dataloader_iter,
        is_gpu=is_on_gpu,
        multihost=multihost,
        data_mesh=data_mesh,
        data_pspec=data_pspec,
        data_sharding=data_sharding,
        prefetch_size=2
    )
    
    last_epoch = 0
    for loop_idx, padded_super_graph in device_iterator:
        
        if use_debug:
            logging.info(f"START_BATCH_TIME: {time.time()}")
        
        batch_idx = start_batch_idx + loop_idx
        local_step = batch_idx * TRAIN_BATCH_SIZE
        global_step = local_step * NUM_TRAIN_SHARDS
        epoch = batch_idx // STEPS_PER_EPOCH_LOCAL
        last_epoch = epoch
        
        loss = train_step(model, padded_super_graph, optimizer)
        
        epoch_avg_train_loss.append(loss)
        
        if batch_idx % log_interval == 0:# and rank==0:
            logging.info(f"batch {batch_idx}, loop_idx={loop_idx}, local step {local_step}, global_step {global_step}, (Epoch {epoch}): Train Loss {loss:.4f}")
            if is_on_gpu:
                logging.info(get_gpu_stats())
            logging.info(get_cpu_stats())

        if (batch_idx + 1) % STEPS_PER_EPOCH_LOCAL == 0:
            logging.info(f"*batch {batch_idx}, local step {local_step}, global_step {global_step}, (Epoch {epoch}): Train Loss {loss:.4f}")
            #finished a train epoch.  calc avg train loss and val metrics
            avg_train_loss = jnp.mean(jnp.array(epoch_avg_train_loss))
            epoch_avg_train_loss.clear()
            
            model.eval()
            train_metrics = eval_step(model, padded_super_graph, top_k)
            
            # val_dataloader is also sharded, so don't isolate this to only shard 0.
            # Also, this is synced across all shards, so all shards have same conditional logic for global_avg_val_metrics below here
            global_avg_val_metrics, n_val_samples = _epoch_validation(model, iter(val_dataloader), top_k, jax_graph_comp_dict)
            model.train()
            
            global_avg_val_loss = global_avg_val_metrics["loss"]
            global_avg_val_mrr = global_avg_val_metrics['mrr']
            global_avg_val_ndcg = global_avg_val_metrics['ndcg']
            global_avg_val_recall = global_avg_val_metrics['recall']
            
            logging.info(f"Epoch {epoch}: Train avg Loss {avg_train_loss:.4f} "
                  f"| train NDCG@{top_k} {train_metrics['ndcg']:.4f} "
                  f"| train MRR@{top_k} {train_metrics['mrr']:.4f} "
                  f"| train recall_{top_k} {train_metrics['recall']:.4f}"
                  f"avg val loss {global_avg_val_loss:.4f} | val NDCG@{top_k} {global_avg_val_ndcg:.4f} "
                  f"| val MRR@{top_k} {global_avg_val_mrr:.4f} | val recall_{top_k} {global_avg_val_recall:.4f}")
            
            metrics_dict = {
                "train_loss":avg_train_loss.item(),
                f"train_{mrr_text}":train_metrics['mrr'].item(),
                f"train_{ndcg_text}" : train_metrics['ndcg'].item(),
                f"train_{recall_text}" : train_metrics['recall'].item(),
                "val_loss":global_avg_val_loss,
                f"val_{mrr_text}":global_avg_val_mrr,
                f"val_{ndcg_text}":global_avg_val_ndcg,
                f"val_{recall_text}":global_avg_val_recall
            }
            
            if save_checkpoints:
                #orbax for checkpointing.  saves latest 2
                convert_fn = partial(convert_to_global, mesh=model_mesh)
                _graphdef, model_state = nnx.split(model)
                _, opt_state = nnx.split(optimizer)
                global_model_state = jax.tree_util.tree_map(convert_fn, model_state)
                global_opt_state = jax.tree_util.tree_map(convert_fn, opt_state)
                global_step_state = jax.tree_util.tree_map(convert_fn,{'global_step': jnp.array(global_step, dtype=jnp.int32)})
                global_rng_state = jax.tree_util.tree_map(convert_fn, nnx.state(rngs))
                mngr_latest.save(
                    epoch,
                    args=ocp.args.Composite(
                        model=ocp.args.StandardSave(global_model_state),
                        opt=ocp.args.StandardSave(global_opt_state),
                        global_step=ocp.args.StandardSave(global_step_state),
                        # NNX RNGs need to be converted to state (dictionary of arrays)
                        rngs=ocp.args.StandardSave(global_rng_state),
                        # Include your dataloader from before
                        train_dataloader=grain.checkpoint.CheckpointSave(train_dataloader_iter),
                        config=ocp.args.JsonSave(config_dict)
                    )
                )
                mngr_latest.wait_until_finished()  # Ensure it's on disk
                if validate_checkpoint_restores:
                    _assert_checkpoints_restore(latest_checkpoint_uri, model, val_dataloader, global_step, top_k)
                    validate_checkpoint_restores = False #only need to check it once
            
            if global_avg_val_ndcg > best_ndcg + 1e-6:
                best_ndcg = global_avg_val_ndcg
                epochs_without_improvement = 0
                if rank == 0:
                    logging.info(f"  New best val NDCG! ({global_avg_val_ndcg})")
                if save_checkpoints:
                    logging.info(f'worker_rank={rank}: saving best checkpoint')
                    mngr_best.save(
                        epoch,
                        args=ocp.args.Composite(
                            model=ocp.args.StandardSave(global_model_state),
                            opt=ocp.args.StandardSave(global_opt_state),
                            global_step=ocp.args.StandardSave(global_step_state),
                            # NNX RNGs need to be converted to state (dictionary of arrays)
                            rngs=ocp.args.StandardSave(global_rng_state),
                            # Include your dataloader from before
                            train_dataloader=grain.checkpoint.CheckpointSave(train_dataloader_iter),
                            config=ocp.args.JsonSave(config_dict)
                        )
                    )
                    mngr_best.wait_until_finished()  # Ensure it's on disk
            elif epoch >= delay:
                epochs_without_improvement += 1
                if rank == 0:
                    logging.info( f"  No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                if rank == 0:
                    logging.info(f"Early stopping triggered at epoch {epoch}.")
                early_stop_triggered[0] = True
                break
            
            if rank == 0:
                logging.info(f'worker_{rank}: log MLFlow metrics')
                
                mlflow.log_metrics(metrics_dict, step=epoch)
                #check for whether Vizier pruning suggests a stop of this trial
                if trial is not None:
                    trial.add_measurement(
                        vz.Measurement(
                            metrics={f'ndcg_{top_k}': global_avg_val_ndcg},
                            steps=epoch,
                            elapsed_secs=(time.perf_counter() - start_time)
                        ))
                    #early stopping not currently implement in vizier study, but if it were:
                    #if epoch >= delay and trial.check_early_stopping():
                    #    early_stop_triggered[0] = True
                    #    break
                        
        if use_debug:
            logging.info(f"END_BATCH_TIME: {time.time()}")
        
        if early_stop_triggered[0]:
            break
            
    logging.info(f'elapsed time for _train_fn in sec = {time.perf_counter() - start_time}.  last_epoch={last_epoch}')

    return best_ndcg

def build_model_optimizer_and_dataloaders(config:dict, rngs:nnx.Rngs) -> Dict[str, Any]:
    """
    build the model, optimizer, and dataloaders and return them in a dictionary that has keys {"rngs", "model", "optimizer", 'train_dataloader',
    'val_dataloader', 'num_users', 'num_movies', 'embed_len'}  where num_users and num_movies are the number of users and movies in the
    entire user and movie catalog represented by the embeddings.
    
    :param config:
    :param rngs:
    :return: dictionary with keys {"rngs", "model", "optimizer", 'train_dataloader',
    'val_dataloader', 'num_users', 'num_movies'}
    """
    if rngs is None:
        raise ValueError('rngs cannot be None')
    
    req_keys = {'user_embeddings_uri', 'movie_embeddings_uri', 'movies_uri',
        'recommendations_uri', 'recommendations_ts_uri', 'ratings_train_liked_uri',
        'ratings_train_3_uri', 'ratings_train_disliked_uri',
        'ratings_val_liked_uri', 'ratings_val_3_uri', 'ratings_val_disliked_uri',
        'max_history', 'num_epochs', 'batch_size', 'seed'}
    for key in req_keys:
        if key not in config:
            raise ValueError(f'missing key {key} in config')
    
    worker_rank = jax.process_index()

    if "num_users" not in config or 'embed_len' not in config:
        num_users, num_movies, embed_len = get_num_users_movies(
            user_embeddings_uri=config['user_embeddings_uri'],
            movie_embeddings_uri=config['movie_embeddings_uri'])
        config['num_users'] = num_users
        config['num_movies'] = num_movies
        config['embed_len'] = embed_len

    train_dataloader, val_dataloader = create_train_and_val_dataloaders(
        num_users = config['num_users'],
        user_embeddings_uri = config['user_embeddings_uri'],
        movie_embeddings_uri = config['movie_embeddings_uri'],
        movies_uri=config['movies_uri'],
        recommendations_uri=config['recommendations_uri'],
        recommendations_ts_uri=config['recommendations_ts_uri'],
        ratings_train_data_uri=config['ratings_train_liked_uri'],
        ratings_train_history_uris=[config['ratings_train_liked_uri'], config['ratings_train_3_uri'],
            config['ratings_train_disliked_uri']],
        ratings_train_disliked_uris=[config['ratings_train_disliked_uri']],
        ratings_val_data_uri=config['ratings_val_liked_uri'],
        ratings_val_history_uris=[
            config['ratings_train_liked_uri'], config['ratings_train_3_uri'],
            config['ratings_train_disliked_uri'],
            config['ratings_val_liked_uri'], config['ratings_val_3_uri'],
            config['ratings_val_disliked_uri']],
        ratings_val_disliked_uris=[config['ratings_train_disliked_uri'], config['ratings_val_disliked_uri']],
        max_history=config['max_history'],
        num_candidates=config['num_candidates'],
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        seed=config.get('seed', 0),)
    
    process_count = jax.process_count()
    num_records = train_dataloader._data_source.__len__()
    steps_per_epoch = num_records // config['batch_size']  # 7,343 steps
    steps_per_epoch_local = steps_per_epoch // process_count
    total_training_steps_local = config['num_epochs'] * steps_per_epoch_local
    warmup_steps = int(0.1 * total_training_steps_local)
    
    nnx.use_eager_sharding(False)
    model_mesh = get_model_mesh()
    with jax.set_mesh(model_mesh):
        
        #each model gets the same rngs so will have the same initialization even though in a different process
        model = GraphRanker(
            emb_in_dim = config['embed_len'],
            num_candidates=config['num_candidates'],
            hidden_features=config['hidden_dim'],
            num_layers=config['num_layers'],
            out_features=config['out_dim'],
            heads=config['num_heads'],
            edge_embed_dim=config['edge_embed_dim'],
            dropout_rate=config['dropout_rate'], rngs=rngs)
        
        #initialize the layers with same fake data
        user_id_range = (1, config['num_users'])
        movie_id_range = (config['num_users'] + 1, config['num_users'] + config['num_movies'])
        
        fake_batch = create_fake_jagged_batch(batch_size=config['batch_size'],
            max_history=config['max_history'],
            num_candidates=config['num_candidates'], user_id_range=user_id_range,
            movie_id_range=movie_id_range,
            movie_embeddings_uri = config['movie_embeddings_uri'],
            user_embeddings_uri = config['user_embeddings_uri'])

        jax_graph_comp_dict = calc_number_jax_graph_components(
            batch_size=config['batch_size'], max_history=config['max_history'],
            num_candidates=config['num_candidates'], n_local_devices=jax.local_device_count())
        
        padded_graph, _ = optimized_batch_and_pad(
            batch=fake_batch,
            max_nodes=jax_graph_comp_dict['max_nodes'],
            max_edges=jax_graph_comp_dict['max_edges'],
            max_graphs=jax_graph_comp_dict['max_graphs'],
        )
        
        model.eval()
        model(padded_graph)
        model.train()
        
        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config['learning_rate'],
            warmup_steps=warmup_steps,
            decay_steps=total_training_steps_local,
            end_value=1e-6  # Minimum learning rate tail
        )
        
        optimizer = nnx.Optimizer(model,
            optax.adamw(learning_rate=lr_scheduler,
                weight_decay=config['weight_decay']), wrt=nnx.Param)
        
        def to_named_sharding(spec):
            # If the layer had no sharding annotation, spec will be None.
            # We replace it with P() to ensure it is fully replicated across the mesh.
            if spec is None:
                spec = P()
            return jax.sharding.NamedSharding(model_mesh, spec)
            
        model_state = nnx.state(model)  # The model's state, a pure pytree.
        pspecs = nnx.get_partition_spec(model_state)  # Strip out the annotations from state.
        sharding_tree = jax.tree.map(to_named_sharding, pspecs)
        sharded_model_state = jax.device_put(model_state, sharding_tree)
        nnx.update(model, sharded_model_state)  # The model is sharded now!
        
        opt_state = nnx.state(optimizer)
        pspecs = nnx.get_partition_spec(opt_state)
        sharding_tree = jax.tree.map(to_named_sharding, pspecs)
        sharded_opt_state = jax.device_put(opt_state, sharding_tree)
        nnx.update(optimizer, sharded_opt_state)
        
    return {"rngs": rngs, "model": model, "optimizer": optimizer,
        'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader,
        'num_users': config['num_users'], 'num_movies': config['num_movies'],
        'embed_len' : config['embed_len']}

def run_train_phase(config: dict, trial:Trial=None, save_checkpoints:bool=False) -> Tuple[float, str]:
    """
    train the model given data and params specified in config dict and return best validation set ndcg@20 metric and
    return the mlflow_run_id
    :param config:
    :param trial:
    :param save_checkpoints:
    :return: val_ndcg_20, mlflow_run_id
    """
    if "phase" not in config:
        raise LookupError(f"config is missing key 'phase'")
    
    #fixed top_k for consistent stats with retrieval and reranker
    config['top_k'] = 20
    
    req_keys = {'user_embeddings_uri', 'movie_embeddings_uri', 'movies_uri',
        'recommendations_uri', 'recommendations_ts_uri',
        'ratings_train_liked_uri',
        'ratings_train_3_uri', 'ratings_train_disliked_uri',
        'ratings_val_liked_uri', 'ratings_val_3_uri',
        'ratings_val_disliked_uri',
        'max_history', 'num_epochs', 'batch_size', 'seed'}
    for key in req_keys:
        if key not in config:
            raise LookupError(f"config is missing {key}")

    #fail quickly if data are not valid
    validate_movies(config['movies_uri'])
    validate_embedding(config['user_embeddings_uri'])
    validate_embedding(config['movie_embeddings_uri'])
    validate_movie_recommendations(config['recommendations_uri'])
    validate_movie_recommendations_timestamps(config['recommendations_ts_uri'])
    validate_ratings(config['ratings_train_liked_uri'])
    validate_ratings(config['ratings_train_3_uri'])
    validate_ratings(config['ratings_train_disliked_uri'])
    validate_ratings(config['ratings_val_liked_uri'])
    validate_ratings(config['ratings_val_3_uri'])
    validate_ratings(config['ratings_val_disliked_uri'])

    worker_rank = jax.process_index()

    logging.info(f'worker_{worker_rank}: train_fn')
    
    if worker_rank == 0:
        for key in {"phase", "mlflow_experiment_name", "mlflow_experiment_id",
            "mlflow_parent_run_id"}:
            if key not in config:
                raise LookupError(f"config is missing {key}")
    
    rngs = nnx.Rngs(config.get('seed', 0))
    
    _dict = build_model_optimizer_and_dataloaders(config, rngs=rngs)
    
    model = _dict['model']
    optimizer = _dict['optimizer']
    train_dataloader = _dict['train_dataloader']
    val_dataloader = _dict['val_dataloader']

    num_users = _dict['num_users']
    num_movies = _dict['num_movies']
    embed_len = _dict['embed_len']
    
    config['num_users'] = num_users
    config['num_movies'] = num_movies
    config['embed_len'] = embed_len
    
    mlflow_run = None
    best_val_ndcg_k = -1.0
    
    run_name = get_canonical_mlflow_run_name(config)
    
    try:
    
        if worker_rank == 0:
            logging.info(f"mlflow set experiment: {config['mlflow_experiment_name']}")
            experiment = mlflow.set_experiment(experiment_name=config['mlflow_experiment_name'])
            # don't use nested=True because the parent isn't in the same thread in production
            logging.info(f"mlflow start run: {run_name}")
            mlflow_run = mlflow.start_run(
                run_name=run_name,
                #tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                tags = {"mlflow.parentRunId" : config['mlflow_parent_run_id']},
                experiment_id=experiment.experiment_id,
            )
            config['mlflow_run_id'] = mlflow_run.info.run_id
            mlflow.set_tag("phase", config["phase"]) #do not move this before start_run
            mlflow.log_params(stringify_mlflow_params(config))
            mlflow.log_text(str(model), "model_summary.txt")
            logging.info(f'worker_{worker_rank}: started MLFlow run_id={mlflow_run.info.run_id}')
        if save_checkpoints:
            # paradigm is that we save checkpoints for "train" phase, but not HPO trial phases
            sfx = f"{config['study_name']}/{run_name}"
            config['best_checkpoint_uri'] = f"{config['best_checkpoint_uri']}/{sfx}"
            config['latest_checkpoint_uri'] = f"{config['latest_checkpoint_uri']}/{sfx}"
            create_dirs_if_is_filepath(config['best_checkpoint_uri'])
            create_dirs_if_is_filepath(config['latest_checkpoint_uri'])
            if worker_rank == 0:
                # cannot update the mlflow logged param, so instead creata tag for the uris
                mlflow.set_tag('best_checkpoint_uri',  config['best_checkpoint_uri'])
                mlflow.set_tag('latest_checkpoint_uri',config['latest_checkpoint_uri'])
                if trial is not None:
                    trial.update_metadata(
                        vz.Metadata({'best_checkpoint_uri': config['best_checkpoint_uri']}))
        
        logging.info( f"expect the model training to start w/ loss = {-log(1. / config['num_candidates'])}")
        
        validate_checkpoint_restores = config.get('validate_checkpoint_restores', False)
        
        best_val_ndcg_k = _train_fn(model=model, train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer, top_k=config['top_k'],
            latest_checkpoint_uri=config['latest_checkpoint_uri'],
            best_checkpoint_uri=config['best_checkpoint_uri'],
            rngs=rngs,
            config_dict=config,
            trial=trial,
            save_checkpoints=save_checkpoints,
            validate_checkpoint_restores=validate_checkpoint_restores)
            
        if "debug" in config and config['debug'] and save_checkpoints:
            logging.info(f"checkpoints save to directories:\n  {config.get('best_checkpoint_uri','')}"
                  f"\n  {config.get('latest_checkpoint_uri','')}")
            
        return best_val_ndcg_k, config.get('mlflow_run_id', "")
    finally:
        logging.info(f'worker_{worker_rank}: finally clause in train_fn')
        if worker_rank==0 and mlflow_run is not None:
            mlflow.log_metric(f"final_ndcg_{config['top_k']}", float(best_val_ndcg_k))
            mlflow.end_run()
            logging.info(f"end mlflow_run_id={mlflow_run.info.run_id}")
    
def stack_val_batches(dataloader, num_steps):
    batches = []
    for i, batch in enumerate(dataloader):
        batches.append(batch)
        if i + 1 == num_steps:
            break
    # Use jax.tree.map to stack all GraphsTuples into one
    # This creates a Pytree where nodes shape is [num_steps, total_nodes, feature_dim]
    stacked_batches = jax.tree.map(lambda *args: jnp.stack(args), *batches)
    return stacked_batches

@nnx.jit
def validation_epoch_compiled(model: GraphRanker, stacked_batches, top_k):
    """
    Compiled validation epoch.
    Processes multiple batches on-device without returning to Python.
    """
    def scan_body(carry, batch):
        # Call your existing eval_step logic
        # Since eval_step is also @nnx.jit, XLA will inline it here
        metrics = eval_step(model, batch, top_k)
        return None, metrics
    
    # jax.lax.scan iterates over the leading dimension of stacked_batches
    _, metrics_history = jax.lax.scan(scan_body, None, stacked_batches)
    
    # metrics_history now contains arrays of shape [steps_per_worker, ...]
    # We average them directly on the GPU
    return jax.tree.map(lambda x: jnp.mean(x), metrics_history)

def run_full_validation(model, val_dataloader, top_k, steps=903, micro_batch_size=100):
    all_step_metrics = []
    
    # Iterate through the loader in chunks
    for _ in range(0, steps, micro_batch_size):
        # 1. Stack a smaller chunk (e.g., 100 batches)
        chunk = stack_val_batches(val_dataloader, micro_batch_size)
        
        # 2. Run compiled validation on this chunk
        # This keeps GPU utilization high without OOMing the Host RAM
        chunk_metrics = validation_epoch_compiled(model, chunk, top_k)
        all_step_metrics.append(chunk_metrics)
    
    # Final average across chunks
    return jax.tree.map(lambda *args: jnp.mean(jnp.array(args)),
        *all_step_metrics)

def restore_items_from_checkpoint(checkpoint_uri:str, get_earliest:bool=False) -> Dict[str, Any]:
    """
    restore the model, dataloader and state from checkpoint_uri.  if get_Earliest is set to True,
    the earlies of the 2 saved runs will be returned.  This is useful for testing continuation of
    training from an earlier checkpoint.
    :param checkpoint_uri:
    :param get_earliest: False by default and so returns latest of the 2 saved checkpoints, else if get_Earliest=True
    returns the earlier of the 2 checkpoints.  note that for "best" rather than "latest" checkpoints, only 1 is saved.
    :return: dictionary holding: 'model', 'optimizer', 'train_dataloader', 'train_dataloader_iter',
            'val_dataloader', 'rngs', 'global_step', 'num_users', 'num_movies', 'config'
    """
    n_keep = 2 if checkpoint_uri.find('latest') > -1 else 1
    
    model_mesh = get_model_mesh()
    
    mngr = ocp.CheckpointManager(checkpoint_uri,
        item_handlers={
            'model': ocp.StandardCheckpointHandler(),
            'opt': ocp.StandardCheckpointHandler(),
            'global_step': ocp.StandardCheckpointHandler(),
            'rngs': ocp.StandardCheckpointHandler(),
            'train_dataloader': grain.checkpoint.CheckpointHandler(),
            'config': ocp.handlers.JsonCheckpointHandler()
        },
        options=ocp.CheckpointManagerOptions(max_to_keep=n_keep)
    )
    
    epoch = mngr.all_steps()[0] if get_earliest else mngr.latest_step()
    if epoch is None:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_uri}")
    
    restore_fn = partial(convert_to_global, mesh=model_mesh, sync=False)
    
    # restore config, then rngs, so can restore model and dataloaders from them
    restored_config = mngr.restore(epoch, args=ocp.args.Composite(config=ocp.args.JsonRestore()))
    config = restored_config['config']

    _dict = build_model_optimizer_and_dataloaders(config, rngs=nnx.Rngs(config.get('seed', 0)))
    model = _dict['model']
    optimizer = _dict['optimizer']
    train_dataloader = _dict['train_dataloader']
    val_dataloader = _dict['val_dataloader']
    num_users = _dict['num_users']
    num_movies = _dict['num_movies']
    embed_len = _dict['embed_len']
    
    #model_mesh = get_model_mesh()
    #graphdef_model, model_state = nnx.get_abstract_model(lambda: model, model_mesh)
    
    #restore state to those objects:
    model_graph, model_state = nnx.split(model)
    opt_graph, opt_state = nnx.split(optimizer)
    rngs = nnx.Rngs(config.get('seed', 0))
    
    global_model_target = jax.tree_util.tree_map(restore_fn, model_state)
    global_opt_target = jax.tree_util.tree_map(restore_fn, opt_state)
    global_step_target = jax.tree_util.tree_map(restore_fn,  {'global_step': jnp.array(0, dtype=jnp.int32)})
    global_rng_target = jax.tree_util.tree_map(restore_fn, nnx.state(rngs))
    
    restored = mngr.restore(
        epoch,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(global_model_target),
            opt=ocp.args.StandardRestore(global_opt_target),
            global_step=ocp.args.StandardRestore(global_step_target),
            # Grain requires the actual iterator object to restore state in-place
            train_dataloader=grain.checkpoint.CheckpointRestore( iter(train_dataloader)),
            rngs=ocp.args.StandardRestore(global_rng_target),
        )
    )
    
    train_dataloader_iter = restored['train_dataloader']
    nnx.update(optimizer, restored['opt'])
    nnx.update(model, restored['model'])
    nnx.update(rngs, restored['rngs'])
    global_step = int(restored['global_step']['global_step'].item())
    
    logging.info(f"worker_rank ={jax.process_index()}: Restored model at step {global_step}")
    
    return {
        'model': model, 'optimizer': optimizer,
        'train_dataloader': train_dataloader,
        'train_dataloader_iter':train_dataloader_iter,
        'val_dataloader': val_dataloader,
        'rngs': rngs,
        'global_step': global_step,
        'num_users': num_users,
        'num_movies': num_movies,
        'embed_len' : embed_len,
        'config': config
    }

def run_test_phase(config: dict):
    
    if "phase" not in config:
        raise LookupError("config requires a 'phase' parameter")
    
    if config['phase'] not in {"test-best", "test-given"}:
        raise ValueError("'phase' must be 'test-best' or 'test-given'")
    
    for key in ('seed', 'ratings_test_liked_uri'):
        if key not in config:
            raise ValueError(f"key {key} is missing from config")
    
    # fixed top_k for consistent stats with retrieval and reranker
    config['top_k'] = 20
    
    logging.info(f'run_test_phase config: {config}')

    worker_rank = jax.process_index()
    
    if worker_rank == 0:
        for key in {"mlflow_experiment_name", "mlflow_experiment_id",
            "mlflow_parent_run_id", "mlflow_tracking_uri"}:
            if key not in config:
                raise LookupError(f"config is missing {key}")
    
    req_keys = {'ratings_test_liked_uri',
        'ratings_test_3_uri', 'ratings_test_disliked_uri'}
    for key in req_keys:
        if key not in config:
            raise LookupError(f"config is missing {key}")

    validate_ratings(config['ratings_test_liked_uri'])
    validate_ratings(config['ratings_test_3_uri'])
    validate_ratings(config['ratings_test_disliked_uri'])

    if config['phase'] == 'test-best':
        restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['best_checkpoint_uri'])
    else:
        #test-given, use given checkpoint path to restore, test_checkpoint_uri
        restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['test_checkpoint_uri'])
    
    num_users = restore_dict['num_users']
    num_movies = restore_dict['num_movies']
    config['num_users'] = num_users
    config['num_movies'] = num_movies
    config['embed_len'] = restore_dict['embed_len']
    
    model = restore_dict['model']
    model.eval()
    
    mlflow_run = None
    run_name = get_canonical_mlflow_run_name(config)
    try:
        if worker_rank == 0:
            mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
            # don't use nested=True because the parent isn't in the same thread in production
            #there may be ACL to solve for this:
            experiment = mlflow.get_experiment_by_name(config['mlflow_experiment_name'])
            mlflow_run = mlflow.start_run(
                run_name=run_name,
                # tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                tags={"mlflow.parentRunId": config['mlflow_parent_run_id']},
                experiment_id=experiment.experiment_id,
            )
            config['mlflow_run_id'] = mlflow_run.info.run_id
            mlflow.set_tag("phase", config["phase"])  # do not move this before start_run
        
        #write to config from restore dict config
        for key in {"max_history", "num_candidates", "batch_size",
            "movie_embeddings_uri",
            "movies_uri", "recommendations_uri", "recommendations_ts_uri",
            "ratings_train_liked_uri", "ratings_train_3_uri", "ratings_train_disliked_uri",
            "ratings_val_liked_uri", "ratings_val_3_uri", "ratings_val_disliked_uri",
            *model_params_trainable_keys}:
            if key in restore_dict['config']:
                config[key] = restore_dict['config'][key]


        #fail quickly if data are not valid
        validate_movies(config['movies_uri'])
        validate_embedding(config['user_embeddings_uri'])
        validate_embedding(config['movie_embeddings_uri'])
        validate_movie_recommendations(config['recommendations_uri'])
        validate_movie_recommendations_timestamps(config['recommendations_ts_uri'])
        validate_ratings(config['ratings_train_liked_uri'])
        validate_ratings(config['ratings_train_3_uri'])
        validate_ratings(config['ratings_train_disliked_uri'])
        validate_ratings(config['ratings_val_liked_uri'])
        validate_ratings(config['ratings_val_3_uri'])
        validate_ratings(config['ratings_val_disliked_uri'])

        max_history = config['max_history']
        num_candidates = config['num_candidates']
        batch_size = config['batch_size']
        
        #these uris are all in config too, excepting test_ratings
        test_dataloader = create_test_dataloader(
            num_users = num_users,
            user_embeddings_uri = config["user_embeddings_uri"],
            movie_embeddings_uri = config["movie_embeddings_uri"],
            movies_uri = config['movies_uri'],
            recommendations_uri = config['recommendations_uri'],
            recommendations_ts_uri = config['recommendations_ts_uri'],
            
            rattings_data_uri = config['ratings_test_liked_uri'],
            ratings_history_uris=[
                config['ratings_train_liked_uri'],
                config['ratings_train_3_uri'],
                config['ratings_train_disliked_uri'],
                config['ratings_val_liked_uri'], config['ratings_val_3_uri'],
                config['ratings_val_disliked_uri'],
                config['ratings_test_liked_uri'],
                config['ratings_test_3_uri'],
                config['ratings_test_disliked_uri'],
            ],
            ratings_disliked_uris=[config['ratings_train_disliked_uri'],
                config['ratings_val_disliked_uri'], config['ratings_test_disliked_uri']],
            
            max_history = max_history,
            num_candidates = num_candidates,
            batch_size = batch_size,
            seed = config.get('seed', 0))
        
        if not isinstance(test_dataloader._sampler, BatchSampler):
            raise ValueError(
                "test_dataloader sampler must be an instance of BatchSampler")
        
        jax_graph_comp_dict = calc_number_jax_graph_components(batch_size,
            max_history, num_candidates, n_local_devices=jax.local_device_count())
        
        global_test_metrics, n_val_samples = _epoch_validation(model, iter(test_dataloader), config['top_k'], jax_graph_comp_dict)
    
        out_dict = {f"test_{key}_{config['top_k']}" : value for key, value in global_test_metrics.items()}
        #to be consitent w/ train, change the loss label:
        out_dict['test_loss'] = out_dict[f"test_loss_{config['top_k']}"]
        del out_dict[f"test_loss_{config['top_k']}"]
    
        if worker_rank == 0:
            if mlflow_run is not None:
                for key, value in out_dict.items():
                    mlflow.log_metric(key, float(value))
        
        return out_dict
        
    finally:
        if worker_rank == 0 and mlflow_run is not None:
            mlflow.end_run()

def resume_train_fn(config: dict, trial: Trial=None, save_checkpoints: bool=False):
    
    worker_rank = jax.process_index()
    
    if worker_rank == 0:
        for key in {"phase", "mlflow_experiment_name", "mlflow_experiment_id",
            "mlflow_parent_run_id"}:
            if key not in config:
                raise LookupError(f"config is missing {key}")
    
    restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['latest_checkpoint_uri'])
    
    #TODO: handle case when  config['phase'] is not same as restore dict phase
    
    config.update(**restore_dict['config'])
    
    req_keys = {'user_embeddings_uri', 'movie_embeddings_uri', 'movies_uri',
        'recommendations_uri', 'recommendations_ts_uri',
        'ratings_train_liked_uri',
        'ratings_train_3_uri', 'ratings_train_disliked_uri',
        'ratings_val_liked_uri', 'ratings_val_3_uri',
        'ratings_val_disliked_uri',
        'max_history', 'num_epochs', 'batch_size', 'seed'}
    for key in req_keys:
        if key not in config:
            raise LookupError(f"config is missing {key}")
    
    logging.info(f'resume_train_fn config: {config}')
    
    best_val_ndcg_k = -1.0
    mlflow_run = None
    run_name = get_canonical_mlflow_run_name(config)
    try:
        if worker_rank == 0:
            mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
            experiment = mlflow.set_experiment(experiment_name=config['mlflow_experiment_name'])
            # Start a run specifically for this HPO trial
            # don't use nested=True because the parent isn't in the same thread in production
            run_id = config.get('mlflow_run_id', None)  # is not None for a "restore, resume training"
            # in production, there may be ACL to solve for this:
            if run_id is None:
                mlflow_run = mlflow.start_run(
                    run_name=run_name,
                    # tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                    tags={"mlflow.parentRunId": config['mlflow_parent_run_id']},
                    experiment_id=experiment.experiment_id,
                )
            else:
                mlflow_run = mlflow.start_run(
                    run_id=run_id,
                    # tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                    tags={"mlflow.parentRunId": config['mlflow_parent_run_id']},
                )
            config['mlflow_run_id'] = mlflow_run.info.run_id
        
        model = restore_dict['model']
        model.train()
        
        best_val_ndcg_k = _train_fn(model=model,
            train_dataloader=restore_dict['train_dataloader'],
            val_dataloader=restore_dict['val_dataloader'],
            optimizer=restore_dict['optimizer'],
            top_k=config['top_k'],
            latest_checkpoint_uri=config['latest_checkpoint_uri'],
            best_checkpoint_uri=config['best_checkpoint_uri'],
            rngs=restore_dict['rngs'],
            trial=trial,
            config_dict=config,
            restored_train_dataloader_iter=restore_dict['train_dataloader_iter'],
            restored_global_step=restore_dict['global_step'],
            save_checkpoints=save_checkpoints,
        )
        return best_val_ndcg_k
        
    finally:
        if mlflow_run is not None:
            mlflow.log_metric(f"final_ndcg_{config['top_k']}",
                float(best_val_ndcg_k))
            mlflow.end_run()
    
def _assert_checkpoints_restore(checkpoint_uri:str, model, val_data_loader, global_step, top_k:int=20):
    
    logging.info(f'worker_rank={jax.process_index()}: begin _assert_checkpoints_restore')
    
    restore_dict = restore_items_from_checkpoint(checkpoint_uri)
    logging.info(f'worker_rank={jax.process_index()}: global_step={global_step}, restored global_step={restore_dict["global_step"]}')
    restored_model = restore_dict['model']
    restored_model.eval()
    model.eval()
    
    jax_graph_comp_dict = calc_number_jax_graph_components(
        restore_dict['config']['batch_size'],
        restore_dict['config']['max_history'],
        restore_dict['config']['num_candidates'], n_local_devices=jax.local_device_count())
    
    import copy
    loader_current = copy.deepcopy(val_data_loader)
    loader_restored = copy.deepcopy(val_data_loader)
    
    # iter(x) makes a new iterator state
    global_avg_val_metrics_current, n_val_samples_current = _epoch_validation(model, iter(loader_current), top_k, jax_graph_comp_dict)
    
    multihost_utils.sync_global_devices( "sync_barrier_for_model_validation")
    
    global_avg_val_metrics_restored, n_val_samples_restored = _epoch_validation(restored_model, iter(loader_restored), top_k, jax_graph_comp_dict)
    
    multihost_utils.sync_global_devices( "sync_barrier_for_restored_model_validation")
    
    logging.info(f'n_val_samples_current={n_val_samples_current}, n_val_samples_restored = {n_val_samples_restored}')
    
    all_similar = True
    for key in ("loss", "mrr", "ndcg", "recall"):
        logging.info(f'worker_rank={jax.process_index()}: key={key}, model={global_avg_val_metrics_current[key]}, restored={global_avg_val_metrics_restored[key]}')
        if not jnp.allclose(global_avg_val_metrics_current[key], global_avg_val_metrics_restored[key]):
            all_similar = False
    
    model.train()
    
    #logging.info(f'worker_rank={jax.process_index()}:\n    summary of model={str(model)}\n    summary of restored={str(restore_dict["model"])}')
    
    # print out model state
    #_graphdef, model_state = nnx.split(model)
    #_graphdef_restored, model_state_restored = nnx.split(restore_dict['model'])
    #logging.info(
    #    f'worker_rank={jax.process_index()}:\n    summary of model_state={model_state}\n    summary of restored model_state={model_state_restored}',
    #    )
    check_model_state_equality(model, restore_dict['model'])
    
    assert(all_similar)
    assert(n_val_samples_current == n_val_samples_restored)
    
    logging.info(f'worker_rank={jax.process_index()}:checkpoint validated for {checkpoint_uri}')

def check_model_state_equality(model_a, model_b, rtol=1e-5, atol=1e-8) -> bool:
    """
    Compares two Flax NNX models or State objects to verify if their
    internal structural shapes and array values are identical.
    """
    # 1. Extract the underlying nnx.State if passed as a full model instance
    #state_a = nnx.state(model_a) if not isinstance(model_a,
    #    nnx.State) else model_a
    #state_b = nnx.state(model_b) if not isinstance(model_b,
    #    nnx.State) else model_b
    #_graphdef_a, state_a = nnx.split(model_a)
    #_graphdef_b, state_b = nnx.split(model_b)
    state_a = nnx.state(model_a) if not isinstance(model_a,
        nnx.State) else model_a
    state_b = nnx.state(model_b) if not isinstance(model_b,
        nnx.State) else model_b
    
    # 2. Use the functional top-level helper to flatten the states safely
    flat_a = dict(nnx.to_flat_state(state_a))
    flat_b = dict(nnx.to_flat_state(state_b))
    
    # 3. Check for structural or key differences first
    if flat_a.keys() != flat_b.keys():
        missing_in_b = set(flat_a.keys()) - set(flat_b.keys())
        missing_in_a = set(flat_b.keys()) - set(flat_a.keys())
        logging.info("worker_rank={jax.process_index()}: ❌ Model structures DO NOT match!")
        if missing_in_b: logging.info(f"   Missing in Restored: {missing_in_b}")
        if missing_in_a: logging.info(f"   Missing in Current: {missing_in_a}")
        return False
    
    # 4. Element-wise value check across every array leaf
    mismatched_keys = []
    mismatched_vals = []
    for key, val_a in flat_a.items():
        val_b = flat_b[key]
        # Pull raw JAX arrays out of NNX Variable wrappers (like nnx.Param)
        arr_a = val_a.value if hasattr(val_a, 'value') else val_a
        arr_b = val_b.value if hasattr(val_b, 'value') else val_b
        if not jnp.allclose(arr_a, arr_b, rtol=rtol, atol=atol):
            mismatched_keys.append(key)
            v = [f'({a:.3e} , {b:.3e})' for a, b in zip(np.asarray(arr_a).ravel()[:10], np.asarray(arr_b).ravel()[:10])]
            mismatched_vals.append(",".join(v))
    if mismatched_keys:
        logging.info(f"worker_rank={jax.process_index()}: ❌ Model structures match, but values differ at {len(mismatched_keys)} parameter paths:")
        #for ii, path in enumerate(mismatched_keys[:5]):  # Limit output log spam
        for ii, path in enumerate(mismatched_keys):
            logging.info(f"worker_rank={jax.process_index()}:   -> Mismatch in layer path: {path}, values={mismatched_vals[ii]}")
        #if len(mismatched_keys) > 5:
        #    print(f"worker_rank={jax.process_index()}:   -> ... and {len(mismatched_keys) - 5} more paths.")
        return False
    
    logging.info("worker_rank={jax.process_index()}: ✅ Success! Both model states are mathematically identical.")
    return True

def create_fake_jagged_batch(batch_size: int,
    max_history: int, num_candidates: int,
    user_id_range:Tuple[int, int],movie_id_range:Tuple[int, int],
    user_embeddings_uri:str, movie_embeddings_uri:str):
    """

    :param movie_embeddings_uri:
    :param user_embeddings_uri:
    :param batch_size:
    :param max_history:
    :param num_candidates:
    :param user_id_range: a tuple of (start_user_id, end_user_id) where the range should be as large as
        batch_size
    :param movie_id_range: a tuple of (start_movie_id, end_movie_id) where the range should be as large as
        batch_size + max_history + 1 + num_candidates
    :return:
    """

    if (movie_id_range[1] - movie_id_range[0] + 1) < (num_candidates + max_history + 1):
        raise ValueError("the range of movie_id_range must be >= (num_history + num_candidates + 1)")

    user_id = np.ndarray((batch_size,), dtype=np.int32)
    movie_id = np.ndarray((batch_size,), dtype=np.int32)
    rating = np.ndarray((batch_size,), dtype=np.int32)
    history_length = np.ndarray((batch_size,), dtype=np.int32)
    history_movie_ids = np.full((batch_size, max_history), -1,dtype=np.int32)
    history_ratings = np.full((batch_size, max_history), -1, dtype=np.int32)
    candidate_ids = np.ndarray((batch_size, num_candidates), dtype=np.int32)
    labels = np.full((batch_size,  num_candidates), 0, dtype=np.int32)

    for i in range(batch_size):
        user_id[i] = user_id_range[0] + i
        movie_id[i] = movie_id_range[0] + i
        rating[i] = 4 + i%2
        history_length[i] = min(i + 1, max_history)
        for j in range(history_length[i] ):
            history_movie_ids[i][j] = movie_id_range[0] + i + 1 + j
            history_ratings[i][j] = 3 + i%2 + j%2
        for j in range(num_candidates):
            candidate_ids[i][j] = movie_id_range[0] + max_history + j
        #choose 1 to be the target positive and rewrite it
        candidate_ids[i][0] = movie_id[i]
        labels[i][0] = 1.0

    inputs = {
        "user_id" : user_id,
        "movie_id" : movie_id,
        "rating" : rating,
        "history_length" : history_length,
        "history_movie_ids" : history_movie_ids,
        "history_ratings" : history_ratings,
        "candidate_ids" : candidate_ids,
        "labels" : labels
    }

    user_movie_embeddings = read_user_movie_embeddings(
        user_embeddings_uri=user_embeddings_uri,
        movie_embeddings_uri=movie_embeddings_uri)

    transform = SparseLocalSubgraphTransform(user_movie_embeddings=user_movie_embeddings)
    graphs : List[jraph.GraphsTuple] = transform.map(inputs)

    return graphs

def create_dummy_super_padded_graph(batch_size: int,
    max_history: int, num_candidates: int, user_id_range:Tuple[int, int],
    movie_id_range:Tuple[int, int], user_embeddings_uri:str,
    movie_embeddings_uri:str):
    
    fake_graph_list = create_fake_jagged_batch(batch_size=batch_size, max_history=max_history,
        num_candidates=num_candidates, user_id_range=user_id_range,
        movie_id_range=movie_id_range,
        movie_embeddings_uri = movie_embeddings_uri,
        user_embeddings_uri = user_embeddings_uri)
    
    jax_graph_comp_dict = calc_number_jax_graph_components(batch_size,
        max_history, num_candidates, n_local_devices=jax.local_device_count())
    
    padded_super_graph, _ = optimized_batch_and_pad(
        batch=fake_graph_list,
        max_nodes=jax_graph_comp_dict['max_nodes'],
        max_edges=jax_graph_comp_dict['max_edges'],
        max_graphs=jax_graph_comp_dict['max_graphs'],
    )
    
    return padded_super_graph