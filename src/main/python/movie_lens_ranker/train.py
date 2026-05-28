"""
tune, train, test functions for a multi-host, multi-process Jax AI Stack model
and dataloader using SPMD paradigm.
"""
import time
from functools import partial
from typing import Dict, Tuple, Union, Any

import mlflow
import optax
from math import log
import jax
from jax.sharding import PartitionSpec as P
from jax import shard_map, Array
import numpy as np

from vizier.service import pyvizier as vz
from vizier.service.clients import Trial
import jraph
import jax.numpy as jnp
from flax import nnx
import rax
import grain
from flax.typing import Array
from grain._src.python.data_loader import DataLoaderIterator

from movie_lens_ranker.BatchSampler import BatchSampler

import orbax.checkpoint as ocp

from movie_lens_ranker.data_loading import create_train_and_val_dataloaders, \
    create_test_dataloader
from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.util import read_embeddings, get_env_resources, \
    stringify_mlflow_params, get_canonical_mlflow_run_name, \
    calc_number_jax_graph_components, get_model_mesh

env_resources, mesh = get_env_resources()
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
    
    arr_to_broadcast = jax.experimental.multihost_utils.broadcast_one_to_all(
        arr_to_broadcast)
    
    if isinstance(arr_to_broadcast, (np.ndarray, np.generic)):
        arr_to_broadcast = jnp.asarray(arr_to_broadcast)
    
    if is_key:
        arr_to_broadcast = jax.random.wrap_key_data(arr_to_broadcast)
    
    def data_callback(index):
        return arr_to_broadcast[index]
    
    return jax.make_array_from_callback(global_shape, global_sharding, data_callback)

def get_nontrainable_train_config(movies_uri:str,
        recommendations_uri:str, recommendations_ts_uri:str,
        ratings_train_uri:str, ratings_val_uri:str,
        train_negatives_uri:str, val_negatives_uri:str,
        latest_checkpoint_uri:str, best_checkpoint_uri:str,
        movie_embeddings_uri:str, user_embeddings_uri:str,
        num_epochs:int=120, batch_size:int=64, seed:int=0) -> Dict[str, Union[str, int, float]]:
    
    config = {}
    config['movies_uri'] = movies_uri
    config['recommendations_uri'] = recommendations_uri
    config['recommendations_ts_uri'] = recommendations_ts_uri
    config['ratings_train_uri'] = ratings_train_uri
    config['ratings_val_uri'] = ratings_val_uri
    config['train_negatives_uri'] = train_negatives_uri
    config['val_negatives_uri'] = val_negatives_uri
    config['latest_checkpoint_uri']= latest_checkpoint_uri
    config['best_checkpoint_uri']= best_checkpoint_uri
    config['movie_embeddings_uri']= movie_embeddings_uri
    config['user_embeddings_uri']= user_embeddings_uri
    config['seed'] = seed
    config['num_epochs'] = num_epochs
    config['batch_size'] = batch_size
   
    return config

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
    
    debug_weight_before = jnp.linalg.norm(model.score_head.kernel.get_value())
    
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
    
    #an optimized pmead averages all gradients:
    loss, grads = nnx.value_and_grad(loss_fn)(model, padded_graph)
    optimizer.update(model, grads)
    
    debug_weight_after = jnp.linalg.norm(model.score_head.kernel.get_value())
    diff = jnp.abs(debug_weight_before - debug_weight_after)
    # if > 1E-4, is a significant change
    # if > 1, exploding gradient or learning rate issue
    jax.debug.print("Weight Norm: Before={b:.6f}, After={a:.6f}, Delta={d:.8f}",
        b=debug_weight_before, a=debug_weight_after, d=diff)
    
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
    
# in_specs=P() tells JAX the input is a scalar
# out_specs=P() (empty) implies the output is a single global (replicated) value
#  and is sharded across the 'data' axis
@jax.jit
@partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P())
def aggregate_metric(scalar_metric):
    # context set by jax.set_mesh() during compilation.
    return jax.lax.pmean(scalar_metric, axis_name='data')

@nnx.jit
def vectorized_epoch_eval(model, mega_batch, top_k):
    # mega_batch here is a chunk of N batches stacked
    v_eval = nnx.vmap(eval_step, in_axes=(None, 0, None))
    return v_eval(model, mega_batch, top_k)

def _epoch_validation(model: GraphRanker, val_dataloader_iter: DataLoaderIterator,
        top_k: int, jax_graph_comp_dict:Dict[str, int]) -> Tuple[jax.tree.map, int]:
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
    global_data_pspec = P(('processes', 'local_devices'))
    model_mesh = get_model_mesh()
    global_avg_metrics_batches = {"loss":[], "mrr":[], "ndcg":[], "recall":[]}
    #the validation set is 1/10th the size of train, here will eval in 1 big batch if possible, else will need to loop
    # over and average results
    n_samples_tot = 0
    for graphs_tuple_batch in val_dataloader_iter:
        
        padded_super_graph_0, n_samples = pad_graph_tuple_batch(graphs_tuple_batch, jax_graph_comp_dict)
        n_samples_tot += n_samples
        
        #shard the data to local devices:
        #padded_super_graph = jax.device_put(padded_super_graph_0, data_sharding)
        padded_super_graph = jax.tree_util.tree_map(
            lambda x: jax.experimental.multihost_utils.host_local_array_to_global_array(
                x, model_mesh, global_data_pspec),
            padded_super_graph_0
        )
        
        val_metrics = eval_step(model, padded_super_graph, top_k)
        
        # val_metrics['loss'] is now an array of shape (Num_Batches,)
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

def pad_graph_tuple_batch(graph_tuple_batch: jraph.GraphsTuple, jax_graph_comp_dict:Dict[str, int],
    drop_remainder:bool=True) -> Tuple[jraph.GraphsTuple, int]:
    """
    pad a batch from the grain data loaders for use in sharding over local devices
    :param graph_tuple_batch:
    :param jax_graph_comp_dict: dictionary created from calc_number_jax_graph_components
    :return: the padded graph tuple, ready for harding across local devices
    """
    n_local_devices = jax.local_device_count()
    batch = graph_tuple_batch
    
    if drop_remainder:
        remainder = len(batch) % n_local_devices
        if remainder != 0:
            batch = batch[:-remainder]  # Drop trailing graphs to make it divisible
        
    n_samples = len(batch)
    
    batch = jraph.batch(batch)
    
    padded_super_graph_0 = jraph.pad_with_graphs(
        batch,
        n_node=jax_graph_comp_dict['max_nodes'],
        n_edge=jax_graph_comp_dict['max_edges'],
        n_graph=n_local_devices + n_samples
    )
    return padded_super_graph_0, n_samples

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
    
    start_time = time.perf_counter()
    
    rank = jax.process_index()
    
    #tracked_fn_1 = chex.assert_max_traces(train_step, n=1)
    #tracked_fn_2 = chex.assert_max_traces(eval_step, n=1)
    
    TRAIN_BATCH_SIZE = train_dataloader._sampler.batch_size
    TOTAL_RECORDS = train_dataloader._sampler.total_records
    STEPS_PER_EPOCH_GLOBAL = train_dataloader._sampler.num_batches_per_epoch  # = 7234
    NUM_TRAIN_SHARDS = train_dataloader._sampler._shard_options.shard_count
    STEPS_PER_EPOCH_LOCAL = STEPS_PER_EPOCH_GLOBAL//NUM_TRAIN_SHARDS
    
    print(f'TRAIN_BATCH_SIZE={TRAIN_BATCH_SIZE}, TOTAL_RECORDS={TOTAL_RECORDS}, NUM_TRAIN_SHARDS={NUM_TRAIN_SHARDS}', flush=True)
    print(f'STEPS_PER_EPOCH_GLOBAL_TRAIN={STEPS_PER_EPOCH_GLOBAL}')
    print(f'STEPS_PER_EPOCH_LOCAL_TRAIN={STEPS_PER_EPOCH_LOCAL}')
    print(f'NUM_EPOCHS to train={train_dataloader._sampler._num_epochs}')
    
    VAL_BATCH_SIZE = train_dataloader._sampler.batch_size
    TOTAL_RECORDS_VAL = val_dataloader._sampler.total_records
    STEPS_PER_EPOCH_GLOBAL_VAL = val_dataloader._sampler.num_batches_per_epoch # 903
    NUM_VAL_SHARDS = val_dataloader._sampler._shard_options.shard_count
    STEPS_PER_EPOCH_LOCAL_VAL = STEPS_PER_EPOCH_GLOBAL_VAL // NUM_VAL_SHARDS
    
    print(f'VAL_BATCH_SIZE={VAL_BATCH_SIZE}, TOTAL_RECORDS_VAL={TOTAL_RECORDS_VAL}, NUM_VAL_SHARDS={NUM_VAL_SHARDS}', flush=True)
    print(f'STEPS_PER_EPOCH_GLOBAL_VAL={STEPS_PER_EPOCH_GLOBAL_VAL}')
    print(f'STEPS_PER_EPOCH_LOCAL_VAL={STEPS_PER_EPOCH_LOCAL_VAL}', flush=True)
    
    if save_checkpoints:
        print(f'worker_rank={rank}: constructing checkpoint managers', flush=True)
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
    else:
        train_dataloader_iter = restored_train_dataloader_iter
        if restored_global_step is None:
            raise RuntimeError('globalrestored_global_step_step cannot be None if restored_train_dataloader_iter because restore is implicit')
        #global_step = batch_idx * NUM_TRAIN_SHARDS
    
    #NOTE: cannot improve efficiency for this outer loop because gradient loss needs to
    # be calculated and updated for each iteration.
    
    global_data_pspec = P(('processes', 'local_devices'))
    model_mesh = get_model_mesh()
    
    n_local_devices = jax.local_device_count()
    
    sharded_batch_size = config_dict['batch_size'] // n_local_devices
    jax_graph_comp_dict = calc_number_jax_graph_components(config_dict['batch_size'],
        config_dict['max_history'], config_dict['num_candidates'])
        
    for batch_idx, graphs_tuple_batch in enumerate(train_dataloader_iter):
        local_step = batch_idx * TRAIN_BATCH_SIZE
        global_step = local_step * NUM_TRAIN_SHARDS
        epoch = batch_idx // STEPS_PER_EPOCH_LOCAL
        
        padded_super_graph_0, n_samples = pad_graph_tuple_batch(graphs_tuple_batch, jax_graph_comp_dict)
        
        #padded_super_graph = jax.device_put(padded_super_graph_0, data_sharding) #can handle pytrees
        # Map the promotion function over the entire GraphsTuple PyTree
        padded_super_graph = jax.tree_util.tree_map(
            lambda x: jax.experimental.multihost_utils.host_local_array_to_global_array(
                x, model_mesh,global_data_pspec),
            padded_super_graph_0
        )
        
        loss = train_step(model, padded_super_graph, optimizer)
        
        epoch_avg_train_loss.append(loss)
        
        if batch_idx % 5 == 0:# and rank==0:
            print(f"batch {batch_idx}, local step {local_step}, global_step {global_step}, (Epoch {epoch}): Train Loss {loss:.4f}", flush=True)
        
        if (batch_idx + 1) % STEPS_PER_EPOCH_GLOBAL == 0:
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
            
            print(f"Epoch {epoch}: Train avg Loss {avg_train_loss:.4f} "
                  f"| train NDCG@{top_k} {train_metrics['ndcg']:.4f} "
                  f"| train MRR@{top_k} {train_metrics['mrr']:.4f} "
                  f"| train recall_{top_k} {train_metrics['recall']:.4f}"
                  f"avg val loss {global_avg_val_loss:.4f} | val NDCG@{top_k} {global_avg_val_ndcg:.4f} "
                  f"| val MRR@{top_k} {global_avg_val_mrr:.4f} | val recall_{top_k} {global_avg_val_recall:.4f}", flush=True)
            
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
                    print(f"  New best val NDCG! ({global_avg_val_ndcg})")
                if save_checkpoints:
                    print(f'worker_rank={rank}: saving best checkpoint', flush=True)
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
                    print( f"  No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                if rank == 0:
                    print(f"Early stopping triggered at epoch {epoch}.")
                early_stop_triggered[0] = True
                break
            
            if rank == 0:
                print(f'worker_{rank}: log MLFlow metrics', flush=True)
                
                mlflow.log_metrics(metrics_dict, step=epoch)
                #check for whether Vizier pruning suggests a stop of this trial
                if trial is not None:
                    trial.add_measurement(
                        vz.Measurement(
                            metrics={f'ndcg_{top_k}': global_avg_val_ndcg},
                            steps=epoch,
                            elapsed_secs=(time.perf_counter() - start_time)
                        ))
                    if epoch >= delay and trial.check_early_stopping():
                        early_stop_triggered[0] = True
                        break
                        
        if early_stop_triggered[0]:
            break

    return best_ndcg

def build_model_optimizer_and_dataloaders(config:dict, rngs:nnx.Rngs) -> Dict[str, Any]:
    if rngs is None:
        raise ValueError('rngs cannot be None')
    
    worker_rank = jax.process_index()
    
    train_dataloader, val_dataloader = create_train_and_val_dataloaders(
        movies_uri=config['movies_uri'],
        recommendations_uri=config['recommendations_uri'],
        recommendations_ts_uri=config['recommendations_ts_uri'],
        train_ratings_uri=config['ratings_train_uri'],
        val_ratings_uri=config['ratings_val_uri'],
        train_negatives_uri=config['train_negatives_uri'],
        val_negatives_uri=config['val_negatives_uri'],
        max_history=config['max_history'],
        num_candidates=config['num_candidates'],
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        seed=config.get('seed', 0),)
    
    # NOTE: these are prepended with a row of zeros so that user_ids and movie_ids are direct indexes to the embeddings
    embeddings = read_embeddings(
        user_embeddings_uri=config['user_embeddings_uri'],
        movie_embeddings_uri=config['movie_embeddings_uri'],
        batch_size=1024)
    
    nnx.use_eager_sharding(True)
    model_mesh = get_model_mesh()
    with jax.set_mesh(model_mesh):
        
        model = GraphRanker(user_movie_embeds=embeddings,
            num_candidates=config['num_candidates'],
            hidden_features=config['hidden_dim'],
            num_layers=config['num_layers'],
            out_features=config['out_dim'],
            heads=config['num_heads'],
            edge_embed_dim=config['edge_embed_dim'],
            dropout_rate=config['dropout_rate'], rngs=rngs)
            
        optimizer = nnx.Optimizer(model,
            optax.adamw(config['learning_rate'],
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
        #sharded_model_state = jax.lax.with_sharding_constraint(model_state, pspecs)
        nnx.update(model, sharded_model_state)  # The model is sharded now!
        
        opt_state = nnx.state(optimizer)
        pspecs = nnx.get_partition_spec(opt_state)
        sharding_tree = jax.tree.map(to_named_sharding, pspecs)
        sharded_opt_state = jax.device_put(opt_state, sharding_tree)
        #sharded_opt_state = jax.lax.with_sharding_constraint(opt_state, pspecs)
        nnx.update(optimizer, sharded_opt_state)
        
    return {"rngs": rngs, "model": model, "optimizer": optimizer,
        'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader}

def train_fn(config: dict, trial:Trial=None, save_checkpoints:bool=False) -> Tuple[float, str]:
    """
    train the model given data and params specified in config dict and return best validation set ndcg@20 metric and
    return the mlflow_run_id
    :param config:
    :param trial:
    :param save_checkpoints:
    :return: val_ndcg_20, mlflow_run_id
    """
    if "phase" not in config:
        raise ValueError(f"config is missing key 'phase'")
    
    #fixed top_k for consistent stats with retrieval and reranker
    config['top_k'] = 20
    
    worker_rank = jax.process_index()

    print(f'worker_{worker_rank}: train_fn', flush=True)
    
    if worker_rank == 0:
        for key in {"phase", "mlflow_experiment_name", "mlflow_experiment_id",
            "mlflow_parent_run_id"}:
            if key not in config:
                raise ValueError(f"config is missing {key}")
    
    rngs = nnx.Rngs(config.get('seed', 0))
    
    _dict = build_model_optimizer_and_dataloaders(config, rngs=rngs)
    
    model = _dict['model']
    optimizer = _dict['optimizer']
    train_dataloader = _dict['train_dataloader']
    val_dataloader = _dict['val_dataloader']
    
    mlflow_run = None
    best_val_ndcg_k = -1.0
    
    run_name = get_canonical_mlflow_run_name(config)
    
    try:
    
        if worker_rank == 0:
            print(f"mlflow set experiment: {config['mlflow_experiment_name']}", flush=True)
            mlflow.set_experiment(
                experiment_name=config['mlflow_experiment_name'],
            )
            # don't use nested=True because the parent isn't in the same thread in production
            print(f"mlflow start run: {run_name}", flush=True)
            mlflow_run = mlflow.start_run(
                run_name=run_name,
                #tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                tags = {"mlflow.parentRunId" : config['mlflow_parent_run_id']}
            )
            config['mlflow_run_id'] = mlflow_run.info.run_id
            mlflow.set_tag("phase", config["phase"]) #do not move this before start_run
            mlflow.log_params(stringify_mlflow_params(config))
            mlflow.log_text(str(model), "model_summary.txt")
            print(f'worker_{worker_rank}: started MLFlow run_id={mlflow_run.info.run_id}', flush=True)
        if save_checkpoints:
            # paradigm is that we save checkpoints for "train" phase, but not HPO trial phases
            sfx = f"{config['study_name']}/{run_name}"
            config['best_checkpoint_uri'] = f"{config['best_checkpoint_uri']}/{sfx}"
            config['latest_checkpoint_uri'] = f"{config['latest_checkpoint_uri']}/{sfx}"
            if worker_rank == 0:
                # cannot update the mlflow logged param, so instead creata tag for the uris
                mlflow.set_tag('best_checkpoint_uri',  config['best_checkpoint_uri'])
                mlflow.set_tag('latest_checkpoint_uri',config['latest_checkpoint_uri'])
                if trial is not None:
                    trial.update_metadata(
                        vz.Metadata({'best_checkpoint_uri': config['best_checkpoint_uri']}))
        
        print( f"expect the model training to start w/ loss = {-log(1. / config['num_candidates'])}", flush=True)
        
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
            print(f"checkpoints save to directories:\n  {config.get('best_checkpoint_uri','')}"
                  f"\n  {config.get('latest_checkpoint_uri','')}")
            
        return best_val_ndcg_k, config.get('mlflow_run_id', "")
    finally:
        print(f'worker_{worker_rank}: finally clause in train_fn', flush=True)
        if worker_rank==0 and mlflow_run is not None:
            mlflow.log_metric(f"final_ndcg_{config['top_k']}", float(best_val_ndcg_k))
            mlflow.end_run()
    
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
            'val_dataloader', 'rngs', 'global_step', 'config'
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
    
    #model_mesh = get_model_mesh()
    #graphdef_model, model_state = nnx.get_abstract_model(lambda: model, model_mesh)
    
    #restore state to those objects:
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)
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
    
    print(f"worker_rank ={jax.process_index()}: Restored model at step {global_step}")
    
    return {
        'model': model, 'optimizer': optimizer,
        'train_dataloader': train_dataloader,
        'train_dataloader_iter':train_dataloader_iter,
        'val_dataloader': val_dataloader,
        'rngs': rngs,
        'global_step': global_step,
        'config': config
    }

def test_fn(config: dict):
    
    if "phase" not in config:
        raise ValueError("config requires a 'phase' parameter")
    
    for key in ('seed', 'ratings_test_uri', 'train_negatives_uri'):
        if key not in config:
            raise ValueError(f"key {key} is missing from config")
    
    # fixed top_k for consistent stats with retrieval and reranker
    config['top_k'] = 20
    
    worker_rank = jax.process_index()
    
    if worker_rank == 0:
        for key in {"mlflow_experiment_name", "mlflow_experiment_id",
            "mlflow_parent_run_id"}:
            if key not in config:
                raise ValueError(f"config is missing {key}")
    
    if config['phase'] == 'test_best':
        restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['best_checkpoint_uri'])
    else:
        #test_given, uese given checkpoint path to restore, test_checkpoint_uri
        restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['test_checkpoint_uri'])
    
    model = restore_dict['model']
    model.eval()
    
    config['phase'] = 'test_best'
    
    mlflow_run = None
    run_name = f"test_{config.get('test_id', 0)}"
    try:
        if worker_rank == 0:
            mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
            # don't use nested=True because the parent isn't in the same thread in production
            #there may be ACL to solve for this:
            mlflow_run = mlflow.start_run(
                run_name=run_name,
                # tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                tags={"mlflow.parentRunId": config['mlflow_parent_run_id']}
            )
            config['mlflow_run_id'] = mlflow_run.info.run_id
            mlflow.set_tag("phase", config["phase"])  # do not move this before start_run
        
        max_history = restore_dict['config']['max_history']
        num_candidates = restore_dict['config']['num_candidates']
        batch_size = restore_dict['config']['batch_size']
        
        #these uris are all in config too, excepting test_ratings
        test_dataloader = create_test_dataloader(
            movies_uri = restore_dict['config']['movies_uri'],
            recommendations_uri = restore_dict['config']['recommendations_uri'],
            recommendations_ts_uri = restore_dict['config']['recommendations_ts_uri'],
            ratings_uri = config['ratings_test_uri'],
            negatives_uri = config['train_negatives_uri'],
            max_history = max_history,
            num_candidates = num_candidates,
            batch_size = batch_size,
            seed = config.get('seed', 0))
        
        if not isinstance(test_dataloader._sampler, BatchSampler):
            raise ValueError(
                "test_dataloader sampler must be an instance of BatchSampler")
        
        jax_graph_comp_dict = calc_number_jax_graph_components(batch_size, max_history, num_candidates)
        
        global_test_metrics, n_val_samples = _epoch_validation(model, iter(test_dataloader), config['top_k'], jax_graph_comp_dict)
    
        out_dict = {f"test_{key}_{config['top_k']}" : value for key, value in global_test_metrics.items()}
    
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
    
    restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['latest_checkpoint_uri'])
    
    best_val_ndcg_k = -1.0
    mlflow_run = None
    try:
        if worker_rank == 0:
            mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
            # Start a run specifically for this HPO trial
            # don't use nested=True because the parent isn't in the same thread in production
            run_id = config.get('mlflow_run_id', None)  # is not None for a "restore, resume training"
            # in production, there may be ACL to solve for this:
            if run_id is None:
                mlflow_run = mlflow.start_run(
                    run_name=f"trial_{config.get('trial_id', 0)}",
                    # tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                    tags={"mlflow.parentRunId": config['mlflow_parent_run_id']}
                )
            else:
                mlflow_run = mlflow.start_run(
                    run_id=run_id,
                    run_name=f"trial_{config.get('trial_id', 0)}",
                    # tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                    tags={"mlflow.parentRunId": config['mlflow_parent_run_id']}
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

def export_model(trained_model: GraphRanker, batch_size:int, max_history:int,
        num_candidates:int, output_uri:str):
    """
    
    :param trained_model:
    :param batch_size: batch_size used for model training
    :param max_history: max_hisory used for model training
    :param num_candidates: num_candidates used for model training
    :return:
    """
    
    '''
    train_dataloader operations stack receives as inputs:
        batch is a tuple of lists of the 4 datums: ([user_ids],[movie_ids],[ratings],[timestamps])
    then output of last operation is:
        batch is a padded graph tuple:
            GraphsTuple(
              nodes={'candidate_mask': array([False, False, False, ..., False, False, False], shape=(67584,)),
                     'ids': array([6035, 7217, 8600, ...,    0,    0,    0], shape=(67584,), dtype=int32),
                     'label': array([0., 0., 0., ..., 0., 0., 0.], shape=(67584,), dtype=float32),
                     'type': array([0, 1, 1, ..., 0, 0, 0], shape=(67584,), dtype=int32)},
              edges={'rating': array([5, 5, 5, ..., 0, 0, 0], shape=(67520,), dtype=int32)},
              receivers=array([    0,     0,     0, ..., 25233, 25233, 25233], shape=(67520,)),
              senders=array([    1,     2,     3, ..., 25233, 25233, 25233], shape=(67520,)),
              globals=None,
              n_node=array([  333,   342,   337,   455,   273,   468,   338,   467,   468,
                     464,   484,   344,   484,   458,   458,   476,   273,   332,
                     480,   476,   339,   458,   484,   474,   457,   335,   339,
                     284,   342,   472,   331,   284,   468,   347,   474,   341,
                     280,   458,   464,   468,   273,   476,   273,   273,   484,
                     336,   280,   464,   280,   480,   458,   345,   280,   333,
                     480,   476,   484,   472,   273,   348,   345,   480,   273,
                     458, 42351], dtype=int32),
               n_edge=array([  332,   341,   336,   454,   272,   467,   337,   466,   467,
                     463,   483,   343,   483,   457,   457,   475,   272,   331,
                     479,   475,   338,   457,   483,   473,   456,   334,   338,
                     283,   341,   471,   330,   283,   467,   346,   473,   340,
                     279,   457,   463,   467,   272,   475,   272,   272,   483,
                     335,   279,   463,   279,   479,   457,   344,   279,   332,
                     479,   475,   483,   471,   272,   347,   344,   479,   272,
                     457, 42351], dtype=int32))
            where len(n_nodes) = len(n_edges)=65
            
            GraphsTuple(
              nodes={'candidate_mask': array( shape=(max_nodes,), dtype=bool),
                     'ids': array( shape=(max_nodes,), dtype=int32),
                     'label': array( shape=(max_nodes,), dtype=float32),
                     'type': array( shape=(max_nodes,), dtype=int32)},
              edges={'rating': array( shape=(max_edges,), dtype=int32)},
              receivers=array( shape=(max_edges,), dtype=int32),
              senders=array( shape=(max_edges,), dtype=int32),
              globals=None,
              n_node=array( shape=(max_graphs,), dtype=int32),
              n_edge=array( shape=(max_graphs,), dtype=int32))
               
    TODO: still need to write a NumPy or C++ method to replace the grain DataLoader transformations
    for production environment.  The data it needs should be placed in a graph database
    and the method reads from that to construct a padded graph input for the tf SavedModel.
    '''
    '''
    #import tensorflow as tf
    from orbax import export
    
    graphdef, model_state = nnx.split(trained_model)
    
    #'max_nodes', 'max_edges', 'max_graphs'
    # 67584,       67520,       65
    #batch_Size=64, max_history=784, num_candidates=270
    jax_graph_comp_dict = calc_number_jax_graph_components(batch_size,
        max_history, num_candidates)
    
    MAX_NODES = jax_graph_comp_dict['max_nodes']
    MAX_EDGES = jax_graph_comp_dict['max_edges']
    MAX_GRAPHS = 1  # Usually 1 if doing single-request real-time ranking
    
    serving_config = export.ServingConfig(
        signature_key="serving_default",
        input_signature=[
            {
                # Nodes attributes
                "node_candidate_mask": tf.TensorSpec(shape=(MAX_NODES,),
                    dtype=tf.bool, name="node_candidate_mask"),
                "node_ids": tf.TensorSpec(shape=(MAX_NODES,), dtype=tf.int32,
                    name="node_ids"),
                "node_label": tf.TensorSpec(shape=(MAX_NODES,),
                    dtype=tf.float32, name="node_label"),
                "node_type": tf.TensorSpec(shape=(MAX_NODES,), dtype=tf.int32,
                    name="node_type"),
                
                # Edges attributes
                "edge_rating": tf.TensorSpec(shape=(MAX_EDGES,),
                    dtype=tf.int32, name="edge_rating"),
                
                # Core Graph Topology
                "receivers": tf.TensorSpec(shape=(MAX_EDGES,), dtype=tf.int32,
                    name="receivers"),
                "senders": tf.TensorSpec(shape=(MAX_EDGES,), dtype=tf.int32,
                    name="senders"),
                
                # Metadata
                "n_node": tf.TensorSpec(shape=(MAX_GRAPHS,), dtype=tf.int32,
                    name="n_node"),
                "n_edge": tf.TensorSpec(shape=(MAX_GRAPHS,), dtype=tf.int32,
                    name="n_edge"),
            }
        ]
    )
    
    # The first argument MUST receive the model state (weights)
    def pure_apply_fn(state, inputs):
        # This recombines your architecture blueprint with the weights dynamically in RAM
        functional_model = nnx.merge(graphdef, state)
        
        # Reconstruct your exact library GraphsTuple inside the pure function boundaries
        graph_batch = GraphsTuple(
            nodes={
                'candidate_mask': inputs["node_candidate_mask"],
                'ids': inputs["node_ids"],
                'label': inputs["node_label"],
                'type': inputs["node_type"]
            },
            edges={'rating': inputs["edge_rating"]},
            receivers=inputs["receivers"],
            senders=inputs["senders"],
            globals=None,
            n_node=inputs["n_node"],
            n_edge=inputs["n_edge"]
        )
        
        # Call your GraphRanker's __call__ method natively
        return functional_model(graph_batch)
    
    # Wrap your NNX State and the pure function into a JaxModule
    jax_module = export.JaxModule(
        params=model_state,
        apply_fn=pure_apply_fn,
        trainable=False
    )
    
    export_manager = export.ExportManager(jax_module, [serving_config])
    export_manager.save(output_uri)
    '''
    pass

def _assert_checkpoints_restore(checkpoint_uri:str, model, val_data_loader, global_step, top_k:int=20):
    
    print(f'worker_rank={jax.process_index()}: begin _assert_checkpoints_restore', flush=True)
    
    restore_dict = restore_items_from_checkpoint(checkpoint_uri)
    print(f'worker_rank={jax.process_index()}: global_step={global_step}, restored global_step={restore_dict["global_step"]}', flush=True)
    restored_model = restore_dict['model']
    restored_model.eval()
    model.eval()
    
    jax_graph_comp_dict = calc_number_jax_graph_components(
        restore_dict['config']['batch_size'],
        restore_dict['config']['max_history'],
        restore_dict['config']['num_candidates'])
    
    import copy
    loader_current = copy.deepcopy(val_data_loader)
    loader_restored = copy.deepcopy(val_data_loader)
    
    # iter(x) makes a new iterator state
    global_avg_val_metrics_current, n_val_samples_current = _epoch_validation(model, iter(loader_current), top_k, jax_graph_comp_dict)
    
    jax.experimental.multihost_utils.sync_global_devices(
        "sync_barrier_for_model_validation")
    
    global_avg_val_metrics_restored, n_val_samples_restored = _epoch_validation(restored_model, iter(loader_restored), top_k, jax_graph_comp_dict)
    
    jax.experimental.multihost_utils.sync_global_devices(
        "sync_barrier_for_restored_model_validation")
    
    print(f'n_val_samples_current={n_val_samples_current}, n_val_samples_restored = {n_val_samples_restored}')
    
    all_similar = True
    for key in ("loss", "mrr", "ndcg", "recall"):
        print(f'worker_rank={jax.process_index()}: key={key}, model={global_avg_val_metrics_current[key]}, restored={global_avg_val_metrics_restored[key]}', flush=True)
        if not jnp.allclose(global_avg_val_metrics_current[key], global_avg_val_metrics_restored[key]):
            all_similar = False
    
    model.train()
    
    #print(f'worker_rank={jax.process_index()}:\n    summary of model={str(model)}\n    summary of restored={str(restore_dict["model"])}', flush=True)
    
    # print out model state
    #_graphdef, model_state = nnx.split(model)
    #_graphdef_restored, model_state_restored = nnx.split(restore_dict['model'])
    #print(
    #    f'worker_rank={jax.process_index()}:\n    summary of model_state={model_state}\n    summary of restored model_state={model_state_restored}',
    #    flush=True)
    check_model_state_equality(model, restore_dict['model'])
    
    assert(all_similar)
    assert(n_val_samples_current == n_val_samples_restored)
    
    print(f'worker_rank={jax.process_index()}:checkpoint validated for {checkpoint_uri}')

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
        print("worker_rank={jax.process_index()}: ❌ Model structures DO NOT match!")
        if missing_in_b: print(f"   Missing in Restored: {missing_in_b}")
        if missing_in_a: print(f"   Missing in Current: {missing_in_a}")
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
        print(f"worker_rank={jax.process_index()}: ❌ Model structures match, but values differ at {len(mismatched_keys)} parameter paths:")
        #for ii, path in enumerate(mismatched_keys[:5]):  # Limit output log spam
        for ii, path in enumerate(mismatched_keys):
            print(f"worker_rank={jax.process_index()}:   -> Mismatch in layer path: {path}, values={mismatched_vals[ii]}")
        #if len(mismatched_keys) > 5:
        #    print(f"worker_rank={jax.process_index()}:   -> ... and {len(mismatched_keys) - 5} more paths.")
        return False
    
    print("worker_rank={jax.process_index()}: ✅ Success! Both model states are mathematically identical.")
    return True
