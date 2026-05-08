import time
from functools import partial
from typing import Dict, Tuple, Union, Any

import mlflow
import optax
from math import log
import jax
import jax.tree_util as jtu
from jax.sharding import PartitionSpec as P
# In JAX 0.8+, shard_map is typically in the main namespace
from jax import shard_map, Array

from vizier.service import pyvizier as vz
from vizier.service.clients import Trial
import jraph
import jax.numpy as jnp
from flax import nnx
import rax
import grain
from flax.typing import Array
from grain._src.python.data_loader import DataLoader

from movie_lens_ranker.BatchSampler import BatchSampler

import orbax.checkpoint as ocp
from movie_lens_ranker.data_loading import create_train_and_val_dataloaders, \
    create_test_dataloader
from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.util import read_embeddings, get_env_resources

env_resources, mesh = get_env_resources()

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
    debug_weight_before = jnp.linalg.norm(model.score_head.kernel.value)
    
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
    
    loss, grads = nnx.value_and_grad(loss_fn)(model, padded_graph)
    optimizer.update(model, grads)
    
    debug_weight_after = jnp.linalg.norm(model.score_head.kernel.value)
    diff = jnp.abs(debug_weight_before - debug_weight_after)
    # if > 1E-4, is a significant change
    # if > 1, exploding gradient or learning rate issue
    jax.debug.print("Weight Norm: Before={b:.6f}, After={a:.6f}, Delta={d:.8f}",
        b=debug_weight_before, a=debug_weight_after, d=diff)
    
    return loss

@nnx.jit
def eval_step(model: GraphRanker, padded_graph: jraph.GraphsTuple, top_k:int) -> \
dict[str, Array]:
    """
    train step over a batch, where padded_graph contains super graph of the batch
    :param model:
    :param padded_graph:
    :param top_k:
    :return:
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

def _epoch_validation_chunked(model, val_dataloader, top_k):
    all_metrics = []
    current_chunk = []
    chunk_size = 8
    
    for i, batch in enumerate(val_dataloader):
        current_chunk.append(batch)
        
        # When chunk is full, process it
        if len(current_chunk) == chunk_size:
            mega_batch = jtu.tree_map(lambda *xs: jnp.stack(xs),
                *current_chunk)
            # Process the chunk in one vectorized GPU call
            chunk_metrics = vectorized_epoch_eval(model, mega_batch, top_k)
            all_metrics.append(chunk_metrics)
            current_chunk = []  # Reset
    
    # Process any remaining batches in the last partial chunk
    if current_chunk:
        mega_batch = jtu.tree_map(lambda *xs: jnp.stack(xs), *current_chunk)
        all_metrics.append(vectorized_epoch_eval(model, mega_batch, top_k))
    
    # Average across all chunks
    # We stack all the results (e.g., [8, 8, 8, 4]) and take the global mean
    final_metrics = jtu.tree_map(
        lambda *xs: jnp.mean(jnp.concatenate([x.reshape(-1) for x in xs])),
        *all_metrics
    )
    return final_metrics

@nnx.jit
def vectorized_epoch_eval(model, mega_batch, top_k):
    # mega_batch here is a chunk of N batches stacked
    v_eval = nnx.vmap(eval_step, in_axes=(None, 0, None))
    return v_eval(model, mega_batch, top_k)

def _epoch_validation(model: GraphRanker, val_dataloader: DataLoader,
        top_k: int):
    """
    calc metrics for val dataset. Note, if this method consumes too much memory, use the
    _epoch_validation_chunked instead.
    
    :param model:
    :param val_dataloader:
    :param top_k:
    :return:
    """
    # 1. Collect all batches from the loader into a list
    # (Assuming memory permits holding one epoch of padded graphs)
    all_batches = [batch for batch in val_dataloader]
    
    # 2. Stack the list of GraphsTuples into a single vectorized GraphsTuple
    # Every leaf will now have shape (Num_Batches, Padded_Size, ...)
    mega_batch = jtu.tree_map(lambda *xs: jnp.stack(xs), *all_batches)
    
    val_metrics = vectorized_epoch_eval(model, mega_batch, top_k)
    # val_metrics['loss'] is now an array of shape (Num_Batches,)
    local_avg_val_metrics = jax.tree.map(jnp.mean, val_metrics)
    global_avg_metrics = jax.tree.map(aggregate_metric, local_avg_val_metrics)
    return global_avg_metrics

def _epoch_validation_simplest(model: GraphRanker, val_dataloader: DataLoader, top_k: int) -> Dict[str, Array]:
    epoch_val_loss = []
    epoch_val_mrr = []
    epoch_val_ndcg = []
    epoch_val_recall = []
    for val_local_step, val_padded_super_graph in enumerate(val_dataloader):
        val_metrics = eval_step(model, val_padded_super_graph, top_k)
        # jax.debug.print('val_metrics={val_metrics}', val_metrics=val_metrics, ordered=True)
        epoch_val_mrr.append(val_metrics['mrr'])
        epoch_val_ndcg.append(val_metrics['ndcg'])
        epoch_val_loss.append(val_metrics["loss"])
        epoch_val_recall.append(val_metrics['recall'])
    #jaxlib._jax.ArrayImpl;  shape=()
    avg_val_loss = jnp.mean(jnp.array(epoch_val_loss))
    avg_val_mrr = jnp.mean(jnp.array(epoch_val_mrr))
    avg_val_ndcg = jnp.mean(jnp.array(epoch_val_ndcg))
    avg_val_recall = jnp.mean(jnp.array(epoch_val_recall))
    
    #jax.debug.print("avg_val_loss shape={}", jnp.shape(avg_val_loss))
    
    local_metrics = {'loss': avg_val_loss, 'ndcg': avg_val_ndcg, 'mrr': avg_val_mrr, 'recall': avg_val_recall}
    
    return local_metrics
    '''
    dict_specs = {k: P() for k in local_metrics.keys()}
    
    @jax.shard_map(mesh, in_specs=(dict_specs,), out_specs=dict_specs, check_vma=False)
    def sync_fn(metrics):
        # Inside shard_map, 'data' is now a bound axis
        return jax.lax.pmean(metrics, axis_name='data')
    
    # Now this call will find the mesh context it needs
    return sync_fn(local_metrics)
    '''

def _train_fn(model, train_dataloader: grain.DataLoader,
        val_dataloader: grain.DataLoader,
        optimizer: nnx.Optimizer,
        top_k:int, latest_checkpoint_uri: str, best_checkpoint_uri:str,
        rngs:nnx.Rngs, config_dict:Dict[str, Union[str, int, float]],
        trial: Trial = None, save_checkpoints: bool=False,
        restored_train_dataloader_iter=None, restored_global_step:int=None) -> float:
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
    print(f'STEPS_PER_EPOCH_LOCAL_VAL={STEPS_PER_EPOCH_LOCAL_VAL}')
    
    if save_checkpoints:
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
            options=ocp.CheckpointManagerOptions(max_to_keep=2)
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
    
    #for batch_idx, padded_super_graph in enumerate(train_dataloader):
    for batch_idx, padded_super_graph in enumerate(train_dataloader_iter):
        local_step = batch_idx * TRAIN_BATCH_SIZE
        global_step = local_step * NUM_TRAIN_SHARDS
        epoch = batch_idx // STEPS_PER_EPOCH_LOCAL
        #jraph.GraphsTuple
        loss = train_step(model, padded_super_graph, optimizer)
        epoch_avg_train_loss.append(loss)
        
        if batch_idx % 5 == 0 and rank==0:
            print(f"batch {batch_idx}, local step {local_step}, global_step {global_step}, (Epoch {epoch}): Train Loss {loss:.4f}")
        
        if (batch_idx + 1) % STEPS_PER_EPOCH_GLOBAL == 0:
            #finished a train epoch.  calc avg train loss and val metrics
            avg_train_loss = jnp.mean(jnp.array(epoch_avg_train_loss))
            epoch_avg_train_loss.clear()
            train_metrics = eval_step(model, padded_super_graph, top_k)
            
            # val_dataloader is also sharded, so don't isolate this to only shard 0.
            # Also, this is synced across all shards, so all shards have same conditional logic for global_avg_val_metrics below here
            global_avg_val_metrics = _epoch_validation(model, val_dataloader, top_k)
            
            global_avg_val_loss = global_avg_val_metrics["loss"]
            global_avg_val_mrr = global_avg_val_metrics['mrr']
            global_avg_val_ndcg = global_avg_val_metrics['ndcg']
            global_avg_val_recall = global_avg_val_metrics['recall']
            
            print(f"Epoch {epoch}: Train avg Loss {avg_train_loss:.4f} "
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
                "val_loss":global_avg_val_loss.item(),
                f"val_{mrr_text}":global_avg_val_mrr.item(),
                f"val_{ndcg_text}":global_avg_val_ndcg.item(),
                f"val_{recall_text}":global_avg_val_recall.item()
            }
            
            if save_checkpoints:
                #orbax for checkpointing.  saves latest 2
                _graphdef, model_state = nnx.split(model)
                _, opt_state = nnx.split(optimizer)
                mngr_latest.save(
                    epoch,
                    args=ocp.args.Composite(
                        model=ocp.args.StandardSave(model_state),
                        opt=ocp.args.StandardSave(opt_state),
                        global_step=ocp.args.StandardSave({'global_step':global_step}),
                        # NNX RNGs need to be converted to state (dictionary of arrays)
                        rngs=ocp.args.StandardSave(nnx.state(rngs)),
                        # Include your dataloader from before
                        train_dataloader=grain.checkpoint.CheckpointSave(train_dataloader_iter),
                        config=ocp.args.JsonSave(config_dict)
                    )
                )
                mngr_latest.wait_until_finished()  # Ensure it's on disk
            
            if global_avg_val_ndcg > best_ndcg + 1e-6:
                best_ndcg = global_avg_val_ndcg.item()
                epochs_without_improvement = 0
                if rank == 0:
                    print(f"  New best val NDCG! ({global_avg_val_ndcg})")
                if save_checkpoints:
                    #all shards write their best
                    _graphdef, model_state = nnx.split(model)
                    _, opt_state = nnx.split(optimizer)
                    mngr_best.save(
                        epoch,
                        args=ocp.args.Composite(
                            model=ocp.args.StandardSave(model_state),
                            opt=ocp.args.StandardSave(opt_state),
                            global_step=ocp.args.StandardSave({'global_step': global_step}),
                            # NNX RNGs need to be converted to state (dictionary of arrays)
                            rngs=ocp.args.StandardSave(nnx.state(rngs)),
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
                mlflow.log_metrics(metrics_dict, step=epoch)
                #check for whether Vizier pruning suggests a stop of this trial
                if trial is not None:
                    trial.add_measurement(
                        vz.Measurement(
                            metrics={f'ndcg_{top_k}': float(global_avg_val_ndcg.item())},
                            steps=epoch,
                            elapsed_secs=(time.perf_counter() - start_time)
                        ))
                    if epoch >= delay and trial.check_early_stopping():
                        early_stop_triggered[0] = True
                        break
                        
        if early_stop_triggered[0]:
            break

    return best_ndcg

def build_model_optimizer_and_dataloaders(config:dict, rngs:nnx.Rngs=None) -> Dict[str, Any]:
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
        seed=config['seed'])
    
    # NOTE: these are prepended with a row of zeros so that user_ids and movie_ids are direct indexes to the embeddings
    embeddings = read_embeddings(
        user_embeddings_uri=config['user_embeddings_uri'],
        movie_embeddings_uri=config['movie_embeddings_uri'],
        batch_size=1024)
    
    if rngs is None:
        rngs = nnx.Rngs(config['seed'])
    
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
    
    return {"rngs": rngs, "model": model, "optimizer": optimizer,
        'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader}

def train_fn(config: dict, trial:Trial=None, save_checkpoints:bool=False, rngs:nnx.Rngs=None) -> Tuple[float, str]:
    """
    train the model given data and params specified in config dict and return best validation set ndcg@20 metric and
    return the mlflow_run_id
    :param config:
    :param save_checkpoints:
    :param rngs:
    :return: val_ndcg_20, mlflow_run_id
    """
    
    #hard wiring top_k for consistent stats with retrieval and reranker
    config['top_k'] = 20
    
    worker_rank = jax.process_index()
    
    _dict = build_model_optimizer_and_dataloaders(config, rngs=rngs)
    model = _dict['model']
    optimizer = _dict['optimizer']
    rngs = _dict['rngs']
    train_dataloader = _dict['train_dataloader']
    val_dataloader = _dict['val_dataloader']
    
    mlflow_run = None
    best_val_ndcg_k = -1.0
    try:
        if worker_rank == 0:
            print(f"mlflow set experiment: {config['mlflow_experiment_name']}")
            mlflow.set_experiment(
                experiment_name=config['mlflow_experiment_name'],
            )
            # don't use nested=True because the parent isn't in the same thread in production
            train_id = config.get('train_id', 0)
            print(f"mlflow start run: train_{train_id}")
            mlflow_run = mlflow.start_run(
                run_name=f"train_{train_id}",
                #tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                tags = {"mlflow.parentRunId" : config['mlflow_parent_run_id']}
            )
            config['mlflow_run_id'] = mlflow_run.info.run_id
            mlflow.set_tag("phase", config["phase"]) #do not move this before start_run
            for k,v in config.items():
                if k.find('?') > -1:
                    print(f"problem key: {k}={v}", flush=True)
            save_dict = {k:v for k,v in config.items() if k.find('?')==-1}
            mlflow.log_params(save_dict)
           
            mlflow.log_text(str(model), "model_summary.txt")
            
            if save_checkpoints:
                sfx = f"{config['study_name']}/trial_{config['trial_id']:04d}"
                config['best_checkpoint_uri'] = f"{config['best_checkpoint_uri']}/{sfx}"
                config['latest_checkpoint_uri'] = f"{config['latest_checkpoint_uri']}/{sfx}"
                mlflow.set_tag('best_checkpoint_uri',
                    config['best_checkpoint_uri'])
                mlflow.set_tag('latest_checkpoint_uri',
                    config['latest_checkpoint_uri'])
                mlflow.log_param("best_checkpoint_uri", config['best_checkpoint_uri'])
                mlflow.log_param("latest_checkpoint_uri", config['latest_checkpoint_uri'])
                #paradigm is that we save checkpoints for "train" phase, but not HPO trial phases
        
        print(
            f"expect the model training to start w/ loss = {-log(1. / config['num_candidates'])}")
        
        best_val_ndcg_k = _train_fn(model=model, train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer, top_k=config['top_k'],
            latest_checkpoint_uri=config['latest_checkpoint_uri'],
            best_checkpoint_uri=config['best_checkpoint_uri'],
            rngs=rngs, config_dict=config,
            trial=trial, save_checkpoints=save_checkpoints)
        return best_val_ndcg_k, config['mlflow_run_id']
    finally:
        if mlflow_run is not None:
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
    restore the model, dataloadern and state from checkpoint_uri.  if get_Earliest is set to True,
    the earlies of the saved runs will be returned.  This is useful for testing continuation of
    training from an earlier checkpoint.
    :param checkpoint_uri:
    :param get_earliest: False by default, else returns earliest of saved checkpoints
    :return: dictionary holding: 'model', 'optimizer', 'train_dataloader', 'train_dataloader_iter',
            'val_dataloader', 'rngs', 'global_step', 'config
    """
    mngr = ocp.CheckpointManager(checkpoint_uri,
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
    
    if get_earliest:
        available_steps = mngr.all_steps()
        epoch = available_steps[0]
    else:
        epoch = mngr.latest_step()
    if epoch is None:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_uri}")
    
    #restore config, then rngs, so can restore model and dataloaders from them
    _ = mngr.restore(
        epoch,
        args=ocp.args.Composite(
            config=ocp.args.JsonRestore(),
        )
    )
    config = _['config']
    rngs = nnx.Rngs(config['seed'])
    _ = mngr.restore(
        epoch,
        args=ocp.args.Composite(
            rngs=ocp.args.StandardRestore(nnx.state(rngs)),
        )
    )
    nnx.update(rngs, _['rngs'])
    
    _dict = build_model_optimizer_and_dataloaders(config, rngs=rngs)
    model = _dict['model']
    optimizer = _dict['optimizer']
    train_dataloader = _dict['train_dataloader']
    val_dataloader = _dict['val_dataloader']
    
    #restore state to those objects:
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)
    restored = mngr.restore(
        epoch,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(model_state),
            opt=ocp.args.StandardRestore(opt_state),
            global_step=ocp.args.StandardRestore({'global_step': 0}),
            # Grain requires the actual iterator object to restore state in-place
            train_dataloader=grain.checkpoint.CheckpointRestore( iter(train_dataloader)),
        )
    )
    
    train_dataloader_iter = restored['train_dataloader']
    nnx.update(optimizer, restored['opt'])
    nnx.update(model, restored['model'])
    global_step = restored['global_step']['global_step']
    
    print(f"Restored model at step {global_step}")
    
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
    
    #hard wiring as is done in train_fn:
    config['top_k'] = 20
    
    worker_rank = jax.process_index()
    
    restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['test_checkpoint_uri'])
    
    model = restore_dict['model']
    
    config['phase'] = 'test_best'
    
    mlflow_run = None
    try:
        if worker_rank == 0:
            mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
            # don't use nested=True because the parent isn't in the same thread in production
            #in production, there may be ACL to solve for this:
            mlflow_run = mlflow.start_run(
                run_name=f"test_{config.get('test_id', 0)}",
                # tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                tags={"mlflow.parentRunId": config['mlflow_parent_run_id']}
            )
            config['mlflow_run_id'] = mlflow_run.info.run_id
            mlflow.set_tag("phase", config["phase"])  # do not move this before start_run
            #mlflow.log_params(config)
            
            #these uris are all in config too, excepting test_ratings
            test_dataloader = create_test_dataloader(
                movies_uri = restore_dict['config']['movies_uri'],
                recommendations_uri = restore_dict['config']['recommendations_uri'],
                recommendations_ts_uri = restore_dict['config']['recommendations_ts_uri'],
                ratings_uri = config['ratings_test_uri'],
                negatives_uri = config['train_negatives_uri'],
                max_history = restore_dict['config']['max_history'],
                num_candidates = restore_dict['config']['num_candidates'],
                batch_size = restore_dict['config']['batch_size'],
                seed = config['seed'])
            
            if not isinstance(test_dataloader._sampler, BatchSampler):
                raise ValueError(
                    "test_dataloader sampler must be an instance of BatchSampler")
                    
            test_metrics = _epoch_validation(model, test_dataloader, restore_dict['config']['top_k'])
            
            out_dict = {f"test_{key}_{config['top_k']}" : value for key, value in test_metrics.items()}
        
            if mlflow_run is not None:
                for key, value in out_dict.items():
                    mlflow.log_metric(key, float(value))
        
            return out_dict
        
    finally:
        if mlflow_run is not None:
            mlflow.end_run()

def resume_train_fn(config: dict, trial: Trial=None, save_checkpoints: bool=False):
    
    worker_rank = jax.process_index()
    
    restore_dict = restore_items_from_checkpoint(checkpoint_uri=config['best_checkpoint_uri'])
    
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
                    run_name=f"trial_{config.get('trial_id', 0):04d}",
                    # tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                    tags={"mlflow.parentRunId": config['mlflow_parent_run_id']}
                )
            else:
                mlflow_run = mlflow.start_run(
                    run_id=run_id,
                    run_name=f"trial_{config.get('trial_id', 0):04d}",
                    # tags = {mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: config['mlflow_parent_run_id']},
                    tags={"mlflow.parentRunId": config['mlflow_parent_run_id']}
                )
            config['mlflow_run_id'] = mlflow_run.info.run_id
            
        best_val_ndcg_k = _train_fn(model=restore_dict['model'],
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
            save_checkpoints=save_checkpoints
        )
        return best_val_ndcg_k
        
    finally:
        if mlflow_run is not None:
            mlflow.log_metric(f"final_ndcg_{config['top_k']}",
                float(best_val_ndcg_k))
            mlflow.end_run()
