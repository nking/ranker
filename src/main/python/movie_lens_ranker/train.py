from typing import Dict, Tuple

import jax
import jax.tree_util as jtu

import jraph
import jax.numpy as jnp
import mlflow
from flax import nnx
import rax
import grain
from flax.typing import Array
from grain._src.python.data_loader import DataLoader
from jax.sharding import NamedSharding, PartitionSpec as P
from pydantic_core.core_schema import dict_schema

from movie_lens_ranker.BatchSampler import BatchSampler

#import chex

#expect 1 Finished compiling <function_name> statement in logs per hit method
#jax.config.update("jax_log_compiles", True)

import orbax.checkpoint as ocp

from movie_lens_ranker.model import GraphRanker

def get_node_graph_index(graph: jraph.GraphsTuple):
    """
    Recreates the mapping of nodes to graph indices.
    If graph.n_node is [3, 2], this returns [0, 0, 0, 1, 1].
    """
    n_graph = graph.n_node.shape[0]
    graph_indices = jnp.arange(n_graph)
    return jnp.repeat(graph_indices, graph.n_node)

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
    
def debug_stats(x, label="Stats"):
    def _print_stats(flat_x):
        mean = jnp.mean(flat_x)
        std = jnp.std(flat_x)
        min_val = jnp.min(flat_x)
        max_val = jnp.max(flat_x)
        print(f"{label} -> Mean: {mean:.4f}, Std: {std:.4f}, Min: {min_val:.4f}, Max: {max_val:.4f}")

    # We use jax.debug.callback to execute Python code (printing)
    # from inside a JIT-compiled function.
    jax.debug.callback(_print_stats, x.reshape(-1))
    
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
    
    def loss_fn(model, padded_graph) -> float:
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
def eval_step(model: GraphRanker, padded_graph: jraph.GraphsTuple, top_k:int) -> Dict[str, float]:
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
    avg_metrics = jax.tree.map(jnp.mean, val_metrics)
    
    return avg_metrics

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
    
    jax.debug.print("avg_val_loss shape={}", jnp.shape(avg_val_loss))
    
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

def train_fn(model, train_dataloader: grain.DataLoader,
        val_dataloader: grain.DataLoader,
        optimizer: nnx.Optimizer,
        top_k:int, latest_checkpoint_dir: str, best_checkpoint_dir:str,
        rngs:nnx.Rngs):
    """
    a shard's portion of the training
    :param model:
    :param train_dataloader:
    :param val_dataloader:
    :param optimizer:
    :param top_k:
    :param latest_checkpoint_dir:
    :param rngs:
    :return:
    """
   
    if not isinstance(train_dataloader._sampler, BatchSampler):
        raise ValueError("train_dataloader sampler must be an instance of BatchSampler")
    if not isinstance(val_dataloader._sampler, BatchSampler):
        raise ValueError("val_dataloader sampler must be an instance of BatchSampler")
        
    rank = jax.process_index()
    
    #tracked_fn_1 = chex.assert_max_traces(train_step, n=1)
    #tracked_fn_2 = chex.assert_max_traces(eval_step, n=1)
    
    TRAIN_BATCH_SIZE = train_dataloader._sampler.batch_size
    TOTAL_RECORDS = train_dataloader._sampler.total_records
    STEPS_PER_EPOCH_GLOBAL = train_dataloader._sampler.num_batches  # = 7234
    NUM_TRAIN_SHARDS = train_dataloader._sampler._shard_options.shard_count
    STEPS_PER_EPOCH_LOCAL = STEPS_PER_EPOCH_GLOBAL//NUM_TRAIN_SHARDS
    
    print(f'TRAIN_BATCH_SIZE={TRAIN_BATCH_SIZE}, TOTAL_RECORDS={TOTAL_RECORDS}', flush=True)
    print(f'STEPS_PER_EPOCH_GLOBAL_TRAIN={STEPS_PER_EPOCH_GLOBAL}')
    print(f'STEPS_PER_EPOCH_LOCAL_TRAIN={STEPS_PER_EPOCH_LOCAL}')
    
    VAL_BATCH_SIZE = train_dataloader._sampler.batch_size
    TOTAL_RECORDS_VAL = val_dataloader._sampler.total_records
    STEPS_PER_EPOCH_GLOBAL_VAL = val_dataloader._sampler.num_batches # 903
    NUM_VAL_SHARDS = val_dataloader._sampler._shard_options.shard_count
    STEPS_PER_EPOCH_LOCAL_VAL = STEPS_PER_EPOCH_GLOBAL_VAL // NUM_VAL_SHARDS
    
    print(f'VAL_BATCH_SIZE={VAL_BATCH_SIZE}, TOTAL_RECORDS_VAL={TOTAL_RECORDS_VAL}', flush=True)
    print(f'STEPS_PER_EPOCH_GLOBAL_VAL={STEPS_PER_EPOCH_GLOBAL_VAL}')
    print(f'STEPS_PER_EPOCH_LOCAL_VAL={STEPS_PER_EPOCH_LOCAL_VAL}')

    #TODO: add back save of best checkpoint now that ray will not be handling it
    mngr_latest = ocp.CheckpointManager(latest_checkpoint_dir,
        item_handlers={
            'model': ocp.StandardCheckpointHandler(),
            'opt': ocp.StandardCheckpointHandler(),
            'global_step': ocp.StandardCheckpointHandler(),
            'rngs': ocp.StandardCheckpointHandler(),
            'dataloader': grain.checkpoint.CheckpointHandler()
        },
        options=ocp.CheckpointManagerOptions(max_to_keep=2)
    )
    mngr_best = ocp.CheckpointManager(best_checkpoint_dir,
        item_handlers={
            'model': ocp.StandardCheckpointHandler(),
            'opt': ocp.StandardCheckpointHandler(),
            'global_step': ocp.StandardCheckpointHandler(),
            'rngs': ocp.StandardCheckpointHandler(),
            'dataloader': grain.checkpoint.CheckpointHandler()
        },
        options=ocp.CheckpointManagerOptions(max_to_keep=2)
    )
    
    ndcg_text = f'ndcg_{top_k}'
    mrr_text = f'mrr_{top_k}'
    recall_text = f'recall_{top_k}'
    
    history = {
        "train_loss": [], f"train_{mrr_text}": [], f"train_{ndcg_text}": [], f"train_{recall_text}":[],
        "val_loss":[], f"val_{mrr_text}": [], f"val_{ndcg_text}": [], f"val_{recall_text}":[]}
    
    #configure for early stopping when ndcg stops changing
    patience = 5
    best_ndcg = -1.0
    epochs_without_improvement = 0
    delay = 10 # min number of epochs to learn
    
    epoch_avg_train_loss = []
    early_stop_triggered = [False]
    
    #stacked_val = stack_val_batches(val_dataloader, steps_per_worker)
    for local_step, padded_super_graph in enumerate(train_dataloader):
        epoch = local_step // STEPS_PER_EPOCH_LOCAL
        #jraph.GraphsTuple
        loss = train_step(model, padded_super_graph, optimizer)
        epoch_avg_train_loss.append(loss)
        
        if local_step % 100 == 0 and rank==0:
            print(f"Step {local_step} (Epoch {epoch}): Train Loss {loss:.4f}")
            # writer.add_scalar("loss", loss, global_step)
        
        if (local_step + 1) % STEPS_PER_EPOCH_LOCAL == 0:
            #finished a train epoch.  calc avg train loss and val metrics
            avg_train_loss = jnp.mean(jnp.array(epoch_avg_train_loss))
            epoch_avg_train_loss.clear()
            
            #stacked_val = stack_val_batches(val_dataloader, STEPS_PER_EPOCH_LOCAL_VAL)
            #avg_metrics = validation_epoch_compiled(state, stacked_val, top_k)
            
            # 3. Synchronize across Ray workers using pmap + pmean
            # Ensure avg_metrics['ndcg'] is a 1D array of size 1 for pmap
            #global_ndcg = pmap(lambda x: pmean(x, "batch"), "batch")(jnp.array([avg_metrics['ndcg']]))[0]
            
            train_metrics = eval_step(model, padded_super_graph, top_k)
            
            val_metrics = _epoch_validation(model, val_dataloader, top_k)
            
            avg_val_loss = val_metrics["loss"]
            avg_val_mrr = val_metrics['mrr']
            avg_val_ndcg = val_metrics['ndcg']
            avg_val_recall = val_metrics['recall']
            
            print(f"Epoch {epoch}: Train avg Loss {avg_train_loss:.4f} "
                  f"| train NDCG@{top_k} {train_metrics['ndcg']:.4f} "
                  f"| train MRR@{top_k} {train_metrics['mrr']:.4f} "
                  f"| train recall_{top_k} {train_metrics['recall']:.4f}"
                  f"avg val loss {avg_val_loss:.4f} | val NDCG@{top_k} {avg_val_ndcg:.4f} "
                  f"| val MRR@{top_k} {avg_val_mrr:.4f} | val recall_{top_k} {avg_val_recall:.4f}")
            
            history["train_loss"].append(avg_train_loss.item())
            history[f"train_{mrr_text}"].append(train_metrics['mrr'].item())
            history[f"train_{ndcg_text}"].append(train_metrics['ndcg'].item())
            history[f"train_{recall_text}"].append(train_metrics['recall'].item())
            history["val_loss"].append(avg_val_loss.item())
            history[f"val_{mrr_text}"].append(avg_val_mrr.item())
            history[f"val_{ndcg_text}"].append(avg_val_ndcg.item())
            history[f"val_{recall_text}"].append(avg_val_recall.item())
            ray_dict = {
                "train_loss":avg_train_loss.item(),
                f"train_{mrr_text}":train_metrics['mrr'].item(),
                f"train_{ndcg_text}" : train_metrics['ndcg'].item(),
                f"train_{recall_text}" : train_metrics['recall'].item(),
                "val_loss":avg_val_loss.item(),
                f"val_{mrr_text}":avg_val_mrr.item(),
                f"val_{ndcg_text}":avg_val_ndcg.item(),
                f"val_{recall_text}":avg_val_recall.item()
            }
            
            #global_val_ndcg = jax.lax.pmean(avg_val_ndcg, axis_name="batch")
            global_val_ndcg = avg_val_ndcg
            
            #orbax for checkpointing.  saves latest 2
            _graphdef, model_state = nnx.split(model)
            _, opt_state = nnx.split(optimizer)
            mngr_latest.save(
                local_step*NUM_TRAIN_SHARDS,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(model_state),
                    opt=ocp.args.StandardSave(opt_state),
                    global_step=ocp.args.StandardSave({'step': local_step*NUM_TRAIN_SHARDS}),
                    # NNX RNGs need to be converted to state (dictionary of arrays)
                    rngs=ocp.args.StandardSave(nnx.state(rngs)),
                    # Include your dataloader from before
                    dataloader=grain.checkpoint.CheckpointSave(iter(train_dataloader))
                )
            )
            mngr_latest.wait_until_finished()  # Ensure it's on disk
            
            #if rank == 0:
            #    mlflow.log_metrics(ray_dict, step=epoch)
                
            if avg_val_ndcg > best_ndcg + 1e-6:
                best_ndcg = global_val_ndcg
                epochs_without_improvement = 0
                if rank == 0:
                    print(f"  New best NDCG!")
                #all shards write their best
                _graphdef, model_state = nnx.split(model)
                _, opt_state = nnx.split(optimizer)
                mngr_best.save(
                    local_step * NUM_TRAIN_SHARDS,
                    args=ocp.args.Composite(
                        model=ocp.args.StandardSave(model_state),
                        opt=ocp.args.StandardSave(opt_state),
                        global_step=ocp.args.StandardSave(
                            {'step': local_step * NUM_TRAIN_SHARDS}),
                        # NNX RNGs need to be converted to state (dictionary of arrays)
                        rngs=ocp.args.StandardSave(nnx.state(rngs)),
                        # Include your dataloader from before
                        dataloader=grain.checkpoint.CheckpointSave(
                            iter(train_dataloader))
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
            
        if early_stop_triggered[0]:
            break
        
    return history

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

def get_stacked_shards(dataloader, steps_per_worker):
    batches = []
    for i, batch in enumerate(dataloader):
        batches.append(batch)
        if i + 1 >= steps_per_worker:
            break
    # Stacks into a single Pytree where every leaf has leading dim 'steps_per_worker'
    return jax.tree.map(lambda *args: jnp.stack(args), *batches)


@nnx.jit
def validation_epoch_compiled(model: nnx.Module, stacked_batches, top_k):
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

def test_fn(model, test_dataloader: grain.DataLoader, top_k:int):
    """
    calculate loss and metrics for the data in test_dataloader. this method expects that test_dataloader
    was constructed for 1 epoch, but also has a stop at end of first epoch.
    :param model:
    :param test_dataloader:
    :param top_k:
    :return:
    """
    
    '''
    model = config['model']
    test_dataloader = config['test_dataloader']
    top_k = config['top_k']
    '''
    
    if not isinstance(test_dataloader._sampler, BatchSampler):
        raise ValueError("test_dataloader sampler must be an instance of BatchSampler")
    
    TEST_BATCH_SIZE = test_dataloader._sampler.batch_size
    TOTAL_RECORDS = test_dataloader._sampler.total_records
    STEPS_PER_EPOCH = TOTAL_RECORDS // TEST_BATCH_SIZE  # = 7234
    
    ndcg_text = f'ndcg_{top_k}'
    mrr_text = f'mrr_{top_k}'
    recall_text = f'recall_{top_k}'
    
    history = {"test_loss": [], f"test_{mrr_text}": [], f"test_{ndcg_text}": [],
        f"test_{recall_text}": []}
    
    epoch_test_loss = []
    epoch_test_mrr = []
    epoch_test_ndcg = []
    epoch_test_recall = []
    
    for global_step, padded_super_graph in enumerate(test_dataloader):
        epoch = global_step // STEPS_PER_EPOCH
        batch_idx = global_step % STEPS_PER_EPOCH
        #jraph.GraphsTuple
        metrics = eval_step(model, padded_super_graph)
        epoch_test_mrr.append(metrics['mrr'])
        epoch_test_ndcg.append(metrics['ndcg'])
        epoch_test_loss.append(metrics["loss"])
        epoch_test_recall.append(metrics['recall'])
        if global_step % 100 == 0:
            ndcg = metrics['ndcg']
            recall = metrics['recall']
            mrr = metrics['mrr']
            print(f"Step {global_step} (Epoch {epoch}): test loss {metrics['loss']:.4f} "
                  f"| test {ndcg_text} {ndcg:.4f} | test {mrr_text} {mrr:.4f} | test {recall_text} {recall:.4f}")
        if global_step > 1 and global_step % STEPS_PER_EPOCH == 0:
            break
        
    avg_test_loss = jnp.mean(jnp.array(epoch_test_loss))
    avg_test_mrr = jnp.mean(jnp.array(epoch_test_mrr))
    avg_test_ndcg = jnp.mean(jnp.array(epoch_test_ndcg))
    avg_test_recall = jnp.mean(jnp.array(epoch_test_recall))
    print(f"Test avg Loss {avg_test_loss:.4f} "
          f"| test avg NDCG@{top_k} {avg_test_mrr:.4f} "
          f"| test avg MRR@{top_k} {avg_test_ndcg:.4f} "
          f"| test avg RECALL@{top_k} {avg_test_recall:.4f}"
          )
    history["test_loss"].append(avg_test_loss)
    history[f"test_{mrr_text}"].append(avg_test_mrr)
    history[f"test_{ndcg_text}"].append(avg_test_ndcg)
    history[f"test_{recall_text}"].append(avg_test_recall)
        
    return history
