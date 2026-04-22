from typing import Dict, Tuple

import jax
import jraph
import jax.numpy as jnp
from flax import nnx
import rax
import grain
from flax.typing import Array

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
        return loss, {f"mrr_{top_k}": mrr, f"ndcg_{top_k}": ndcg, f"recall_{top_k}": recall}
    
    # has_aux is necessary when loss_fn returns more than scalar loss
    (loss, metrics_dict), grads = loss_fn(model, padded_graph)
    metrics_dict['loss'] = loss
    return metrics_dict

def train_fn(model, train_dataloader: grain.DataLoader,
        val_dataloader: grain.DataLoader,
        optimizer: nnx.Optimizer,
        top_k:int, latest_checkpoint_dir: str, best_checkpoint_dir:str, rngs:nnx.Rngs):
    
    if not isinstance(train_dataloader._sampler, BatchSampler):
        raise ValueError("train_dataloader sampler must be an instance of BatchSampler")
    if not isinstance(val_dataloader._sampler, BatchSampler):
        raise ValueError("val_dataloader sampler must be an instance of BatchSampler")
    
    #tracked_fn_1 = chex.assert_max_traces(train_step, n=1)
    #tracked_fn_2 = chex.assert_max_traces(eval_step, n=1)
    
    TRAIN_BATCH_SIZE = train_dataloader._sampler.batch_size
    TOTAL_RECORDS = train_dataloader._sampler.__len__()
    STEPS_PER_EPOCH = TOTAL_RECORDS // TRAIN_BATCH_SIZE  # = 7234
    
    VAL_BATCH_SIZE = train_dataloader._sampler.batch_size
    TOTAL_RECORDS_VAL = val_dataloader._sampler.__len__()
    STEPS_PER_EPOCH_VAL = TOTAL_RECORDS_VAL // VAL_BATCH_SIZE # 903
    
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
        options=ocp.CheckpointManagerOptions(max_to_keep=1)
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
    best_state = None  # To store the best weights
    delay = 10 # min number of epochs to learn
    
    epoch_avg_train_loss = []
    early_stop_triggered = [False]
    
    for global_step, padded_super_graph in enumerate(train_dataloader):
        epoch = global_step // STEPS_PER_EPOCH
        batch_idx = global_step % STEPS_PER_EPOCH
        #jraph.GraphsTuple
        loss = train_step(model, padded_super_graph, optimizer)
        epoch_avg_train_loss.append(loss)
        
        if global_step % 100 == 0:
            print(f"Step {global_step} (Epoch {epoch}): Train Loss {loss:.4f}")
            # writer.add_scalar("loss", loss, global_step)
        
        if global_step > 1 and global_step % STEPS_PER_EPOCH == 0:
            #finished a train epoch.  calc avg train loss and val metrics
            avg_train_loss = jnp.mean(jnp.array(epoch_avg_train_loss))
            train_metrics = eval_step(model, padded_super_graph, top_k)
            jax.debug.print("Epoch {epoch}: Train Loss {avg_train_loss:.4f}", avg_train_loss=avg_train_loss, ordered=True)
            epoch_avg_train_loss.clear()
            
            epoch_val_loss = [], epoch_val_mrr = [], epoch_val_ndcg = [], epoch_val_recall = []
            for val_global_step, val_padded_super_graph in enumerate(val_dataloader):
                val_metrics = eval_step(model, val_padded_super_graph, top_k)
                epoch_val_mrr.append(val_metrics[mrr_text])
                epoch_val_ndcg.append(val_metrics[ndcg_text])
                epoch_val_loss.append(val_metrics["loss"])
                epoch_val_recall.append(val_metrics[recall_text])
                if val_global_step > 1 and val_global_step % STEPS_PER_EPOCH_VAL == 0:
                    #finished an epoch
                    break
            avg_val_loss = jnp.mean(jnp.array(epoch_val_loss))
            avg_val_mrr = jnp.mean(jnp.array(epoch_val_mrr))
            avg_val_ndcg = jnp.mean(jnp.array(epoch_val_ndcg))
            avg_val_recall = jnp.mean(jnp.array(epoch_val_recall))
            
            print(f"Epoch {epoch}: Train avg Loss {avg_train_loss:.4f} "
                  f"| train NDCG@{top_k} {train_metrics[{ndcg_text}]:.4f} "
                  f"| train MRR@{top_k} {train_metrics[{mrr_text}]:.4f} "
                  f"| train recall_{top_k} {train_metrics[{recall_text}]:.4f}"
                  f"avg val loss {avg_val_loss:.f} | val NDCG@{top_k} {avg_val_ndcg:.4f} "
                  f"| val MRR@{top_k} {avg_val_mrr:.4f} | val recall_{top_k} {avg_val_recall:.4f}")
            
            history["train_loss"].append(avg_train_loss)
            history[f"train_{mrr_text}"].append(train_metrics[{mrr_text}])
            history[f"train_{ndcg_text}"].append(train_metrics[{ndcg_text}])
            history[f"train_{recall_text}"].append(train_metrics[{recall_text}])
            history["val_loss"].append(avg_val_loss)
            history[f"val_{mrr_text}"].append(avg_val_mrr)
            history[f"val_{ndcg_text}"].append(avg_val_ndcg)
            history[f"val_{recall_text}"].append(avg_val_recall)
        
            if avg_val_ndcg > best_ndcg:
                best_ndcg = avg_val_ndcg
                epochs_without_improvement = 0
                best_state = model.split()
                print(f"  New best NDCG! Saving model...")
                mngr_best.save(
                    global_step,
                    args=ocp.args.Composite(
                        model=ocp.args.StandardSave(nnx.state(model)),
                        opt=ocp.args.StandardSave(nnx.state(optimizer)),
                        global_step=ocp.args.StandardSave(global_step),
                        # NNX RNGs need to be converted to state (dictionary of arrays)
                        rngs=ocp.args.StandardSave(nnx.state(rngs)),
                        # Include your dataloader from before
                        dataloader=grain.checkpoint.CheckpointSave(iter(train_dataloader))
                    )
                )
                mngr_best.wait_until_finished()
            elif epoch >= delay:
                epochs_without_improvement += 1
                print( f"  No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                model.update(best_state)
                early_stop_triggered[0] = True
                break
            #write checkpoints
            
            #orbax for checkpointing.  saves latest 2
            _graphdef, model_state = nnx.split(model)
            _, opt_state = nnx.split(optimizer)
            mngr_latest.save(
                global_step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(model_state),
                    opt=ocp.args.StandardSave(opt_state),
                    global_step=ocp.args.StandardSave(global_step),
                    # NNX RNGs need to be converted to state (dictionary of arrays)
                    rngs=ocp.args.StandardSave(nnx.state(rngs)),
                    # Include your dataloader from before
                    dataloader=grain.checkpoint.CheckpointSave(iter(train_dataloader))
                )
            )
            mngr_latest.wait_until_finished()  # Ensure it's on disk
                
        if early_stop_triggered[0]:
            break
        
    return history

def test_fn(model, test_dataloader: grain.DataLoader, top_k:int):
    """
    calculate loss and metrics for the data in test_dataloader. this method expects that test_dataloader
    was constructed for 1 epoch, but also has a stop at end of first epoch.
    :param model:
    :param test_dataloader:
    :param top_k:
    :return:
    """
    if not isinstance(test_dataloader._sampler, BatchSampler):
        raise ValueError("test_dataloader sampler must be an instance of BatchSampler")
    
    TEST_BATCH_SIZE = test_dataloader._sampler.batch_size
    TOTAL_RECORDS = test_dataloader._sampler.__len__()
    STEPS_PER_EPOCH = TOTAL_RECORDS // TEST_BATCH_SIZE  # = 7234
    
    ndcg_text = f'ndcg_{top_k}'
    mrr_text = f'mrr_{top_k}'
    recall_text = f'recall_{top_k}'
    
    history = {"test_loss": [], f"test_{mrr_text}": [], f"test_{ndcg_text}": [],
        f"test_{recall_text}": []}
    
    epoch_test_loss = [], epoch_test_mrr = [], epoch_test_ndcg = [], epoch_test_recall = []
    
    for global_step, padded_super_graph in enumerate(test_dataloader):
        epoch = global_step // STEPS_PER_EPOCH
        batch_idx = global_step % STEPS_PER_EPOCH
        #jraph.GraphsTuple
        metrics = eval_step(model, padded_super_graph)
        epoch_test_mrr.append(metrics[mrr_text])
        epoch_test_ndcg.append(metrics[ndcg_text])
        epoch_test_loss.append(metrics["loss"])
        epoch_test_recall.append(metrics[recall_text])
        if global_step % 100 == 0:
            ndcg = metrics[ndcg_text]
            recall = metrics[recall_text]
            mrr = metrics[mrr_text]
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
