from typing import Dict, Tuple

import jax
import jraph
import jax.numpy as jnp
from flax import nnx
import rax
import grain
from flax.typing import Array
from tqdm import tqdm

import chex

import numpy as np

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
def eval_step(model: GraphRanker, padded_graph: jraph.GraphsTuple,
        optimizer: nnx.Optimizer) -> Tuple[float, Dict[str, float]]:
    """
    train step over a batch, where padded_graph contains super graph of the batch
    :param model:
    :param padded_graph:
    :param optimizer:
    :return:
    """
    def loss_fn(model, padded_graph) -> Tuple[Array, Dict[str, Array]]:
        scores_2d, labels_2d, main_mask = score_and_shape_results(
            model, padded_graph)
        
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
            safe_scores, labels_2d, where=main_mask, topn=20, reduce_fn=jnp.mean)
        ndcg = rax.ndcg_metric(
            safe_scores, labels_2d, where=main_mask, topn=20, reduce_fn=jnp.mean)
        recall = rax.recall_metric(
            safe_scores, labels_2d, where=main_mask, topn=20, reduce_fn=jnp.mean)
        return loss, {"mrr": mrr, "ndcg": ndcg, "recall": recall}
    
    # has_aux is necessary when loss_fn returns more than scalar loss
    (loss, metrics_dict), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, padded_graph)
    optimizer.update(model, grads)
    metrics_dict['loss'] = loss
    return metrics_dict

def debug_train_fn(model, num_epochs: int, train_dataloader: grain.DataLoader,
        val_dataloader: grain.DataLoader,
        optimizer: nnx.Optimizer, batch_size: int, max_history: int, num_candidates: int):
    """
    attempt to get the model to overfit to make sure it can learn
    - training 1 batch elminiates variance, so we know that the math is wrong if the loss doesn't decrease.
    - uf loss starys flat, implies the gradients are essentially 0 and there may be a masking bug hiding the positive label.
    """
    single_train_batch = next(iter(train_dataloader))
    
    history = {"train_loss": [], "val_loss": [], "val_mrr": [], "val_ndcg": [],
        "val_recall": []}
    
    for epoch in range(num_epochs):
        epoch_train_loss = []
        loss = train_step(model, single_train_batch, optimizer)
        epoch_train_loss.append(loss)
        
        if epoch % 5 == 0:
            # We use block_until_ready to get accurate timing/sync for the print
            loss_val = jax.device_get(loss)
            print(f"Epoch {epoch}: Overfit Loss {loss_val:.6f}")
            
            # Optional: Exit early if we've successfully overfit
        if loss < 1e-5:
            print(f"✅ Successfully overfit batch at epoch {epoch}")
            break
    
    return history
    
def train_fn(model, num_epochs: int, train_dataloader: grain.DataLoader,
        val_dataloader: grain.DataLoader,
        optimizer: nnx.Optimizer, batch_size: int, max_history: int, num_candidates: int):
    
    jax.debug.print("expect the model training to start w/ loss = {}", -np.log(1./num_candidates))
    
    #DEBUG:
    if False:
        return debug_train_fn(model, num_epochs, train_dataloader, val_dataloader,
        optimizer, batch_size, max_history, num_candidates)
    
    history = {"train_loss": [], "val_loss":[], "val_mrr": [], "val_ndcg": [], "val_recall":[]}
    
    #configure for early stopping when ndcg stops changing
    patience = 5
    best_ndcg = -1.0
    epochs_without_improvement = 0
    best_state = None  # To store the best weights
    delay = 10 # min number of epochs to learn
    
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        epoch_train_loss = []
        for padded_super_graph in pbar:
            #jraph.GraphsTuple
            loss = train_step(model, padded_super_graph, optimizer)
            epoch_train_loss.append(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
        avg_train_loss = jnp.mean(jnp.array(epoch_train_loss))
        jax.debug.print("Epoch {epoch}: Train Loss {avg_train_loss:.4f}", avg_train_loss=avg_train_loss, ordered=True)
        epoch_val_loss = [], epoch_val_mrr = [], epoch_val_ndcg = [], epoch_val_recall = []
        
        for padded_super_graph_val in  val_dataloader:
            metrics = eval_step(model, padded_super_graph_val, optimizer)
            epoch_val_mrr.append(metrics["mrr"])
            epoch_val_ndcg.append(metrics["ndcg"])
            epoch_val_loss.append(metrics["val_loss"])
            epoch_val_recall.append(metrics["val_recall"])
        
        avg_val_loss = jnp.mean(jnp.array(epoch_val_loss))
        avg_val_mrr = jnp.mean(jnp.array(epoch_val_mrr))
        avg_val_ndcg = jnp.mean(jnp.array(epoch_val_ndcg))
        avg_val_recall = jnp.mean(jnp.array(epoch_val_recall))
        jax.debug.print("Epoch {epoch}: Train Loss {avg_train_loss:.4f} |Val Loss {avg_val_loss:.4f} "
            "| Val NDCG {avg_val_ndcg:.4f} | Val MRR {avg_val_mrr:.4f} | Val recall {avg_val_recall:.4f}",
            epoch=epoch, avg_train_loss=avg_train_loss, avg_val_loss=avg_val_loss,
            avg_val_ndcg=avg_val_ndcg, avg_val_mrr=avg_val_mrr, avg_val_recall=avg_val_recall)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_mrr"].append(avg_val_mrr)
        history["val_ndcg"].append(avg_val_ndcg)
        history["val_recall"].append(avg_val_recall)
        
        #early stopping
        if avg_val_ndcg > best_ndcg:
            best_ndcg = avg_val_ndcg
            epochs_without_improvement = 0
            best_state = model.split()
            print(f"  New best NDCG! Saving model...")
        elif epoch >= delay:
            epochs_without_improvement += 1
            print( f"  No improvement for {epochs_without_improvement} epoch(s).")
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            model.update(best_state)
            break
        
    return history

def test_fn(model, num_epochs: int, test_dataloader: grain.DataLoader,
        optimizer: nnx.Optimizer, batch_size: int, max_history: int,
        num_candidates: int):
    
    history = {"test_loss": [], "test_mrr": [], "test_ndcg": [], "test_f1": []}
    epoch_test_loss = [], epoch_test_mrr = [], epoch_test_ndcg = [], epoch_test_f1 = []
    for epoch in range(num_epochs):
        pbar = tqdm(test_dataloader, desc=f"Epoch {epoch}")
        for padded_super_graph in pbar:
            metrics = eval_step(model, padded_super_graph, optimizer)
            epoch_test_mrr.append(metrics["mrr"])
            epoch_test_ndcg.append(metrics["ndcg"])
            epoch_test_loss.append(metrics["val_loss"])
            epoch_test_f1.append(metrics["val_f1"])
            pbar.set_postfix({"test_loss": f"{metrics['loss']:.4f}",
                "test_ndcg": f"{metrics['ndcg']:.4f}",
                "test_mrr": f"{metrics['mrr']:.4f}",
                "test_recall": f"{metrics['recall']:.4f}",
            })
        
        avg_test_loss = jnp.mean(jnp.array(epoch_test_loss))
        avg_test_mrr = jnp.mean(jnp.array(epoch_test_mrr))
        avg_test_ndcg = jnp.mean(jnp.array(epoch_test_ndcg))
        avg_test_f1 = jnp.mean(jnp.array(epoch_test_f1))
        print(f"Epoch {epoch}: Test Loss {avg_test_loss:.4f} "
            f"| test NDCG {avg_test_ndcg:.4f} | test MRR {avg_test_mrr:.4f} | test F1 {avg_test_f1:.4f}")
        history["test_loss"].append(avg_test_loss)
        history["test_mrr"].append(avg_test_mrr)
        history["test_ndcg"].append(avg_test_ndcg)
        history["test_f1"].append(avg_test_f1)
        
    return history
