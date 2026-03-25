from typing import Dict, Tuple

import jraph
import jax.numpy as jnp
from flax import nnx
import math
import rax
import grain
from flax.typing import Array

from movie_lens_ranker.model import GraphRanker


def calc_number_jax_graph_components(batch_size: int, max_history: int,
        num_candidates: int) -> Dict[str, int]:
    # 40->50, #123->200, #1234->2000, #12345->20000
    def next_64(x) -> int:
        return 64*(1+int(x//64))
    
    max_nodes = next_64(
        batch_size * (1 + max_history + num_candidates))
    max_edges = next_64(batch_size * (max_history + num_candidates))
    max_graphs = batch_size + 1
    return {'max_nodes': max_nodes, 'max_edges': max_edges,
        'max_graphs': max_graphs}

def pad_batch_for_jax(batch: list[jraph.GraphsTuple],
        max_nodes: int, max_edges: int,
        max_graphs: int) -> jraph.GraphsTuple:
    super_graph = jraph.batch(batch)
    # n_graph is usually batch_size + 1 (the padding graph)
    padded_graph = jraph.pad_with_graphs(
        super_graph,
        n_node=max_nodes,
        n_edge=max_edges,
        n_graph=max_graphs
    )
    return padded_graph


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
    all_scores = model(padded_graph)
    
    num_total_graphs = padded_graph.n_node.shape[0]  # batch_size + 1
    K = model.K  # num_candidates from data loading
    
    total_candidate_slots = num_total_graphs * K
    
    # Extract Candidate Data. length is K * num_total_graphs
    cand_indices = jnp.where(
        padded_graph.nodes["type"] == 2,
        size=total_candidate_slots
    )[0]
    
    # lengths are K * num_total_graphs
    labels_flat = padded_graph.nodes["label"][cand_indices]
    record_mask_flat = padded_graph.nodes["candidate_mask"][cand_indices]
    
    # Reshape everything to [Batch, K]
    scores_2d = all_scores.reshape(num_total_graphs, K)
    labels_2d = labels_flat.reshape((num_total_graphs, K))
    record_mask_2d = record_mask_flat.reshape((num_total_graphs, K))
    
    # Create Batch Mask (Ignore the last JAX padding graph)
    # real_graph_indices: [0, 1, 2] -> [True, True, False]
    is_real_graph = jnp.arange(num_total_graphs) < (
                num_total_graphs - 1)
    
    # Broadcast to [3, K]
    batch_mask = jnp.broadcast_to(is_real_graph[:, None],
        (num_total_graphs, K))
    
    #  Combine Masks
    # Master mask is True only for real candidates in real graphs
    final_mask = record_mask_2d & batch_mask
    
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
    def loss_fn(model, padded_graph) -> float:
        scores_2d, labels_2d, master_mask = score_and_shape_results(model, padded_graph)
        # Rax Ranking Loss & Metrics
        # Rax is ignores entries where master_mask is False
        loss = rax.softmax_loss(
            scores=scores_2d,
            labels=labels_2d,
            where=master_mask,
            reduce_fn=jnp.mean
        )
        return loss
    loss, grads = nnx.value_and_grad(loss_fn)(model, padded_graph)
    optimizer.update(model, grads)
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
        scores_2d, labels_2d, master_mask = score_and_shape_results(
            model, padded_graph)
        
        # Rax Ranking Loss & Metrics
        # Rax is designed to ignore entries where master_mask is False
        loss = rax.softmax_loss(
            scores=scores_2d,
            labels=labels_2d,
            where=master_mask,
            reduce_fn=jnp.mean
        )
        
        mrr = rax.mrr_metric(
            scores_2d, labels_2d, where=master_mask, reduce_fn=jnp.mean)
        ndcg = rax.ndcg_metric(
            scores_2d, labels_2d, where=master_mask, reduce_fn=jnp.mean)
        
        return loss, {"mrr": mrr, "ndcg": ndcg}
    
    # has_aux is necessary when loss_fn returns more than scalar loss
    (loss, metrics_dict), grads = nnx.value_and_grad(loss_fn,
        has_aux=True)(model, padded_graph)
    optimizer.update(model, grads)
    metrics_dict['loss'] = loss
    return metrics_dict

def train_fn(model, num_epochs: int, train_dataloader: grain.DataLoader,
        val_dataloader: grain.DataLoader,
        optimizer: nnx.Optimizer, batch_size: int, max_history: int, num_candidates: int):
    
    jax_graph_comp_dict = calc_number_jax_graph_components(batch_size,
        max_history, num_candidates)
    
    history = {"train_loss": [], "val_loss":[], "val_mrr": [], "val_ndcg": [], "val_f1":[]}
    
    #configure for early stopping when ndcg stops changing
    patience = 5
    best_ndcg = -1.0
    epochs_without_improvement = 0
    best_state = None  # To store the best weights
    delay = 10 # min number of epochs to learn
    
    for epoch in range(num_epochs):
        epoch_train_loss = []
        for batch_list in train_dataloader:
            padded_super_graph = pad_batch_for_jax(
                batch_list,
                max_nodes=jax_graph_comp_dict['max_nodes'],
                max_edges=jax_graph_comp_dict['max_edges'],
                max_graphs=jax_graph_comp_dict['max_graphs'],
            )
            loss = train_step(model, padded_super_graph, optimizer)
            epoch_train_loss.append(loss)
            
        avg_train_loss = jnp.mean(jnp.array(epoch_train_loss))
        epoch_val_loss = [], epoch_val_mrr = [], epoch_val_ndcg = [], epoch_val_f1 = []
        for val_batch_list in val_dataloader:
            padded_super_graph = pad_batch_for_jax(
                val_batch_list,
                max_nodes=jax_graph_comp_dict['max_nodes'],
                max_edges=jax_graph_comp_dict['max_edges'],
                max_graphs=jax_graph_comp_dict['max_graphs'],
            )
            metrics = eval_step(model, padded_super_graph, optimizer)
            epoch_val_mrr.append(metrics["mrr"])
            epoch_val_ndcg.append(metrics["ndcg"])
            epoch_val_loss.append(metrics["val_loss"])
            epoch_val_f1.append(metrics["val_f1"])
        
        avg_val_loss = jnp.mean(jnp.array(epoch_val_loss))
        avg_val_mrr = jnp.mean(jnp.array(epoch_val_mrr))
        avg_val_ndcg = jnp.mean(jnp.array(epoch_val_ndcg))
        avg_val_f1 = jnp.mean(jnp.array(epoch_val_f1))
        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} |Val Loss {avg_val_loss:.4f} "
              f"| Val NDCG {avg_val_ndcg:.4f} | Val MRR {avg_val_mrr:.4f} | Val F1 {avg_val_f1:.4f}")
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_mrr"].append(avg_val_mrr)
        history["val_ndcg"].append(avg_val_ndcg)
        history["val_f1"].append(avg_val_f1)
        
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
    jax_graph_comp_dict = calc_number_jax_graph_components(batch_size,
        max_history, num_candidates)
    
    history = {"test_loss": [], "test_mrr": [], "test_ndcg": [], "test_f1": []}
    epoch_test_loss = [], epoch_test_mrr = [], epoch_test_ndcg = [], epoch_test_f1 = []
    for epoch in range(num_epochs):
        for test_batch_list in test_dataloader:
            padded_super_graph = pad_batch_for_jax(
                test_batch_list,
                max_nodes=jax_graph_comp_dict['max_nodes'],
                max_edges=jax_graph_comp_dict['max_edges'],
                max_graphs=jax_graph_comp_dict['max_graphs'],
            )
            metrics = eval_step(model, padded_super_graph, optimizer)
            epoch_test_mrr.append(metrics["mrr"])
            epoch_test_ndcg.append(metrics["ndcg"])
            epoch_test_loss.append(metrics["val_loss"])
            epoch_test_f1.append(metrics["val_f1"])
        
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
