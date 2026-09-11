"""
in interest of replacing train_step and eval_step with point-wise losses
instead of list-wise.

the reason for considering pointwise loss is to improve the stability of
the ndcg_tail_20 training.
"""
import jax
import rax
from flax import nnx
import jax.numpy as jnp
import numpy as np
from jax import Array
import jraph
from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.train import score_and_shape_results


@nnx.jit
def train_step(model: GraphRanker, padded_graph: jraph.GraphsTuple,
               optimizer: nnx.Optimizer,
               movie_tiers:np.ndarray, movie_offset:int = 6040+1,
               tier_weights_config: jnp.ndarray = jnp.array([0.25, 0.55, 0.2]),
               focal_loss_gamma: float = 2.0,
               ) -> Array:

    normalized_weights = tier_weights_config / jnp.sum(tier_weights_config)

    def loss_fn(model, padded_graph) -> Array:

        #labels_2d = 1 for target movie_id, else 0
        scores_2d, labels_2d, main_mask, cand_ids_2d = score_and_shape_results(model, padded_graph)
        safe_scores = jnp.where(main_mask, scores_2d, -1e9)

        # FOCAL WEIGHTING (Pointwise adaptation) ---
        # Calculate independent probability p_i via sigmoid
        probs = jax.nn.sigmoid(safe_scores)

        # Calculate target probability p_t:
        # p_t = p_i if label=1 (positive), 1-p_i if label=0 (negative)
        target_probs = jnp.where(labels_2d > 0.5, probs, 1.0 - probs)

        # Scales gradient by hard/easy predictions across ALL candidates independently
        focal_weights = jnp.power(1.0 - target_probs, focal_loss_gamma)

        # IPW TIER WEIGHTING (Unchanged) ---
        # Look up tier weight based on target item ID.
        batch_target_ids = jnp.sum(cand_ids_2d * labels_2d, axis=1) - movie_offset
        safe_target_ids = jnp.clip(batch_target_ids, 0, normalized_weights.shape[0] - 1)
        batch_target_tiers = movie_tiers[safe_target_ids]

        # Shape: (batch_size, 1). This broadcasts the target's tier weight
        # to all negatives evaluated in that query's list.
        ipw_weights = normalized_weights[batch_target_tiers][:, None]

        # COMBINED LOSS ---
        # focal_weights is (batch_size, num_candidates), ipw_weights is (batch_size, 1)
        combined_weights = focal_weights * ipw_weights

        # Swap to Pointwise Sigmoid Loss (BCE)
        loss = rax.pointwise_sigmoid_loss(
            scores=safe_scores,
            labels=labels_2d,
            where=main_mask,
            weights=combined_weights,
            reduce_fn=jnp.mean
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

@nnx.jit(static_argnames=('top_k',))
def eval_step(model: GraphRanker, padded_graph: jraph.GraphsTuple,
              movie_tiers:np.ndarray, movie_offset:int,
              tier_weights_config: jnp.ndarray = jnp.array([0.25, 0.55, 0.2]),
              top_k:int=20) -> dict[str, Array]:
    """
    train step over a batch, where padded_graph contains super graph of the batch
    :param model:
    :param padded_graph:
    :param movie_tiers: array of movie tiers where indicies are movie_id-movie_offset and values are 0, 1, 2 for
        head, torso, tail, respectively of the movie frequency distribution (where distribution
        was determined from train dataset).
    :param movie_offset: offset from 0 of movie_ids
    :param tier_weights_config : holds the weights of the tiers head, torso, and tail, respectively.  They will be normalized to sum to 1 if not already.
    :param top_k:
    :return: dictionary with keys:
        "loss",
        "mrr_{top_k}",
        "ndcg_{top_k}", "ndcg_head_{top_k}", "ndec_torso_{top_k}", "ndcg_tail_{top_k}"
        "recall_{top_k}", "recall_head_{top_k}",, "recall_torso_{top_k}", ,"recall_tail_{top_k}",
        "precision_{top_k}", "precision_1", "precision_5",
        "logit_mean"
        "logit_std"
        "logit_min"
        "logit_max"
    """

    #FIXED values decided in the TwoTowerDNN bi-encoder:
    normalized_weights = tier_weights_config / jnp.sum(tier_weights_config)
    w_head:float= normalized_weights[0] # 0.25
    w_torso:float= normalized_weights[1] # 0.55
    w_tail:float= normalized_weights[2]  # 0.2

    #shapes: (total number of graphs including dummy grpahs, model.num_candidates).
    # main_mask is True for real data and False for dummy graph data
    scores_2d, labels_2d, main_mask, cand_ids_2d = score_and_shape_results(model, padded_graph)
    safe_scores = jnp.where(main_mask, scores_2d, -1e9)

    # Swap to Pointwise Sigmoid Loss (BCE)
    loss = rax.pointwise_sigmoid_loss(
        scores=safe_scores,
        labels=labels_2d,
        where=main_mask,
        reduce_fn=jnp.mean
    )

    per_query_ndcg = rax.ndcg_metric(
        safe_scores, labels_2d, where=main_mask, topn=top_k, reduce_fn=None)
    per_query_recall = rax.recall_metric(
        safe_scores, labels_2d, where=main_mask, topn=top_k, reduce_fn=None)
    mrr = rax.mrr_metric(
        safe_scores, labels_2d, where=main_mask, topn=top_k, reduce_fn=jnp.mean)
    prec_1 = rax.precision_metric(
        safe_scores, labels_2d, where=main_mask, topn=1, reduce_fn=jnp.mean)
    prec_5 = rax.precision_metric(
        safe_scores, labels_2d, where=main_mask, topn=5, reduce_fn=jnp.mean)
    prec_k = rax.precision_metric(
        safe_scores, labels_2d, where=main_mask, topn=top_k, reduce_fn=jnp.mean)

    #statistics to track the score head and logistics
    safe_num_valid = jnp.maximum(jnp.sum(main_mask), 1.0)
    masked_scores = jnp.where(main_mask, scores_2d, 0.0)

    # Mean and Standard Deviation over valid unpadded candidates
    logit_mean = jnp.sum(masked_scores) / safe_num_valid
    #logit_mean = jnp.mean(jnp.where(main_mask, scores_2d, jnp.inf))
    logit_var = jnp.sum(jnp.where(main_mask, jnp.square(scores_2d - logit_mean), 0.0)) / safe_num_valid
    logit_std = jnp.sqrt(logit_var)
    # Extreme values bounded strictly within the masked region
    logit_min = jnp.min(jnp.where(main_mask, scores_2d, jnp.inf))
    logit_max = jnp.max(jnp.where(main_mask, scores_2d, -jnp.inf))

    # ==========================================
    # TARGET ID EXTRACTION
    # ==========================================
    # labels_2d is exactly 1 at the target movie_id index and 0 elsewhere.
    # Multiplying and summing across the row isolates the target ID.
    batch_target_ids = jnp.sum(cand_ids_2d * labels_2d, axis=1) - movie_offset

    safe_target_ids = jnp.clip(batch_target_ids, 0, movie_tiers.shape[0] - 1)
    #in safe_target_ids, all non-target candidates map to index=0
    batch_target_tiers = movie_tiers[safe_target_ids]

    # Create Slice Masks for the target movie_id
    valid_query_mask = (jnp.sum(labels_2d * main_mask, axis=-1) > 0)
    head_mask = valid_query_mask & (batch_target_tiers == 0)
    torso_mask = valid_query_mask & (batch_target_tiers == 1)
    tail_mask = valid_query_mask & (batch_target_tiers == 2)

    def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
        count = jnp.sum(mask)
        return jnp.where(count > 0, jnp.sum(values * mask) / count, 0.0)

    metrics_dict = {
        f"loss" : loss,
        f"ndcg_{top_k}": masked_mean(per_query_ndcg, valid_query_mask),
        f"ndcg_head_{top_k}": masked_mean(per_query_ndcg, head_mask),
        f"ndcg_torso_{top_k}": masked_mean(per_query_ndcg, torso_mask),
        f"ndcg_tail_{top_k}": masked_mean(per_query_ndcg, tail_mask),
        f"recall_{top_k}": masked_mean(per_query_recall, valid_query_mask),
        f"recall_head_{top_k}": masked_mean(per_query_recall, head_mask),
        f"recall_torso_{top_k}": masked_mean(per_query_recall, torso_mask),
        f"recall_tail_{top_k}": masked_mean(per_query_recall, tail_mask),
        f"precision_{top_k}": prec_k,
        f"precision_1": prec_1,
        f"precision_5": prec_5,
        f"mrr_{top_k}": mrr,
        "logit_mean": logit_mean,
        "logit_std": logit_std,
        "logit_min": logit_min,
        "logit_max": logit_max,
    }

    metrics_dict[f"composite_ndcg_{top_k}"] = (
            w_head * metrics_dict[f"ndcg_head_{top_k}"] +
            w_torso * metrics_dict[f"ndcg_torso_{top_k}"] +
            w_tail * metrics_dict[f"ndcg_tail_{top_k}"]
    )

    return metrics_dict