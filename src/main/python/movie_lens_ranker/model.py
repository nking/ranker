import jraphx
import jax
import jax.numpy as jnp
from flax import nnx
import jraph
from array_record.python import array_record_module
from movie_lens_ranker.data_loading import *


class GraphRanker(nnx.Module):
    def __init__(self, user_movie_embeds: jnp.ndarray,
            num_candidates: int,
            hidden_features: int = 128, num_layers: int = 2,
            out_features: int = 64, heads: int = 4,
            dropout_rate: float = 0.1, rngs: nnx.Rngs = nnx.Rngs(0)):
        """
        :param user_movie_embeds: concat of [user embeddings and movie_embeddings]
        :param num_candidates:
        :param hidden_features:
        :param num_layers:
        :param out_features:
        :param heads:
        :param dropout_rate:
        :param rngs:
        """
        self.embed_in_dim = user_movie_embeds.shape[1]
        self.user_movie_embeddings = nnx.Variable(user_movie_embeds)
        
        self.num_candidates_per_user = num_candidates
        
        self.gatv2 = jraphx.nn.GAT(
            in_features=self.embed_in_dim,
            hidden_features=hidden_features,
            # 2 * embed_in_dim is probably good
            num_layers=num_layers,
            out_features=out_features,
            heads=heads,
            act_first=False,
            v2=True,  # Use GATv2 if True
            dropout_rate=dropout_rate,
            norm="layer_norm",
            jk="max",  # JumpingKnowledge aggregation
            edge_dim=None,
            rngs=rngs
        )
        self.score_head = nnx.Linear(out_features * 2, 1, rngs=rngs)
    
    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        x = jnp.take(a=self.user_movie_embeddings.value,
            indices=graph.nodes["ids"], axis=0)
        
        #replace the nodes dictionary of "ids", "label", "type", "candidate_mask" with
        # the
        # Returns (num_nodes, out_features)
        #returns node embeddings as final representation of each node after
        # all message-passing layers.
        node_repr = self.gatv2(graph._replace(nodes=x))
        
        #but graph is still original graph, that is, graph.nodes['type'] still exists
        
        #The "Cross" Step: Pair User with Candidates
        # We need the User embedding for every graph in the batch
        is_user = (graph.nodes['type'] == 0)
        user_reprs = node_repr[
            is_user]  # Shape: [Num_Graphs, Hidden_Dim]
        
        # Repeat each user's representation for each of their candidates
        # This aligns the User vector with the Movie vectors
        # broadcast_to works because jraph.batch maintains the graph order
        user_reprs_expanded = jnp.repeat(user_reprs,
            self.num_candidates_per_user, axis=0)
        
        # Get only the candidate movie representations
        is_candidate = (graph.nodes['type'] == 2)
        candidate_reprs = node_repr[
            is_candidate]  # Shape: [Num_Graphs * K, Hidden_Dim]
        
        # 5. Concatenate (The actual Cross-Encoding)
        # cross_repr shape: [Num_Graphs * K, Hidden_Dim * 2]
        cross_repr = jnp.concatenate(
            [user_reprs_expanded, candidate_reprs], axis=-1)
        
        # 6. Final Scoring
        # score_head should now be nnx.Linear(Hidden_Dim * 2, 1)
        scores = self.score_head(cross_repr)
        
        # Return flat scores for the candidates only
        return jnp.squeeze(scores, axis=-1)
