import jraphx
import jax
import jax.numpy as jnp
from flax import nnx
import jraph
from array_record.python import array_record_module
import msgpack
from movie_lens_ranker.data_loading import *

'''
the class GAT inherits from BasicGNN which has this call method:
_call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray | None = None,
        edge_attr: jnp.ndarray | None = None,
        batch: jnp.ndarray | None = None,
        batch_size: int | None = None,
    ) -> jnp.ndarray:
    
nnx.Module call structure
'''

class GraphRanker(nnx.Module):
    def __init__(self, user_embeds, movie_embeds, num_candidates:int,
        in_features:int, hidden_features:int=128, num_layers:int=2,
        out_features:int=64, heads:int=4, dropout_rate:float=0.1, rngs:nnx.Rngs=nnx.Rngs(0)):
        
        self.user_embedding = nnx.Variable(user_embeds)
        self.movie_embedding = nnx.Variable(movie_embeds)
        
        self.num_candidates_per_user = num_candidates
        
        self.gat = jraphx.nn.models.GAT(
            in_features = in_features, hidden_features = hidden_features,
            num_layers = num_layers,
            out_features = out_features,
            heads = heads, v2 = False,  # Use GATv2 if True
            dropout_rate =dropout_rate,norm = "layer_norm",
            jk = "max", # JumpingKnowledge aggregation
            rngs = rngs
        )
        self.score_head = nnx.Linear(out_features  * 2, 1, rngs=rngs)
    
    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        #TODO: to speed this up, one could:
        #  (1) re-number the movie_ids throughout the Ranker application
        #      such that they are n_users + 1, n_users + 2, ... n_users + n_movies
        # (2) concatenate the embeddings here
        # (3) x = jnp.take(self.full_embeddings.value, graph.nodes, axis=0)
        # (4) updated_graph = self.gat(graph.replace(nodes=x))
        u_feat = jnp.take(self.user_weights.value, graph.nodes, axis=0)
        m_feat = jnp.take(self.movie_weights.value, graph.nodes, axis=0)
        
        # Masking to pick user vs movie features
        # type 0 is user_id, types 1 and 2 are movie_ids
        x = jnp.where(graph.nodes['type'][:, None] == 0, u_feat, m_feat)
        
        # Returns (num_nodes, out_features)
        updated_graph = self.gat(graph.replace(nodes=x))
        node_repr = updated_graph.nodes
        
        #The "Cross" Step: Pair User with Candidates
        # We need the User embedding for every graph in the batch
        is_user = (graph.nodes['type'] == 0)
        user_reprs = node_repr[is_user] # Shape: [Num_Graphs, Hidden_Dim]
        
        # Repeat each user's representation for each of their candidates
        # This aligns the User vector with the Movie vectors
        # broadcast_to works because jraph.batch maintains the graph order
        user_reprs_expanded = jnp.repeat(user_reprs,
            self.num_candidates_per_user, axis=0)
        
        # Get only the candidate movie representations
        is_candidate = (graph.nodes['type'] == 2)
        candidate_reprs = node_repr[is_candidate]  # Shape: [Num_Graphs * K, Hidden_Dim]
        
        # 5. Concatenate (The actual Cross-Encoding)
        # cross_repr shape: [Num_Graphs * K, Hidden_Dim * 2]
        cross_repr = jnp.concatenate(
            [user_reprs_expanded, candidate_reprs], axis=-1)
        
        # 6. Final Scoring
        # score_head should now be nnx.Linear(Hidden_Dim * 2, 1)
        scores = self.score_head(cross_repr)
        
        # Return flat scores for the candidates only
        return jnp.squeeze(scores, axis=-1)
        