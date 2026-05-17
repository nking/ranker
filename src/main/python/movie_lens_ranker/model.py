import jraphx
import jraph
from array_record.python import array_record_module
from movie_lens_ranker.data_loading import *
import jax.numpy as jnp

class GraphRanker(nnx.Module):
    def __init__(self, user_movie_embeds: jnp.ndarray,
            num_candidates: int,
            hidden_features: int = 128, num_layers: int = 2,
            out_features: int = 64, heads: int = 4, edge_embed_dim:int=8,
            dropout_rate: float = 0.1, rngs: nnx.Rngs = nnx.Rngs(0)):
        """
        :param user_movie_embeds: concat of [user embeddings and movie_embeddings]
        :param num_candidates: number per user of negatives + positive to use for their final graph
        :param hidden_features: size of hidden layers per head in the GATv2 layer
        :param num_layers: number of layers in the GATv2 layer
        :param out_features: utput dimension of the score head dense layer
        :param heads: number of attention heads in the GATv2 layer
        :param edge_embed_dim: typically a value in range 4 to 16. size of output of GATv2 layer
        :param dropout_rate: the dropout probability of a layer in the GATv2 layer
        :param rngs: the pseudo random number generator
        """
        self.edge_embed_dim = edge_embed_dim
        self.embed_in_dim = user_movie_embeds.shape[1]
        self.user_movie_embeddings = nnx.Variable(user_movie_embeds)
        
        self.K = num_candidates
        
        # 6 embeddings: 0 (No Rating/Candidate), 1, 2, 3, 4, 5 (Ratings)
        self.rating_embed = nnx.Embed(num_embeddings=6, features=edge_embed_dim, rngs=rngs)
        
        self.gatv2 = jraphx.nn.GAT(
            in_features=self.embed_in_dim,
            hidden_features=hidden_features,
            edge_dim=edge_embed_dim,
            # 2 * embed_in_dim is probably good
            num_layers=num_layers,
            out_features=out_features,
            heads=heads,
            act_first=False,
            v2=True,  # Use GATv2 if True
            dropout_rate=dropout_rate,
            norm="layer_norm",
            jk="max",  # JumpingKnowledge aggregation
            rngs=rngs
        )
        self.score_head = nnx.Linear(out_features * 2, 1, rngs=rngs)
    
    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        """
        always returns a static shape of (max_graphs * K)
        :param graph:
        :return:
        """
        #[ len(graph.nodes["ids"]) X embed_in_dim ]
        x = jnp.take(a=self.user_movie_embeddings.get_value(), indices=graph.nodes["ids"], axis=0)
        
        # Convert edge ratings to integers (0-5)
        # Ensure they are int32 so the embedding layer can use them as indices
        edge_indices = graph.edges["rating"].astype(jnp.int32)
        
        edge_attr = self.rating_embed(edge_indices)
        
        num_total_nodes = x.shape[0]
        batch_indices = jnp.repeat(
            jnp.arange(len(graph.n_node)), #batch_size + 1 dummy = 3
            graph.n_node, # ndarray of [21, 21, 86]
            total_repeat_length=num_total_nodes #128
        )
        #batch_indices length is num_total_nodes
        
        # Returns (num_nodes, out_features)
        #returns node embeddings as final representation of each node after
        # all message-passing layers.
        node_repr = self.gatv2(
            x=x,
            edge_index = jnp.stack([graph.senders, graph.receivers]),
            edge_weight = None,
            edge_attr = edge_attr,
            batch=batch_indices,
            batch_size=graph.n_node.shape[0]
        )
       
        num_total_graphs = len(graph.n_node) #batch_size + 1
        num_total_candidates = num_total_graphs * self.K # K is num_candidates from data loading stage
        
        user_indices = jnp.where(graph.nodes["type"] == 0, size=num_total_graphs)[0]
        cand_indices =  jnp.where(graph.nodes["type"] == 2, size=num_total_candidates)[0]
        
        user_reprs = node_repr[user_indices]
        cand_reprs = node_repr[cand_indices]
        
        # Cross-Encoder Concatenation
        # Repeat each user K times to pair with their respective candidates
        # Result: [U1, U1... (K times), U2, U2... (K times), U_pad, U_pad... (K times)]
        user_expanded = jnp.repeat(user_reprs, self.K, axis=0)
        
        combined = jnp.concatenate([user_expanded, cand_reprs], axis=-1)
        
        scores = self.score_head(combined)
        return jnp.squeeze(scores, axis=-1)