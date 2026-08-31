import jraphx
import jraph
from array_record.python import array_record_module
from flax import nnx
import jax.numpy as jnp

class GraphRanker(nnx.Module):
    def __init__(self,
            emb_in_dim:int,
            num_candidates: int,
            hidden_features: int = 128, num_layers: int = 2,
            out_features: int = 64, heads: int = 4, edge_embed_dim:int=8,
            mlp_hidden_dim : float = 0.5,
            dropout_rate: float = 0.1,
            temperature: float= 0.1,
            rngs: nnx.Rngs = nnx.Rngs(0)):
        """
        :param num_candidates: number per user of negatives + positive to use for their final graph
        :param hidden_features: size of hidden layers per head in the GATv2 layer
        :param num_layers: number of layers in the GATv2 layer
        :param out_features: output dimension of the score head dense layer
        :param heads: number of attention heads in the GATv2 layer
        :param edge_embed_dim: typically a value in range 4 to 16. size of output of GATv2 layer.
        :param mlp_hidden_dim the output dimension of the 2 layer MLP that precedes the score head expressed
            as a multiple of out_features value.
        :param dropout_rate: the dropout probability of a layer in the GATv2 layer
        :param temperature: a factor to use after L2 normalizing the user and candidate representations
        :param rngs: the pseudo random number generator
        """
        self.edge_embed_dim = edge_embed_dim
        #self.embed_in_dim = user_movie_embeds.shape[1]
        self.embed_in_dim = emb_in_dim
        self.num_candidates = num_candidates
        self.temperature = temperature

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

        # nearest multiple of 8, then lower bound of 16
        mlp_hidden_dim_int = max(16, int(round(mlp_hidden_dim * out_features / 8.0) * 8.0))

        # The Hidden Layer (Accepts U, C, U*C, |U-C|)
        self.interaction_hidden = nnx.Linear(
            out_features * 4,
            mlp_hidden_dim_int,
            use_bias=True,
            rngs=rngs,
            bias_init=nnx.initializers.zeros_init(),
            kernel_init=nnx.initializers.he_normal() # He init is better for ReLU/GELU
        )

        ## Output projection accepts 4 * out_features (u, c, u*c, |u-c|)
        # pure data paralellism, no sharding of the model:
        self.score_head = nnx.Linear(
            mlp_hidden_dim_int,
            1,
            use_bias=False,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
    
    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        """
        always returns a static shape of (max_graphs * K)
        :param graph:padded super graph of graphs from a batch.
        :return: scores of shape (max_graphs * self.num_candidates)
        """
        #[ len(graph.nodes["ids"]) X embed_in_dim ]

        x = graph.nodes["embeddings"]
        
        # Convert edge ratings to integers (0-5)
        # Ensure they are int32 so the embedding layer can use them as indices
        edge_indices = graph.edges["rating"].astype(jnp.int32)
        
        edge_attr = self.rating_embed(edge_indices)
        
        num_total_nodes = x.shape[0]
        batch_indices = jnp.repeat(
            jnp.arange(len(graph.n_node)),
            graph.n_node,
            total_repeat_length=num_total_nodes
        )
        #batch_indices length is num_total_nodes

        #flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
        #Inward: senders array holds the history movie edges and have implied receiver of user (where type==1)
        #Outward: receivers array holds the candidate movie edges and have implied send of user (where type==1)
        #How the GATV2 here acts as a cross-endcoder via multi-hop flow:
        #num_layers=2
        #    Layer 1 (these happen in parallel):
        #       h_u': user node gathers information from its history node embeddings, learning the user profile.
        #       h_c': the candidate nodes gather information from the user node embeddings, to learn who is evaluating them.
        #    Layer 2 (these happen in parallel):
        #       h_u'': user node gathers information from history node embeddings again
        #       h_c'': the candidate nodes gather information from h_u'  <==== interaction to inform candidates about user
        # Returns (num_nodes, out_features)
        #returns node embeddings as final representation of each node after
        # all message-passing layers.

        node_repr = self.gatv2(
            x=x,
            edge_index = jnp.stack([graph.senders, graph.receivers]), #must be shape [2, num_edges]
            edge_weight = None,
            edge_attr = edge_attr,
            batch=batch_indices,
            batch_size=graph.n_node.shape[0]
        )
        #NOTE: cannot append to the sends and receives candidates -> user edges because it would cause
        # target leakage.  also it would consume alot of memory.

        num_total_graphs = len(graph.n_node) # number of batches + number of dummy padding graphs
        num_total_candidates = num_total_graphs * self.num_candidates

        # We use fill_value=0. For dummy graphs that don't have type==1 or type==3,
        # JAX will pad the remaining required indices with 0.
        # This safely points phantom entities to the 0th node embedding.
        user_indices = jnp.where(
            graph.nodes["type"] == 1,
            size=num_total_graphs,
            fill_value=0
        )[0]

        cand_indices = jnp.where(
            graph.nodes["type"] == 3,
            size=num_total_candidates,
            fill_value=0
        )[0]

        user_reprs = node_repr[user_indices]
        cand_reprs = node_repr[cand_indices]

        # NaN-Safe L2 Normalization (epsilon INSIDE the square root)
        def safe_l2_normalize(x, eps=1e-8):
            norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + eps)
            return x / norm

        user_norm = safe_l2_normalize(user_reprs)
        cand_norm = safe_l2_normalize(cand_reprs)

        user_expanded = jnp.repeat(user_norm, self.num_candidates, axis=0)

        dot_product = user_expanded * cand_norm         #element-wise cosine similarity
        abs_diff = jnp.sqrt(jnp.square(user_expanded - cand_norm) + 1e-8) #prevents gradient explosion if values diffs are 0

        ## Concatenate into a Lean Matching Vector [U, C, U*C, |U-C|]
        matching_vector = jnp.concatenate([user_expanded, cand_norm, dot_product, abs_diff], axis=-1)

        ## the following was used for linear interaction for a 2 * out_features improvement in rumtime.
        ## the savings during training is nearly negigible though if the T4x2 GPUs are used because the bottleneck is loading data onto the GPUs
        ## and then the math is done extremely quickly.
        ## so commenting out this linear interaction to add a non-linear interaction using 1 2 layer MLP
        # Single Bias-Free Linear Projection + Temperature Scaling
        #scores = self.score_head(matching_vector)
        #scores = jnp.squeeze(scores, axis=-1) / self.temperature

        hidden = self.interaction_hidden(matching_vector)

        # Apply non-linearity (GELU is standard for modern recsys, ReLU is slightly faster on CPU)
        hidden = nnx.gelu(hidden)

        # Project to final scalar score
        scores = self.score_head(hidden)

        # --- SCALING ---
        scores = jnp.squeeze(scores, axis=-1) / self.temperature

        return scores

        return scores
