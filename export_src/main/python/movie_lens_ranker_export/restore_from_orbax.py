import jax
from flax import nnx
import orbax.checkpoint as ocp

from typing import Dict, Any

from functools import partial

from build.lib.movie_lens_ranker.util import get_model_mesh
from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.train import create_fake_jagged_batch, convert_to_global, create_dummy_super_padded_graph
from movie_lens_ranker.util import calc_number_jax_graph_components

from movie_lens_ranker.util_np import optimized_batch_and_pad

def restore_model_from_checkpoint(checkpoint_uri:str, replace_embeddings_gs_uri:str=None) -> Dict[str, Any]:

    model_mesh = get_model_mesh()

    mngr = ocp.CheckpointManager(checkpoint_uri,
         item_handlers={
             'model': ocp.StandardCheckpointHandler(),
             'rngs': ocp.StandardCheckpointHandler(),
             'config': ocp.handlers.JsonCheckpointHandler()
         },)

    epoch = mngr.latest_step()
    if epoch is None:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_uri}")

    restore_fn = partial(convert_to_global, mesh=model_mesh, sync=False)

    # restore config, then rngs, so can restore model from them
    restored_config = mngr.restore(epoch, args=ocp.args.Composite(config=ocp.args.JsonRestore()))
    config = restored_config['config']

    if replace_embeddings_gs_uri:
        #replace gs:// with
        config['user_embeddings_uri'] = config['user_embeddings_uri'].replace("gs://", replace_embeddings_gs_uri)
        config['movie_embeddings_uri'] = config['movie_embeddings_uri'].replace("gs://", replace_embeddings_gs_uri)

    _dict = _build_model_only(config, rngs=nnx.Rngs(config.get('seed', 0)))
    model = _dict['model']
    num_users = _dict['num_users']
    num_movies = _dict['num_movies']
    embed_len = _dict['embed_len']

    #restore state to those objects:
    model_graph, model_state = nnx.split(model)
    rngs = nnx.Rngs(config.get('seed', 0))

    global_model_target = jax.tree_util.tree_map(restore_fn, model_state)
    global_rng_target = jax.tree_util.tree_map(restore_fn, nnx.state(rngs))

    restored = mngr.restore(
        epoch,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(global_model_target),
            rngs=ocp.args.StandardRestore(global_rng_target),
        )
    )

    nnx.update(model, restored['model'])
    nnx.update(rngs, restored['rngs'])

    return {
        'model': model,
        'rngs': rngs,
        'num_users': num_users,
        'num_movies': num_movies,
        'embed_len' : embed_len,
        'config': config
    }


def _build_model_only(config:dict, rngs:nnx.Rngs) -> Dict[str, Any]:
    """
    build the model,and return them in a dictionary that has keys {"rngs", "model", , 'num_users', 'num_movies', 'embed_len'}
     where num_users and num_movies are the number of users and movies in the
    entire user and movie catalog represented by the embeddings.

    :param config:
    :param rngs:
    :return: dictionary with keys {"rngs", "model", "optimizer", 'train_dataloader',
    'val_dataloader', 'num_users', 'num_movies'}
    """
    if rngs is None:
        raise ValueError('rngs cannot be None')

    req_keys = {'max_history', 'num_epochs', 'batch_size', 'seed', 'num_users', 'num_movies', 'embed_len'}
    for key in req_keys:
        if key not in config:
            raise ValueError(f'missing key {key} in config')

    #each model gets the same rngs so will have the same initialization even though in a different process
    model = GraphRanker(
        emb_in_dim = config['embed_len'],
        num_candidates=config['num_candidates'],
        hidden_features=config['hidden_dim'],
        num_layers=config['num_layers'],
        out_features=config['out_dim'],
        heads=config['num_heads'],
        edge_embed_dim=config['edge_embed_dim'],
        dropout_rate=config['dropout_rate'], rngs=rngs)

    '''
    #initialize the layers with same fake data
    user_id_range = (1, config['num_users'])
    movie_id_range = (config['num_users'] + 1, config['num_users'] + config['num_movies'])

    fake_padded_graph = create_dummy_super_padded_graph(batch_size=config['batch_size'],
                                          max_history=config['max_history'],
                                          num_candidates=config['num_candidates'],
                                          user_id_range=user_id_range,
                                          movie_id_range=movie_id_range,
                                          movie_embeddings_uri = config['movie_embeddings_uri'],
                                          user_embeddings_uri = config['user_embeddings_uri'])

    model.eval()
    model(fake_padded_graph)
    '''

    return {"rngs": rngs, "model": model,
            'num_users': config['num_users'], 'num_movies': config['num_movies'], 'embed_len' : config['embed_len']}



