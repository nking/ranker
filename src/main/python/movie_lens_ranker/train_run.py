import jax
import optax
from flax import nnx
import grain
from math import log
from movie_lens_ranker.data_loading import create_train_and_val_dataloaders
from movie_lens_ranker.model import GraphRanker
from movie_lens_ranker.train import train_fn
from movie_lens_ranker.util import read_embeddings

# runs on every worker (GPU/TPU)
def train_loop_per_worker(config):
    
    worker_rank =  jax.process_index()
    #config should contain these:
    data_params_nontrainable_keys = {'movies_uri', 'recommendations_uri', 'recommendations_ts_uri',
        'ratings_train_uri', 'ratings_val_uri', 'train_negatives_uri', 'val_negatives_uri',
        'seed'
    }
    data_params_trainable_keys = {'max_history', 'num_candidates', 'num_epochs', 'batch_size'}
    model_params_nontrainable_keys = {'latest_checkpoint_dir', 'log_dir,'
        'movie_embeddings_uri', 'user_embeddings_uri', 'mlflow_config'}
    model_params_trainable_keys = {'top_k', 'learning_rate', 'weight_decay',
        'out_dim', 'hidden_dim', 'num_layers', 'num_heads', 'dropout_rate',
    }
    #mlflow_config
    
    '''
    if worker_rank == 0:
        ray.train.mlflow.setup_mlflow(
            tracking_uri=config['mlflow_config']['tracking_uri'],
            registry_uri=config['mlflow_config']['registry_uri'],
            experiment_id=config['mlflow_config']['experiment_id'],
            experiment_name=config['mlflow_config']['experiment_name'],
            tracking_token=config['mlflow_config']['tracking_token'],
            artifact_location=config['mlflow_config']['artifact_location'],
            create_experiment_if_not_exists=config['mlflow_config']['create_experiment_if_not_exists']
        )
    '''
    
    data_params_nontrainable = config['data_params_nontrainable']
    data_params_trainable = config['data_params_trainable']
    model_params_nontrainable = config['model_params_nontrainable']
    model_params_trainable = config['model_params_trainable']

    train_dataloader, val_dataloader = create_train_and_val_dataloaders(
        movies_uri=data_params_nontrainable['movies_uri'],
        recommendations_uri=data_params_nontrainable['recommendations_uri'],
        recommendations_ts_uri=data_params_nontrainable['recommendations_ts_uri'],
        train_ratings_uri=data_params_nontrainable['ratings_train_uri'],
        val_ratings_uri=data_params_nontrainable['ratings_val_uri'],
        train_negatives_uri=data_params_nontrainable['train_negatives_uri'],
        val_negatives_uri=data_params_nontrainable['val_negatives_uri'],
        max_history=data_params_trainable['max_history'],
        num_candidates=data_params_trainable['num_candidates'],
        num_epochs=data_params_trainable['num_epochs'],
        batch_size=data_params_trainable['batch_size'],
        seed=data_params_nontrainable['seed'])
    
    # NOTE: these are prepended with a row of zeros so that user_ids and movie_ids are direct indexes to the embeddings
    embeddings = read_embeddings(
        user_embeddings_uri=model_params_nontrainable['user_embeddings_uri'],
        movie_embeddings_uri=model_params_nontrainable['movie_embeddings_uri'],
        batch_size=1024)
    
    rngs = nnx.Rngs(data_params_nontrainable['seed'])
    
    model = GraphRanker(user_movie_embeds=embeddings,
        num_candidates=data_params_trainable['num_candidates'],
        hidden_features=model_params_trainable['hidden_dim'],
        num_layers=model_params_trainable['num_layers'],
        out_features=model_params_trainable['out_dim'],
        heads=model_params_trainable['num_heads'],
        dropout_rate=model_params_trainable['dropout_rate'], rngs=rngs)
    
    optimizer = nnx.Optimizer(model,
        optax.adamw(model_params_trainable['learning_rate'],
        weight_decay=model_params_trainable['weight_decay']), wrt=nnx.Param)
    
    print(f"expect the model training to start w/ loss = {-log(1. / data_params_trainable['num_candidates'])}")
    
    train_metrics = train_fn(model=model, train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer, top_k=model_params_trainable['top_k'],
        latest_checkpoint_dir=model_params_nontrainable['latest_checkpoint_dir'],
        rngs=rngs)
