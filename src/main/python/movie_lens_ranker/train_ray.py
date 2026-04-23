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
    nontrainable_data_keys = {'movies_uri', 'recommendations_uri', 'recommendations_ts_uri',
        'ratings_train_uri', 'ratings_val_uri', 'train_negatives_uri', 'val_negatives_uri',
        'seed'
    }
    trainable_data_keys = {'max_history', 'num_candidates', 'num_epochs', 'batch_size'}
    nontrainable_model_keys = {'latest_checkpoint_dir', 'log_dir,'
        'movie_embeddings_uri', 'user_embeddings_uri', 'mlflow_config'}
    trainable_model_keys = {'top_k', 'learning_rate', 'weight_decay',
        'out_dim', 'hidden_dim', 'num_layers', 'num_heads', 'dropout_rate',
    }
    
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
    
    train_dataloader, val_dataloader = create_train_and_val_dataloaders(
        movies_uri=config['movies_uri'],
        recommendations_uri=config['recommendations_uri'],
        recommendations_ts_uri=config['recommendations_ts_uri'],
        train_ratings_uri=config['ratings_train_uri'],
        val_ratings_uri=config['ratings_val_uri'],
        train_negatives_uri=config['train_negatives_uri'],
        val_negatives_uri=config['val_negatives_uri'],
        max_history=config['max_history'], num_candidates=config['num_candidates'],
        num_epochs=config['num_epochs'], batch_size=config['batch_size'], seed=config['seed'])
    
    # NOTE: these are prepended with a row of zeros so that user_ids and movie_ids are direct indexes to the embeddings
    embeddings = read_embeddings(
        user_embeddings_uri=config['user_embeddings_uri'],
        movie_embeddings_uri=config['movie_embeddings_uri'],
        batch_size=1024)
    
    rngs = nnx.Rngs(config['seed'])
    
    model = GraphRanker(user_movie_embeds=embeddings,
        num_candidates=config['num_candidates'],
        hidden_features=config['hidden_dim'], num_layers=config['num_layers'],
        out_features=config['out_dim'], heads=config['num_heads'],
        dropout_rate=config['dropout_rate'], rngs=rngs)
    
    optimizer = nnx.Optimizer(model,
        optax.adamw(config['learning_rate'], weight_decay=config['weight_decay']), wrt=nnx.Param)
    
    print(f"expect the model training to start w/ loss = {-log(1. / config['num_candidates'])}")
    
    train_metrics = train_fn(model=model, train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer, top_k=config['top_k'],
        latest_checkpoint_dir=config['latest_checkpoint_dir'],
        rngs=rngs)
