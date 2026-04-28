import threading

import mlflow
from absl import flags
from movie_lens_ranker.train import train_fn, test_fn, get_optuna_suggestions

FLAGS = flags.FLAGS

import optuna
from absl import app
from optuna import Trial

def get_or_create_mlflow_experiment(experiment_name:str):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
def main(_):
    """
    set-up optuna trial
    :param _:
    :return:
    """
    #contains same keys as get_nontrainable_train_config() and
    config = FLAGS.flag_values_dict()
    
    if FLAGS.phase == 'test':
        return test_fn(config)
    
    # Connect to the study created by the launcher
    study = optuna.load_study(
        study_name=config['study_name'],
        storage=config['optuna_storage_uri'],
    )
    
    # Optuna's DB locking ensures each container gets unique params
    trial: Trial = study.ask()
    config.update(trial.params)
    
    config['mlflow_experiment_name'] = config['study_name']
    config['mlflow_experiment_id'] = get_or_create_mlflow_experiment(config['mlflow_experiment_name'])
    
    #NOTE: this is specific to disks and assumes have permission to mkdir...wold be different for cloud storage
    #append trial id to uris:
    config['best_checkpoint_dir'] = f"{config['best_checkpoint_dir']}/{config['study_name']}/trial_{config['trial_id']}"
    config['latest_checkpoint_dir'] = f"{config['latest_checkpoint_dir']}/{config['study_name']}/trial_{config['trial_id']}"
    
    # get trial suggestions
    optuna_params = get_optuna_suggestions(trial)
    config.update(optuna_params)
    
    best_val_ndcg_k, STATE = train_fn(config, trial)
    
    study.tell(trial, values=float(best_val_ndcg_k), state=STATE)

if __name__ == '__main__':
    app.run(main)