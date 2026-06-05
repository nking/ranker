import os
from json import dumps
import yaml
from k8s_train_util import run_train_job_phase
from kind_util import setup_cluster, delete_cluster, find_executable_path
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#USAGE:
#   python3 run_app_trainer.py

#NOTE: script assumes output_hyperparams_uri and output_metrics_uri use uri pattern
#       gs://hpo-results-bucket/<project_id>/<study_name>/<tune|train|test>/hpo_<hparams|metrics>.json
#       in order to make dynamic changes to train_job.yaml

# ====================================================================
# CONFIGURATION
# ====================================================================
NUM_TRIALS = 4
NUM_TRIALS_PER_WORKER = 2
KUBEFLOW_VERSION = "v2.2.0"
TRAINJOB_GROUP = "trainer.kubeflow.org" # Verify this matches your CRD
TRAINJOB_VERSION = "v1alpha1"           # Verify this matches your CRD
TRAINJOB_PLURAL = "trainjobs"
NAMESPACE = "ranker-ns"
PROJECT_ROOT = os.path.abspath("../../")

# ====================================================================
# MAIN LIFECYCLE
# ====================================================================

def run_training_loop(manifest_str:str):
   
    manifest = yaml.safe_load(manifest_str)
    if "spec" not in manifest or "trainer" not in manifest["spec"] or "args" not in manifest["spec"][
        "trainer"]:
        raise ValueError(f"{infile} is missing spec.trainer.args")
    
    namespace = manifest['metadata']['namespace']
    
    for i in range(0, NUM_TRIALS, NUM_TRIALS_PER_WORKER):
        
        trial_ids = dumps(list([val for val in range(i, i + NUM_TRIALS_PER_WORKER) if val < NUM_TRIALS]))
        
        run_train_job_phase(train_job_yaml_content=manifest_str,
            namespace=namespace,
            phase='tune',
            trial_ids=trial_ids, output_log_dir_uri="./")

def assert_logs():
    for i, log_file in enumerate(["tune-master-logs.txt", "tune-worker-1-logs.txt"]):
        with open(log_file, "r") as f:
            file_str = f.read()
            assert(file_str.find("'trial_ids': '[0, 1]'") > -1)
            assert (file_str.find("'trial_ids': '[2, 3]'") > -1)
            assert (file_str.count("Epoch 2:") == 4)
            assert(file_str.count('finally clause in train_fn') == 4)
            if i == 0:
                assert (file_str.find("worker_0") > -1)
                assert (file_str.count("mlflow start run: trial_") == 4)
                assert (file_str.count("New best val NDCG") == 4)
            else:
                assert (file_str.find("worker_1") > -1)
    
# ====================================================================
# EXECUTION ENTRY POINT
# ====================================================================
if __name__ == "__main__":
    kind_path = find_executable_path("kind")
    kubectl_path = find_executable_path("kubectl")
    finished = False
    try:
        # Clear old logs
        for log_file in ["tune-master-logs.txt", "tune-worker-1-logs.txt"]:
            if os.path.exists(log_file):
                os.remove(log_file)
        
        setup_cluster(kind_path=kind_path, kubectl_path=kubectl_path, PROJECT_ROOT=PROJECT_ROOT,
            KUBEFLOW_VERSION=KUBEFLOW_VERSION, NAMESPACE=NAMESPACE)
        
        infile = f"{PROJECT_ROOT}/k8s/kind_k8s/train_job.yaml"
        with open(infile, "r") as f:
            manifest_str = f.read()
        manifest = yaml.safe_load(manifest_str)
        namespace = manifest['metadata']['namespace']
        
        #config.load_kube_config() is in setup_cluster
        run_training_loop(manifest_str)
       
        logging.info("\nExtract HPO results:")
        run_train_job_phase(
            train_job_yaml_content=manifest_str,
            namespace=namespace,
            phase='export-hpo-results')
        
        logging.info("\nTrain model using best HPO results:")
        run_train_job_phase(
            train_job_yaml_content=manifest_str,
            namespace=namespace,
            phase='train-best')
        '''
        
        logging.info("\nExtract train results:")
        run_train_job_phase(
            train_job_yaml_content=manifest_str,
            namespace=namespace,
            phase='export-train-results')
        
        logging.info("\nTest best trained model:")
        run_train_job_phase(
            train_job_yaml_content=manifest_str,
            namespace=namespace,
            phase='test-best')
        
        logging.info("\nExtract test results:")
        run_train_job_phase(
            train_job_yaml_content=manifest_str,
            namespace=namespace,
            phase='export-test-results')
        
        finished = True
        
    except KeyboardInterrupt:
        logging.info("\n⚠️ Interrupted by user.")
    except Exception as e:
        logging.exception(f"\n❌ Unhandled Exception: {e}")
    finally:
        delete_cluster(kind_path)
        if finished:
            assert_logs()
