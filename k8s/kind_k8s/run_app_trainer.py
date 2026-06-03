import os
import time
import subprocess
from json import dumps

import yaml
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException

from kind_util import setup_cluster, delete_cluster, find_executable_path

#USAGE:
#   python3 run_app_trainer.py

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

def run_training_loop():
    config.load_kube_config()
    crd_api = client.CustomObjectsApi()
    core_v1 = client.CoreV1Api()
    
    # Apply the custom TrainJob
    with open("train_job.yaml", "r") as f:
        manifest_str = f.read()
    manifest = yaml.safe_load(manifest_str)
    if "spec" not in manifest or "trainer" not in manifest["spec"] or "args" not in manifest["spec"][
        "trainer"]:
        raise ValueError("train_job.yaml is missing spec.trainer.args")
    modify_i = -1
    for i, arg in enumerate(manifest["spec"]["trainer"]["args"]):
        if arg.find("--trial_ids") == 0:
            modify_i = i
            break
    if modify_i == -1:
        modify_i = len(manifest["spec"]["trainer"]["args"])
        manifest["spec"]["trainer"]["args"].append("placeholder")
    
    for i in range(0, NUM_TRIALS, NUM_TRIALS_PER_WORKER):
        
        trial_ids = [val for val in range(i, i + NUM_TRIALS_PER_WORKER) if val < NUM_TRIALS]
        print(f"\n🚀 Launching JobGroup chunk with trial_ids={trial_ids}")
        
        manifest["spec"]["trainer"]["args"][modify_i] = f"--trial_ids={dumps(list(trial_ids))}"
        
        crd_api.create_namespaced_custom_object(
            group=TRAINJOB_GROUP,
            version=TRAINJOB_VERSION,
            namespace=NAMESPACE,
            plural=TRAINJOB_PLURAL,
            body=manifest
        )

        print("⏳ Waiting for TrainJob to complete...")
        job_name = manifest['metadata']['name']
        
        # Poll for Custom Resource Completion status
        completed = False
        while not completed:
            time.sleep(10)
            try:
                job_status = crd_api.get_namespaced_custom_object(
                    group=TRAINJOB_GROUP, version=TRAINJOB_VERSION, 
                    namespace=NAMESPACE, plural=TRAINJOB_PLURAL, name=job_name
                )
                conditions = job_status.get('status', {}).get('conditions', [])
                for condition in conditions:
                    if condition.get('type') == 'Complete' and condition.get('status') == 'True':
                        completed = True
                        break
                    if condition.get('type') == 'Failed' and condition.get('status') == 'True':
                        print("❌ TrainJob FAILED!")
                        completed = True 
                        break
            except ApiException as e:
                print(f"API Error fetching TrainJob: {e}")

        # Fetch Logs Programmatically
        print("📥 Fetching logs from pods...")
        pods = core_v1.list_namespaced_pod(NAMESPACE).items
        
        master_pod = next((p.metadata.name for p in pods if "node-0-0" in p.metadata.name), None)
        worker_pod = next((p.metadata.name for p in pods if "node-0-1" in p.metadata.name), None)

        if master_pod:
            master_logs = core_v1.read_namespaced_pod_log(name=master_pod, namespace=NAMESPACE)
            with open("chunk_trainer_master_logs.txt", "a") as f:
                f.write(master_logs)
        
        if worker_pod:
            worker_logs = core_v1.read_namespaced_pod_log(name=worker_pod, namespace=NAMESPACE)
            with open("chunk_trainer_worker-0_logs.txt", "a") as f:
                f.write(worker_logs)

        # Delete the TrainJob
        print("🧹 Cleaning up chunk...")
        crd_api.delete_namespaced_custom_object(
            group=TRAINJOB_GROUP, version=TRAINJOB_VERSION, 
            namespace=NAMESPACE, plural=TRAINJOB_PLURAL, name=job_name
        )
        time.sleep(5) # Give cluster time to terminate pods

def extract_and_shutdown(kind_path:str):
    print("\n🏁 Script finished. Running HPO results extraction...")
    result = subprocess.run(["python3", "extract_hpo_results.py"], capture_output=True, text=True)
    
    with open("hpo_results.txt", "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    if result.returncode == 0:
        print("✅ HPO results extracted successfully.")
    else:
        print("❌ ERROR: extract_hpo_results.py crashed! Check hpo_results.txt.")
        input("🛑 DEBUG PAUSE: Press [Enter] to tear down cluster...")

    delete_cluster(kind_path)
    
def assert_logs():
    for i, log_file in enumerate(["chunk_trainer_master_logs.txt", "chunk_trainer_worker-0_logs.txt"]):
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
        for log_file in ["chunk_trainer_master_logs.txt", "chunk_trainer_worker-0_logs.txt"]:
            if os.path.exists(log_file):
                os.remove(log_file)
        setup_cluster(kind_path=kind_path, kubectl_path=kubectl_path, PROJECT_ROOT=PROJECT_ROOT,
            KUBEFLOW_VERSION=KUBEFLOW_VERSION, NAMESPACE=NAMESPACE)
        
        run_training_loop()
        finished = True
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unhandled Exception: {e}")
    finally:
        extract_and_shutdown(kind_path)
        if finished:
            assert_logs()
