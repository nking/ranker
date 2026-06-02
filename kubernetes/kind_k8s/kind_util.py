#install kubernetes and PyYAML
import os
import shutil
import sys
import time
import subprocess

import yaml
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException

def find_executable_path(binary_name:str):
    """Run a shell command and print output."""
    path = shutil.which(binary_name)
    if path:
        return path
    
    # If not found, explicitly check common installation locations
    home = os.path.expanduser("~")
    fallback_locations = [
        os.path.join(home, "go", "bin", "kind"),  # Default Go binary path
        f"/snap/bin/{binary_name}",
        f"/usr/local/bin/{binary_name}",  # Standard Linux path
        f"/usr/bin/{binary_name}",  # Alternate Linux path
        f"/opt/homebrew/bin/{binary_name}",  # macOS Apple Silicon Homebrew
        f"/usr/local/Homebrew/bin/{binary_name}",  # macOS Intel Homebrew
        os.path.join(home, ".local", "bin", binary_name)  # Local user bin
    ]
    
    for path in fallback_locations:
        # os.path.exists checks if it's there, kindos.access checks if it is executable
        if os.path.exists(path) and os.access(path, os.X_OK):
            print(f"⚠️ Found 'kind' via fallback path: {path}")
            return path
    
    # If we exhaust all options, raise a clear error
    raise FileNotFoundError(
        "Could not find the 'kind' executable in PATH or fallback directories.")

def run_cmd(cmd, check=True, timeout:float=None):
    """Run a shell command and print output."""
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, timeout=timeout)
    if check and result.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}.  result: {result.stderr}", flush=True)
        sys.exit(1)
    return result

def apply_templated_yaml(api_client, filepath, replacements):
    """Read a YAML, replace string variables, and apply it."""
    with open(filepath, 'r') as f:
        manifest_str = f.read()
    
    for key, val in replacements.items():
        manifest_str = manifest_str.replace(key, str(val))
    
    # Parse and apply all documents in the YAML file
    docs = yaml.safe_load_all(manifest_str)
    for doc in docs:
        if doc:
            try:
                utils.create_from_dict(api_client, doc)
            except utils.FailToCreateError as e:
                # Ignore "AlreadyExists" errors
                if "AlreadyExists" not in str(e):
                    raise e

def wait_for_deployment(apps_v1, name, namespace, timeout=180):
    """Poll a deployment until readyReplicas matches availableReplicas."""
    print(f"⏳ Waiting for deployment {name} in {namespace} to roll out...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            dep = apps_v1.read_namespaced_deployment(name, namespace)
            if dep.status.ready_replicas == dep.status.replicas and dep.status.replicas is not None:
                print(f"✅ Deployment {name} is ready!")
                return
        except ApiException as e:
            if e.status != 404:
                print(f"API Error checking {name}: {e}")
        time.sleep(5)
    
    print(f"❌ Timeout waiting for {name}. Please debug cluster.")
    input("🛑 SETUP DEBUG PAUSE: Press [Enter] to exit and teardown...")
    sys.exit(1)

def setup_cluster(kind_path:str, kubectl_path:str, PROJECT_ROOT:str, KUBEFLOW_VERSION:str, NAMESPACE:str):
    print("🌐 Checking internet connection...")
    run_cmd(["ping", "-c", "1", "-W", "3", "google.com"])

    print("🚀 Creating Kind cluster...")
    # Using subprocess to pipe envsubst into kind
    cmd = f"envsubst '$PROJECT_ROOT' < kind-cluster.yaml | {kind_path} create cluster --config -"
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, env={**os.environ, "PROJECT_ROOT": PROJECT_ROOT})
    if result.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}.  result: {result.stderr}", flush=True)
        sys.exit(1)
        
    print("⏳ Waiting for nodes...")
    run_cmd([kubectl_path, "wait", "--for=condition=Ready", "nodes", "--all", "--timeout=120s"])

    # Install Kubeflow via Kustomize (subprocess is best here)
    print("📦 Installing Kubeflow Trainer Controller...")
    run_cmd([kubectl_path, "apply", "--server-side", "-k", f"https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref={KUBEFLOW_VERSION}"])
    time.sleep(3)

    # Initialize Kubernetes Python Client
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()

    print("🩹 Patching JobSet Image...")
    patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [{"name": "manager", "image": "registry.k8s.io/jobset/jobset:v0.12.0"}]
                }
            }
        }
    }
    apps_v1.patch_namespaced_deployment("jobset-controller-manager", "kubeflow-system", patch)

    wait_for_deployment(apps_v1, "kubeflow-trainer-controller-manager", "kubeflow-system")
    wait_for_deployment(apps_v1, "jobset-controller-manager", "kubeflow-system")

    print("📦 Installing Kubeflow Training Runtimes...")
    run_cmd([kubectl_path, "apply", "--server-side", "-k", f"https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref={KUBEFLOW_VERSION}"])

    print("🐳 Sideloading Docker Images...") #the tags, when not :latest, result in looking for image locally first
    cluster_name = "graphranker-tune-train-test-cluster"
    run_cmd([kind_path, "load", "docker-image", "ranker-app:local", "--name", cluster_name])
    run_cmd([kind_path, "load", "docker-image", "vizier-server:local", "--name", cluster_name])

    print("🗄️ Deploying Databases...")
    # Create namespace if not exists
    try:
        core_v1.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=NAMESPACE)))
    except ApiException:
        pass # Already exists
        
    api_client = client.ApiClient()
    apply_templated_yaml(api_client, "secrets.yaml", {})
    apply_templated_yaml(api_client, "dbs.yaml", {"${PROJECT_ROOT}": PROJECT_ROOT})

    wait_for_deployment(apps_v1, "local-db-store", NAMESPACE)
    wait_for_deployment(apps_v1, "gcs-emulator", NAMESPACE)
    wait_for_deployment(apps_v1, "vizier-server", NAMESPACE)

def delete_cluster(kind_path:str):
    print("🗑️ Deleting Kind cluster...")
    subprocess.run([kind_path, "delete", "cluster", "--name", "graphranker-tune-train-test-cluster"])
