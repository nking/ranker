#install kubernetes and PyYAML
import os
import shutil
import sys
import time
import subprocess
import yaml
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import subprocess
import logging

def image_exists(docker_path:str, image_name: str) -> bool:
    """Checks if a docker image exists locally without parsing output.
    :param docker_path: path to docker binary
    :param image_name: name of image to build.  e.g. run_phase:local
    """
    cmd = [docker_path, "image", "inspect", image_name]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0

def prepare_container_image(kind_path : str, docker_path : str, docker_file_path : str,
        image_name : str, cluster_name : str = "graphranker-tune-train-test-cluster"):
    """
    Prepares a container image if doesn't exist and loads it into kind
    :param kind_path: path for kind binary
    :param docker_path: path to docker binary
    :param image_name: name of image to build.  e.g. run_phase:local
    :param cluster_name: name of k8s cluster
    """
    build_image = not image_exists(docker_path, image_name)
    
    if build_image:
        logging.info(f"🏗️ Image '{image_name}' not found. Building...")
        try:
            subprocess.run([docker_path, "build", "-f", docker_file_path, "-t", image_name, "."], check=True)
            logging.info(f"✅ Build successful.")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Docker build failed: {e}")
            raise
    
    if is_image_loaded_in_kind(kind_path, docker_path, image_name, cluster_name):
        logging.info("🚀 Image is already in Kind! Skipping load.")
    else:
        logging.info(f"🚚 Loading image into cluster '{cluster_name}'...")
        try:
            run_cmd(cmd=[kind_path, "load", "docker-image", image_name, "--name",
                cluster_name], check=True)
            logging.info(f"✅ Image loaded successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Kind load failed: {e}")
            raise
  
def is_image_loaded_in_kind(kind_path : str, docker_path : str, image_name: str, cluster_name: str) -> bool:
    """
    Checks if a specific image exists inside the Kind cluster nodes.
    """
    try:
        # Get the list of nodes in the Kind cluster
        nodes_output = subprocess.check_output(
            [kind_path, "get", "nodes", "--name", cluster_name],
            text=True
        )
        nodes = nodes_output.strip().splitlines()
        if not nodes:
            return False
        
        #['graphranker-tune-train-test-cluster-control-plane', 'graphranker-tune-train-test-cluster-worker2', 'graphranker-tune-train-test-cluster-worker']
        
        # Check the first node (Kind clusters typically pull/load to all nodes)
        target_node = nodes[0]
        
        # Use docker exec to check the node's internal image store
        # We use 'docker image inspect' inside the node's environment
        check_cmd = [docker_path, "exec", target_node, "docker", "image", "inspect", image_name]
        
        result = subprocess.run(
            check_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        return result.returncode == 0
        
    except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
        # Either Kind isn't installed, the cluster isn't running, or command failed
        return False
    
def find_executable_path(binary_name:str):
    """Run a shell command and print output.
    :param binary_name: name of binary path to resolve
    """
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
            logging.info(f"⚠️ Found 'kind' via fallback path: {path}")
            return path
    
    # If we exhaust all options, raise a clear error
    raise FileNotFoundError(
        "Could not find the 'kind' executable in PATH or fallback directories.")

def run_cmd(cmd, check=True, timeout: float = None, max_retries: int = 0,
        base_delay: float = 3.0, backoff_factor: float = 2.0):
    """
    Run a shell command with exponential backoff for retries.

    :param max_retries: Number of times to retry on failure (0 means no retries).
    :param base_delay: Initial wait time in seconds before the first retry.
    :param backoff_factor: Multiplier for the delay after each subsequent failure.
    """
    attempt = 0
    current_delay = base_delay
    
    while True:
        logging.info(f"Executing: {' '.join(cmd)}")
        
        # We capture stderr so we can log the exact failure reason if it crashes
        result = subprocess.run(cmd, text=True, timeout=timeout, stderr=subprocess.PIPE)
        
        # Success case
        if result.returncode == 0:
            if attempt > 0:
                logging.info(f"✅ Command succeeded on attempt {attempt + 1}")
            return result
        
        # Failure case - Check if we have retries left
        if attempt < max_retries:
            attempt += 1
            logging.warning(
                f"⚠️ Command failed (Attempt {attempt}/{max_retries}). "
                f"Retrying in {current_delay}s... Reason: {result.stderr.strip()}")
            time.sleep(current_delay)
            current_delay *= backoff_factor  # Exponential backoff (3s -> 6s -> 12s)
        else:
            # Out of retries
            break
    
    # If we reach here, all retries failed (or max_retries was 0)
    if check:
        logging.error(
            f"❌ Command failed permanently after {max_retries + 1} attempts: {' '.join(cmd)}")
        logging.error(f"Standard Error output:\n{result.stderr}")
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

def wait_for_deployment(apps_v1, name, namespace, timeout=240):
    """Poll a deployment until readyReplicas matches availableReplicas."""
    logging.info(f"⏳ Waiting for deployment {name} in {namespace} to roll out...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            dep = apps_v1.read_namespaced_deployment(name, namespace)
            if dep.status.ready_replicas == dep.status.replicas and dep.status.replicas is not None:
                logging.info(f"✅ Deployment {name} is ready!")
                return
        except ApiException as e:
            if e.status != 404:
                logging.info(f"API Error checking {name}: {e}")
        time.sleep(5)
    
    logging.info(f"❌ Timeout waiting for {name}. Please debug cluster.")
    input("🛑 SETUP DEBUG PAUSE: Press [Enter] to exit and teardown...")
    sys.exit(1)

def setup_cluster(kind_path:str, kubectl_path:str, PROJECT_ROOT:str, KUBEFLOW_VERSION:str, NAMESPACE:str):
    logging.info("🌐 Checking internet connection...")
    #time.google.com is 216.239.35.0
    run_cmd(["ping", "-c", "1", "-W", "3", "216.239.35.0"])

    try:
        logging.info("🚀 Creating Kind cluster...")
        # Using subprocess to pipe envsubst into kind
        cmd = f"envsubst '$PROJECT_ROOT' < kind-cluster.yaml | {kind_path} create cluster --config -"
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True,
            text=True, env={**os.environ, "PROJECT_ROOT": PROJECT_ROOT})
        if result.returncode != 0:
            logging.exception(f"❌ Command failed: {' '.join(cmd)}.  result: {result.stderr}")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ KIND COMMAND FAILED!")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}\n")
        raise e
    
    logging.info("⏳ Waiting for nodes...")
    run_cmd([kubectl_path, "wait", "--for=condition=Ready", "nodes", "--all", "--timeout=120s"])
    
    # Install Kubeflow via Kustomize (subprocess is best here)
    logging.info("📦 Installing Kubeflow Trainer Controller...")
    run_cmd([kubectl_path, "apply", "--server-side", "-k", f"https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref={KUBEFLOW_VERSION}"],
        max_retries=3)
    time.sleep(3)

    # Initialize Kubernetes Python Client
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()

    logging.info("🩹 Patching JobSet Image...")
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

    logging.info("📦 Installing Kubeflow Training Runtimes...")
    run_cmd([kubectl_path, "apply", "--server-side", "-k", f"https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref={KUBEFLOW_VERSION}"],
        max_retries=3)

    logging.info("🐳 Sideloading Docker Images...") #the tags, when not :latest, result in looking for image locally first
    cluster_name = "graphranker-tune-train-test-cluster"
    run_cmd([kind_path, "load", "docker-image", "ranker-app:local", "--name", cluster_name])
    run_cmd([kind_path, "load", "docker-image", "vizier-server:local", "--name", cluster_name])

    logging.info("🗄️ Deploying Databases...")
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
    logging.info("🗑️ Deleting Kind cluster...")
    subprocess.run([kind_path, "delete", "cluster", "--name", "graphranker-tune-train-test-cluster"])
