import os
import argparse
import socket
import subprocess
import time
from typing import Tuple

import yaml
import fsspec

from util_kfpv2 import setup_kfpv2_backend
from util_kind import setup_cluster, delete_cluster, find_executable_path, \
    prepare_container_image
#pip install kfp==2.16.1
from kfp import dsl, compiler, client as kfp_client
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_project_dir() -> str:
    cwd = os.getcwd()
    head = cwd
    proj_dir = ""
    while head and head != os.sep:
        head, tail = os.path.split(head)
        if tail:  # Add only if not an empty string (e.g., from root or multiple separators)
            if tail == "ranker":
                proj_dir = os.path.join(head, tail)
                break
    return proj_dir

KUBEFLOW_VERSION = "v2.2.0"
NAMESPACE = "ranker-ns"
PROJECT_ROOT = get_project_dir()

def setup_rbac(namespace: str = "ranker-ns"):
    """Grants KFP permissions to manage TrainJob custom resources."""
    from kubernetes import client, config
    config.load_kube_config()
    rbac_api = client.RbacAuthorizationV1Api()
    
    logging.info(f"🔐 Setting up RBAC for KFP in namespace: {namespace}...")
    
    role = client.V1Role(
        metadata=client.V1ObjectMeta(name="kfp-trainjob-manager",
            namespace=namespace),
        rules=[
            client.V1PolicyRule(
                api_groups=["trainer.kubeflow.org"],
                resources=["trainjobs"],
                verbs=["create", "get", "list", "watch", "delete", "patch",
                    "update"]
            )
        ]
    )
    
    binding = client.V1RoleBinding(
        metadata=client.V1ObjectMeta(name="kfp-trainjob-binding", namespace=namespace),
        subjects=[
            client.RbacV1Subject(kind="ServiceAccount", name="pipeline-runner", namespace=namespace),
            client.RbacV1Subject(kind="ServiceAccount", name="default", namespace=namespace)
        ],
        role_ref=client.V1RoleRef(
            api_group="rbac.authorization.k8s.io",
            kind="Role",
            name="kfp-trainjob-manager"
        )
    )
    
    try:
        rbac_api.create_namespaced_role(namespace=namespace, body=role)
        rbac_api.create_namespaced_role_binding(namespace=namespace,
            body=binding)
        logging.info("✅ RBAC applied successfully.")
    except Exception as e:
        if "AlreadyExists" not in str(e):
            logging.info(f"⚠️ RBAC creation warning: {e}")

def setup_rbac_yaml(rbac_yaml_uri: str = "./rbac.yaml", namespace: str = "ranker-ns"):
    """Grants KFP permissions by applying a YAML manifest."""
    from kubernetes import client, config, utils
    
    config.load_kube_config()
    
    # create_from_yaml requires an ApiClient object, not a specific API group
    k8s_client = client.ApiClient()
    
    logging.info(f"🔐 Setting up RBAC from {rbac_yaml_uri} in namespace: {namespace}...")
    
    try:
        # The 'namespace' arg here overrides the default and injects into the YAML
        utils.create_from_yaml(
            k8s_client,
            yaml_file=rbac_yaml_uri,
            namespace=namespace
        )
        logging.info("✅ RBAC applied successfully.")
    except utils.FailToCreateError as e:
        # create_from_yaml wraps API exceptions in FailToCreateError
        if "AlreadyExists" in str(e):
            logging.info("ℹ️ RBAC resources already exist. Skipping creation.")
        else:
            logging.error(f"⚠️ RBAC creation warning: {e}")
            raise e
        
# ====================================================================
# THE COMPONENTS
# These functions define what runs INSIDE the Kubernetes pods.
# ====================================================================
@dsl.component(base_image='python:3.12-slim')
def generate_trial_ids(trial_ids_str:str = None, num_trials: int = None) -> str:
    """
    if trial_ids_str is not None, it will check the format and return it, else if num_trials is given but trial_ids is None,
    the method will create a string  of list of trial ids 0 through num_trials - 1, else.
    :param trial_ids_str: a string of trial_ids to be used in HPO to identify trials.  e.g. '[0,1,2,3]'
    :param num_trials: if trial_ids_str is None, this is used to create a string list of trial_ids.
    e.g. num_trials=3 results in '[0,1,2]'
    :return: string of list of trial ids
    """
    from json import loads
    if trial_ids_str is not None:
        trial_ids = loads(trial_ids_str)
        if not isinstance(trial_ids, list):
            raise ValueError("trial_ids_str is expected to be string made from json.dumps(list(int_list))")
        if not all(isinstance(x, int) and x > -1 for x in trial_ids):
            raise ValueError("trial_ids must only contain non-negative integers")
        return trial_ids_str
    if num_trials is None:
        raise ValueError("trial_ids_str or num_trials must be given")
    from json import dumps
    return dumps(list(range(num_trials)))

@dsl.component(base_image="run_phase:local")
def run_train_job(
        train_job_yaml_content: str,
        namespace: str = "ranker-ns",
        phase: str = "tune",
        trial_ids:str=None,
        output_log_dir_uri: str = None,
):
    """
    run train_job.yaml for given phase.
    :param train_job_yaml_content:
    :param namespace:
    :param phase: can be "tune" or "train-best" or "test-best".  other options not yet implemented
    :param trial_ids: the list of HPO trials to make if phase =="tune", else is None if phase is not "tune"
    :param output_log_dir_uri: uri where to write output logs with name job_name if not None
    """
    
    # Authenticate within the cluster
    from kubernetes import config
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    import os
    import sys
    logging.info(f"DEBUG: Current working directory: {os.getcwd()}")
    logging.info(f"DEBUG: Python sys.path: {sys.path}")
    logging.info(f"DEBUG: Files in CWD: {os.listdir(os.getcwd())}")
    try:
        # Try to import it manually to catch the specific underlying error
        import importlib
        util = importlib.import_module("util_k8s_train")
        from util_k8s_train import run_train_job_phase
    except ImportError as e:
        # THIS will reveal the truth
        logging.error(f"❌ Actual ImportError: {e}")
        # Print the traceback so you can see exactly which line is failing
        import traceback
        traceback.print_exc()
        raise e
    
    try:
        config.load_incluster_config()
        logging.info("Loaded in-cluster Kubernetes configuration.")
    except config.ConfigException:
        config.load_kube_config()
        logging.info("Loaded local kube_config for development.")
    
    import base64
    from util_k8s_train import run_train_job_phase
    
    train_job_yaml_content = base64.b64decode(train_job_yaml_content).decode('utf-8')
    
    run_train_job_phase(train_job_yaml_content, namespace, phase, trial_ids, output_log_dir_uri)

# ====================================================================
#  THE PIPELINE
# ====================================================================

@dsl.pipeline(
    name="graphranker-sequential-hpo-train-test",
    description="Sequential HPO execution loop powered by embedded layout manifests.",
)
def hpo_train_test_pipeline(train_job_yaml_content:str="", namespace:str = 'ranker-ns',
    trial_ids_str:str=None, num_trials:int=20):
    """
    define the pipeline tasks by using the given train_job yaml content, the given namespace and for trials
    given as trial_ids_str or enumerated by num_trials.
    :param trial_ids_str: a string of trial_ids to be used in HPO to identify trials.  e.g. '[0,1,2,3]'
    :param num_trials: if trial_ids_str is None, this is used to create a string list of trial_ids.
    e.g. num_trials=3 results in '[0,1,2]'
    :param train_job_yaml_content: the string of the train_job.yaml file contents.
    :param namespace: the namespace for the k8s objects within the cluster
    """
    
    ## demonstrating sequential use of 2 list of trial_ids given to 2 nodes, each with 2 local devices (==2 hax processes).
    # the code uses SPMD to partition the data into 2 nodes * 2 local devices = 4
    
    # a dsl component returns a task
    
    prepare_ids_task = generate_trial_ids(trial_ids_str=trial_ids_str, num_trials=num_trials)
    
    hpo_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase='tune',
        trial_ids = prepare_ids_task.output,
    )
    
    # Single-node extractor using the same image
    tune_extraction_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase="export-hpo-results")
    
    train_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase="train-best")
    
    train_extraction_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase="export-train-results")
    
    test_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase="test-best")
    
    test_extraction_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase="export-test-results")
    
    hpo_task.after(prepare_ids_task)
    tune_extraction_task.after(hpo_task)
    train_task.after(tune_extraction_task)
    train_extraction_task.after(train_task)
    test_task.after(train_extraction_task)
    test_extraction_task.after(test_task)
    
@dsl.component(base_image="python:3.12-slim")
def cleanup_cluster_resources(kind_path: str):
    """
    delete the cluster
    :param kind_path: path to kind executable on local machine
    """
    delete_cluster(kind_path)

def compile_pipeline_yaml(output_pipeline_yaml_uri: str = 'graphranker_pipeline.yaml',
    train_job_yaml_uri:str = "./train_job.yaml", trial_ids_str:str=None, num_trials:int=20) -> Tuple[str, str]:
    """
    compile the pipeline to yaml and return the namespace.  the train_job.yaml contains the namespace to use for the Trainer v2 job.
    the HPO trials are identified using trial_ids_str or enumerated by num_trials.
    :param trial_ids_str: a string of trial_ids to be used in HPO to identify trials.  e.g. '[0,1,2,3]'
    :param num_trials: if trial_ids_str is None, this is used to create a string list of trial_ids.
    e.g. num_trials=3 results in '[0,1,2]'
    :param output_pipeline_yaml_uri: name of the pipeline yaml file to write to
    :return: the namespace parsed from the train_job.yaml file train_job_yaml_uri and the content of the train job yaml file as a string
    """
    logging.info( f"🛠️ Ingesting train job template and compiling pipeline to {output_pipeline_yaml_uri}...")
    TRAIN_JOB_YAML_PATH = train_job_yaml_uri
    
    with fsspec.open(TRAIN_JOB_YAML_PATH, "r") as f:
        train_job_yaml_content = f.read()
    
    manifest = yaml.safe_load(train_job_yaml_content)
    namespace = manifest['metadata']['namespace']
    
    compiler.Compiler().compile(
        pipeline_func=hpo_train_test_pipeline,
        package_path=output_pipeline_yaml_uri,
        pipeline_parameters={
            'namespace': namespace,
            #'train_job_yaml_content': train_job_yaml_content,#<-- pass in at runtime instead
            'trial_ids_str' : trial_ids_str,
            'num_trials' : num_trials
        }
    )
    logging.info(f'wrote pipeline to uri={output_pipeline_yaml_uri}')
    return namespace, train_job_yaml_content
    
def run_pipeline_local(train_job_yaml_uri:str = "./train_job.yaml", trial_ids_str = None, num_trials:int=20):
    import kfp.local
    
    # You can use DockerRunner() to run components in their containers,
    # or SubprocessRunner() to run them directly in your local python env.
    kfp.local.init(runner=kfp.local.SubprocessRunner())
    
    with fsspec.open(train_job_yaml_uri, "r") as f:
        train_job_yaml_content = f.read()
    
    manifest = yaml.safe_load(train_job_yaml_content)
    namespace = manifest['metadata']['namespace']
    
    #this is immediately started in kfp
    gr_pipeline = hpo_train_test_pipeline(
        namespace=namespace,
        train_job_yaml_content = train_job_yaml_content,
        trial_ids_str = trial_ids_str,
        num_trials = num_trials
    )
    print("Pipeline execution complete!")
    print(f"Final status: {gr_pipeline.state}")

def wait_for_port(port:int, host:str='127.0.0.1', timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0: return True
        time.sleep(0.5)
    return False

def run_pipeline_on_kfpv2(output_pipeline_yaml_uri: str = 'graphranker_pipeline.yaml',
    train_job_yaml_uri:str = "./train_job.yaml", trial_ids_str = None, num_trials:int=20):
    """
    run the pipeline defined by tasks that are using the train_job yaml.
    the HPO trials are identified using trial_ids_str or enumerated by num_trials.
    :param trial_ids_str: a string of trial_ids to be used in HPO to identify trials.  e.g. '[0,1,2,3]'
    :param num_trials: if trial_ids_str is None, this is used to create a string list of trial_ids.
    e.g. num_trials=3 results in '[0,1,2]'
    :param output_pipeline_yaml_uri: where to write the compiled pipeline yaml to
    :param train_job_yaml_uri: uri for input train_job.yaml file
    """
    #TODO: revisit namespace use and make sure its consistent
    tunnel = None
    try:
        namespace, train_job_yaml_content = compile_pipeline_yaml(
            output_pipeline_yaml_uri,
            train_job_yaml_uri, trial_ids_str=trial_ids_str,
            num_trials=num_trials)
        
        kind_path = find_executable_path("kind")
        kubectl_path = find_executable_path("kubectl")
        docker_path = find_executable_path("docker")
        
        # DEBUG: uncomment when done debugging
        '''
        setup_cluster(kind_path=kind_path, kubectl_path=kubectl_path,
            PROJECT_ROOT=PROJECT_ROOT,
            KUBEFLOW_VERSION=KUBEFLOW_VERSION, NAMESPACE=NAMESPACE)
        '''
        
        prepare_container_image(kind_path=kind_path, docker_path=docker_path,
            docker_file_path="./Dockerfile_kfp", image_name= "run_phase:local",
            cluster_name = "graphranker-tune-train-test-cluster")
        
        setup_rbac_yaml(rbac_yaml_uri="./rbac.yaml", namespace=namespace)
        
        # DEBUG: uncomment when done debugging
        '''
        setup_kfpv2_backend(kubectl_path)
        '''
        
        logging.info("create port-forward tunnel")
        tunnel = subprocess.Popen(
            [kubectl_path, "port-forward", "svc/ml-pipeline-ui", "-n",
                "kubeflow", "8080:80"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        if wait_for_port(8080):
            logging.info("Tunnel is active!")
        
        # Send the compiled asset to the Kind KFP engine
        logging.info("📤 Submitting pipeline run to local backend...")
        
        import base64
        train_job_yaml_content = base64.b64encode(train_job_yaml_content.encode('utf-8')).decode('utf-8')
        
        try:
            client = kfp_client.Client(host="http://localhost:8080")
            
            kfp_experiment = client.create_experiment("GraphRanker-HPO")
            
            #asynchronous:
            run = client.run_pipeline(
                experiment_id=kfp_experiment.experiment_id,
                job_name="hermetic-sequential-hpo-run",
                pipeline_package_path=output_pipeline_yaml_uri,
                params={
                    # Inject the massive string here at runtime
                    "train_job_yaml_content": train_job_yaml_content
                }
            )

            dashboard_url = f"http://localhost:8080/#/runs/details/{run.run_id}"
            logging.info(f"🎉 Run initiated! Dashboard URL: {dashboard_url}")
            
            logging.info(f"⏳ Waiting for run {run.run_id} to complete...")
            client.wait_for_run_completion(run_id=run.run_id, timeout=3600)  # 1 hour timeout
            
            logging.info("✅ Pipeline finished!")
            
        except Exception as e:
            logging.exception((f"⚠️ Automatic submission skipped/failed: {e}"
                "You can manually upload 'graphranker_pipeline.yaml' directly inside the KFP UI website."))
        
    finally:
        if tunnel:
            tunnel.terminate()
            
        logging.info(f'deleting cluster')
        #DEBUG: uncomment when done debugging
        #cleanup_cluster_resources(kind_path)
        
# ====================================================================
# EXECUTION ENTRY POINT
# ====================================================================
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="GraphRanker Pipeline Runner")
    parser.add_argument(
        '--num_trials',
        help="number of HPO trials to run",
        type=int,
        default=4
    )
    parser.add_argument(
        '--trial_ids_str',
        help="string list of integer of HPO trial ids to run.  e.g. '[0,1,2,3]'.  can be safely"
             "constructed with json.dumps(list(int_list))",
        type=str,
        default=None
    )
    parser.add_argument(
        '--output_pipeline_yaml_uri',
        help="output path to write the compile yaml to",
        type=str,
        default="./output_compiled_pipeline.yaml"
    )
    parser.add_argument(
        '--input_train_job_yaml_uri',
        help="uri for train_job.yaml",
        type=str,
        default="./train_job.yaml"
    )
    args = parser.parse_known_args()
    args_dict = vars(args[0])
    
    run_pipeline_on_kfpv2(output_pipeline_yaml_uri=args_dict['output_pipeline_yaml_uri'],
        train_job_yaml_uri=args_dict['input_train_job_yaml_uri'],
        trial_ids_str=args_dict['trial_ids_str'],
        num_trials=args_dict['num_trials'])
    
