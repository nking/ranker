#NOTE: in unit test choice of interprete,  interpreter with kubernetes installed in it. such as xmanager_py311
import os
import time
import subprocess
from json import dumps
from typing import Dict, List
import argparse
import yaml
import fsspec

from k8s_train_util import run_train_job_phase
from kind_util import setup_cluster, delete_cluster, find_executable_path

#pip install kfp==2.16.1
from kfp import dsl, compiler, client as kfp_client

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
    
    print(f"🔐 Setting up RBAC for KFP in namespace: {namespace}...")
    
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
        metadata=client.V1ObjectMeta(name="kfp-trainjob-binding",
            namespace=namespace),
        subjects=[
            client.V1Subject(kind="ServiceAccount", name="pipeline-runner",
                namespace=namespace),
            client.V1Subject(kind="ServiceAccount", name="default",
                namespace=namespace)
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
        print("✅ RBAC applied successfully.")
    except Exception as e:
        if "AlreadyExists" not in str(e):
            print(f"⚠️ RBAC creation warning: {e}", flush=True)

# ====================================================================
# THE COMPONENTS
# These functions define what runs INSIDE the Kubernetes pods.
# ====================================================================

@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["kubernetes==30.1.0", "pyyaml==6.0.3", "fsspec==2026.2.0", "gcsfs==2026.2.0"]
)
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
    :param phase: can be "tune" or "train_best" or "test_best".  other options not yet implemented
    :param trial_ids: the list of HPO trials to make if phase =="tune", else is None if phase is not "tune"
    :param output_log_dir_uri: uri where to write output logs with name job_name if not None
    """
    
    # Authenticate within the cluster
    from kubernetes import config
    try:
        config.load_incluster_config()
        print("Loaded in-cluster Kubernetes configuration.")
    except config.ConfigException:
        config.load_kube_config()
        print("Loaded local kube_config for development.")
    
    run_train_job_phase(train_job_yaml_content, namespace, phase, trial_ids, output_log_dir_uri)

# ====================================================================
#  THE PIPELINE
# ====================================================================

@dsl.pipeline(
    name="graphranker-sequential-hpo-train-test",
    description="Sequential HPO execution loop powered by embedded layout manifests.",
)
def hpo_train_test_pipeline(train_job_yaml_content:str, namespace:str = 'ranker-ns',
    num_trials:int=20):
    
    ## demonstrating sequential use of 2 list of trial_ids given to 2 nodes, each with 2 local devices (==2 hax processes).
    # the code uses SPMD to partition the data into 2 nodes * 2 local devices = 4
    
    # a dsl component returns a task
    
    hpo_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase='tune',
        trial_ids = dumps(list([i for i in range(num_trials)])),
    )
    
    # Single-node extractor using the same image
    extraction_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase="export_hpo_results")
    
    train_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase="train_best")
    
    test_task = run_train_job(
        train_job_yaml_content=train_job_yaml_content,
        namespace=namespace,
        phase="test_best")
    
    extraction_task.after(hpo_task)
    train_task.after(extraction_task)
    test_task.after(train_task)
    
    #TODO: make targets for "export_train_results" and "export_test_results"
    
    
@dsl.component(base_image="python:3.12-slim")
def cleanup_cluster_resources(kind_path: str):
    delete_cluster(kind_path)

def compile_pipeline_yaml(output_pipeline_yaml_uri: str = 'graphranker_pipeline.yaml',
    train_job_yaml_uri:str = "./train_job.yaml", num_trials:int=20) -> str:
    """
    compile the pipeline to yaml and return the namespace.  the local train_job.yaml is the internal input.
    :param output_pipeline_yaml_uri: name of the pipeline yaml file to write to
    :return: the namespace parsed from the local train_job.yaml
    """
    print( f"🛠️ Ingesting train job template and compiling pipeline to {output_pipeline_yaml_uri}...")
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
            'train_job_yaml_content': train_job_yaml_content,
            'num_trials' : num_trials
        }
    )
    return namespace
    
def run_pipeline(output_pipeline_yaml_uri: str = 'graphranker_pipeline.yaml',
    train_job_yaml_uri:str = "./train_job.yaml", num_trials:int=20):
    
    try:
        namespace = compile_pipeline_yaml(output_pipeline_yaml_uri, train_job_yaml_uri, num_trials)
        
        kind_path = find_executable_path("kind")
        kubectl_path = find_executable_path("kubectl")
        
        setup_cluster(kind_path=kind_path, kubectl_path=kubectl_path,
            PROJECT_ROOT=PROJECT_ROOT,
            KUBEFLOW_VERSION=KUBEFLOW_VERSION, NAMESPACE=NAMESPACE)
        
        setup_rbac(namespace)
        
        # Send the compiled asset to the Kind KFP engine
        print("📤 Submitting pipeline run to local backend...")
        try:
            client = kfp_client.Client(host="http://localhost:8080")
            experiment = client.create_experiment("GraphRanker-HPO")
            run = client.run_pipeline(
                experiment_id=experiment.id,
                job_name="hermetic-sequential-hpo-run",
                pipeline_package_path=output_pipeline_yaml_uri
            )
            print(f"🎉 Run initiated! Dashboard URL: {run.url}", flush=True)
        except Exception as e:
            print(f"⚠️ Automatic submission skipped/failed: {e}")
            print(
                "You can manually upload 'graphranker_pipeline.yaml' directly inside the KFP UI website.")
    finally:
        print(f'deleting cluster')
        cleanup_cluster_resources(kind_path)
        
# ====================================================================
# EXECUTION ENTRY POINT
# ====================================================================
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="GraphRanker Pipeline Runner")
    parser.add_argument(
        '--num_trials',
        help="number of HPO trials to run",
        type=int,
        default=20
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
    
    run_pipeline(output_pipeline_yaml_uri=args.output_pipeline_yaml_uri,
        train_job_yaml_uri=args.input_train_job_yaml_uri,
        num_trials=args.num_trials)
    