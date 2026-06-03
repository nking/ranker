#NOTE: in unit test choice of interprete,  interpreter with kubernetes installed in it. such as xmanager_py311
import os
import time
import subprocess
from json import dumps
from typing import Dict
import argparse
import yaml
from kind_util import setup_cluster, delete_cluster, find_executable_path

#pip install kfp==2.16.1
from kfp import dsl, compiler, client as kfp_client

KUBEFLOW_VERSION = "v2.2.0"
NAMESPACE = "ranker-ns"
PROJECT_ROOT = os.path.abspath("../../")

HPO_RESULTS_EXTRACTOR_IMAGE = "ranker-app:local"

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

#TODO: these need versions
@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["kubernetes==30.1.0", "pyyaml==6.0.3"]
)
def run_trainjob_chunk(
        trial_ids: list,
        train_job_yaml_content: str,
        namespace: str = "ranker-ns"
):
    import time
    import yaml
    from kubernetes import client, config
    
    # Authenticate within the cluster
    config.load_incluster_config()
    crd_api = client.CustomObjectsApi()
    
    # Parse the embedded template string into a Python dictionary
    manifest = yaml.safe_load(train_job_yaml_content)
    
    if "spec" not in manifest or "trainer" not in manifest["spec"] or "args" not in manifest["spec"][
        "trainer"]:
        raise ValueError("train_job.yaml is missing spec.trainer.args")
    trial_ids_str = f"--trial_ids={dumps(list(trial_ids))}"
    for i, arg in enumerate(manifest["spec"]["trainer"]["args"]):
        if arg.find("--trial_ids") == 0:
            manifest["spec"]["trainer"]["args"][i] = trial_ids_str
            trial_ids_str = None
            break
    if trial_ids_str:
        manifest["spec"]["trainer"]["args"].append(trial_ids_str)
    
    job_name = f"jax-hpo-chunk-{trial_ids[0]}"
    manifest['metadata']['name'] = job_name
    manifest['metadata']['namespace'] = namespace
    
    # Deploy to Kubernetes
    print(f"🚀 Launching TrainJob: {job_name}")
    crd_api.create_namespaced_custom_object(
        group="trainer.kubeflow.org", version="v1alpha1",
        namespace=namespace, plural="trainjobs", body=manifest
    )
    
    #  Monitor execution
    completed = False
    while not completed:
        time.sleep(10)
        status = crd_api.get_namespaced_custom_object(
            group="trainer.kubeflow.org", version="v1alpha1",
            namespace=namespace, plural="trainjobs", name=job_name
        )
        conditions = status.get('status', {}).get('conditions', [])
        for condition in conditions:
            if condition.get('type') == 'Complete' and condition.get(
                    'status') == 'True':
                print(f"✅ TrainJob {job_name} completed successfully!")
                completed = True
                break
            elif condition.get('type') == 'Failed' and condition.get(
                    'status') == 'True':
                raise RuntimeError(f"❌ TrainJob {job_name} failed!")
    
    # Lifecycle cleanup
    print(f"🧹 Tearing down TrainJob custom resource: {job_name}")
    crd_api.delete_namespaced_custom_object(
        group="trainer.kubeflow.org", version="v1alpha1",
        namespace=namespace, plural="trainjobs", name=job_name
    )

@dsl.container_component
def extract_hpo_results(namespace: str = "ranker-ns"):
    """Uses the same unified image variable to invoke extraction logic."""
    ## inputs.parameters['namespace']
    return dsl.ContainerSpec(
        image=HPO_RESULTS_EXTRACTOR_IMAGE, # <-- KFPv2 undocumented requirement for a static string for docker image
        args=[
            "--phase", "extract_hpo_results",
            "--namespace", namespace
        ]
    )

# ====================================================================
#  THE PIPELINE
# ====================================================================

@dsl.pipeline(
    name="graphranker-sequential-hpo",
    description="Sequential HPO execution loop powered by embedded layout manifests.",
)
def hpo_pipeline(train_job_yaml_content:str, namespace:str = 'ranker-ns'):
    
    ## demonstrating sequential use of 2 list of trial_ids given to 2 nodes, each with 2 local devices (==2 hax processes).
    # the code uses SPMD to partition the data into 2 nodes * 2 local devices = 4
    
    chunks = [[0, 1], [2, 3]]
    
    previous_task = None
    
    for chunk in chunks:
        #dsl compoment returns task
        train_task = run_trainjob_chunk(
            trial_ids=chunk,
            train_job_yaml_content=train_job_yaml_content,
            namespace=namespace
        )
        
        if previous_task is not None:
            train_task.after(previous_task)
        
        previous_task = train_task
    
    # Single-node extractor using the same image
    extraction_task = extract_hpo_results(namespace=namespace)
    
    if previous_task is not None:
        extraction_task.after(previous_task)
    
@dsl.component(base_image="python:3.12-slim")
def cleanup_cluster_resources(kind_path: str):
    delete_cluster(kind_path)

def compile_pipeline_yaml(pipeline_filename:str, train_job_yaml_path:str) -> str:
    """
    compile the pipeline to yaml and return the namespace.  the local train_job.yaml is the internal input.
    :param pipeline_filename: name of the pipeline yaml file to write to
    :return: the namespace parsed from the local train_job.yaml
    """
    print( f"🛠️ Ingesting template and compiling pipeline to {pipeline_filename}...")
    TRAIN_JOB_YAML_PATH = train_job_yaml_path
    
    if not os.path.exists(TRAIN_JOB_YAML_PATH):
        raise FileNotFoundError(
            f"Missing train job yaml file: {TRAIN_JOB_YAML_PATH}")
    
    with open(TRAIN_JOB_YAML_PATH, "r") as f:
        train_job_yaml_content = f.read()
    
    manifest = yaml.safe_load(train_job_yaml_content)
    namespace = manifest['metadata']['namespace']
    
    compiler.Compiler().compile(
        pipeline_func=hpo_pipeline,
        package_path=pipeline_filename,
        pipeline_parameters={
            'namespace': namespace,
            'train_job_yaml_content': train_job_yaml_content
        }
    )
    return namespace
    
def run_pipeline(pipeline_filename: str = 'graphranker_pipeline.yaml', train_job_yaml_path:str = "./train_job.yaml"):
    try:
        namespace = compile_pipeline_yaml(pipeline_filename, train_job_yaml_path)
        
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
                pipeline_package_path=pipeline_filename
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
    run_pipeline()
    