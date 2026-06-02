import os
import time
import subprocess
from json import dumps
from typing import Dict

import yaml
from kind_util import setup_cluster, delete_cluster, find_executable_path

from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import \
    TUNED_BLOCK_SIZES
#pip install kfp==2.16.1
from kfp import dsl, compiler, client as kfp_client

KUBEFLOW_VERSION = "v2.2.0"
NAMESPACE = "ranker-ns"
PROJECT_ROOT = os.path.abspath("../../")
# ====================================================================
# THE INFRASTRUCTURE
# These functions only run on your laptop, NEVER inside the cluster.
# You can bypass these completely when moving to Vertex AI or GKE.
# ====================================================================
def atart_local_cluster():
    kind_path = find_executable_path("kind")
    kubectl_path = find_executable_path("kubectl")
    setup_cluster(kind_path=kind_path, kubectl_path=kubectl_path,
        PROJECT_ROOT=PROJECT_ROOT,
        KUBEFLOW_VERSION=KUBEFLOW_VERSION, NAMESPACE=NAMESPACE)
    
def stop_local_cluster(kind_path:str):
    delete_cluster(kind_path)
    
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
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes", "pyyaml"]
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
def extract_hpo_results(target_image: str, namespace: str = "ranker-ns"):
    """Uses the same unified image variable to invoke extraction logic."""
    return dsl.ContainerSpec(
        image=target_image,
        args=[
            "--phase=extract_hpo_results",
            f"--namespace={namespace}"
        ]
    )


# ====================================================================
# PART 3: THE PIPELINE ("The Script")
# ====================================================================

@dsl.pipeline(
    name="graphranker-sequential-hpo",
    description="Sequential HPO execution loop powered by embedded layout manifests."
)
def hpo_pipeline(train_job_yaml_content:str, namespace:str = 'ranker-ns', target_image:str = 'ranker-app:local'):
    
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
    extraction_task = extract_hpo_results(target_image=target_image, namespace=namespace)
    
    if previous_task is not None:
        extraction_task.after(previous_task)

# ====================================================================
# EXECUTION ENTRY POINT
# ====================================================================
if __name__ == '__main__':
    
    TRAIN_JOB_YAML_PATH = "./train_job.yaml"
    
    if not os.path.exists():
        raise FileNotFoundError(
            f"Missing trian job yaml file file: {TRAIN_JOB_YAML_PATH}")
    
    with open(TRAIN_JOB_YAML_PATH, "r") as f:
        train_job_yaml_content = f.read()
    
    manifest = yaml.safe_load(train_job_yaml_content)
    target_image = manifest['spec']['trainer']['image']
    namespace = manifest['metadata']['namespace']
    
    kind_path = find_executable_path("kind")
    kubectl_path = find_executable_path("kubectl")
    
    # Setup Local Infrastructure and RBAC
    with dsl.ExitHandler(
            exit_task=stop_local_cluster(kind_path=kind_path),
            name="infrastructure-guard"
    ):
        
        setup_cluster(kind_path=kind_path, kubectl_path=kubectl_path,
            PROJECT_ROOT=PROJECT_ROOT,
            KUBEFLOW_VERSION=KUBEFLOW_VERSION, NAMESPACE=NAMESPACE)
        
        setup_rbac(namespace)
        
        # Compile the Pipeline
        pipeline_filename = 'graphranker_pipeline.yaml'
        print( f"🛠️ Ingesting template and compiling pipeline to {pipeline_filename}...")
        compiler.Compiler().compile(
            pipeline_func=hpo_pipeline,
            package_path=pipeline_filename,
            # Bind our locally loaded variables as default pipeline configuration entries
            pipeline_parameters={
                'namespace': namespace,
                'target_image': target_image,
                'train_job_yaml_content': train_job_yaml_content
            }
        )
        
        # 4. Send the compiled asset to your Kind KFP engine
        print("📤 Submitting pipeline run to local backend...")
        try:
            client = kfp_client.Client(host="http://localhost:8080")
            experiment = client.create_experiment("GraphRanker-HPO")
            run = client.run_pipeline(
                experiment_id=experiment.id,
                job_name="hermetic-sequential-hpo-run",
                pipeline_package_path=pipeline_filename
            )
            print(f"🎉 Run initiated! Dashboard URL: {run.url}")
        except Exception as e:
            print(f"⚠️ Automatic submission skipped/failed: {e}")
            print("You can manually upload 'graphranker_pipeline.yaml' directly inside the KFP UI website.")