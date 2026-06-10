import os
import subprocess
from pathlib import Path

from air_gapped_kfpv2_prep import delete_namespace_kubeflow, \
    sideload_image_without_kind, vendor_manifests, extract_and_sideload_images, \
    deploy_offline, build_kustomize_if_not_found
from util_kind import find_executable_path, run_cmd
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

"""
NOTE:  the use of kubectl and github fetches began to fail this week as
       the latencies of github increased greatly.
       The workaround is an "air-gapped" approach:
            download and install kustomize
            docker pull images
            docker load images
            kustomize and kubectl to inject KFPv2 backend into cluster
"""

def setup_kfpv2_backend(kubectl_path:str, kind_path:str, docker_path:str, PROJECT_ROOT:str, CLUSTER_NAME:str):
    KFP_VERSION = "2.16.1"
    KUSTOMIZE_VERSION = "v5.8.1" #if do not use kubectl -k
    #KUSTOMIZE_VERSION = "v5.7.1" #if do use kubectl -k
    
    git_path = find_executable_path("git")
    
    delete_namespace_kubeflow(kubectl_path=kubectl_path)
    
    problem_images = ["mysql:8.4", "chrislusf/seaweedfs:4.00"]
    for img in problem_images:
        sideload_image_without_kind(docker_path=docker_path, kind_path=kind_path,
            image_name=img, cluster_name=CLUSTER_NAME)
    
    kustomize_path = os.path.join(PROJECT_ROOT, "orchestration/k8s/kustomize")
    build_kustomize_if_not_found(kustomize_path=kustomize_path,
        version=KUSTOMIZE_VERSION)
    
    VENDOR_DIR = Path(os.path.join(PROJECT_ROOT, "orchestration/k8s/vendor"))
    cluster_scoped_path = Path(os.path.join(VENDOR_DIR, "kubeflow-pipelines",
        "manifests", "kustomize", "cluster-scoped-resources"))
    apps_path = Path(os.path.join(VENDOR_DIR, "kubeflow-pipelines",
        "manifests", "kustomize", "env", "dev"))
    
    # (Clones kubeflow-pipelines and argo-workflows to VENDOR_DIR)
    vendor_manifests(git_path=git_path, vendor_dir=VENDOR_DIR, kfp_version=KFP_VERSION)
    
    # Extract and sideload all images automatically
    extract_and_sideload_images(
        kustomize_path=kustomize_path,
        docker_path=docker_path,
        kind_path=kind_path,
        vendor_dirs=[cluster_scoped_path, apps_path],
        cluster_name=CLUSTER_NAME
    )
        
    deploy_offline(kustomize_path=kustomize_path, kubectl_path=kubectl_path, deploy_dir=cluster_scoped_path)
    
    deploy_offline(kustomize_path=kustomize_path, kubectl_path=kubectl_path, deploy_dir=apps_path)

    run_cmd([kubectl_path, "rollout", "status", "deployment/ml-pipeline", "-n", "kubeflow", "--timeout=1200s"], timeout=1200)
    run_cmd([kubectl_path, "rollout", "status", "deployment/ml-pipeline-ui", "-n", "kubeflow"])
    
    print("✅ KFP is ready!")
    
    run_cmd([kubectl_path, "get", "pods", "-n", "kubeflow"])
    
    run_cmd([kubectl_path, "get", "svc", "-n", "kubeflow"])
    
