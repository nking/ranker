import time
from util_kind import run_cmd
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

"""
NOTE: if setup_kfpv2_backend seems to be stuck at downloading the very large mysl:
    download manually: 
       docker pull mysql:8.4
       docker pull chrislusf/seaweedfs:4.00
    workaround architecture bug:
       docker save mysql:8.4 | docker exec -i graphranker-tune-train-test-cluster-control-plane ctr -n k8s.io images import -
       docker save mysql:8.4 | docker exec -i graphranker-tune-train-test-cluster-worker ctr -n k8s.io images import -
       docker save mysql:8.4 | docker exec -i graphranker-tune-train-test-cluster-worker2 ctr -n k8s.io images import -

       docker save chrislusf/seaweedfs:4.00 | docker exec -i graphranker-tune-train-test-cluster-control-plane ctr -n k8s.io images import -
       docker save chrislusf/seaweedfs:4.00 | docker exec -i graphranker-tune-train-test-cluster-worker ctr -n k8s.io images import -
       docker save chrislusf/seaweedfs:4.00 | docker exec -i graphranker-tune-train-test-cluster-worker2 ctr -n k8s.io images import -
    then:
       kubectl delete namespace kubeflow
       kubectl create namespace kubeflow
    and run the script again
"""

def setup_kfpv2_backend(kubectl_path:str):
    KFP_VERSION="2.16.1"
    
    # Apply cluster-scoped resources (like CRDs)
    run_cmd([kubectl_path, "apply", "-k",
        f"https://github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref={KFP_VERSION}"],
        max_retries=3)
    
    # Wait for the Application CRD to be established
    run_cmd([kubectl_path, "wait", "--for=condition=established","crd/applications.app.k8s.io", "--timeout=120s"])
    
    #Deploy the core KFP services
    run_cmd([kubectl_path, "apply", "-k",
        f"https://github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref={KFP_VERSION}"],
        max_retries=3)
    
    run_cmd([kubectl_path, "rollout", "status", "deployment/ml-pipeline", "-n", "kubeflow"])
    run_cmd([kubectl_path, "rollout", "status", "deployment/ml-pipeline-ui", "-n", "kubeflow"])
    
    print("✅ KFP is ready!")
    
    run_cmd([kubectl_path, "get", "pods", "-n", "kubeflow"])
    
    run_cmd([kubectl_path, "get", "svc", "-n", "kubeflow"])
    
