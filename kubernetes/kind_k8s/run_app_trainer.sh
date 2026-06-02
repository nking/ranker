#!/bin/bash

# this script runs the container image ranker-app:latest using a Kubeflow Trainer API v2.2
#    as a jax-distributed Trainer using kubectl
# USAGE:
#    use in a venv that has vizier installed
#    ./run_app_trainer.sh

echo "Checking internet connection, needed to pull docker images..."
if ! ping -c 1 -W 3 google.com &> /dev/null; then
  echo "❌ WARNING: No internet connection or DNS failure detected."
  echo "Exiting to prevent partial/broken cluster deployment."
  exit 1
fi
echo "✅ Internet connection verified."

# shellcheck disable=SC2155
export PROJECT_ROOT=$(realpath "../../")

date
echo "found project directory = $PROJECT_ROOT"

run_code="true"

NUM_TRIALS=4
NUM_TRIALS_PER_WORKER=2

#if see binding errors, make sure that containers using the same ports
# are shutdown with docker compose down.
# look for ports: sudo lsof -i :5432

extract_and_shutdown() {
    echo "Script finished or interrupted."

    #fetch and report
    if [ "$run_code" = "true" ]; then
        echo "Running HPO results extraction..."

        #TODO: this script, extract_hpo_results.py, running outside of the ranker-app:local
        # could be replaced by reading in app-runner.yaml and replacing:
        # spec.replicas --> override value with "1"
        # spec.template.spec.containers.env.JAX_NUM_PROCESSES --> override value with "1"
        # spec.template.spec.containers.args.phase --> override value to use 'extract_hpo'
        # then invoke similarly as below with envsubst '$PROJECT_ROOT $TRIAL_IDS' < app-runner_modified.yaml | kubectl apply -f -
        # where have to figure out how to pass the modified string as the yaml...

        # Runs python script, redirects stdout and stderr to hpo_results.txt, and tests exit code
        if python3 extract_hpo_results.py > hpo_results.txt 2>&1; then
            echo "✅ HPO results extracted successfully to hpo_results.txt"
        else
            echo "❌ ERROR: extract_hpo_results.py crashed!"
            echo "📝 Check 'hpo_results.txt' to view the python traceback error statements."
            echo "🛑 DEBUG PAUSE: Keeping the Kind cluster alive so you can inspect databases/logs."
            read -p "Press [Enter] to tear down the cluster and exit..."
        fi

        echo "Cleaning up local cluster..."
        kind delete cluster --name graphranker-tune-train-test-cluster

        ## assert contents of chunk_logs_app_0.txt and chunk_logs_app_1.txt
        FILES=("chunk_trainer_master_logs.txt" "chunk_trainer_worker-0_logs.txt")
        for FILE in "${FILES[@]}"; do
            if ! grep -q "'trial_ids': '\[0, 1\]'" $FILE; then
                echo "missing trial_ids from $FILE"
            fi
            if ! grep -q "'trial_ids': '\[2, 3\]'" $FILE; then
                echo "missing trial_ids from $FILE"
            fi
            PHRASE="Epoch 2:"
            EXPECTED=4
            count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
            if ! [ "$count" -eq "$EXPECTED" ]; then
                echo "❌ Assertion failed: Expected $EXPECTED of $PHRASE, but found $count in $FILE"
                exit 1
            fi
            PHRASE="finally clause in train_fn"
            count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
            if ! [ "$count" -eq "$EXPECTED" ]; then
                echo "❌ Assertion failed: Expected $EXPECTED of $PHRASE, but found $count in $FILE"
                exit 1
            fi
        done

        FILE="chunk_trainer_master_logs.txt"
        PHRASE="mlflow start run: trial_"
        EXPECTED=4
        count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
        if ! [ "$count" -eq "$EXPECTED" ]; then
            echo "❌ Assertion failed: Expected $EXPECTED of $PHRASE, but found $count in $FILE"
            exit 1
        fi
        PHRASE="worker_0"
        EXPECTED=4
        count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
        if ! [ "$count" -eq "$EXPECTED" ]; then
            echo "❌ Assertion failed: Expected $EXPECTED of $PHRASE, but found $count in $FILE"
            exit 1
        fi
        PHRASE="worker_0"
        EXPECTED=4
        count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
        if ! [ "$count" -eq "$EXPECTED" ]; then
            echo "❌ Assertion failed: Expected $EXPECTED of $PHRASE, but found $count."
            exit 1
        fi
        FILE="chunk_trainer_worker-0_logs.txt"
        count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
        if ! [ "$count" -eq "$EXPECTED" ]; then
            echo "❌ Assertion failed: Expected $EXPECTED of $PHRASE, but found $count."
            exit 1
        fi
        date
    fi
}

trap extract_and_shutdown EXIT

rm -f chunk_trainer_master_logs.txt chunk_trainer_worker-0_logs.txt

if [ "$run_code" = "true" ]; then

    echo "creating cluster"
    envsubst '$PROJECT_ROOT' < kind-cluster.yaml | kind create cluster --config -
    echo "waiting for nodes"
    kubectl wait --for=condition=Ready nodes --all --timeout=120s

    # ====================================================================
    # INSTALL KUBEFLOW TRAINER V2
    # ====================================================================
    export VERSION=v2.2.0
    echo "1. Installing Kubeflow Trainer Controller (and bundled JobSet)..."
    kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"

    # Give the API server a few seconds to register the deployment objects before patching
    sleep 3

    echo "Applying Hotfix for broken JobSet staging image..."
    # This overrides the deleted staging image with the official, stable v0.12.0 image
    kubectl set image deployment/jobset-controller-manager manager=registry.k8s.io/jobset/jobset:v0.12.0 -n kubeflow-system

    echo "Waiting for Controllers to be Ready..."
    # Wait for Trainer Controller
    while [[ $(kubectl get deployment kubeflow-trainer-controller-manager -n kubeflow-system -o 'jsonpath={.status.readyReplicas}') != "1" ]]; do
      echo "Trainer Controller not ready yet... checking again in 5s"
      sleep 5
    done
    echo "Trainer Controller is ready!"

    # Wait for JobSet Controller
    while [[ $(kubectl get deployment jobset-controller-manager -n kubeflow-system -o 'jsonpath={.status.readyReplicas}') != "1" ]]; do
      echo "JobSet Controller not ready yet... checking again in 5s"
      sleep 5
    done
    echo "JobSet Controller is ready!"

    echo "Installing Kubeflow Training Runtimes..."
    kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=${VERSION}"

    echo "Waiting for Kubeflow components to start..."
    kubectl wait --for=condition=Available deployment/kubeflow-trainer-controller-manager -n kubeflow-system --timeout=120s
    # ====================================================================

    echo "Sideloading local docker image into Kind..."
    #note if doesn't exist, cd to project root and use docker build -t ranker-app:local -f Dockerfile_cpu .
    kind load docker-image ranker-app:local --name graphranker-tune-train-test-cluster
    kind load docker-image vizier-server:local --name graphranker-tune-train-test-cluster

    echo "deploying databases"
    kubectl create namespace ranker-ns --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f secrets.yaml -n ranker-ns
    envsubst '$PROJECT_ROOT' < dbs.yaml | kubectl apply -f -

    #echo "waiting for readiness of databases"
    #kubectl rollout status deployment/local-db-store -n ranker-ns --timeout=60s || exit 1
    #kubectl rollout status deployment/gcs-emulator -n ranker-ns --timeout=60s || exit 1
    #kubectl rollout status deployment/vizier-server -n ranker-ns --timeout=60s || exit 1
    echo "waiting for readiness of databases (timeout is 3m)"
    # If any of these fail, pause so you can debug instead of instantly exiting and deleting the cluster
    if ! kubectl rollout status deployment/local-db-store -n ranker-ns --timeout=180s; then
        echo "❌ ERROR: local-db-store failed to roll out."
        echo "🛑 SETUP DEBUG PAUSE: Run 'kubectl get pods -n ranker-ns' in another terminal to inspect."
        read -p "Press [Enter] to allow the script to exit and clean up..."
        exit 1
    fi

    if ! kubectl rollout status deployment/gcs-emulator -n ranker-ns --timeout=180s; then
        echo "❌ ERROR: gcs-emulator failed to roll out."
        read -p "Press [Enter] to allow the script to exit and clean up..."
        exit 1
    fi

    if ! kubectl rollout status deployment/vizier-server -n ranker-ns --timeout=180s; then
        echo "❌ ERROR: vizier-server failed to roll out."
        read -p "Press [Enter] to allow the script to exit and clean up..."
        exit 1
    fi
fi

(
    date
    for (( i=0; i<NUM_TRIALS; i+=NUM_TRIALS_PER_WORKER )); do
        #format the trial_ids string to give a worker several trials to process
        TRIAL_IDS="["
        for (( j=0; j<NUM_TRIALS_PER_WORKER; j++ )); do
            TRIAL_VAL=$((i + j))
            if [ "$TRIAL_VAL" -ge "$NUM_TRIALS" ]; then
                break
            fi
            if [ "$j" -gt 0 ]; then
                TRIAL_IDS+=", "
            fi
            TRIAL_IDS+="$TRIAL_VAL"
        done
        TRIAL_IDS+="]"

        export TRIAL_IDS
        echo "Launching JobGroup chunk with trial_ids=${TRIAL_IDS}"

        if [ "$run_code" = "true" ]; then

            # Apply the Kubeflow manifest
            envsubst '$TRIAL_IDS' < train_job.yaml | kubectl apply -f -

            echo "🚀 Training Job submitted to Kubeflow! Waiting for completion..."

            sleep 3

            echo "get service for JAX_COORDINATOR_ADDRESS:"
            kubectl get svc -n ranker-ns
            kubectl get pods -n ranker-ns

            #NOTE: job-role=master is the rank=0 worker and worker=0 is the 2nd worker.
            # the jax_process=0 is the 'master' and jax_process=1 is the 'worker' with replica index 0
            MASTER_POD=$(kubectl get pods -n ranker-ns -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep "node-0-0")

            if [ -n "$MASTER_POD" ]; then
                echo "📡 Live-streaming logs to live_logs.txt..."
                kubectl logs -f $MASTER_POD -n ranker-ns --tail=-1 > live_logs.txt 2>&1 &
            fi

            # Wait natively for the TrainJob to finish!
            # Kubeflow automatically sets the "Complete" condition when Rank 0 finishes successfully.
            kubectl wait --for=condition=Complete trainjob/graphranker-jax-training -n ranker-ns --timeout=1h

            WORKER_POD=$(kubectl get pods -n ranker-ns -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep "node-0-1")

            echo "Saving logs for $MASTER_POD and $WORKER_POD..."

            kubectl logs $MASTER_POD -n ranker-ns --tail=-1 >> chunk_trainer_master_logs.txt 2>&1
            kubectl logs $WORKER_POD -n ranker-ns --tail=-1 >> chunk_trainer_worker-0_logs.txt 2>&1

            echo "Chunk finished!"
            kubectl delete -f train_job.yaml --ignore-not-found

        fi
    done
    date
)

##debugging: 
#kubectl get pods -n kubeflow-system
#kubectl delete validatingwebhookconfiguration validator.trainer.kubeflow.org
#kubectl rollout restart deployment/kubeflow-trainer-controller-manager -n kubeflow-system

# kubectl describe trainjob -n ranker-ns graphranker-jax-training
# kubectl get pods -n ranker-ns
# kubectl logs <pod_name from get pods> -n ranker-ns
