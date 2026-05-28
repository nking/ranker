#!/bin/bash

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
        python3 extract_hpo_results.py >& hpo_results.txt

        ##this is handled in cluster shutdown already, so save 10-15 sec by deleting the cluster only
        ##kubectl delete -f dbs.yaml

        # You could add a check here to extract data if it wasn't already
        #kind delete cluster --name graphranker-tune-train-test-cluster
    fi
}

trap extract_and_shutdown EXIT

rm chunk_logs_app_0.txt chunk_logs_app_1.txt

if [ "$run_code" = "true" ]; then

    echo "creating cluster"
    envsubst '$PROJECT_ROOT' < kind-cluster.yaml | kind create cluster --config -
    echo "waiting for nodes"
    kubectl wait --for=condition=Ready nodes --all --timeout=120s

    echo "Sideloading local docker image into Kind..."
    kind load docker-image ranker-app:latest --name graphranker-tune-train-test-cluster
    kind load docker-image ranker-vizier_server:latest --name graphranker-tune-train-test-cluster

    echo "deploying databases"
    kubectl create namespace ranker-ns --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f secrets.yaml -n ranker-ns
    envsubst '$PROJECT_ROOT' < dbs.yaml | kubectl apply -f -
    echo "waiting for readiness of databases"
    kubectl rollout status deployment/local-db-store -n ranker-ns --timeout=60s || exit 1
    kubectl rollout status deployment/gcs-emulator -n ranker-ns --timeout=60s || exit 1
    kubectl rollout status deployment/vizier-server -n ranker-ns --timeout=60s || exit 1

fi

(
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
            #for HPO, we use a new instance per chunk, though could refactor to keep container if wanted...
            envsubst '$PROJECT_ROOT $TRIAL_IDS' < app-runner.yaml | kubectl apply -f -

            echo "Waiting for chunk to finish..."

            kubectl wait --for=condition=Initialized pod -l app=app-runner -n ranker-ns --timeout=30s

            ## stream container statements in background and save its Process ID (PID)
            #kubectl logs -f -l app=app-runner -c app-runner -n ranker-ns --prefix=true &
            #LOGS_PID=$!

            echo "Waiting for chunk to finish..."
            # 1-hour execution safeguard blocking the foreground
            kubectl wait --for=condition=ready=false pod -l app=app-runner -n ranker-ns --timeout=1h

            kubectl logs pod/app-runner-0 -n ranker-ns >> chunk_logs_app_0.txt
            kubectl logs pod/app-runner-1 -n ranker-ns >> chunk_logs_app_1.txt

            # --- DEBUG PAUSE ---
            echo "🛑 DEBUG PAUSE: Pods have finished or crashed."
            read -p "Press [Enter] to delete the StatefulSet and continue to the next chunk..."
            # ----------------------------

            ## Kill the background log stream now that the chunk is done
            #kill $LOGS_PID 2>/dev/null

            echo "Chunk finished!"
            kubectl delete -f app-runner.yaml --ignore-not-found
        fi
    done
)
