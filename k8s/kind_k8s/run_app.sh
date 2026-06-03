#!/bin/bash

# this script runs the container image ranker-app:local using a StatefulSet
#    and kubectl

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
        FILES=("chunk_logs_app_0.txt" "chunk_logs_app_1.txt")
        for FILE in "${FILES[@]}"; do
            if ! grep -q "'trial_ids': '\[0, 1\]'" $FILE; then
                echo "missing trial_ids in $FILE"
            fi
            if ! grep -q "'trial_ids': '\[2, 3\]'" $FILE; then
                echo "missing trial_ids in $FILE"
            fi
            PHRASE="Epoch 2:"
            EXPECTED=4
            count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
            if ! [ "$count" -eq "$EXPECTED" ]; then
                echo "❌ Assertion failed: Expected $EXPECTED for $PHRASE, but found $count in $FILE"
                exit 1
            fi
            PHRASE="finally clause in train_fn"
            count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
            if ! [ "$count" -eq "$EXPECTED" ]; then
                echo "❌ Assertion failed: Expected $EXPECTED for $PHRASE, but found $count in $FILE"
                exit 1
            fi
        done

        FILE="chunk_logs_app_0.txt"
        PHRASE="mlflow start run: trial_"
        EXPECTED=4
        count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
        if ! [ "$count" -eq "$EXPECTED" ]; then
            echo "❌ Assertion failed: Expected $EXPECTED for $PHRASE, but found $count in $FILE"
            exit 1
        fi
        PHRASE="worker_0"
        count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
        if [ "$count" -eq 0 ]; then
            echo "❌ Assertion failed: Expected > 0 for $PHRASE, but found $count in $FILE"
            exit 1
        fi
        FILE="chunk_logs_app_1.txt"
        PHRASE="worker_1"
        count=$(grep -oF "$PHRASE" "$FILE" | wc -l)
        if [ "$count" -eq 0 ]; then
            echo "❌ Assertion failed: Expected > 0 for $PHRASE, but found $count in $FILE"
            exit 1
        fi
        date
    fi
}

fetch_logs() {
    local pod_name=$1
    local output_file=$2
    local temp_log="temp_log_$(date +%s%N).txt"

    echo "Attempting to fetch logs for $pod_name..."

    kubectl logs "pod/$pod_name" -n ranker-ns > "$temp_log" 2>/dev/null

    # 2. Check if file is empty (size is zero)
    if [ ! -s "$temp_log" ]; then
        echo "⚠️  No current logs found for $pod_name. Trying --previous (-p)..."
        kubectl logs "pod/$pod_name" -n ranker-ns -p > "$temp_log" 2>/dev/null
    fi

    # 3. Append whatever we got to the final file
    if [ -s "$temp_log" ]; then
        cat "$temp_log" >> "$output_file"
        echo "✅ Logs for $pod_name appended to $output_file."
    else
        echo "❌ No logs available for $pod_name (current or previous)."
    fi

    # 4. Clean up
    rm -f "$temp_log"
}

# --- Main Script Execution ---

# Wait for completion
kubectl wait --for=condition=ready=false pod -l app=app-runner -n ranker-ns --timeout=1h

# Append logs for this chunk iteration
fetch_logs "app-runner-0" "chunk_logs_app_0.txt"
fetch_logs "app-runner-1" "chunk_logs_app_1.txt"

trap extract_and_shutdown EXIT

rm -f chunk_logs_app_0.txt chunk_logs_app_1.txt

if [ "$run_code" = "true" ]; then

    echo "creating cluster"
    envsubst '$PROJECT_ROOT' < kind-cluster.yaml | kind create cluster --config -
    echo "waiting for nodes"
    kubectl wait --for=condition=Ready nodes --all --timeout=120s

    echo "Sideloading local docker image into Kind..."
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

            envsubst '$TRIAL_IDS' < app-runner.yaml | kubectl apply -f -

            echo "Waiting for all ML workers to roll out..."
            # blocks until the statefulset meets its entire replica availability goal
            kubectl rollout status statefulset/app-runner -n ranker-ns --timeout=300s

            echo "🚀 Chunk executing! Waiting for training workloads to complete..."
            # 1-hour safeguard blocking the script until the pods finish executing (Ready=False)
            kubectl wait --for=condition=ready=false pod -l app=app-runner -n ranker-ns --timeout=1h

            kubectl get pods -n ranker-ns

            # Append logs for this chunk iteration
            fetch_logs "app-runner-0" "chunk_logs_app_0.txt"
            fetch_logs "app-runner-1" "chunk_logs_app_1.txt"

            ## --- DEBUG PAUSE.... comment out when done debugging ---
            #echo "🛑 DEBUG PAUSE: Pods have finished or crashed."
            #read -p "Press [Enter] to delete the StatefulSet and continue to the next chunk..."
            ## ----------------------------

            echo "Chunk finished!"
            kubectl delete -f app-runner.yaml --ignore-not-found

        fi
    done
    date
)
