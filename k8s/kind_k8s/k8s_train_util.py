import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _append_or_replace_spec_trainer_args(manifest: dict, arg_key:str, arg_value:str):
    mod_i = -1
    key = f"--{arg_key}"
    for i, arg in enumerate(manifest["spec"]["trainer"]["args"]):
        if arg.find(key) == 0:
            mod_i = i
            break
    if mod_i == -1:
        manifest["spec"]["trainer"]["args"].append(f"{key}={arg_value}")
    else:
        manifest["spec"]["trainer"]["args"][mod_i] = f"{key}={arg_value}"

def _append_or_replace_spec_trainer_env(manifest: dict, env_key:str, env_value:str):
    mod_i = -1
    for i, env_dict in enumerate(manifest["spec"]["trainer"]["env"]):
        if env_dict["name"] == env_key:
            mod_i = i
            break
    if mod_i == -1:
        manifest["spec"]["trainer"]["env"].append({"name": env_key, "value": env_value})
    else:
        manifest["spec"]["trainer"]["env"][mod_i]["value"] = env_value

def run_train_job_phase(
        train_job_yaml_content: str,
        namespace: str = "ranker-ns",
        phase: str = "tune",
        trial_ids: str = None,
        output_log_dir_uri: str = None,
):
    """
    run train_job.yaml for given phase. Authenticate within the cluster before invoking this method.
    phase based dynamic changes to train_job yaml are performed internally.
    note that this assumes that yaml args for output_hyperparams_uri and output_metrics_uri use uri pattern
      gs://hpo-results-bucket/<project_id>/<study_name>/<tune|train|test>/hpo_<hparams|metrics>.json
    :param train_job_yaml_content:
    :param namespace:
    :param phase: can be "tune" or "train-best" or "test-best".  other options not yet implemented
    :param trial_ids: the list of HPO trials to make if phase =="tune", else is None if phase is not "tune'
    :param output_log_dir_uri: uri where to write output logs with name job_name if not None.  the logs
       will be written as output_log_dir_uri/{phase}-<master|worker-podNumber>-logs.txt
    """
    import time
    import yaml
    from kubernetes import client, config
    from kubernetes.client import ApiException
    from json import loads
    
    allowed_phases = ("tune", "train-best", "test-best", "export-hpo-results", "export-train-results", "export-test-results")
    if phase not in allowed_phases:
        raise ValueError(
            f"phase must be one of {allowed_phases}")
    
    if phase == "tune" and trial_ids is None:
        raise ValueError("trial_ids cannot be None when phase is 'tune'")
    
    job_name = f"graphranker-{phase}-job"
    
    # Parse the embedded template string into a Python dictionary
    try:
        manifest = yaml.safe_load(train_job_yaml_content)
    except Exception as ex:
        if not isinstance(train_job_yaml_content, str):
            raise ValueError("train_job_yaml_content must be the yaml file read into a string")
        raise ex
        
    ## ---------------------------- begin dynamic edits to the yaml -------------------------------
    if phase == "tune":
        trial_ids_str = trial_ids
        trial_ids = loads(trial_ids)
        if not isinstance(trial_ids, list):
            raise ValueError("trial_ids is expected to be string made from json.dumps(list(int_list))")
        if not all(isinstance(x, int) and x > -1 for x in trial_ids):
            raise ValueError("trial_ids must only contain non-negative integers")
        if ("spec" not in manifest or "trainer" not in
                manifest["spec"] or "args" not in manifest["spec"]["trainer"]):
            raise KeyError("train_job.yaml is missing spec.trainer.args")
        job_name = f'{job_name}-{trial_ids[0]}'
        _append_or_replace_spec_trainer_args(manifest, "trial_ids", trial_ids_str)
        
    elif phase in {"export-hpo-results", "export-train-results", "export-test-results"}:
        # only needs to run on 1 node no matter what is in train_job.yaml
        manifest["spec"]["trainer"]["numNodes"] = 1
        _append_or_replace_spec_trainer_env(manifest, "JAX_NUM_PROCESSES", "1")
        
    if phase != "tune":
        #remove the environment variable in train_job.yaml
        for i, arg in enumerate(manifest["spec"]["trainer"]["args"]):
            if arg.find("--trial_ids") == 0:
                del manifest["spec"]["trainer"]["args"][i]
                break
       
    _append_or_replace_spec_trainer_args(manifest, "phase", phase)
    manifest['metadata']['name'] = job_name
    manifest['metadata']['namespace'] = namespace
    _append_or_replace_spec_trainer_env(manifest, "JAX_COORDINATOR_ADDRESS", f"{job_name}-node-0-0.{job_name}:8888")
    
    #gs://hpo-results-bucket/<project_id>/<study_name>/<tune|train|test>/hpo_<hparams|metrics>.json
    if phase in {"export-train-results", "export-test-results"}:
        mod_is = []
        for i, arg in enumerate(manifest["spec"]["trainer"]["args"]):
            if arg.find("--output_hyperparams_uri") == 0:
                mod_is.append(i)
            elif arg.find("--output_metrics_uri") == 0:
                mod_is.append(i)
        repl = "train" if phase == "export-train-results" else "test"
        import re
        # Match any non-slash characters that are preceded by a slash
        # and followed by exactly one slash and the end of the string
        # example entry: text = "--output_hyperparams_uri=gs://hpo-results-bucket/tune-kind-01/GraphRanker_tuning_kind/tune/hpo_hparams.json"
        pattern = r'(?<=/)[^/]+(?=/[^/]+$)'
        for mod_i in mod_is:
            manifest["spec"]["trainer"]["args"][mod_i] = re.sub(pattern, repl, manifest["spec"]["trainer"]["args"][mod_i])
        
    ## ---------------------- end dynamic yaml edits ------------------------------
    
    logging.info(f'deploying {job_name}')
    
    crd_api = client.CustomObjectsApi()
    
    try:
        
        # Deploy to Kubernetes
        logging.info(f"🚀 Launching TrainJob: {job_name}")
        crd_api.create_namespaced_custom_object(
            group="trainer.kubeflow.org", version="v1alpha1",
            namespace=namespace, plural="trainjobs", body=manifest
        )
        
        #  Monitor execution
        completed = False
        while not completed:
            time.sleep(10)
            try:
                status = crd_api.get_namespaced_custom_object(
                    group="trainer.kubeflow.org", version="v1alpha1",
                    namespace=namespace, plural="trainjobs", name=job_name
                )
                conditions = status.get('status', {}).get('conditions', [])
                for condition in conditions:
                    if condition.get('type') == 'Complete' and condition.get( 'status') == 'True':
                        logging.info(f"✅ TrainJob {job_name} completed successfully!")
                        completed = True
                        break
                    elif condition.get('type') == 'Failed' and condition.get( 'status') == 'True':
                        reason = condition.get('reason', 'UnknownReason')
                        message = condition.get('message','No error message provided.')
                        # FETCH POD LOGS BEFORE TEARDOWN
                        try:
                            core_v1_api = client.CoreV1Api()
                            # Find all pods belonging to this TrainJob
                            label_selector = f"jobset.sigs.k8s.io/jobset-name={job_name}"
                            pods = core_v1_api.list_namespaced_pod(namespace=namespace,
                                label_selector=label_selector)
                            if not pods.items:
                                logging.warning( f"⚠️ No pods found for TrainJob {job_name}. "
                                                 f"The job may have failed before pods could be scheduled.")
                            # Iterate through them and dump the logs
                            for pod in pods.items:
                                pod_name = pod.metadata.name
                                logging.info(f"\n{'=' * 20} Logs for Pod: {pod_name} {'=' * 20}")
                                try:
                                    # Try fetching current logs
                                    pod_logs = core_v1_api.read_namespaced_pod_log(
                                        name=pod_name, namespace=namespace)
                                    logging.info(f"\n{pod_logs}")
                                except Exception as current_err:
                                    logging.warning( f"Could not get current logs, attempting "
                                                     f"to fetch previous (crash-looped) logs...")
                                    try:
                                        # Fallback: Try fetching previous logs (equivalent to kubectl logs -p)
                                        pod_logs_prev = core_v1_api.read_namespaced_pod_log(
                                            name=pod_name, namespace=namespace,
                                            previous=True)
                                        logging.info(f"\n[PREVIOUS CONTAINER INSTANCE]\n{pod_logs_prev}")
                                    except Exception as prev_err:
                                        logging.error(f"Could not fetch any logs for {pod_name}: {prev_err}")
                                logging.info(f"{'=' * 60}\n")
                        except Exception as e3:
                            logging.exception(
                                f"⚠️ Failed to retrieve pod logs during error handling: {e3}")
                        raise RuntimeError(f"❌ TrainJob {job_name} failed! reason: {reason}, message: {message}")
            except ApiException as e:
                logging.exception(f"API Error fetching TrainJob: {e}")
        
        if output_log_dir_uri is not None:
            logging.info(f"writing logs to {output_log_dir_uri}")
            import fsspec
            core_v1 = client.CoreV1Api()
            pods = core_v1.list_namespaced_pod(namespace).items
            master_pod = next((p.metadata.name for p in pods if "node-0-0" in p.metadata.name), None)
            if master_pod:
                master_logs = core_v1.read_namespaced_pod_log(name=master_pod, namespace=namespace)
                with fsspec.open(f"{output_log_dir_uri}/{phase}-master-logs.txt","a") as f:
                    f.write(master_logs)
            
            num_nodes = manifest["spec"]["trainer"]["numNodes"] #is already int
            for i in range(1, num_nodes):
                worker_pod = next((p.metadata.name for p in pods if f"node-0-{i}" in p.metadata.name), None)
                if worker_pod:
                    worker_logs = core_v1.read_namespaced_pod_log(name=worker_pod, namespace=namespace)
                    with fsspec.open(f"{output_log_dir_uri}/{phase}-worker-{i}-logs.txt", "a") as f:
                        f.write(worker_logs)
                       
    except Exception as e2:
        logging.exception(f'Error during Train Job: {e2}')
        raise e2
    finally:
        # Lifecycle cleanup'
        logging.info(f"🧹 Tearing down TrainJob custom resource: {job_name}")
        crd_api.delete_namespaced_custom_object(
            group="trainer.kubeflow.org", version="v1alpha1",
            namespace=namespace, plural="trainjobs", name=job_name
        )
