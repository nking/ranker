import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_train_job_phase(
        train_job_yaml_content: str,
        namespace: str = "ranker-ns",
        phase: str = "tune",
        trial_ids: str = None,
        output_log_dir_uri: str = None,
):
    """
    run train_job.yaml for given phase. Authenticate within the cluster before invoking this method.
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
    
    if phase not in ("tune", "train-best", "test-best", "export-hpo-results"):
        raise ValueError(
            f"phase must be one of {('tune', 'train-best', 'test-best', 'export-hpo-results')}")
    
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
        
    ## ---- begin dynamic edits to the yaml ------
    if phase == "tune":
        trial_ids_str = trial_ids
        trial_ids = loads(trial_ids)
        if not all(isinstance(x, int) and x > -1 for x in trial_ids):
            raise ValueError("trial_ids must only contain non-negative integers")
        if "spec" not in manifest or "trainer" not in manifest[
            "spec"] or "args" not in manifest["spec"][
            "trainer"]:
            raise ValueError("train_job.yaml is missing spec.trainer.args")
        job_name = f'{job_name}-{trial_ids[0]}'
        for i, arg in enumerate(manifest["spec"]["trainer"]["args"]):
            if arg.find("--trial_ids") == 0:
                mod_i = i
                break
        manifest["spec"]["trainer"]["args"][mod_i] = f"--trial_ids={trial_ids_str}"
    elif phase == "export-hpo-results":
        # this one only needs to run on 1 node no matter what is in train_job.yaml
        manifest["spec"]["trainer"]["numNodes"] = 1
        for env_dict in manifest["spec"]["trainer"]["env"]:
            if env_dict["name"] == "JAX_NUM_PROCESSES":
                env_dict["value"] = "1"
    
    if phase != "tune":
        #remove the environment variable in train_job.yaml
        for i, arg in enumerate(manifest["spec"]["trainer"]["args"]):
            if arg.find("--trial_ids") == 0:
                rm_i = i
                break
        del manifest["spec"]["trainer"]["args"][rm_i]
       
    #modify the phase in args
    for i, arg in enumerate(manifest["spec"]["trainer"]["args"]):
        if arg.find("--phase") == 0:
            mod_i = i
            break
    manifest["spec"]["trainer"]["args"][mod_i] = f"--phase={phase}"
    
    manifest['metadata']['name'] = job_name
    manifest['metadata']['namespace'] = namespace
    for env_dict in manifest["spec"]["trainer"]["env"]:
        if env_dict["name"] == "JAX_COORDINATOR_ADDRESS":
            env_dict["value"] = f"{job_name}-node-0-0.{job_name}:8888"
            break
        # {'name': 'XLA_FLAGS', 'value': '--xla_force_host_platform_device_count=2'}
    
    ## ---- end dynamic yaml edits ------
    
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
        # Lifecycle cleanup
        logging.info(f"🧹 Tearing down TrainJob custom resource: {job_name}")
        crd_api.delete_namespaced_custom_object(
            group="trainer.kubeflow.org", version="v1alpha1",
            namespace=namespace, plural="trainjobs", name=job_name
        )
        
