
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
    :param phase: can be "tune" or "train_best" or "test_best".  other options not yet implemented
    :param trial_ids: the list of HPO trials to make if phase =="tune", else is None if phase is not "tune'
    :param output_log_dir_uri: uri where to write output logs with name job_name if not None.  the logs
       will be written as output_log_dir_uri/{phase}-<master|worker-podNumber>-logs.txt
    """
    import time
    import yaml
    from kubernetes import client, config
    from kubernetes.client import ApiException
    from json import loads
    
    if phase not in ("tune", "train_best", "test_best", "export_hpo_results"):
        raise ValueError(
            f"phase must be one of {('tune', 'train_best', 'test_best')}")
    
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
        
    if phase == "tune":
        trial_ids_str = trial_ids
        trial_ids = loads(trial_ids_str)
        if not all(isinstance(x, int) and x > -1 for x in trial_ids):
            raise ValueError("trial_ids must only contain non-negative integers")
        if "spec" not in manifest or "trainer" not in manifest[
            "spec"] or "args" not in manifest["spec"][
            "trainer"]:
            raise ValueError("train_job.yaml is missing spec.trainer.args")
        for i, arg in enumerate(manifest["spec"]["trainer"]["args"]):
            if arg.find("--trial_ids") == 0:
                manifest["spec"]["trainer"]["args"][i] = trial_ids_str
                trial_ids_str = None
                break
        if trial_ids_str:
            manifest["spec"]["trainer"]["args"].append(trial_ids_str)
        job_name = f'{job_name}-{trial_ids[0]}'
    elif phase == "export_hpo_results":
        # this one only needs to run on one node no matter what is in train_job.yaml
        print(f'type={type(manifest["spec"]["trainer"]["num_nodes"])}')
        manifest["spec"]["trainer"]["num_nodes"] = "2"
        manifest["spec"]["trainer"]["env"]['JAX_NUM_PROCESSES'] = "1"
    
    manifest['metadata']['name'] = job_name
    manifest['metadata']['namespace'] = namespace
    
    crd_api = client.CustomObjectsApi()
    
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
        try:
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
        except ApiException as e:
            print(f"API Error fetching TrainJob: {e}")
    
    if output_log_dir_uri:
        print(f"writing logs to {output_log_dir_uri}")
        import fsspec
        core_v1 = client.CoreV1Api()
        pods = core_v1.list_namespaced_pod(namespace).items
        master_pod = next(
            (p.metadata.name for p in pods if "node-0-0" in p.metadata.name), None)
        if master_pod:
            master_logs = core_v1.read_namespaced_pod_log(name=master_pod,
                namespace=namespace)
            with fsspec.open(
                    f"{output_log_dir_uri}/{phase}-master-logs.txt",
                    "a") as f:
                f.write(master_logs)
        
        num_nodes = int(manifest["spec"]["trainer"]["num_nodes"])
        for i in range(1, num_nodes):
            worker_pod = next((p.metadata.name for p in pods if
            f"node-0-{i}" in p.metadata.name), None)
            if worker_pod:
                worker_logs = core_v1.read_namespaced_pod_log(name=worker_pod,
                    namespace=namespace)
                with fsspec.open(f"{output_log_dir_uri}/{phase}-worker-{i}-logs.txt",
                        "a") as f:
                    f.write(worker_logs)
    
    # Lifecycle cleanup
    print(f"🧹 Tearing down TrainJob custom resource: {job_name}")
    crd_api.delete_namespaced_custom_object(
        group="trainer.kubeflow.org", version="v1alpha1",
        namespace=namespace, plural="trainjobs", name=job_name
    )
