"""
main runner for the tuning, training, and testing of a Jax AI stack
model with JaxAI stack dataloader under SPMD paradigm with multi-host, multi-process
abilities.
"""

# Force Python to spawn clean workers instead of cloning the GPU context.
import multiprocessing as mp
import os
import sys
import logging

def init_multiprocessing():
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    try:
        mp.get_logger().setLevel(logging.DEBUG)
    except Exception:
        pass
    
    # Handle JAX platform naming as you did before
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        os.environ.pop("JAX_PLATFORM_NAME", None)
    
    current_ppath = os.environ.get("PYTHONPATH", "")
    new_paths = ":".join(sys.path)
    
    if current_ppath:
        # Avoid duplication if the path is already there
        os.environ["PYTHONPATH"] = f"{current_ppath}:{new_paths}"
    else:
        os.environ["PYTHONPATH"] = new_paths

import jax
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def safe_jax_init():
    """
    cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES", "UNDEFINED")
    logging.info( f"DEBUG: safe_jax_init running in process with CUDA_VISIBLE_DEVICES='{cuda_val}'")
    
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        logging.info("Skipping JAX distributed init for Grain worker process.")
        return
        
        # If already initialized (by the parent process before spawn), skip it.
    if jax.distributed.is_initialized():
        logging.info("JAX distributed already initialized, skipping.")
        return
    """
    if jax.distributed.is_initialized():
        return
    def get_process_id():
        import re
        #print(f'POD_NAME={os.environ.get("POD_NAME", "")}', flush=True)
        #print(f'MASTER_ADDR={os.environ.get("MASTER_ADDR", "")}', flush=True)
        #print(f'MASTER_PORT={os.environ.get("MASTER_PORT", "")}', flush=True)
        #print(f'RANK={os.environ.get("RANK", "")}', flush=True)
        #print(f'LOCAL_RANK={os.environ.get("LOCAL_RANK", "")}', flush=True)
        if "JAX_PROCESS_ID" in os.environ and os.environ.get("JAX_PROCESS_ID").strip() != "":
            return int(os.environ.get("JAX_PROCESS_ID"))
        pod_name = os.environ.get("POD_NAME", "")
        # Standard JobSet/StatefulSet pattern: name-replicatedjob-index-podindex
        # the last digit represents the pod index/rank
        match = re.search(r'-(\d+)-[a-z0-9]+$', pod_name)
        if match:
            return int(match.group(1))
        return 0  # Fallback
        
    try:
        if "LOCAL_SIMULATION" in os.environ and os.environ.get("LOCAL_SIMULATION") == "True":
            logging.info("🛠️ Detected local simulation. Applying manual jax initialization...")
            process_id = get_process_id()
            coord_addr = os.environ.get("JAX_COORDINATOR_ADDRESS")
            num_processes = int(os.environ.get("JAX_NUM_PROCESSES", 1))
            
            logging.info(f'process_id = {process_id} coord_addr={coord_addr} num_processes={num_processes}')
            
            jax.distributed.initialize(
                coordinator_address=coord_addr,
                num_processes=num_processes,
                process_id=process_id
            )
    
        # Try jax[k8s] auto-discovery if no coordinator is provided
        elif 'KUBERNETES_SERVICE_HOST' in os.environ:
            logging.info("Initializing JAX via jax[k8s] auto-discovery...")
            jax.distributed.initialize()
        
        # Standard local run (e.g., unit tests on your laptop)
        else:
            logging.info("No distributed environment detected. Running locally.")
    
    except RuntimeError as e:
        #absorb the error to avoid failure from more than one init attempt
        logging.exception(f'WARNING while trying to initialize JAX distributed: {e}')
        
    logging.info(f"jax devices={jax.devices()}, jax.loca_devices={jax.local_devices()}")
    
if __name__ == '__main__':
    
    from movie_lens_ranker.util import define_flags
    
    define_flags()
    
    #mp.log_to_stderr()
    #logger = mp.get_logger()
    #logger.setLevel(logging.DEBUG)
    
    init_multiprocessing()
    safe_jax_init()
    
    from movie_lens_ranker.app_runner_inner import main
    from absl import app
    
    app.run(main)
    
