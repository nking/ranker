import asyncio
import os
from xmanager import xm
from xmanager import xm_local
from xmanager.contrib import parameter_controller
from dotenv import dotenv_values

import logging
from absl import logging as absl_logging, app

absl_logging.set_verbosity(absl_logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

def main(_):
    # This provides the event loop that was missing in your tt103 logs
    with xm_local.create_experiment(experiment_title='test_pattern') as experiment:
        
        num_trials = 4  # 20
        num_trials_per_worker = 2
        num_processes = 2
        print(f'JAX_NUM_PROCESSES={num_processes}', flush=True)
        docker_bridge_gateway = "172.17.0.1"
        env_config = {
            **dotenv_values(".env_unittests"),
            'PYTHONUNBUFFERED': '1',
            'JAX_NUM_PROCESSES': str(num_processes),
            'XLA_FLAGS': f'--xla_force_host_platform_device_count={num_processes}',
            'PYTHONIOENCODING': 'UTF-8',
            'TF_CPP_MIN_LOG_LEVEL': '0',
            'JAX_LOG_LEVEL': 'debug',
        }
        
        @parameter_controller.controller(
            executor=xm_local.Local(
                docker_options=xm_local.DockerOptions(
                    # This bridge is essential for local runs
                    volumes={'/var/run/docker.sock': '/var/run/docker.sock'}
                ),
            ),
            controller_args=env_config,
            controller_env_vars=env_config,
            package_path='.',
        )
        async def parameter_controller_example(experiment: xm.Experiment):
            executable = experiment.package([
                #xm.Packageable(
                #    executable_spec=xm.Container(image_path='docker.io/library/hello-world:latest'),
                #    executor_spec=xm_local.Local.Spec(),
                #),
                xm.Packageable(
                    executable_spec=xm.Binary(path='./hello_world'),
                    executor_spec=xm_local.Local.Spec(),
                ),
            ])[0]
            
            
            for i in range(3):
                print(f"Scheduling Job {i}...")
                job = xm.Job(
                    executable=executable,
                    executor=xm_local.Local(),
                    args={'trial_id': i, 'learning_rate': 0.01 * (i + 1)},
                    env_vars={f'print_{i}' : f"/tmp/print_{i}"},
                )
                handle = await experiment.add(job)
                await handle.wait_until_complete()
                print(f'finished job_{i}')
                
            print("All phase 1 jobs complete.")
        
        experiment.add(parameter_controller_example())  # pylint: disable=no-value-for-parameter
        
    # Trigger the async runner
    #xm_local.run(run_experiment)

if __name__ == '__main__':
    from absl import app
    
    app.run(main)