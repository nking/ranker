#adapted from https://github.com/google-deepmind/xmanager/blob/63a2ee86bca0fa847787f362f421b8bc4d2a6eb8/examples/parameter_controller/launcher.py#L86
# which has the following copyright:
#
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio

from xmanager import xm
from xmanager import xm_local
from xmanager.contrib import parameter_controller
from dotenv import dotenv_values

import logging
from absl import logging as absl_logging, app

absl_logging.set_verbosity(absl_logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

"""
1) start the postgres db server
2) invoke this at base of project directory, which is one level above this directory:
    xmanager launch xmngr_controller/test_pipeline_pattern.py -- --xm_db_yaml_config_path=db_config.yaml
"""

def main(_):
    
    with xm_local.create_experiment(experiment_title='test_pattern') as experiment:
        
        env_config = {
            **dotenv_values(".env_unittests"), #includes postgres user and password as needed
            'PYTHONUNBUFFERED': '1',
            'PYTHONIOENCODING': 'UTF-8',
            'JAX_LOG_LEVEL': 'debug',
        }
        
        @parameter_controller.controller(
            executor=xm_local.Local(
                docker_options=xm_local.DockerOptions(
                    # for local runs
                    volumes={'/var/run/docker.sock': '/var/run/docker.sock'}
                ),
            ),
            controller_args=env_config,
            controller_env_vars=env_config,
            package_path='.',
        )
        async def run_pipeline(experiment: xm.Experiment):
            executable = experiment.package([
                xm.Packageable(
                    executable_spec=xm.Container(image_path='docker.io/library/hello-world:latest'),
                    executor_spec=xm_local.Local.Spec(),
                ),
                #xm.Packageable(
                #    executable_spec=xm.Binary(path='./hello_world'), #a c++ compiled hello world that sleeps for 3 sec at end
                #    executor_spec=xm_local.Local.Spec(),
                #),
            ])[0]
            
            for i in range(3):
                logging.info(f"Scheduling Job {i}...")
                job = xm.Job(
                    executable=executable,
                    executor=xm_local.Local(),
                    args={},
                    env_vars={},
                )
                handle = await experiment.add(job)
                await handle.wait_until_complete()
                logging.info(f'finished job_{i}')
                
            logging.info("All phase 1 jobs complete.")
            
            for i in range(30, 34, 1):
                logging.info(f"Scheduling Job {i}...")
                job = xm.Job(
                    executable=executable,
                    executor=xm_local.Local(),
                    args={},
                    env_vars={},
                )
                handle = await experiment.add(job)
                await handle.wait_until_complete()
                logging.info(f'finished job_{i}')
                
            logging.info("All phase 2 jobs complete.")
            
        experiment.add(run_pipeline())
        
        logging.info("pipeline done.")
        
if __name__ == '__main__':
    app.run(main)