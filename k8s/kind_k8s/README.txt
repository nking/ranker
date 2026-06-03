in this directory are 3 scripts demonstrating different ways to run 
the container holding app_runner.py (the tune, train, test code).

note that the data and docker image preparation steps from 
the project_root README.md are needed before using these.

----------------------------------------------------------------

to run the container using multiple hosts and mulstiple processes
as a k82 StatfeulSet, invoked by kubectl commands in a bash script
and run locally on Kind:
  ./run_app.sh

to run the container using Kubeflow Trainer v2 as a jax-distributed
CustomResource, invoked by kubectl commands in a bash script
and run locally on Kind:
  ./run_app_trainer.sh

to run the container using Kubeflow Trainer v2 as a jax-distributed
CustomResource, invoked by commands from python Kubernetes client SDK
and run locally on Kind:
  python3 run_app_trainer.py

to run the container using Kubeflow Trainer v2 as a jax-distributed
CustomResource, compiled into Kubeflow Pipeline (KFP) yaml and run locally on Kind:
  python3 run_app_trainer_kfp.py

===============================================
versions:
   kind:
       see end of kind_notes.txt for kind versions

   kubernetes:
       36.0.1, 36.0.0, 35.0.0

  kubeflow trainer v2.2.0 (dynamic)

  python:
     >= 3.11

  KFP:
     2.16.1  or dynamic (https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/)

