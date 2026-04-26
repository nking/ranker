#!/usr/bin/env bash
conda create -q --name xmanager_py311 python=3.11 -y

conda activate xmanager_py311
# consider adding: wait
pip install xmanager==0.7.1
pip install optuna==2.10.1 -c orchestrator_constraints.txt
