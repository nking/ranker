#!/bin/bash
conda create -q --name kubeflow_py312 python=3.12 -y

conda activate kubeflow_py312
# consider adding: wait
pip install kubeflow==  editing
