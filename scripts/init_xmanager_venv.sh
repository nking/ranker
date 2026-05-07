#!/bin/bash
conda create -q --name xmanager_py311 python=3.11 -y

conda activate xmanager_py311
# consider adding: wait
pip install xmanager==0.7.1
