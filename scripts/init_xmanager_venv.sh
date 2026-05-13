#!/bin/bash
conda create -q --name xmanager_py311 python=3.11 -y

conda activate xmanager_py311
# consider adding: wait
pip install xmanager==0.7.1
pip instal dotenv
pip install psycopg2-binary==2.9.5
pip install cloudpickle==3.1.2
