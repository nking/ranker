#!/bin/bash
conda create -q --name xmanager_py311_sqla_1_4 python=3.11 -y

conda activate xmanager_py311_sqla_1_4
# consider adding: wait
pip install xmanager==0.7.1
pip instal dotenv
pip install psycopg2-binary==2.9.5
pip install SQLAlchemy==1.4.54
