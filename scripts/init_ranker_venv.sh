#!/bin/bash
conda create -q --name ranker_py312 python=3.12 -y

conda activate ranker_py312
# consider adding: wait
cd ..
#pip install -r requirements2.txt
pip install -c requirements.txt -e .
pip install equinox==0.13.2 alembic==1.18.4
conda deactivate
