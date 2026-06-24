#!/bin/bash
echo "invoke from project root directory"
mkdir TMP
cp requirements-kaggle-gpu.txt TMP/
cp pyproject.toml TMP/
cp -rf src TMP/
rm -rf TMP/src/drafts
rm -rf TMP/src/test
rm -rf TMP/src/main/resources
rm -rf TMP/src/main/python/movie_lens_ranker.egg-info
rm -rf TMP/src/main/python/movie_lens_ranker/__pycache__
cd TMP
tar -cvf ra.tar .
gzip ra.tar
rm -rf pypro* req* src
#rename to avoid parsing problems on kaggle side:
mv ra.tar.gz ra.tar.bin
#create dataset-metadata.json:
conda activate kaggle_py312
kaggle datasets init -p .

#then edit title and id for the name of the kaggle dataset:
# "title": "ranker-app3",
# "id": "your_kaggle_user_name/ranker-app3",
#
# create new:
#   kaggle datasets create -h
# or update:
#   kaggle datasets version -p . -m "Your update message"
