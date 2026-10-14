the dir data/bie-encoder contains tower-products for versions
of the trained tower that coincide with the 
model_repositories/saved_model_formats/bi-encoder/<model>/ versions

to load tmp_mounts/fake_gcs_data for the trained model specific products
cp -f data/bi-encoder/<version>/* to the tmp_mounts/fake_gcs_data directory
before running tests and zipping it for Kaggle.

user_recommendations*array_records record are a list of 
msgpack serialized rows containing
[int, list[int]] of user_id, recommended_movie_ids.
The files were written in repository
github.com/nking/retrieval.git
in file src/test/python/movie_lens_retrieval/write_tower_products.py
There is a metadata file to track the two-tower model which made them.

<user|movie>_emb*array_records record are a list of 
msgpack serialized rows containing
[int, list[float]] of id, embedding vector.
The files were written in repository
github.com/nking/retrieval.git
in file src/test/python/movie_lens_retrieval/write_tower_products.py
There is a metadata file to track the two-tower model which made them.

ratings_<*>.array_record are a list of
msgpack serialized rows containing
[int, int, int, int] of user_id, movie_id, rating, timestamp
The files that were written in repository
github.com/nking/recommender_systems.git
in file src/test/python/movie_lens_tfx/WriteRankerInputArrayRecords.py

movie_ids.array_record is a list of
msgpack serialized rows, each containing an int movie_id
The file that was written in repository
github.com/nking/recommender_systems.git
in file src/test/python/movie_lens_tfx/WriteRankerInputArrayRecords.py

movie_tiers files are 
   movie_id and tier
where tier are zones of the movie frequency distribution
from the train ratings partition.
tier=0 is head of distribution (the 20% most frequently rated),
tier=1 is torso of distribution 
tier=2 is tail of distribution (the 20 % least frequently rated),
the files were created in respository
github.com/nking/recommender_systems.git
in file src/test/python/movie_lens_tfx/WriteRankerInputArrayRecords.py

the model_repositories contents:
-  the bi-encoder query model was built from the TFX pipeline in
   the github repository 
   github.com/nking/recommender_systems.git
- the cross-encoder graph-ranker model was built from
  export_src/test/python/movie_lens_ranker_export/test_export.py

the checkpoint-bucket contents were trained on Kaggle T4x2 GPUs:
   https://www.kaggle.com/code/nicholeasuniquename/ranker-cross-encoder-w-gatv2

users.bin was written in the project 
github.com/nking/recommender_systems.git
in file src/test/python/movie_lens_tfx/write_users_bin.py
it holds gender, age, occupation and the implied user_id is index+1

