user_recommendations*array_records record are a list of 
msgpack serialized rows containing
[int, list[int]] of user_id, recommended_movie_ids.
The files were written in repository
github.com/nking/retrieval.git
in file src/test/python/movie_lens_retrieval/write_user_recommendations_and_negatives.py

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

the model_repositories contents:
-  the bi-encoder query model was built from the TFX pipeline in
   the github repository 
   github.com/nking/recommender_systems.git
- the cross-encoder graph-ranker model was built from
  export_src/test/python/movie_lens_ranker_export/test_export.py

