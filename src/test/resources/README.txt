user_recommendations*array_records record are a list of 
msgpack serialized rows containing
[int, list[int]] of user_id, recommended_movie_ids.
The files were written in repository
github.com/nking/retrieval.git
in file src/test/python/movie_lens_retrieval/test_Retriever.py
in method test_eval_all 

ratings_part_[1|2].array_record are a list of
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

[user|movie]_embeddings.array_record is a list of
msgpack serialized rows, each containing an int id and a list of embeddings.
The files were written in repository
github.com/nking/retrieval.git
in file src/test/python/movie_lens_retrieval/test_Retriever.py
in method test_write_movie_embeddings 
