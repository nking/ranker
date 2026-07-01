
/*
inputs:
     user_history
     max_history

     batch of ratings

 outputs:
     shape(batch_size, )
         'user_id',
         'movie_id',
         'rating',
         'timestamp',
     shape (batch_size, max_history)
            "history_movie_ids",
            "history_ratings",
            "history_length"

inputs:
    num_candidates
     user_history
     user_history_negatives
     shape(num_movies,)
         all_movie_ids
     shape(num_users, 1)
         recommended_movies
     shape(batch_size, )
         'user_id',
         'movie_id',
         'rating',
         'timestamp',
     shape (batch_size, max_history)
            "history_movie_ids",
            "history_ratings",
            "history_length"
outputs:

 */