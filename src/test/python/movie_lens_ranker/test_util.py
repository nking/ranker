from unittest import TestCase
import numpy as np
from movie_lens_ranker.util import read_embeddings_length, read_movie_tiers_uri, read_movies_array_record
from helper import *

class UtilTest(TestCase):
    
    def test_read_embeddings_length(self):
        movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        embed_dim = read_embeddings_length(movie_embeddings_uri)
        self.assertTrue(embed_dim >= 16 and embed_dim <= 64 and embed_dim % 8 == 0)

    def test_read_movie_tiers(self):
        movie_tiers_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_tiers-00000-of-00001.array_record")

        tiers, movie_offset, num_catalog_movies =  read_movie_tiers_uri(movie_tiers_uri)

        self.assertTrue(len(tiers), num_catalog_movies)
        self.assertTrue(len(tiers) > 0)
        self.assertTrue(movie_offset > 0)

        #look up a popular movie and assert it is in head
        movies_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movies-00000-of-00001.array_record")

        movie_dict = read_movies_array_record(movies_uri=movies_uri, ret_ids_only=False)

        inv_movie_dict = {v[0]:k for k, v in movie_dict.items()}
        test_movie_id = inv_movie_dict["Matrix, The (1999)"]
        self.assertIsNotNone(test_movie_id)

        test_movie_tier = tiers[test_movie_id - movie_offset]
        self.assertEqual(0, test_movie_tier)


