from unittest import TestCase
import numpy as np
from movie_lens_ranker.util import read_embeddings_length
from helper import *

class UtilTest(TestCase):
    
    def test_read_embeddings_length(self):
        movie_embeddings_uri = os.path.join(get_project_dir(),
            "src/test/resources/data/movie_emb-00000-of-00001.array_record")
        embed_dim = read_embeddings_length(movie_embeddings_uri)
        self.assertTrue(embed_dim >= 16 and embed_dim <= 64 and embed_dim % 8 == 0)

        