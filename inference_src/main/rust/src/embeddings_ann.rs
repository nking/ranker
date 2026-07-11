use std::{fs};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use usearch::ffi::Matches;
use crate::embeddings_util::read_movie_embeddings;

const PERSISTED_INDEX_PATH: &'static str = "./movie_embeddings_indexer";

const TOPK : usize = 20;

pub struct Searcher {
    indexer : Index,
    movie_embeddings_catalog : Vec<f32>,
    num_catalog_movies : usize,
    embed_len : usize
}

impl Searcher {

    // static constructor
    pub fn new(movie_embeddings_uri: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let (movie_embeddings_catalog, num_movies, embed_len) = read_movie_embeddings(&movie_embeddings_uri);
        let indexer = Self::load(&movie_embeddings_catalog, embed_len);
        match indexer {
            Ok(indexer) => Ok(Self{
                indexer: indexer, movie_embeddings_catalog : movie_embeddings_catalog,
                num_catalog_movies: num_movies, embed_len : embed_len}),
            Err(error) => {panic!("There was a problem loading the indexes: {:?}", error)}
        }
    }

    pub fn get_num_catalog_movies(&self) -> usize {
        self.num_catalog_movies
    }
    pub fn get_embed_len(&self) -> usize {
        self.embed_len
    }

    pub fn restore(&self) -> Result<Index, Box<dyn std::error::Error>> {
        match fs::exists(&PERSISTED_INDEX_PATH) {
            Ok(true) => Ok(Index::restore(&PERSISTED_INDEX_PATH)?),
            Ok(false) => Err(Box::from("File does not exist.")),
            Err(e) => Err(Box::from("Error checking file: {e}")),
        }
    }

    pub fn get_movies_embedding_catalog_ref(&self) -> &Vec<f32> {
        &self.movie_embeddings_catalog
    }

    fn construct_index(embed_len : usize, capacity: usize) -> Result<Index, Box<dyn std::error::Error>> {
        let mut options = IndexOptions::default();
        options.dimensions = embed_len;
        options.metric = MetricKind::IP; // inner product
        options.quantization = ScalarKind::F32; // Use 32-bit floating point numbers
        options.connectivity = 16; //HNSW degree

        let mut index = Index::new(&options)?;
        index.reserve(capacity)?;

        Ok(index)
    }

    pub fn load(movie_embeddings_catalog : &[f32], embed_len : usize) -> Result<Index, Box<dyn std::error::Error>> {
        let num_catalog_movies = movie_embeddings_catalog.len() / embed_len;
        let mut index = Self::construct_index(embed_len, num_catalog_movies)?;
        for (id, chunk) in movie_embeddings_catalog.chunks_exact(embed_len).enumerate() {
            index.add(id as u64, chunk)?;
        }
        index.save(&PERSISTED_INDEX_PATH).expect("serialize index to disk failed");
        Ok(index)
    }

    pub fn search(&self, query: &[f32]) -> Result<Vec<Matches>, Box<dyn std::error::Error>> {
        let num_queries = query.len() / self.embed_len;
        let mut results: Vec<Matches> = Vec::with_capacity(num_queries);

        for i in 0..num_queries {
            let q = &query[i*self.embed_len .. (i+1)*self.embed_len];
            let r = self.indexer.search(&q, TOPK)?;
            results.push(r);
        }
        Ok(results)
    }
}

// all of the libraries have disk serialization:
//     USearch: Uses .save(path) and .load(path)
// FAISS IVF-PQ has incredible memory savings if trying to squeeze in more to a single RAM machine
//if on a disk with NVMe SSDs, USearch's memory-mapping can lead to good performance as long
//    as don't have frequent page reloads  (.e.g page faults from always new queries, random queries,...)