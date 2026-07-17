use std::{path::{Path, PathBuf}};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use usearch::ffi::Matches;
use crate::embeddings_util::read_movie_embeddings;

pub struct Searcher {
    indexer : Index,
    movie_embeddings_catalog : Vec<f32>,
    num_catalog_movies : usize,
    embed_len : usize,
    top_k : usize,
    persisted_index_path: PathBuf,
}

impl Searcher {

    // static constructor
    pub fn new(movie_embeddings_uri: &str, top_k: usize, persisted_index_path: impl AsRef<Path>)
        -> Result<Self, Box<dyn std::error::Error+ Send + Sync>> {

        let (movie_embeddings_catalog, num_movies, embed_len) = read_movie_embeddings(&movie_embeddings_uri);
        let path_buf = persisted_index_path.as_ref().to_path_buf();
        let indexer = if path_buf.exists() {
            println!("Restoring index from {:?}", path_buf);
            let path_str = path_buf.to_str().ok_or("Path contains invalid UTF-8 characters")?;
            Index::restore(path_str)?
        } else {
            // Otherwise, build and save
            println!("Building new index at {:?}", path_buf);
            Self::build_and_save(&movie_embeddings_catalog, embed_len, &path_buf)?
        };
        Ok(Self{
            indexer: indexer, movie_embeddings_catalog : movie_embeddings_catalog,
            num_catalog_movies: num_movies, embed_len : embed_len, top_k: top_k,
            persisted_index_path: path_buf,
        })
    }

    fn build_and_save(catalog: &[f32], embed_len: usize, path: &Path)
        -> Result<Index, Box<dyn std::error::Error + Send + Sync>> {

        let num_catalog_movies = catalog.len() / embed_len;
        let mut index: Index = Self::construct_index(embed_len, num_catalog_movies)?;

        for (id, chunk) in catalog.chunks_exact(embed_len).enumerate() {
            index.add(id as u64, chunk)?;
        }

        index.save(path.to_str().unwrap())?;
        Ok(index)
    }

    pub fn restore(&self) -> Result<Index, Box<dyn std::error::Error>> {
        // 1. Check existence first
        if !self.persisted_index_path.exists() {
            return Err(Box::from("File does not exist."));
        }

        // 2. Convert to string safely (handle non-UTF-8 paths)
        let path_str = self.persisted_index_path
            .to_str()
            .ok_or("Path contains invalid UTF-8 characters")?;

        // 3. Restore and propagate errors with '?'
        Ok(Index::restore(path_str)?)
    }

    pub fn get_num_catalog_movies(&self) -> usize {
        self.num_catalog_movies
    }
    pub fn get_embed_len(&self) -> usize {
        self.embed_len
    }

    pub fn get_persisted_index_path(&self) -> PathBuf {
        self.persisted_index_path.clone()
    }

    pub fn get_movies_embedding_catalog_ref(&self) -> &Vec<f32> {
        &self.movie_embeddings_catalog
    }

    fn construct_index(embed_len : usize, capacity: usize) -> Result<Index, Box<dyn std::error::Error + Send + Sync>> {
        let mut options = IndexOptions::default();
        options.dimensions = embed_len;
        options.metric = MetricKind::IP; // inner product
        options.quantization = ScalarKind::F32; // Use 32-bit floating point numbers
        options.connectivity = 16; //HNSW degree

        let mut index: Index = Index::new(&options)?;
        index.reserve(capacity)?;

        Ok(index)
    }

    pub fn search(&self, query: &[f32]) -> Result<Matches, Box<dyn std::error::Error>> {
        let r = self.indexer.search(&query, self.top_k)?;
        Ok(r)
    }

    pub fn search_batch(&self, query: &[f32]) -> Result<Vec<Matches>, Box<dyn std::error::Error>> {
        let num_queries = query.len() / self.embed_len;
        let mut results: Vec<Matches> = Vec::with_capacity(num_queries);

        for i in 0..num_queries {
            let q = &query[i*self.embed_len .. (i+1)*self.embed_len];
            let r = self.indexer.search(&q, self.top_k)?;
            results.push(r);
        }
        Ok(results)
    }
}

// all of the ANN libraries have disk serialization:
//     USearch: Uses .save(path) and .load(path)
// FAISS IVF-PQ has incredible memory savings if trying to squeeze in more to a single RAM machine
//if on a disk with NVMe SSDs, USearch's memory-mapping can lead to good performance as long
//    as don't have frequent page reloads  (.e.g page faults from always new queries, random queries,...)