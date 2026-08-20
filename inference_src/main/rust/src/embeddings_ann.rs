use std::{path::{Path, PathBuf}};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use usearch::ffi::Matches;
use crate::embeddings_util::read_movie_embeddings;
use std::fmt;

pub struct Searcher {
    indexer : Index,
    movie_embeddings_catalog : Vec<f32>,
    num_catalog_movies : usize,
    embed_len : usize,
    num_candidates: usize,
    persisted_index_path: PathBuf,
}

impl fmt::Debug for Searcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Searcher")
            .field("indexer_size", &self.indexer.size())         // Current vector count
            .field("indexer_capacity", &self.indexer.capacity()) // Max allocated capacity
            .field("indexer_dimensions", &self.indexer.dimensions())
            .field("movie_embeddings_catalog", &self.movie_embeddings_catalog)
            .field("num_catalog_movies", &self.num_catalog_movies)
            .field("embed_len", &self.embed_len)
            .field("num_candidates", &self.num_candidates)
            .field("persisted_index_path", &self.persisted_index_path)
            .finish()
    }
}

impl Searcher {

    // static constructor
    pub fn new(movie_embeddings_uri: &str, num_candidates: usize, persisted_index_path: impl AsRef<Path>)
        -> Result<Self, Box<dyn std::error::Error+ Send + Sync>> {

        let (movie_embeddings_catalog, num_movies, embed_len) = read_movie_embeddings(&movie_embeddings_uri);

        println!("Movies embeddings embed_len: {}", embed_len);

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
            indexer: indexer,
            movie_embeddings_catalog : movie_embeddings_catalog,
            num_catalog_movies: num_movies,
            embed_len : embed_len,
            num_candidates : num_candidates,
            persisted_index_path: path_buf,
        })
    }

    /// build the ANN index for embedding vectors of length embed_len using inner_product
    /// (which results in cosine similarity distances if the embedding vectors ar already normalized)
    /// and save the index to path for fast re-loading abilities.
    ///
    /// # Arguments
    ///
    /// * `catalog`: the movie_embeddings catalog as a 1D array
    /// * `embed_len`: the length of each embedding vector.
    /// * `path`: where the index wll be persisted to
    ///
    /// returns: Result<Index, Box<dyn Error+Send+Sync, Global>>
    fn build_and_save(catalog: &[f32], embed_len: usize, path: &Path)
        -> Result<Index, Box<dyn std::error::Error + Send + Sync>> {

        let num_catalog_movies = catalog.len() / embed_len;
        let index: Index = Self::construct_index(embed_len, num_catalog_movies)?;

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

    ///construct a Userach index for inner_product.  If the embeddings catalog are normalized, the results
    /// are cosin_similarity distances, else the results are simply dot product distances.
    /// The Index is configured to use HNSW degree for connectivity and expect F32 type embedding vectors.
    /// # Arguments
    ///
    /// * `embed_len`: length of an embedding vector
    /// * `capacity`: the expected number of embeddings to be stored in the index.
    ///
    /// returns: Result<Index, Box<dyn Error+Send+Sync, Global>>
    fn construct_index(embed_len : usize, capacity: usize) -> Result<Index, Box<dyn std::error::Error + Send + Sync>> {
        let mut options = IndexOptions::default();
        options.dimensions = embed_len;
        options.metric = MetricKind::IP; // inner product
        options.quantization = ScalarKind::F32; // Use 32-bit floating point numbers
        options.connectivity = 16; //HNSW degree

        let index: Index = Index::new(&options)?;
        index.reserve(capacity)?;

        Ok(index)
    }

    /// given a flat vector of embeddings as query, search for the k nearest neighbors for each embedding
    /// in the query.  let n be the number of embeddings in query, then the search
    /// returns a Match structure having keys : Vec<k*n:u64> and distances: Vec<k*n:f32>
    ///
    /// # Arguments
    ///
    /// * `query`: a flat vector of user embeddings to search for ANNs for.
    /// * `k`: the number of approximate nearest neighbors to return for each embedding in query
    ///
    /// returns: Result<Matches, Box<dyn Error, Global>>
    ///   let n be the number of embeddings in query,
    //    returns a Match structure having keys : Vec<k*n:u64> and distances: Vec<k*n:f32>
    pub fn search(&self, query: &[f32], k : Option<usize>) -> Result<Matches, Box<dyn std::error::Error>> {
        let n = k.unwrap_or(self.num_candidates);
        let r = self.indexer.search(&query, n)?;
        Ok(r)
    }

    pub fn search_batch(&self, query: &[f32], k : Option<usize>) -> Result<Vec<Matches>, Box<dyn std::error::Error>> {
        let num_queries = query.len() / self.embed_len;
        let mut results: Vec<Matches> = Vec::with_capacity(num_queries);
        let n = k.unwrap_or(self.num_candidates);
        for i in 0..num_queries {
            let q = &query[i*self.embed_len .. (i+1)*self.embed_len];
            let r = self.indexer.search(&q, n)?;
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