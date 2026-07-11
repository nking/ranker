//use pyo3::prelude::*;
use numpy::{PyArrayDyn};

pub mod user_history;
pub mod recommended_movies;
pub mod graph_builder;

pub mod embeddings_util;

pub mod util;
pub mod states;
pub mod main;
pub mod client;
pub mod orchestrator;
pub mod embeddings_ann;

pub mod pb {
    tonic::include_proto!("recommender");
}

/*
#[pyfunction]
pub fn process_batch(py: Python, raw_bytes: Vec<&[u8]>) -> PyResult<Bound<PyArrayDyn<f32>>> {
    // processing logic goes here.
    // When finished, convert results to a NumPy array.
    // Example: ... (your logic)
    Ok(your_result.to_pyarray_bound(py))
}

#[pymodule]
fn inference_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_batch, m)?)?;
    Ok(())
}
*/