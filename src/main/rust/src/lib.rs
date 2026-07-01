use pyo3::prelude::*;
use numpy::{PyArrayDyn};

pub mod user_history;
pub mod recommended_movies;
mod transforms;
mod util;
/*
#[pyfunction]
pub fn process_batch(py: Python, raw_bytes: Vec<&[u8]>) -> PyResult<Bound<PyArrayDyn<f32>>> {
    // processing logic goes here.
    // When finished, convert results to a NumPy array.
    // Example: ... (your logic)
    Ok(your_result.to_pyarray_bound(py))
}

#[pymodule]
fn prep_inputs_for_graphranker(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_batch, m)?)?;
    Ok(())
}
*/