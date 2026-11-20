use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::reranker_prompt_formatter::RerankerPromptFormatter;

#[pyclass(name = "RerankerPromptFormatter", frozen)]
pub struct PyRerankerPromptFormatter {
    inner: RerankerPromptFormatter,
}

#[pymethods]
impl PyRerankerPromptFormatter {
    #[new]
    #[pyo3(signature = (
        movies_path,
        movie_tiers_path,
        query_yaml_path,
        passages_yaml_path,
        emb_placeholder_token = "<|vision_pad|>",
        max_history = 10,
        top_k = 20,
    ))]
    fn new(
        movies_path: &str,
        movie_tiers_path: &str,
        query_yaml_path: &str,
        passages_yaml_path: &str,
        emb_placeholder_token: &str,
        max_history: usize,
        top_k: usize,
    ) -> PyResult<Self> {
        let inner = RerankerPromptFormatter::from_yaml_files(
            movies_path,
            movie_tiers_path,
            query_yaml_path,
            passages_yaml_path,
            emb_placeholder_token,
            max_history,
            top_k,
        )
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    #[staticmethod]
    fn candidate_labels(top_k: usize) -> Vec<String> {
        RerankerPromptFormatter::candidate_labels(top_k)
    }

    #[pyo3(signature = (
        user_ids,
        movie_ids,
        timestamps,
        hist_movie_ids,
        hist_ratings,
        hist_timestamps,
        pad_value,
        candidate_ids,
        candidate_scores,
        user_embeddings,
        candidate_embeddings,
        indices = None,
        seed = 42,
        training = true,
        preserve_positive = true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn format_batch<'py>(
        &self,
        py: Python<'py>,
        user_ids: PyReadonlyArray1<'py, i32>,
        movie_ids: PyReadonlyArray1<'py, i32>,
        timestamps: PyReadonlyArray1<'py, i64>,
        hist_movie_ids: PyReadonlyArray2<'py, i32>,
        hist_ratings: PyReadonlyArray2<'py, f32>,
        hist_timestamps: PyReadonlyArray2<'py, i64>,
        pad_value: i32,
        candidate_ids: PyReadonlyArray2<'py, i32>,
        candidate_scores: PyReadonlyArray2<'py, f32>,
        user_embeddings: PyReadonlyArray2<'py, f32>,
        candidate_embeddings: PyReadonlyArray3<'py, f32>,
        indices: Option<Vec<u64>>,
        seed: u64,
        training: bool,
        preserve_positive: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch_size = user_ids.shape()[0];
        let h_width = hist_movie_ids.shape()[1];
        let num_candidates = candidate_ids.shape()[1];
        let dim = user_embeddings.shape()[1];

        if movie_ids.shape() != [batch_size] || timestamps.shape() != [batch_size] {
            return Err(PyValueError::new_err(format!(
                "movie_ids and timestamps must have shape [{batch_size}]"
            )));
        }
        if hist_movie_ids.shape() != [batch_size, h_width]
            || hist_ratings.shape() != [batch_size, h_width]
            || hist_timestamps.shape() != [batch_size, h_width]
        {
            return Err(PyValueError::new_err(format!(
                "history arrays must all have shape [{batch_size}, {h_width}]"
            )));
        }
        if candidate_ids.shape() != [batch_size, num_candidates]
            || candidate_scores.shape() != [batch_size, num_candidates]
            || user_embeddings.shape() != [batch_size, dim]
            || candidate_embeddings.shape() != [batch_size, num_candidates, dim]
        {
            return Err(PyValueError::new_err(format!(
                "candidate/embedding arrays must match batch={batch_size}, num_candidates={num_candidates}, dim={dim}"
            )));
        }
        if let Some(ref idx_vec) = indices {
            if idx_vec.len() != batch_size {
                return Err(PyValueError::new_err(format!(
                    "indices length ({}) must match batch_size ({batch_size})",
                    idx_vec.len()
                )));
            }
        }

        let u_ids = user_ids
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("user_ids not contiguous: {e}")))?
            .to_vec();
        let m_ids = movie_ids
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("movie_ids not contiguous: {e}")))?
            .to_vec();
        let ts = timestamps
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("timestamps not contiguous: {e}")))?
            .to_vec();
        let h_mids = hist_movie_ids
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("hist_movie_ids not contiguous: {e}")))?
            .to_vec();
        let h_rats = hist_ratings
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("hist_ratings not contiguous: {e}")))?
            .to_vec();
        let h_ts = hist_timestamps
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("hist_timestamps not contiguous: {e}")))?
            .to_vec();
        let cand_ids = candidate_ids
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("candidate_ids not contiguous: {e}")))?
            .to_vec();
        let cand_scores = candidate_scores
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("candidate_scores not contiguous: {e}")))?
            .to_vec();
        let user_embs = user_embeddings
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("user_embeddings not contiguous: {e}")))?
            .to_vec();
        let cand_embs = candidate_embeddings
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("candidate_embeddings not contiguous: {e}")))?
            .to_vec();

        let row_indices = indices.unwrap_or_default();

        // Release the GIL while rayon formats the batch across CPU cores.
        let out = py.detach(|| {
            self.inner.format_batch_from_padded_f32(
                &u_ids,
                &m_ids,
                &ts,
                &h_mids,
                &h_rats,
                &h_ts,
                h_width,
                pad_value,
                &cand_ids,
                &cand_scores,
                &user_embs,
                &cand_embs,
                &row_indices,
                seed,
                training,
                preserve_positive,
            )
        });

        let kept = out.kept_indices.len();
        let eff_k = num_candidates.min(self.inner.top_k());

        let ranker_embs_arr =
            PyArray1::from_vec(py, out.ranker_embs).reshape((kept, 1 + eff_k, dim))?;
        let cand_scores_arr =
            PyArray1::from_vec(py, out.candidate_scores).reshape((kept, eff_k))?;

        let dict = PyDict::new(py);
        dict.set_item("prompt", out.prompts)?;
        dict.set_item("completion", out.completions)?;
        dict.set_item("labels_list", out.labels_list)?;
        dict.set_item("ranker_embs", ranker_embs_arr)?;
        dict.set_item("candidate_scores", cand_scores_arr)?;
        dict.set_item("movie_id", out.movie_ids)?;
        dict.set_item("pos_tier", out.pos_tiers)?;
        dict.set_item("kept_indices", out.kept_indices)?;

        if training {
            let rels_arr = PyArray1::from_vec(py, out.label_rels).reshape((kept, eff_k))?;
            dict.set_item("label_rels_list", rels_arr)?;
        }

        Ok(dict)
    }
}

#[pymodule]
fn reranker_prompt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRerankerPromptFormatter>()?;
    Ok(())
}