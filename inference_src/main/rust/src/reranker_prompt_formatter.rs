use std::io;

pub use reranker_prompt::movie_catalog::MovieCatalog;
pub use reranker_prompt::reranker_prompt_formatter::*;

use crate::movie_db::MovieDb;
use crate::user_history::UserHistory;

/// Constructs a `RerankerPromptFormatter` sharing the existing `Arc<MovieCatalog>` owned by `movie_db`.
pub fn from_movie_db_and_yaml_files(
    movie_db: &MovieDb,
    movie_tiers_path: &str,
    query_yaml_path: &str,
    passages_yaml_path: &str,
    emb_placeholder_token: impl Into<String>,
    max_history: usize,
    top_k: usize,
) -> io::Result<RerankerPromptFormatter> {
    RerankerPromptFormatter::from_catalog_and_yaml_files(
        movie_db.catalog(),
        movie_tiers_path,
        query_yaml_path,
        passages_yaml_path,
        emb_placeholder_token,
        max_history,
        top_k,
    )
}

/// Convenience helper for `inference_engine`: queries `UserHistory` for padded
/// `(movie_ids, ratings, timestamps)` ordered most-recent-first and delegates to
/// `reranker_prompt::RerankerPromptFormatter`.
/// # Arguments
///
/// * `formatter`:
/// * `user_history`: includes all history, not jsut the positive
/// * `user_ids`:
/// * `movie_ids`:
/// * `timestamps`:
/// * `candidate_ids_batch`:
/// * `candidate_scores_batch`:
/// * `user_embeddings_batch`:
/// * `candidate_embeddings_batch`:
/// * `preserve_positive`:
///
/// returns: FormattedBatch
///
/// # Examples
///
/// ```
///
/// ```
#[allow(clippy::too_many_arguments)]
pub fn format_batch_with_user_history(
    formatter: &RerankerPromptFormatter,
    user_history: &UserHistory,
    user_ids: &[i32],
    movie_ids: &[i32],
    timestamps: &[i64],
    candidate_ids_batch: &[i32],
    candidate_scores_batch: &[f32],
    user_embeddings_batch: &[f32],
    candidate_embeddings_batch: &[f32],
    preserve_positive: bool,
) -> FormattedBatch {
    let hist_width = formatter.max_history();
    let (hist_movie_ids, hist_ratings, hist_timestamps) = user_history
        .get_all_history_before_timestamp(user_ids, timestamps, hist_width, true);

    /*
    example row indices in training:
        pf = pq.ParquetFile(parquet_path)
        row_offset = 0
        for record_batch in pf.iter_batches(batch_size=2048):
            n = record_batch.num_rows
            indices = list(range(row_offset, row_offset + n))
            row_offset += n
            ...
            out = rust_fmt.format_batch(
                ...,
                indices=indices,
                seed=42,
                training=True,
                preserve_positive=True,
            )

     */
    formatter.format_batch_from_padded_i32(
        user_ids,
        movie_ids,
        timestamps,
        &hist_movie_ids,
        &hist_ratings,
        &hist_timestamps,
        hist_width,
        user_history.pad_value,
        candidate_ids_batch,
        candidate_scores_batch,
        user_embeddings_batch,
        candidate_embeddings_batch,
        &[], // use &[] for inference
        42,
        false,
        preserve_positive,
    )
}