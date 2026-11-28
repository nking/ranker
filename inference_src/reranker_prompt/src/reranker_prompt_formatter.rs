use std::fmt::Write as _;
use std::fs;
use std::io;
use std::sync::Arc;

use minijinja::{context, Environment};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::movie_catalog::MovieCatalog;
use crate::movie_tiers::load_from_file;

const DOW_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTH_ABBRS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const SEASONS: [&str; 12] = [
    "Winter", "Winter", "Spring", "Spring", "Spring", "Summer",
    "Summer", "Summer", "Fall",   "Fall",   "Fall",   "Winter",
];

#[derive(Debug, Clone, Deserialize)]
pub struct QueryYaml {
    pub system_template: String,
    pub template: String,
    pub completion_template: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PassagesYaml {
    pub history_template: String,
    pub candidate_template: String,
}

/// Raw history tuple `(movie_id, rating, timestamp)` ordered most-recent-first.
#[derive(Debug, Clone, Copy)]
pub struct HistoryItemRaw {
    pub movie_id: i32,
    pub rating: f32,
    pub timestamp: i64,
}

/// Contiguous batch output shared by Rust inference and Python training.
#[derive(Debug, Default)]
pub struct FormattedBatch {
    /// Original batch row indices retained (empty-history rows are skipped).
    pub kept_indices: Vec<usize>,
    pub prompts: Vec<String>,
    /// Populated when `training = true`, `None` for inference.
    pub completions: Vec<Option<String>>,
    pub labels_list: Vec<Vec<String>>,
    /// Populated when `training = true`: flat `[kept_len * effective_top_k]`.
    pub label_rels: Vec<i32>,
    /// Flat contiguous `[kept_len, 1 + effective_top_k, ranker_out_features]`.
    pub ranker_embs: Vec<f32>,
    /// Flat contiguous `[kept_len, effective_top_k]`.
    pub candidate_scores: Vec<f32>,
    /// Flat contiguous `[kept_len, effective_top_k]`.
    pub candidate_ids: Vec<i32>,
    pub movie_ids: Vec<i32>,
    pub pos_tiers: Vec<i32>,
    pub effective_top_k: usize,
    pub ranker_out_features: usize,
}

pub struct RerankerPromptFormatter {
    movie_tiers: FxHashMap<i32, i32>,
    movie_catalog: Arc<MovieCatalog>,
    top_k: usize,
    max_history: usize,
    emb_placeholder_token: String,
    candidate_labels: Vec<String>,
    rendered_system_prompt: String,
    completion_uses_join: bool,
    env: Environment<'static>,
}

impl RerankerPromptFormatter {
    pub fn from_yaml_files(
        movies_path: &str,
        movie_tiers_path: &str,
        query_yaml_path: &str,
        passages_yaml_path: &str,
        emb_placeholder_token: impl Into<String>,
        max_history: usize,
        top_k: usize,
    ) -> io::Result<Self> {
        let q_str = fs::read_to_string(query_yaml_path)?;
        let p_str = fs::read_to_string(passages_yaml_path)?;
        Self::from_yaml_strs(
            movies_path,
            movie_tiers_path,
            &q_str,
            &p_str,
            emb_placeholder_token,
            max_history,
            top_k,
        )
    }

    /// Reuses an existing `Arc<MovieCatalog>` (e.g., shared with `MovieDb`) and loads YAML + tiers.
    pub fn from_catalog_and_yaml_files(
        movie_catalog: Arc<MovieCatalog>,
        movie_tiers_path: &str,
        query_yaml_path: &str,
        passages_yaml_path: &str,
        emb_placeholder_token: impl Into<String>,
        max_history: usize,
        top_k: usize,
    ) -> io::Result<Self> {
        let q_str = fs::read_to_string(query_yaml_path)?;
        let p_str = fs::read_to_string(passages_yaml_path)?;
        let query_yaml: QueryYaml = serde_yaml::from_str(&q_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let passages_yaml: PassagesYaml = serde_yaml::from_str(&p_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let movie_tiers = load_from_file(movie_tiers_path)?;

        Self::from_loaded(
            movie_catalog,
            movie_tiers,
            query_yaml,
            passages_yaml,
            emb_placeholder_token,
            max_history,
            top_k,
        )
    }

    pub fn from_yaml_strs(
        movies_path: &str,
        movie_tiers_path: &str,
        query_yaml_str: &str,
        passages_yaml_str: &str,
        emb_placeholder_token: impl Into<String>,
        max_history: usize,
        top_k: usize,
    ) -> io::Result<Self> {
        let query_yaml: QueryYaml = serde_yaml::from_str(query_yaml_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let passages_yaml: PassagesYaml = serde_yaml::from_str(passages_yaml_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let movie_catalog = Arc::new(MovieCatalog::from_parquet(movies_path)?);
        let movie_tiers = load_from_file(movie_tiers_path)?;

        Self::from_loaded(
            movie_catalog,
            movie_tiers,
            query_yaml,
            passages_yaml,
            emb_placeholder_token,
            max_history,
            top_k,
        )
    }

    pub fn from_loaded(
        movie_catalog: Arc<MovieCatalog>,
        movie_tiers: FxHashMap<i32, i32>,
        query_yaml: QueryYaml,
        passages_yaml: PassagesYaml,
        emb_placeholder_token: impl Into<String>,
        max_history: usize,
        top_k: usize,
    ) -> io::Result<Self> {
        let candidate_labels = Self::candidate_labels(top_k);
        let first_label = candidate_labels.first().map(String::as_str).unwrap_or("A");
        let last_label = candidate_labels.last().map(String::as_str).unwrap_or("A");

        let mut env = Environment::new();
        env.add_template_owned("sys".to_string(), query_yaml.system_template.trim().to_string())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        env.add_template_owned("hist".to_string(), passages_yaml.history_template.trim().to_string())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        env.add_template_owned("cand".to_string(), passages_yaml.candidate_template.trim().to_string())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        env.add_template_owned("query".to_string(), query_yaml.template.trim_start().to_string())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        env.add_template_owned("comp".to_string(), query_yaml.completion_template.trim().to_string())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let rendered_system_prompt = env
            .get_template("sys")
            .unwrap()
            .render(context!(first_label => first_label, last_label => last_label))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .trim()
            .to_string();

        let completion_uses_join = query_yaml.completion_template.contains("join");

        Ok(Self {
            movie_tiers,
            movie_catalog,
            top_k,
            max_history,
            emb_placeholder_token: emb_placeholder_token.into(),
            candidate_labels,
            rendered_system_prompt,
            completion_uses_join,
            env,
        })
    }

    #[inline]
    pub fn movie_catalog(&self) -> Arc<MovieCatalog> {
        Arc::clone(&self.movie_catalog)
    }

    #[inline]
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    #[inline]
    pub fn max_history(&self) -> usize {
        self.max_history
    }

    pub fn candidate_labels(top_k: usize) -> Vec<String> {
        (0..top_k)
            .map(|idx| {
                if idx < 26 {
                    ((b'A' + idx as u8) as char).to_string()
                } else {
                    format!("C{idx}")
                }
            })
            .collect()
    }

    /// Formats a batch from flat padded history slices with `i32` ratings
    /// (matches `inference_engine::user_history::UserHistory::get_all_history_before_timestamp`).
    /// 
    /// param: row_indices:
    /// 
    ///     example row indices in training:
    ///         pf = pq.ParquetFile(parquet_path)
    ///         row_offset = 0
    ///         for record_batch in pf.iter_batches(batch_size=2048):
    ///             n = record_batch.num_rows
    ///             indices = list(range(row_offset, row_offset + n))
    ///             row_offset += n
    ///             ...
    ///             out = rust_fmt.format_batch(
    ///                 ...,
    ///                 indices=indices,
    ///                 seed=42,
    ///                 training=True,
    ///                 preserve_positive=True,
    ///             )
    /// 
    #[allow(clippy::too_many_arguments)]
    pub fn format_batch_from_padded_i32(
        &self,
        user_ids: &[i32],
        movie_ids: &[i32],
        timestamps: &[i64],
        hist_movie_ids: &[i32],
        hist_ratings: &[i32],
        hist_timestamps: &[i64],
        hist_width: usize,
        pad_value: i32,
        candidate_ids_batch: &[i32],
        candidate_scores_batch: &[f32],
        user_embeddings_batch: &[f32],
        candidate_embeddings_batch: &[f32],
        row_indices: &[u64],
        seed: u64,
        training: bool,
        preserve_positive: bool,
    ) -> FormattedBatch {
        let batch_size = user_ids.len();
        assert_eq!(hist_movie_ids.len(), batch_size * hist_width);
        assert_eq!(hist_ratings.len(), batch_size * hist_width);
        assert_eq!(hist_timestamps.len(), batch_size * hist_width);

        let mut histories = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let base = b * hist_width;
            let mut row_hist = Vec::with_capacity(hist_width.min(self.max_history));
            for j in 0..hist_width {
                let hm = hist_movie_ids[base + j];
                if hm == pad_value {
                    break;
                }
                row_hist.push(HistoryItemRaw {
                    movie_id: hm,
                    rating: hist_ratings[base + j] as f32,
                    timestamp: hist_timestamps[base + j],
                });
            }
            histories.push(row_hist);
        }

        self.format_batch_with_histories(
            &histories,
            user_ids,
            movie_ids,
            timestamps,
            candidate_ids_batch,
            candidate_scores_batch,
            user_embeddings_batch,
            candidate_embeddings_batch,
            row_indices,
            seed,
            training,
            preserve_positive,
        )
    }

    /// Formats a batch from flat padded history slices with `f32` ratings (used by PyO3 bindings).
    #[allow(clippy::too_many_arguments)]
    pub fn format_batch_from_padded_f32(
        &self,
        user_ids: &[i32],
        movie_ids: &[i32],
        timestamps: &[i64],
        hist_movie_ids: &[i32],
        hist_ratings: &[f32],
        hist_timestamps: &[i64],
        hist_width: usize,
        pad_value: i32,
        candidate_ids_batch: &[i32],
        candidate_scores_batch: &[f32],
        user_embeddings_batch: &[f32],
        candidate_embeddings_batch: &[f32],
        row_indices: &[u64],
        seed: u64,
        training: bool,
        preserve_positive: bool,
    ) -> FormattedBatch {
        let batch_size = user_ids.len();
        assert_eq!(hist_movie_ids.len(), batch_size * hist_width);
        assert_eq!(hist_ratings.len(), batch_size * hist_width);
        assert_eq!(hist_timestamps.len(), batch_size * hist_width);

        let mut histories = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let base = b * hist_width;
            let mut row_hist = Vec::with_capacity(hist_width.min(self.max_history));
            for j in 0..hist_width {
                let hm = hist_movie_ids[base + j];
                if hm == pad_value {
                    break;
                }
                row_hist.push(HistoryItemRaw {
                    movie_id: hm,
                    rating: hist_ratings[base + j],
                    timestamp: hist_timestamps[base + j],
                });
            }
            histories.push(row_hist);
        }

        self.format_batch_with_histories(
            &histories,
            user_ids,
            movie_ids,
            timestamps,
            candidate_ids_batch,
            candidate_scores_batch,
            user_embeddings_batch,
            candidate_embeddings_batch,
            row_indices,
            seed,
            training,
            preserve_positive,
        )
    }

    /// Core parallel formatter shared by Rust inference and Python training.
    #[allow(clippy::too_many_arguments)]
    pub fn format_batch_with_histories(
        &self,
        histories: &[Vec<HistoryItemRaw>],
        user_ids: &[i32],
        movie_ids: &[i32],
        timestamps: &[i64],
        candidate_ids_batch: &[i32],
        candidate_scores_batch: &[f32],
        user_embeddings_batch: &[f32],
        candidate_embeddings_batch: &[f32],
        row_indices: &[u64],
        seed: u64,
        training: bool,
        preserve_positive: bool,
    ) -> FormattedBatch {
        let batch_size = user_ids.len();
        if batch_size == 0 {
            return FormattedBatch::default();
        }

        assert_eq!(histories.len(), batch_size);
        assert_eq!(movie_ids.len(), batch_size);
        assert_eq!(timestamps.len(), batch_size);
        assert_eq!(candidate_ids_batch.len() % batch_size, 0);
        assert_eq!(user_embeddings_batch.len() % batch_size, 0);

        let num_candidates = candidate_ids_batch.len() / batch_size;
        let ranker_out_features = user_embeddings_batch.len() / batch_size;
        let effective_k = num_candidates.min(self.top_k);

        assert_eq!(candidate_scores_batch.len(), batch_size * num_candidates);
        assert_eq!(
            candidate_embeddings_batch.len(),
            batch_size * num_candidates * ranker_out_features
        );

        struct RowOut {
            orig_idx: usize,
            prompt: String,
            completion: Option<String>,
            label_rels: Vec<i32>,
            ranker_embs: Vec<f32>,
            cand_scores: Vec<f32>,
            cand_ids: Vec<i32>,
            movie_id: i32,
            pos_tier: i32,
        }

        let rows: Vec<RowOut> = (0..batch_size)
            .into_par_iter()
            .filter_map(|i| {
                let hist_items = &histories[i];
                if hist_items.is_empty() {
                    return None;
                }

                let mid = movie_ids[i];
                let q_ts = timestamps[i];
                let row_idx = row_indices.get(i).copied().unwrap_or(i as u64);

                let hist_tmpl = self.env.get_template("hist").unwrap();
                let cand_tmpl = self.env.get_template("cand").unwrap();
                let query_tmpl = self.env.get_template("query").unwrap();

                let mut title_buf = String::with_capacity(96);
                let mut genres_buf = String::with_capacity(64);
                let mut recency_buf = String::with_capacity(16);

                // 1. Render history block
                let mut history_block = String::with_capacity(hist_items.len() * 80);
                for (h_idx, h) in hist_items.iter().take(self.max_history).enumerate() {
                    if h_idx > 0 {
                        history_block.push('\n');
                    }
                    let (raw_title, raw_genres) =
                        self.movie_catalog.get_title_and_genres(h.movie_id);
                    title_buf.clear();
                    write_movielens_title(raw_title, &mut title_buf);
                    genres_buf.clear();
                    write_movielens_genres(raw_genres, &mut genres_buf);
                    recency_buf.clear();
                    write_relative_delta(q_ts, h.timestamp, &mut recency_buf);

                    let rating_str = format_rating(h.rating);
                    let line = hist_tmpl
                        .render(context!(
                            title => title_buf.as_str(),
                            genres => genres_buf.as_str(),
                            rating => rating_str,
                            recency => recency_buf.as_str(),
                        ))
                        .expect("history_template render failed");
                    history_block.push_str(line.trim());
                }

                // 2. Candidate truncation & positive-preservation swap
                let cand_base = i * num_candidates;
                let row_cand_ids = &candidate_ids_batch[cand_base..cand_base + num_candidates];
                let row_cand_scores =
                    &candidate_scores_batch[cand_base..cand_base + num_candidates];

                let mut slot_order: Vec<usize> = (0..effective_k).collect();
                if (training || preserve_positive) && num_candidates > effective_k {
                    if let Some(pos_idx) = row_cand_ids.iter().position(|&id| id == mid) {
                        if pos_idx >= effective_k {
                            slot_order[0] = pos_idx;
                        }
                    }
                }

                // 3. Training relevance labels & deterministic candidate permutation
                let mut label_rels = Vec::new();
                if training {
                    label_rels = vec![0i32; effective_k];
                    let pos_slot = slot_order
                        .iter()
                        .position(|&s| row_cand_ids[s] == mid)
                        .unwrap_or(0);
                    label_rels[pos_slot] = 1;

                    let mut perm: Vec<usize> = (0..effective_k).collect();
                    deterministic_shuffle(&mut perm, seed.wrapping_add(row_idx));

                    let old_slots = slot_order.clone();
                    let old_rels = label_rels.clone();
                    for k in 0..effective_k {
                        slot_order[k] = old_slots[perm[k]];
                        label_rels[k] = old_rels[perm[k]];
                    }
                }

                // 4. Render candidates block
                let mut candidates_block = String::with_capacity(effective_k * 80);
                for (label_idx, &slot) in slot_order.iter().enumerate() {
                    if label_idx > 0 {
                        candidates_block.push('\n');
                    }
                    let c_mid = row_cand_ids[slot];
                    let (raw_title, raw_genres) = self.movie_catalog.get_title_and_genres(c_mid);
                    title_buf.clear();
                    write_movielens_title(raw_title, &mut title_buf);
                    genres_buf.clear();
                    write_movielens_genres(raw_genres, &mut genres_buf);

                    let line = cand_tmpl
                        .render(context!(
                            cand_emb => self.emb_placeholder_token.as_str(),
                            label => self.candidate_labels[label_idx].as_str(),
                            title => title_buf.as_str(),
                            genres => genres_buf.as_str(),
                        ))
                        .expect("candidate_template render failed");
                    candidates_block.push_str(line.trim());
                }

                // 5. Render full prompt
                let mut query_time = String::with_capacity(32);
                write_query_time(q_ts, &mut query_time);

                let mut prompt = query_tmpl
                    .render(context!(
                        system_prompt => self.rendered_system_prompt.as_str(),
                        query_time => query_time.as_str(),
                        user_emb => self.emb_placeholder_token.as_str(),
                        history_block => history_block.as_str(),
                        candidates_block => candidates_block.as_str(),
                    ))
                    .expect("query_template render failed");
                if !prompt.ends_with('\n') {
                    prompt.push('\n');
                }

                // 6. Pack embeddings & candidate arrays
                let mut ranker_embs = Vec::with_capacity((1 + effective_k) * ranker_out_features);
                let u_start = i * ranker_out_features;
                ranker_embs.extend_from_slice(
                    &user_embeddings_batch[u_start..u_start + ranker_out_features],
                );

                let cand_emb_row_base = i * num_candidates * ranker_out_features;
                let mut cand_ids = Vec::with_capacity(effective_k);
                let mut cand_scores = Vec::with_capacity(effective_k);
                for &slot in &slot_order {
                    let c_start = cand_emb_row_base + slot * ranker_out_features;
                    ranker_embs.extend_from_slice(
                        &candidate_embeddings_batch[c_start..c_start + ranker_out_features],
                    );
                    cand_ids.push(row_cand_ids[slot]);
                    cand_scores.push(row_cand_scores[slot]);
                }

                // 7. Render completion (training only)
                let completion = if training {
                    let mut order_idx: Vec<usize> = (0..effective_k).collect();
                    order_idx.sort_by(|&a, &b| {
                        label_rels[b]
                            .cmp(&label_rels[a])
                            .then_with(|| cand_scores[b].total_cmp(&cand_scores[a]))
                    });
                    let ranked_labels: Vec<String> = order_idx
                        .iter()
                        .map(|&idx| self.candidate_labels[idx].clone())
                        .collect();
                    let comp_tmpl = self.env.get_template("comp").unwrap();
                    let rendered = if self.completion_uses_join {
                        comp_tmpl
                            .render(context!(ranked_labels => ranked_labels))
                            .expect("completion_template render failed")
                    } else {
                        let joined = ranked_labels.join(",");
                        comp_tmpl
                            .render(context!(ranked_labels => joined))
                            .expect("completion_template render failed")
                    };
                    Some(rendered)
                } else {
                    None
                };

                let pos_tier = self
                    .movie_tiers
                    .get(&mid)
                    .copied()
                    .unwrap_or_else(|| mid.rem_euclid(3));

                Some(RowOut {
                    orig_idx: i,
                    prompt,
                    completion,
                    label_rels,
                    ranker_embs,
                    cand_scores,
                    cand_ids,
                    movie_id: mid,
                    pos_tier,
                })
            })
            .collect();

        let kept = rows.len();
        let mut out = FormattedBatch {
            kept_indices: Vec::with_capacity(kept),
            prompts: Vec::with_capacity(kept),
            completions: Vec::with_capacity(kept),
            labels_list: Vec::with_capacity(kept),
            label_rels: if training {
                Vec::with_capacity(kept * effective_k)
            } else {
                Vec::new()
            },
            ranker_embs: Vec::with_capacity(kept * (1 + effective_k) * ranker_out_features),
            candidate_scores: Vec::with_capacity(kept * effective_k),
            candidate_ids: Vec::with_capacity(kept * effective_k),
            movie_ids: Vec::with_capacity(kept),
            pos_tiers: Vec::with_capacity(kept),
            effective_top_k: effective_k,
            ranker_out_features,
        };

        let row_labels = self.candidate_labels[..effective_k].to_vec();
        for r in rows {
            out.kept_indices.push(r.orig_idx);
            out.prompts.push(r.prompt);
            out.completions.push(r.completion);
            out.labels_list.push(row_labels.clone());
            if training {
                out.label_rels.extend_from_slice(&r.label_rels);
            }
            out.ranker_embs.extend_from_slice(&r.ranker_embs);
            out.candidate_scores.extend_from_slice(&r.cand_scores);
            out.candidate_ids.extend_from_slice(&r.cand_ids);
            out.movie_ids.push(r.movie_id);
            out.pos_tiers.push(r.pos_tier);
        }

        out
    }
}

#[inline]
fn format_rating(r: f32) -> String {
    if (r - r.trunc()).abs() < 1e-6 {
        format!("{}", r as i32)
    } else {
        format!("{r}")
    }
}

#[inline]
fn deterministic_shuffle(slice: &mut [usize], mut state: u64) {
    let mut next_u64 = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    for i in (1..slice.len()).rev() {
        let j = (next_u64() as usize) % (i + 1);
        slice.swap(i, j);
    }
}

/// Converts `"Matrix, The (1999)"` -> `"The Matrix (1999)"` without regex allocation.
#[inline]
pub fn write_movielens_title(raw_title: &str, out: &mut String) {
    let s = raw_title.trim();
    let bytes = s.as_bytes();
    if bytes.len() >= 6 && bytes[bytes.len() - 1] == b')' && bytes[bytes.len() - 6] == b'(' {
        let year_digits = &bytes[bytes.len() - 5..bytes.len() - 1];
        if year_digits.iter().all(u8::is_ascii_digit) {
            let year_part = &s[s.len() - 6..];
            let before_year = s[..s.len() - 6].trim_end();
            if let Some((prefix, after_comma)) = before_year.rsplit_once(',') {
                let article = after_comma.trim_start();
                if matches!(article, "The" | "A" | "An") {
                    out.push_str(article);
                    out.push(' ');
                    out.push_str(prefix);
                    out.push(' ');
                    out.push_str(year_part);
                    return;
                }
            }
        }
    }
    out.push_str(s);
}

/// Converts `"Action|Sci-Fi|Thriller"` -> `"Action, Sci-Fi, Thriller"`.
#[inline]
pub fn write_movielens_genres(raw_genres: &str, out: &mut String) {
    let mut parts = raw_genres.trim().split('|');
    if let Some(first) = parts.next() {
        out.push_str(first);
        for part in parts {
            out.push_str(", ");
            out.push_str(part);
        }
    }
}

/// Formats UTC Unix timestamp as `"Fall, Sat Night (Oct)"`.
#[inline]
pub fn write_query_time(ts: i64, out: &mut String) {
    let days_since_epoch = ts.div_euclid(86_400);
    let sec_of_day = ts.rem_euclid(86_400);

    let dow_idx = (days_since_epoch + 3).rem_euclid(7) as usize;
    let hour = sec_of_day / 3_600;
    let tod = match hour {
        5..=11 => "Morning",
        12..=16 => "Afternoon",
        17..=21 => "Evening",
        _ => "Night",
    };

    let month_idx = (month_from_days(days_since_epoch) - 1) as usize;
    let _ = write!(
        out,
        "{}, {} {} ({})",
        SEASONS[month_idx], DOW_NAMES[dow_idx], tod, MONTH_ABBRS[month_idx]
    );
}

#[inline]
fn month_from_days(days_since_epoch: i64) -> u32 {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    m as u32
}

/// Formats `(query_ts - hist_ts)` into compact relative units (`"20m ago"`, `"10d ago"`).
#[inline]
pub fn write_relative_delta(query_ts: i64, hist_ts: i64, out: &mut String) {
    let diff_sec = (query_ts - hist_ts).max(0);
    if diff_sec < 3_600 {
        let _ = write!(out, "{}m ago", (diff_sec / 60).max(1));
    } else if diff_sec < 86_400 {
        let _ = write!(out, "{}h ago", diff_sec / 3_600);
    } else if diff_sec < 86_400 * 30 {
        let _ = write!(out, "{}d ago", diff_sec / 86_400);
    } else {
        let _ = write!(out, "{}mo ago", diff_sec / (86_400 * 30));
    }
}