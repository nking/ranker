use std::sync::Arc;

use rayon::prelude::*;

use crate::recommended_movies::RecommendedMovies;
use crate::user_history::UserHistory;
use crate::util_sampling::{
    build_negative_pool, generate_type_4_negatives, row_wise_intersect,
    row_wise_sortedset_subtract, simultaneous_shuffle, SplitMix64,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingStrategy {
    MixedHard,
    WorstRecommended,
}

impl SamplingStrategy {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "mixed_hard" => Ok(Self::MixedHard),
            "worst_recommended" => Ok(Self::WorstRecommended),
            other => Err(format!("Unsupported SamplingStrategy: {other}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledCandidates {
    /// Shape: `(batch_size, num_candidates)`
    pub candidate_ids: Vec<Vec<i32>>,
    /// Shape: `(batch_size, num_candidates)` — `1` at the positive item, `0` for negatives
    pub labels: Vec<Vec<i32>>,
}

/// Converts a flat row-major `(n_rows * row_width)` buffer into `n_rows` trimmed `Vec<i32>` rows
/// with all `pad_value` entries removed.
fn flat_to_rows(flat: &[i32], n_rows: usize, row_width: usize, pad_value: i32) -> Vec<Vec<i32>> {
    if n_rows == 0 || row_width == 0 {
        return vec![Vec::new(); n_rows];
    }
    debug_assert_eq!(flat.len(), n_rows * row_width);
    flat.chunks_exact(row_width)
        .map(|row| row.iter().copied().filter(|&v| v != pad_value).collect())
        .collect()
}

/// Reads each user's valid movie IDs strictly `< target_ts` directly from `UserHistory`'s
/// read-only dense arrays without allocating an intermediate `(batch_size * max_history)`
/// padded buffer. Optionally prepends `prepend_ids[i]` at column 0 of each row.
fn extract_history_before_timestamp(
    history: &UserHistory,
    user_ids: &[i32],
    timestamps: &[i64],
    prepend_ids: Option<&[i32]>,
) -> Vec<Vec<i32>> {
    let max_hist = history.max_history;
    let pad_val = history.pad_value;
    let ts_pad = history.ts_pad_value;

    if max_hist == 0 {
        return match prepend_ids {
            Some(ids) => ids.iter().map(|&id| vec![id]).collect(),
            None => vec![Vec::new(); user_ids.len()],
        };
    }

    user_ids
        .par_iter()
        .zip(timestamps.par_iter())
        .enumerate()
        .map(|(i, (&uid, &target_ts))| {
            let user_idx = history.resolve_user_idx(uid);
            let start = user_idx * max_hist;
            let end = start + max_hist;

            let user_movies = &history.movie_ids[start..end];
            let user_timestamps = &history.timestamps[start..end];

            let cutoff_ts = target_ts.min(ts_pad);
            let valid_end = user_timestamps.partition_point(|&ts| ts < cutoff_ts);

            let extra = usize::from(prepend_ids.is_some());
            let mut row = Vec::with_capacity(extra + valid_end);
            if let Some(ids) = prepend_ids {
                row.push(ids[i]);
            }
            for &m in &user_movies[..valid_end] {
                if m != pad_val {
                    row.push(m);
                }
            }
            row
        })
        .collect()
}

/// Port of `HardNegativeSamplingTransform.py` wired directly to shared read-only `UserHistory`
/// handles and owned catalog / `RecommendedMovies` data.
pub struct HardNegativeSamplingTransform {
    history_lookup: Arc<UserHistory>,
    history_lookup_disliked: Arc<UserHistory>,
    all_movie_ids: Vec<i32>,
    recommendations: RecommendedMovies,
    num_candidates: usize,
    strategy: SamplingStrategy,
    num_1_negatives: usize,
    num_2_negatives: usize,
    num_3_negatives: usize,
    pad_value: i32,
}

impl HardNegativeSamplingTransform {
    pub fn new(
        history_lookup: Arc<UserHistory>,
        history_lookup_disliked: Arc<UserHistory>,
        all_movie_ids: Vec<i32>,
        recommendations: RecommendedMovies,
        num_candidates: usize,
        strategy: SamplingStrategy,
    ) -> Result<Self, String> {
        if num_candidates == 0 {
            return Err("num_candidates must be >= 1".to_string());
        }
        if all_movie_ids.is_empty() {
            return Err("all_movie_ids must not be empty".to_string());
        }

        let pad_value = history_lookup.pad_value;
        if history_lookup_disliked.pad_value != pad_value {
            return Err(format!(
                "history_lookup_disliked.pad_value ({}) must match history_lookup.pad_value ({})",
                history_lookup_disliked.pad_value, pad_value
            ));
        }
        if recommendations.pad_value != pad_value {
            return Err(format!(
                "recommendations.pad_value ({}) must match history_lookup.pad_value ({})",
                recommendations.pad_value, pad_value
            ));
        }
        if recommendations.num_users > 0
            && recommendations.movie_ids.len() % recommendations.num_users != 0
        {
            return Err(format!(
                "recommendations.movie_ids.len() ({}) is not divisible by num_users ({})",
                recommendations.movie_ids.len(),
                recommendations.num_users
            ));
        }

        let n_negs = num_candidates.saturating_sub(1).max(1);
        let num_1_negatives = ((n_negs as f64) * 0.15).round_ties_even() as usize;
        let num_2_negatives = ((n_negs as f64) * 0.55).round_ties_even() as usize;
        let num_3_negatives = ((n_negs as f64) * 0.15).round_ties_even() as usize;

        Ok(Self {
            history_lookup,
            history_lookup_disliked,
            all_movie_ids,
            recommendations,
            num_candidates,
            strategy,
            num_1_negatives,
            num_2_negatives,
            num_3_negatives,
            pad_value,
        })
    }

    pub fn pad_value(&self) -> i32 {
        self.pad_value
    }

    pub fn num_candidates(&self) -> usize {
        self.num_candidates
    }

    pub fn num_negatives(&self) -> usize {
        self.num_candidates - 1
    }

    #[inline]
    fn rec_row_length(&self) -> usize {
        if self.recommendations.num_users == 0 {
            0
        } else {
            self.recommendations.movie_ids.len() / self.recommendations.num_users
        }
    }

    /// Equivalent to Python `RecommendedMovies.get_all_recommended_movies(user_id=user_ids)`.
    /// Uses `RecommendedMovies`'s 1-based user indexing (`user_idx = (uid - 1) as usize`).
    fn get_all_recommended_movies_rows(&self, user_ids: &[i32]) -> Vec<Vec<i32>> {
        let row_len = self.rec_row_length();
        if row_len == 0 {
            return vec![Vec::new(); user_ids.len()];
        }

        user_ids
            .par_iter()
            .map(|&uid| {
                if uid < 1 || (uid as usize) > self.recommendations.num_users {
                    return Vec::new();
                }
                let user_idx = (uid - 1) as usize;
                let start = user_idx * row_len;
                let end = start + row_len;
                self.recommendations.movie_ids[start..end]
                    .iter()
                    .copied()
                    .filter(|&m| m != self.pad_value)
                    .collect()
            })
            .collect()
    }

    /// Shared internal sampler used by both `sample_candidates_with_pos` and
    /// `sample_negative_candidates`. Returns `(negatives, rng)` where `negatives` has shape
    /// `(user_ids.len(), self.num_candidates - 1)` with any remaining `pad_value` slots
    /// backfilled from `self.all_movie_ids`.
    fn sample_negatives_internal(
        &self,
        user_ids: &[i32],
        positive_ids: Option<&[i32]>,
        timestamps: &[i64],
        seed: u64,
    ) -> Result<(Vec<Vec<i32>>, SplitMix64), String> {
        let n_users = user_ids.len();
        if timestamps.len() != n_users {
            return Err(format!(
                "Batch length mismatch: user_ids={}, timestamps={}",
                n_users,
                timestamps.len()
            ));
        }
        if let Some(pos_ids) = positive_ids {
            if pos_ids.len() != n_users {
                return Err(format!(
                    "Batch length mismatch: user_ids={}, positive_ids={}",
                    n_users,
                    pos_ids.len()
                ));
            }
        }

        let mut rng = SplitMix64::new(seed);
        if n_users == 0 {
            return Ok((Vec::new(), rng));
        }

        let n_negs = self.num_candidates - 1;
        if n_negs == 0 {
            return Ok((vec![Vec::new(); n_users], rng));
        }

        // Read directly from shared read-only UserHistory buffers in one pass (no 2048-wide temp alloc)
        let watched_and_positive = extract_history_before_timestamp(
            &self.history_lookup,
            user_ids,
            timestamps,
            positive_ids,
        );
        let movie_histories_disliked = extract_history_before_timestamp(
            &self.history_lookup_disliked,
            user_ids,
            timestamps,
            None,
        );

        let seed1 = rng.next_u64() & 0x7FFF_FFFF;
        let type_4_neg = generate_type_4_negatives(
            &self.all_movie_ids,
            &watched_and_positive,
            n_negs,
            self.pad_value,
            seed1,
        );

        let seed2 = rng.next_u64() & 0x7FFF_FFFF;
        let rec_row_len = self.rec_row_length();

        let mut negatives = match self.strategy {
            SamplingStrategy::WorstRecommended => {
                let worst_k = (n_negs * 2).min(rec_row_len);
                let worst_unseen = if worst_k > 0 {
                    let flat = self.recommendations.get_lowest_scoring_unseen_movies(
                        user_ids,
                        timestamps,
                        worst_k,
                    );
                    flat_to_rows(&flat, n_users, worst_k, self.pad_value)
                } else {
                    vec![Vec::new(); n_users]
                };

                let worst_filtered = row_wise_sortedset_subtract(
                    &worst_unseen,
                    &watched_and_positive,
                    self.pad_value,
                )?;
                let empty_pool = vec![Vec::new(); n_users];
                build_negative_pool(
                    &worst_filtered,
                    &empty_pool,
                    &empty_pool,
                    &type_4_neg,
                    n_negs,
                    0,
                    0,
                    n_negs,
                    self.pad_value,
                    seed2,
                )?
            }
            SamplingStrategy::MixedHard => {
                let all_recommendations = self.get_all_recommended_movies_rows(user_ids);
                let unseen_k = (n_negs * 2).max(20).min(rec_row_len);
                let unseen_recommendations = if unseen_k > 0 {
                    let flat = self.recommendations.get_unseen_movies(
                        user_ids,
                        timestamps,
                        unseen_k,
                    );
                    flat_to_rows(&flat, n_users, unseen_k, self.pad_value)
                } else {
                    vec![Vec::new(); n_users]
                };

                // Type 1: "hard negatives" = recommended intersect user's disliked (< timestamp)
                let type_1_neg = row_wise_intersect(
                    &all_recommendations,
                    &movie_histories_disliked,
                    self.pad_value,
                )?;

                // Type 2: "implicit hard negatives" = unseen recommended - (watch history U positive)
                let type_2_neg = row_wise_sortedset_subtract(
                    &unseen_recommendations,
                    &watched_and_positive,
                    self.pad_value,
                )?;

                // Type 3: "out of distr negatives" = disliked - all recommended
                let type_3_neg = row_wise_sortedset_subtract(
                    &movie_histories_disliked,
                    &all_recommendations,
                    self.pad_value,
                )?;

                build_negative_pool(
                    &type_1_neg,
                    &type_2_neg,
                    &type_3_neg,
                    &type_4_neg,
                    self.num_1_negatives,
                    self.num_2_negatives,
                    self.num_3_negatives,
                    n_negs,
                    self.pad_value,
                    seed2,
                )?
            }
        };

        // Backfill any leftover pad_values from the catalog, avoiding the positive item and duplicates
        let n_catalog = self.all_movie_ids.len();
        for (row_idx, neg_row) in negatives.iter_mut().enumerate() {
            let excluded_pos = positive_ids.map(|ids| ids[row_idx]);
            for slot_idx in 0..neg_row.len() {
                if neg_row[slot_idx] == self.pad_value {
                    let mut chosen = self.all_movie_ids[rng.gen_range_usize(n_catalog)];
                    for _ in 0..64 {
                        let cand = self.all_movie_ids[rng.gen_range_usize(n_catalog)];
                        if cand != self.pad_value
                            && excluded_pos != Some(cand)
                            && !neg_row.contains(&cand)
                        {
                            chosen = cand;
                            break;
                        }
                    }
                    neg_row[slot_idx] = chosen;
                }
            }
        }

        Ok((negatives, rng))
    }

    /// Samples only the `num_candidates - 1` negative candidate IDs per user.
    ///
    /// Pass `positive_ids = Some(ids)` to ensure the ground-truth positive item is excluded
    /// from the sampled negatives, or `None` to exclude only prior watch history.
    pub fn sample_negative_candidates(
        &self,
        user_ids: &[i32],
        positive_ids: Option<&[i32]>,
        timestamps: &[i64],
        seed: u64,
        shuffle_negatives: bool,
    ) -> Result<Vec<Vec<i32>>, String> {
        let (mut negatives, mut rng) =
            self.sample_negatives_internal(user_ids, positive_ids, timestamps, seed)?;

        if shuffle_negatives && !negatives.is_empty() {
            let seed3 = rng.next_u64() & 0x7FFF_FFFF;
            negatives.par_iter_mut().enumerate().for_each(|(i, row)| {
                let num_cols = row.len();
                if num_cols <= 1 {
                    return;
                }
                let mut row_rng = SplitMix64::new(seed3.wrapping_add(i as u64));
                for j in (1..num_cols).rev() {
                    let k = row_rng.gen_range_usize(j + 1);
                    row.swap(j, k);
                }
            });
        }

        Ok(negatives)
    }

    /// Maps a batch of `(user_ids, positive_ids, timestamps)` to `candidate_ids` and `labels`
    /// of shape `(batch_size, num_candidates)` containing the positive item and `num_candidates - 1`
    /// sampled negatives.
    pub fn sample_candidates_with_pos(
        &self,
        user_ids: &[i32],
        positive_ids: &[i32],
        timestamps: &[i64],
        seed: u64,
        shuffle_candidates: bool,
    ) -> Result<SampledCandidates, String> {
        let (negatives, mut rng) =
            self.sample_negatives_internal(user_ids, Some(positive_ids), timestamps, seed)?;

        let n_users = user_ids.len();
        if n_users == 0 {
            return Ok(SampledCandidates {
                candidate_ids: Vec::new(),
                labels: Vec::new(),
            });
        }

        let n_catalog = self.all_movie_ids.len();
        let mut candidate_ids: Vec<Vec<i32>> = positive_ids
            .iter()
            .zip(negatives.into_iter())
            .map(|(&pos_id, neg_row)| {
                let mut row = Vec::with_capacity(self.num_candidates);
                let safe_pos = if pos_id == self.pad_value {
                    self.all_movie_ids[rng.gen_range_usize(n_catalog)]
                } else {
                    pos_id
                };
                row.push(safe_pos);
                row.extend(neg_row);
                row
            })
            .collect();

        let mut labels: Vec<Vec<i32>> = (0..n_users)
            .map(|_| {
                let mut row = vec![0i32; self.num_candidates];
                row[0] = 1;
                row
            })
            .collect();

        if shuffle_candidates {
            let seed3 = rng.next_u64() & 0x7FFF_FFFF;
            simultaneous_shuffle(&mut candidate_ids, &mut labels, seed3)?;
        }

        Ok(SampledCandidates {
            candidate_ids,
            labels,
        })
    }
}