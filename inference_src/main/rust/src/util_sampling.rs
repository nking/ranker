use rayon::prelude::*;

/// Fast, deterministic 64-bit PRNG for per-row parallel sampling (no `rand` version coupling).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Unbiased uniform integer in `0..bound`.
    #[inline]
    pub fn gen_range_usize(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        let b = bound as u64;
        let threshold = b.wrapping_neg() % b;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return (r % b) as usize;
            }
        }
    }
}

/// Row-wise intersection of `arr1` with `arr2`, preserving element order in `arr1`.
/// Stops scanning each row at the first occurrence of `pad_value` (if present).
pub fn row_wise_intersect(
    arr1: &[Vec<i32>],
    arr2: &[Vec<i32>],
    pad_value: i32,
) -> Result<Vec<Vec<i32>>, String> {
    if arr1.len() != arr2.len() {
        return Err(format!(
            "arr1 and arr2 must have the same number of rows ({} != {})",
            arr1.len(),
            arr2.len()
        ));
    }

    let result = arr1
        .par_iter()
        .zip(arr2.par_iter())
        .map(|(row1, row2)| {
            let mut out = Vec::with_capacity(row1.len());
            for &val1 in row1 {
                if val1 == pad_value {
                    break;
                }
                for &val2 in row2 {
                    if val2 == pad_value {
                        break;
                    }
                    if val1 == val2 {
                        out.push(val1);
                        break;
                    }
                }
            }
            out
        })
        .collect();

    Ok(result)
}

/// Row-wise set subtraction (`arr1 - arr2`), preserving element order in `arr1`.
/// Stops scanning each row at the first occurrence of `pad_value` (if present).
pub fn row_wise_sortedset_subtract(
    arr1: &[Vec<i32>],
    arr2: &[Vec<i32>],
    pad_value: i32,
) -> Result<Vec<Vec<i32>>, String> {
    if arr1.len() != arr2.len() {
        return Err(format!(
            "arr1 and arr2 must have the same number of rows ({} != {})",
            arr1.len(),
            arr2.len()
        ));
    }

    let result = arr1
        .par_iter()
        .zip(arr2.par_iter())
        .map(|(row1, row2)| {
            let mut out = Vec::with_capacity(row1.len());
            for &val1 in row1 {
                if val1 == pad_value {
                    break;
                }
                let mut exists_in_arr2 = false;
                for &val2 in row2 {
                    if val2 == pad_value {
                        break;
                    }
                    if val1 == val2 {
                        exists_in_arr2 = true;
                        break;
                    }
                }
                if !exists_in_arr2 {
                    out.push(val1);
                }
            }
            out
        })
        .collect();

    Ok(result)
}

/// Extracts non-pad, non-duplicate items from `source_row`, Fisher-Yates shuffles them,
/// and appends up to `max_take` items into `pool_row[write_idx..num_negatives]`.
pub fn _extract_shuffle_and_append(
    source_row: &[i32],
    max_take: usize,
    pool_row: &mut [i32],
    mut write_idx: usize,
    num_negatives: usize,
    pad_value: i32,
    buffer: &mut Vec<i32>,
    rng: &mut SplitMix64,
) -> usize {
    if max_take == 0 || write_idx >= num_negatives {
        return write_idx;
    }

    buffer.clear();
    for &val in source_row {
        if val == pad_value {
            break;
        }
        if !pool_row[..write_idx].contains(&val) && !buffer.contains(&val) {
            buffer.push(val);
        }
    }

    let count = buffer.len();
    if count == 0 {
        return write_idx;
    }

    // In-place O(N) Fisher-Yates shuffle
    for j in (1..count).rev() {
        let k = rng.gen_range_usize(j + 1);
        buffer.swap(j, k);
    }

    let available_space = num_negatives - write_idx;
    let items_to_take = max_take.min(count).min(available_space);
    if items_to_take > 0 {
        pool_row[write_idx..write_idx + items_to_take].copy_from_slice(&buffer[..items_to_take]);
        write_idx += items_to_take;
    }

    write_idx
}

/// Combines Type 1, 2, 3, and 4 negative pools per user using `target1`, `target2`, `target3`,
/// filling any remaining slots up to `num_negatives` from `arr4` (and padding with `pad_value`
/// if fewer than `num_negatives` unique items are available).
#[allow(clippy::too_many_arguments)]
pub fn build_negative_pool(
    arr1: &[Vec<i32>],
    arr2: &[Vec<i32>],
    arr3: &[Vec<i32>],
    arr4: &[Vec<i32>],
    target1: usize,
    target2: usize,
    target3: usize,
    num_negatives: usize,
    pad_value: i32,
    seed: u64,
) -> Result<Vec<Vec<i32>>, String> {
    let num_users = arr1.len();
    if arr2.len() != num_users || arr3.len() != num_users || arr4.len() != num_users {
        return Err(format!(
            "All 4 negative arrays must have the same number of rows ({}, {}, {}, {})",
            arr1.len(),
            arr2.len(),
            arr3.len(),
            arr4.len()
        ));
    }

    let pool = (0..num_users)
        .into_par_iter()
        .map(|i| {
            let mut rng = SplitMix64::new(seed.wrapping_add(i as u64));
            let mut pool_row = vec![pad_value; num_negatives];
            let mut buffer = Vec::new();
            let mut write_idx = 0;

            write_idx = _extract_shuffle_and_append(
                &arr1[i],
                target1,
                &mut pool_row,
                write_idx,
                num_negatives,
                pad_value,
                &mut buffer,
                &mut rng,
            );
            write_idx = _extract_shuffle_and_append(
                &arr2[i],
                target2,
                &mut pool_row,
                write_idx,
                num_negatives,
                pad_value,
                &mut buffer,
                &mut rng,
            );
            write_idx = _extract_shuffle_and_append(
                &arr3[i],
                target3,
                &mut pool_row,
                write_idx,
                num_negatives,
                pad_value,
                &mut buffer,
                &mut rng,
            );

            let remainder_needed = num_negatives.saturating_sub(write_idx);
            if remainder_needed > 0 {
                _extract_shuffle_and_append(
                    &arr4[i],
                    remainder_needed,
                    &mut pool_row,
                    write_idx,
                    num_negatives,
                    pad_value,
                    &mut buffer,
                    &mut rng,
                );
            }

            pool_row
        })
        .collect();

    Ok(pool)
}

/// Generates Type 4 "easy negatives" (`all_movie_ids - exclude[i]`) via rejection sampling.
/// Falls back to a linear scan if rejection sampling stalls on a small catalog.
pub fn generate_type_4_negatives(
    all_movie_ids: &[i32],
    exclude: &[Vec<i32>],
    n_negs: usize,
    pad_value: i32,
    seed: u64,
) -> Vec<Vec<i32>> {
    let n_movies = all_movie_ids.len();
    if n_movies == 0 || n_negs == 0 {
        return vec![vec![pad_value; n_negs]; exclude.len()];
    }

    let max_attempts = n_negs.saturating_mul(64).max(1024);

    exclude
        .par_iter()
        .enumerate()
        .map(|(i, excl_row)| {
            let mut rng = SplitMix64::new(seed.wrapping_add(i as u64));
            let mut row = vec![pad_value; n_negs];
            let mut count = 0;
            let mut attempts = 0;

            // Active slice of excluded items before trailing pad_value
            let excl_len = excl_row
                .iter()
                .position(|&v| v == pad_value)
                .unwrap_or(excl_row.len());
            let excl_active = &excl_row[..excl_len];

            while count < n_negs && attempts < max_attempts {
                attempts += 1;
                let candidate = all_movie_ids[rng.gen_range_usize(n_movies)];
                if candidate == pad_value {
                    continue;
                }
                if !excl_active.contains(&candidate) && !row[..count].contains(&candidate) {
                    row[count] = candidate;
                    count += 1;
                }
            }

            // Fallback scan if user watched almost the entire catalog (e.g. tiny unit tests)
            if count < n_negs {
                let start = rng.gen_range_usize(n_movies);
                for offset in 0..n_movies {
                    if count >= n_negs {
                        break;
                    }
                    let candidate = all_movie_ids[(start + offset) % n_movies];
                    if candidate != pad_value
                        && !excl_active.contains(&candidate)
                        && !row[..count].contains(&candidate)
                    {
                        row[count] = candidate;
                        count += 1;
                    }
                }
            }

            row
        })
        .collect()
}

/// In-place, row-wise Fisher-Yates shuffle on `candidates` and `labels` simultaneously
/// to preserve element alignment.
pub fn simultaneous_shuffle(
    candidates: &mut [Vec<i32>],
    labels: &mut [Vec<i32>],
    seed: u64,
) -> Result<(), String> {
    if candidates.len() != labels.len() {
        return Err(format!(
            "candidates and labels must have the same number of rows ({} != {})",
            candidates.len(),
            labels.len()
        ));
    }

    for (idx, (c, l)) in candidates.iter().zip(labels.iter()).enumerate() {
        if c.len() != l.len() {
            return Err(format!(
                "Row {idx} length mismatch between candidates ({}) and labels ({})",
                c.len(),
                l.len()
            ));
        }
    }

    candidates
        .par_iter_mut()
        .zip(labels.par_iter_mut())
        .enumerate()
        .for_each(|(i, (row_cand, row_label))| {
            let num_cols = row_cand.len();
            if num_cols <= 1 {
                return;
            }
            let mut rng = SplitMix64::new(seed.wrapping_add(i as u64));
            for j in (1..num_cols).rev() {
                let k = rng.gen_range_usize(j + 1);
                row_cand.swap(j, k);
                row_label.swap(j, k);
            }
        });

    Ok(())
}

/// Output of `build_graph_arrays`, matching the 9-tuple returned by `util_numba.build_graph_arrays`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphArrays {
    pub senders: Vec<i32>,
    pub receivers: Vec<i32>,
    pub edge_features: Vec<i32>,
    pub node_ids: Vec<i32>,
    pub node_labels: Vec<i32>,
    pub node_types: Vec<i32>,
    pub candidate_mask: Vec<bool>,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// Builds the star-graph node/edge arrays for a single user query (User <-> History + Candidates).
pub fn build_graph_arrays(
    user_id: i32,
    n_real_history: usize,
    candidate_ids: &[i32],
    history_ratings: &[i32],
    history_movie_ids: &[i32],
    labels: &[i32],
) -> GraphArrays {
    let n_candidates = candidate_ids.len();
    debug_assert!(history_ratings.len() >= n_real_history);
    debug_assert!(history_movie_ids.len() >= n_real_history);
    debug_assert_eq!(labels.len(), n_candidates);

    let total_nodes = 1 + n_real_history + n_candidates;
    let total_edges = n_real_history + n_candidates;

    let mut senders = vec![0i32; total_edges];
    let mut receivers = vec![0i32; total_edges];
    let mut edge_features = vec![0i32; total_edges];

    // History -> User (Inward)
    for i in 0..n_real_history {
        senders[i] = (i + 1) as i32;
        receivers[i] = 0;
        edge_features[i] = history_ratings[i];
    }

    // User -> Candidates (Outward)
    for i in 0..n_candidates {
        let idx = n_real_history + i;
        senders[idx] = 0;
        receivers[idx] = (1 + n_real_history + i) as i32;
        edge_features[idx] = 0;
    }

    // Nodes: [User, History..., Candidates...]
    let mut node_ids = Vec::with_capacity(total_nodes);
    node_ids.push(user_id);
    node_ids.extend_from_slice(&history_movie_ids[..n_real_history]);
    node_ids.extend_from_slice(candidate_ids);

    // Labels: 0 for user & history, `labels` for candidates
    let mut node_labels = vec![0i32; total_nodes];
    node_labels[1 + n_real_history..].copy_from_slice(labels);

    // Types: 1 = user/target, 2 = real history, 3 = candidate
    let mut node_types = Vec::with_capacity(total_nodes);
    node_types.push(1);
    node_types.resize(1 + n_real_history, 2);
    node_types.resize(total_nodes, 3);

    let mut candidate_mask = vec![false; total_nodes];
    for m in &mut candidate_mask[1 + n_real_history..] {
        *m = true;
    }

    GraphArrays {
        senders,
        receivers,
        edge_features,
        node_ids,
        node_labels,
        node_types,
        candidate_mask,
        total_nodes,
        total_edges,
    }
}