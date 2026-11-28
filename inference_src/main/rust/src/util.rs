use std::sync::Arc;
use object_store::ObjectStore;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use std::time::{SystemTime, UNIX_EPOCH};

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Helper to split URI into ObjectStore backend and ObjectStore Path
/// # Arguments
/// * `uri_string` - The uri of the file to read
pub fn parse_uri(uri_string: &str) -> (Arc<dyn ObjectStore>, object_store::path::Path) {
    if uri_string.starts_with("gs://") {
        let trimmed = uri_string.trim_start_matches("gs://");
        let parts: Vec<&str> = trimmed.splitn(2, '/').collect();
        let bucket_name = parts[0];
        let file_path = if parts.len() > 1 { parts[1] } else { "" };
        let gcs = GoogleCloudStorageBuilder::new()
            .with_bucket_name(bucket_name)
            .build()
            .expect("Failed to build GCS backend");
        (Arc::new(gcs), object_store::path::Path::from(file_path))
    } else {
        // Local file system
        // Strip file:// if present, otherwise assume absolute/relative path
        let path = uri_string.trim_start_matches("file://");
        (Arc::new(LocalFileSystem::new()), object_store::path::Path::from(path))
    }
}

/// given a 2D array flattened into 1D and the number of rows, and number of columns,
/// for each row, count the number of non-padded elements.
///
/// # Arguments
///
/// * `num_rows`: number of rows in the figurative 2D array
/// * `num_cols`: number of columns in the figurative 2D array
/// * `arr`: the figurative 2D array, flattened to 1D
/// * `pad_value`:  the value given to an empty element
///
/// returns: the non-padded lengths of each row as Vec<i32, Global>
///
/// # Examples
///
/// ```
///
/// ```
pub fn get_non_padded_lengths_of_flattened_arrays(num_rows:usize, num_cols:usize,
    arr : &Vec<i32>, pad_value:i32) -> Vec<usize> {

    let mut arr_length : Vec<usize> = Vec::with_capacity(num_rows);
    for j in 0..num_rows {
        let offset = j * num_cols;
        let mut len = 0;
        //number of elements in history_movie_ids that are not user_history.pad_value, breaking at first that is
        for k in offset..offset+num_cols {
            if arr[k] == pad_value {
                break;
            }
            len += 1;
        }
        arr_length.push(len);
    }
    arr_length

}


/// calculate the padding for graph components.  note that the number of local devices is considered in order
//     to make max_graphs divisible by jax.local_devices_count() to give an integer quotient.
///
/// # Arguments
///
/// * `batch_size`:
/// * `max_history`:
/// * `num_candidates`:
/// * `n_local_devices`:
///
/// returns: max_nodes,  max_edges, max_graphs as (usize, usize, usize)
///
/// # Examples
///
/// ```
///
/// ```
pub fn calc_number_jax_graph_components(batch_size: usize, max_history: usize,
    num_candidates: usize,  n_local_devices:usize) -> (usize, usize, usize) {

    let max_nodes = next_64(batch_size * (1 + max_history + num_candidates));
    let max_edges = next_64(batch_size * (max_history + num_candidates));

    //batch_size + 1 extra for every local device + padd up to integer quotient of local_devices
    let add_to = n_local_devices - (batch_size % n_local_devices);
    let max_graphs = batch_size + n_local_devices + add_to;

    (max_nodes,  max_edges, max_graphs)
}

pub fn next_64(x : usize) -> usize {
    64 * (1 + (x / 64))
}

/// given and array of ids and an array of scores which are flat arrays containing n_users*num_candidates
/// elements each, for each user, sort the movie_ids and scores in order descending scores
///
/// # Arguments
///
/// * `ids`:
/// * `scores`:
///
/// returns: (Vec<i32, Global>, Vec<f32, Global>)
///
/// # Examples
///
/// ```
///
/// ```
pub fn sort_in_place_by_desc_scores(movie_ids : &mut [i32], scores : &mut [f32], num_candidates: usize) {

    movie_ids.chunks_exact_mut(num_candidates)
        .zip(scores.chunks_exact_mut(num_candidates))
        .for_each(|(movie_chunk, score_chunk)| {

            let mut paired: Vec<(i32, f32)> = movie_chunk.iter()
                .zip(score_chunk.iter())
                .map(|(&id, &score)| (id, score))
                .collect();

            //  Sort the pairs by score in descending order
            // Using total_cmp prevents NaN panics and optimizes sorting speed
            paired.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

            // npack the sorted values back into the original slices in place
            for (i, (id, score)) in paired.into_iter().enumerate() {
                movie_chunk[i] = id;
                score_chunk[i] = score;
            }
        });

}

/// Given slices of ids and scores containing n_users * num_candidates elements,
/// sorts the candidates for each user in descending order by score, and extracts
/// the top_k results into new flattened vectors.
///
/// # Arguments
///
/// * `movie_ids`: Flattened slice of movie IDs
/// * `scores`: Flattened slice of scores
/// * `num_candidates`: The original number of candidates per user
/// * `top_k`: The maximum number of top candidates to keep per user
///
/// returns: (Vec<i32>, Vec<f32>) - The truncated, flattened top_k arrays
pub fn get_top_k_desc_scores(
    movie_ids: &[i32],
    scores: &[f32],
    num_candidates: usize,
    top_k: usize
) -> (Vec<i32>, Vec<f32>) {

    // Ensure top_k is strictly bounded by the actual number of candidates available
    let safe_top_k = top_k.min(num_candidates);
    let n_users = movie_ids.len() / num_candidates;

    // Pre-allocate the exact capacity needed for the truncated flat output
    let mut out_movie_ids = Vec::with_capacity(n_users * safe_top_k);
    let mut out_scores = Vec::with_capacity(n_users * safe_top_k);

    // OPTIMIZATION: Pre-allocate a single sorting buffer to avoid
    // re-allocating a new Vec for every single user in the batch.
    let mut paired: Vec<(i32, f32)> = Vec::with_capacity(num_candidates);

    // We use a standard for-loop instead of .for_each() so we can mutably borrow the 'paired' buffer
    for (movie_chunk, score_chunk) in movie_ids.chunks_exact(num_candidates).zip(scores.chunks_exact(num_candidates)) {

        // Clear the buffer and refill it with the current user's candidates
        paired.clear();
        paired.extend(
            movie_chunk.iter()
                .zip(score_chunk.iter())
                .map(|(&id, &score)| (id, score))
        );

        // Sort the pairs by score in descending order
        paired.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Extract only the top K elements and push them to the flattened outputs
        for &(id, score) in paired.iter().take(safe_top_k) {
            out_movie_ids.push(id);
            out_scores.push(score);
        }
    }

    (out_movie_ids, out_scores)
}

/// Extracts the top K candidates per user, sorting their scores, movie IDs,
/// and candidate representations in descending order of score.
/// runtime complexity is O(n_users * top_k * log(top_k))
///
/// # Arguments
/// * `movie_ids` - Shape: (n_users * num_candidates)
/// * `scores` - Shape: (n_users * num_candidates)
/// * `cand_reprs` - Shape: (n_users * num_candidates * out_features)
/// * `num_candidates` - Original number of candidates per user
/// * `top_k` - Number of candidates to keep (top_k < num_candidates)
/// * `out_features` - The embedding dimension of the candidate representations
///
/// Returns: (top_movie_ids, top_scores, top_cand_reprs)
pub fn get_top_k_desc_diagnostic(
    n_users: usize,
    movie_ids: &[i32],
    scores: &[f32],
    cand_reprs: &[f32],
    num_candidates: usize,
    top_k: usize,
    out_features: usize,
) -> (Vec<i32>, Vec<f32>, Vec<f32>) {

    // Safety check for empty inputs
    if num_candidates == 0 || top_k == 0 || n_users == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let safe_top_k = top_k.min(num_candidates);

    // Pre-allocate exact capacities
    let mut out_movie_ids = Vec::with_capacity(n_users * safe_top_k);
    let mut out_scores = Vec::with_capacity(n_users * safe_top_k);
    let mut out_cand_reprs = Vec::with_capacity(n_users * safe_top_k * out_features);

    // Pre-allocate a single index buffer to avoid allocating memory inside the loop
    let mut indices: Vec<usize> = (0..num_candidates).collect();

    for u in 0..n_users {
        // Isolate the slices belonging to this specific user
        let start = u * num_candidates;
        let end = start + num_candidates;

        let score_chunk = &scores[start..end];
        let movie_chunk = &movie_ids[start..end];

        // Isolate candidate representations for this user
        let c_start = start * out_features;
        let c_end = end * out_features;
        let cand_chunk = &cand_reprs[c_start..c_end];

        // Reset the index buffer for the current user
        for i in 0..num_candidates {
            indices[i] = i;
        }

        // Determine which slice of indices to iterate over
        let slice_to_iterate = if safe_top_k < num_candidates {
            // O(N) Partitioning: Mutates `indices` so the top elements are at the front
            indices.select_nth_unstable_by(safe_top_k - 1, |&a, &b| {
                score_chunk[b].total_cmp(&score_chunk[a]) // Descending order
            });

            // Capture the full top K slice (which includes both the left_slice AND the pivot)
            let top_k_indices = &mut indices[0..safe_top_k];

            // O(K log K): Sort ONLY the tiny top K slice
            top_k_indices.sort_unstable_by(|&a, &b| {
                score_chunk[b].total_cmp(&score_chunk[a])
            });
            top_k_indices
        } else {
            // Fallback: If top_k == num_candidates, sort the entire array
            indices.sort_unstable_by(|&a, &b| {
                score_chunk[b].total_cmp(&score_chunk[a])
            });
            &mut indices[..]
        };

        // Extract the sorted elements into the flat output vectors
        for &idx in slice_to_iterate.iter() {
            out_scores.push(score_chunk[idx]);
            out_movie_ids.push(movie_chunk[idx]);

            // Copy the full feature vector for this candidate using fast slice copying
            let cand_start = idx * out_features;
            out_cand_reprs.extend_from_slice(&cand_chunk[cand_start .. cand_start + out_features]);
        }
    }

    (out_movie_ids, out_scores, out_cand_reprs)
}

/// Preserves candidate index `0` (the positive item) at position `0` for each user,
/// and selects the `top_k - 1` highest-scoring negatives from indices `1..num_candidates`
/// in descending order of score.
///
/// Runtime complexity is O(n_users * (num_candidates + top_k * log(top_k))).
///
/// # Arguments
/// * `n_users` - Number of users in this batch
/// * `movie_ids` - Shape: `(n_users * num_candidates)`, with positive at col 0 per user
/// * `scores` - Shape: `(n_users * num_candidates)`
/// * `cand_reprs` - Shape: `(n_users * num_candidates * out_features)`
/// * `num_candidates` - Original number of candidates per user (1 positive + `num_candidates - 1` negatives)
/// * `top_k` - Total candidates to keep per user (1 positive + `top_k - 1` hardest negatives)
/// * `out_features` - Embedding dimension of each candidate representation
///
/// Returns: `(out_movie_ids, out_scores, out_cand_reprs)` with `safe_top_k` items per user.
pub fn get_pos_and_top_k_neg_desc_diagnostic(
    n_users: usize,
    movie_ids: &[i32],
    scores: &[f32],
    cand_reprs: &[f32],
    num_candidates: usize,
    top_k: usize,
    out_features: usize,
) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    if num_candidates == 0 || top_k == 0 || n_users == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    debug_assert_eq!(movie_ids.len(), n_users * num_candidates);
    debug_assert_eq!(scores.len(), n_users * num_candidates);
    debug_assert_eq!(cand_reprs.len(), n_users * num_candidates * out_features);

    let safe_top_k = top_k.min(num_candidates);
    let neg_to_keep = safe_top_k - 1;
    let n_neg_avail = num_candidates - 1;

    let mut out_movie_ids = Vec::with_capacity(n_users * safe_top_k);
    let mut out_scores = Vec::with_capacity(n_users * safe_top_k);
    let mut out_cand_reprs = Vec::with_capacity(n_users * safe_top_k * out_features);

    // Reusable buffer holding negative candidate indices `1..num_candidates`
    let mut neg_indices: Vec<usize> = (1..num_candidates).collect();

    for u in 0..n_users {
        let start = u * num_candidates;
        let end = start + num_candidates;

        let score_chunk = &scores[start..end];
        let movie_chunk = &movie_ids[start..end];
        let cand_chunk = &cand_reprs[start * out_features..end * out_features];

        // 1. Always preserve the positive candidate at position 0
        out_movie_ids.push(movie_chunk[0]);
        out_scores.push(score_chunk[0]);
        out_cand_reprs.extend_from_slice(&cand_chunk[0..out_features]);

        if neg_to_keep == 0 {
            continue;
        }

        // 2. Reset negative index buffer to `1..num_candidates`
        for (slot, idx) in neg_indices.iter_mut().enumerate() {
            *idx = slot + 1;
        }

        // 3. Select and sort the top `neg_to_keep` highest-scoring negatives
        let selected_neg_indices = if neg_to_keep < n_neg_avail {
            neg_indices.select_nth_unstable_by(neg_to_keep - 1, |&a, &b| {
                score_chunk[b].total_cmp(&score_chunk[a])
            });
            let top_negs = &mut neg_indices[0..neg_to_keep];
            top_negs.sort_unstable_by(|&a, &b| score_chunk[b].total_cmp(&score_chunk[a]));
            top_negs
        } else {
            neg_indices.sort_unstable_by(|&a, &b| score_chunk[b].total_cmp(&score_chunk[a]));
            &mut neg_indices[..]
        };

        for &idx in selected_neg_indices.iter() {
            out_movie_ids.push(movie_chunk[idx]);
            out_scores.push(score_chunk[idx]);
            let cand_start = idx * out_features;
            out_cand_reprs.extend_from_slice(&cand_chunk[cand_start..cand_start + out_features]);
        }
    }

    (out_movie_ids, out_scores, out_cand_reprs)
}

/// get timestamp in seconds for "now".  timestamp is the number of seconds since
/// January 1st, 1970 at UTC.
///
/// returns: u64
pub fn timestamp_now()-> u64 {

    let now = SystemTime::now();

    // Calculate duration since the epoch
    let since_the_epoch = now
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");

    // Get the timestamp in seconds
    let timestamp_secs = since_the_epoch.as_secs();

    // Get the timestamp in milliseconds
    //let timestamp_millis = since_the_epoch.as_millis();

    timestamp_secs
}

/// find the largest index in a sorted array arr where arr[index] <= srch
///
/// # Arguments
///
/// * `arr`: an array whose values are sorted ascending
/// * `srch`: the value to search for the largest index in arr for which the value <= srch
///
/// returns: usize
///
/// # Examples
/// ```text
///       arr       = [1, 2, 3, 3, 4]
///          ceil of srch=3 is index 3
///          ceil of srch=4 is index 4
///          ceil of srch=9 is index 5
///          ceil of srch=0 is index 0
/// ```
pub fn ceiling_search(arr: &[i64], srch: i64) -> usize {
    // Find the first index where the element is STRICTLY GREATER than `srch`.
    let p = arr.partition_point(|&x| x <= srch);

    // If the element just before this index equals our target,
    //    return that index (the last occurrence). Otherwise, return the insertion point.
    if p > 0 && arr[p - 1] == srch {
        p - 1
    } else {
        p
    }
}

pub(crate) fn check_path(path_str: &str, description: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = std::path::Path::new(path_str);
    if !path.exists() {
        // We use canonicalize() to show the absolute path, which helps
        // debugging when working with relative paths in different environments
        let full_path = path.canonicalize()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| path.to_string_lossy().to_string());

        eprintln!("CRITICAL ERROR: {} not found.", description);
        eprintln!("Looked for: {}", full_path);

        return Err(format!("Required file missing: {}", path_str).into());
    }
    Ok(())
}

#[test]
fn test_get_pos_and_top_k_neg_desc_diagnostic() {
    // 2 users, 5 candidates each (col 0 = positive, cols 1..5 = negatives), out_features = 2
    // User 0's positive (id 100) has the LOWEST score (-10.0) and must still stay at index 0.
    let movie_ids = vec![
        100, 101, 102, 103, 104,
        200, 201, 202, 203, 204,
    ];
    let scores = vec![
        -10.0, 0.2, 0.9, 0.1, 0.7,
        0.5, 0.8, 0.3, 0.95, 0.4,
    ];
    let cand_reprs: Vec<f32> = movie_ids
        .iter()
        .flat_map(|&id| [id as f32, -(id as f32)])
        .collect();

    let (out_ids, out_scores, out_reprs) =
        get_pos_and_top_k_neg_desc_diagnostic(2, &movie_ids, &scores, &cand_reprs, 5, 3, 2);

    // User 0: pos=100, top-2 negs = 102 (0.9), 104 (0.7)
    // User 1: pos=200, top-2 negs = 203 (0.95), 201 (0.8)
    assert_eq!(out_ids, vec![100, 102, 104, 200, 203, 201]);
    assert_eq!(out_scores, vec![-10.0, 0.9, 0.7, 0.5, 0.95, 0.8]);
    assert_eq!(
        out_reprs,
        vec![
            100.0, -100.0, 102.0, -102.0, 104.0, -104.0,
            200.0, -200.0, 203.0, -203.0, 201.0, -201.0,
        ]
    );

    // Edge case: top_k = 1 keeps only positives
    let (pos_only_ids, pos_only_scores, pos_only_reprs) =
        get_pos_and_top_k_neg_desc_diagnostic(2, &movie_ids, &scores, &cand_reprs, 5, 1, 2);
    assert_eq!(pos_only_ids, vec![100, 200]);
    assert_eq!(pos_only_scores, vec![-10.0, 0.5]);
    assert_eq!(pos_only_reprs, vec![100.0, -100.0, 200.0, -200.0]);
}

pub fn resolve_unique_file(
    dir_path: &Path,
    pattern: &str,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let (prefix, suffix) = match pattern.split_once('*') {
        Some((p, s)) => {
            if s.contains('*') {
                return Err(format!("Only one '*' wildcard is supported: '{pattern}'").into());
            }
            (p, Some(s))
        }
        None => (pattern, None),
    };

    let mut matched: Option<PathBuf> = None;

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };

        let is_match = match suffix {
            Some(suf) => {
                name.len() >= prefix.len() + suf.len()
                    && name.starts_with(prefix)
                    && name.ends_with(suf)
            }
            None => name == prefix,
        };
        if !is_match {
            continue;
        }

        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        if let Some(prev) = matched {
            return Err(format!(
                "Expected 1 file matching '{}' in '{}', found multiple ('{}', '{}')",
                pattern,
                dir_path.display(),
                prev.display(),
                path.display()
            )
                .into());
        }
        matched = Some(path);
    }

    matched.ok_or_else(|| {
        format!(
            "No file matching '{}' found in '{}'",
            pattern,
            dir_path.display()
        )
            .into()
    })
}