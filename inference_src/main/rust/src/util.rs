use std::sync::Arc;
use object_store::ObjectStore;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::local::LocalFileSystem;
use std::time::{SystemTime, UNIX_EPOCH};

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
/// for each row, coult the number of non-padded elements.
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

/// given and array of ids and an array of scores, both of same length, make an ascending
/// sort of both arrays by the values in scores array.
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
pub fn sort_by_scores(ids : &[i32], scores : &[f32]) -> (Vec<i32>, Vec<f32>) {
    let mut indices: Vec<usize> = (0..ids.len()).collect();
    indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let sorted_ids: Vec<i32> = indices.clone().into_iter().map(|idx| ids[idx]).collect();
    let sorted_scores: Vec<f32> = indices.into_iter().map(|idx| scores[idx]).collect();
    (sorted_ids, sorted_scores)
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
///
///       arr       = [1, 2, 3, 3, 4]
//          indexes =  0  1  2  3  4
//
//          ceil of srch=3 is index 3
//          ceil of srch=4 is index 4
//          ceil of srch=9 is index 5
//          ceil of srch=0 is index 0
///
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



