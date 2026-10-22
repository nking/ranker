use rustc_hash::FxHashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use object_store::ObjectStoreExt;
use arrow_array::{Int32Array, Int64Array};
use futures::stream::StreamExt;
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use std::cmp::{min};
use std::path::Path;
use futures::future::join_all;
use crate::util;

#[derive(Debug, Clone)]
pub struct UserMapEntry {
    pub movie_ids: Vec<i32>,
    pub ratings: Vec<i32>,
    pub timestamps: Vec<i64>,
}

/// Stores per-user interaction histories in flattened dense row-major arrays
/// of shape `(n_users + 1, max_history)` where outer row index `i` maps directly
/// to `user_id = i as i32`.
///
/// - Row `0` (`0..max_history`) is reserved for `user_id = 0` and out-of-range
///   cold-start user IDs (`<= 0` or `> n_users`), and is permanently filled with
///   `pad_value` (-1) and `ts_pad_value` (2524608000).
/// - Valid catalog users `1..=n_users` index directly into row `user_id as usize`.
#[derive(Debug, Clone)]
pub struct UserHistory {
    pub movie_ids: Vec<i32>,
    pub ratings: Vec<i32>,
    pub timestamps: Vec<i64>,
    pub n_users: usize,
    pub max_history: usize,
    pub pad_value: i32,
    pub ts_pad_value: i64,
}

impl UserHistory {
    /// Maps a requested `user_id` to its dense row index in `0..=self.n_users`.
    /// Any `user_id <= 0` or `user_id > self.n_users` is remapped to row `0`
    /// (the permanent padded sentinel row).
    #[inline]
    pub fn resolve_user_idx(&self, user_id: i32) -> usize {
        if user_id >= 1 && (user_id as usize) <= self.n_users {
            user_id as usize
        } else {
            0
        }
    }

    /*
     * Retrieves up to `requested_max_hist` movies and ratings for given users where
     * the movie's timestamp is strictly less than the provided target timestamp.
     * Shape of returned data is conceptually (user_ids.len(), requested_max_hist),
     * but flattened into a 1D Vec for performance.
     * resulting vector lengths are user_ids.len() * requested_max_history
     */
    pub fn get_history_before_timestamp(
        &self,
        user_ids: &[i32],
        timestamps: &[i64],
        requested_max_hist: usize
    ) -> (Vec<i32>, Vec<i32>) {
        let n_requests = user_ids.len();

        // Pre-allocate output arrays filled with pad_value (-1)
        let mut ret_movie_ids = vec![self.pad_value; n_requests * requested_max_hist];
        let mut ret_ratings: Vec<i32> = vec![self.pad_value; n_requests * requested_max_hist];

        if self.max_history == 0 || requested_max_hist == 0 {
            return (ret_movie_ids, ret_ratings);
        }

        // Iterate through each requested user_id and target timestamp
        for (i, (&target_uid, &target_ts)) in user_ids.iter().zip(timestamps.iter()).enumerate() {
            // Remap out-of-range or 0 user_ids to row 0 (padded sentinel row)
            let user_idx = self.resolve_user_idx(target_uid);

            let start_idx = user_idx * self.max_history;
            let end_idx = start_idx + self.max_history;

            let user_movies = &self.movie_ids[start_idx..end_idx];
            let user_ratings = &self.ratings[start_idx..end_idx];
            let user_timestamps = &self.timestamps[start_idx..end_idx];

            let mut count = 0;
            let out_offset = i * requested_max_hist;

            for j in 0..self.max_history {
                let ts = user_timestamps[j];

                // Stop early if we hit padding (including row 0 at j=0) or collected enough history
                if ts == self.ts_pad_value || count >= requested_max_hist {
                    break;
                }

                // Timestamps are sorted ascending; any ts >= target_ts means subsequent entries are also >= target_ts
                if ts < target_ts {
                    ret_movie_ids[out_offset + count] = user_movies[j];
                    ret_ratings[out_offset + count] = user_ratings[j];
                    count += 1;
                } else {
                    break;
                }
            }
        }

        debug_assert!(ret_movie_ids.len() == n_requests * requested_max_hist);
        debug_assert!(ret_ratings.len() == n_requests * requested_max_hist);

        (ret_movie_ids, ret_ratings)
    }

    pub fn get_history_count_before_timestamp(
        &self,
        user_ids: &[i32],
        timestamps: &[i64],
    ) -> Vec<usize> {
        let n_requests = user_ids.len();
        let mut ret_counts: Vec<usize> = vec![0; n_requests];

        if self.max_history == 0 {
            return ret_counts;
        }

        for (i, (&target_uid, &target_ts)) in user_ids.iter().zip(timestamps.iter()).enumerate() {
            // Remap out-of-range or 0 user_ids to row 0 (padded sentinel row)
            let user_idx = self.resolve_user_idx(target_uid);

            let start_idx = user_idx * self.max_history;
            let end_idx = start_idx + self.max_history;
            let user_timestamps = &self.timestamps[start_idx..end_idx];

            // Row 0 and empty users have user_timestamps[0] == self.ts_pad_value and exit immediately
            if user_timestamps.is_empty()
                || user_timestamps[0] == self.ts_pad_value
                || user_timestamps[0] > target_ts
            {
                continue;
            }

            let count = util::ceiling_search(user_timestamps, target_ts);
            ret_counts[i] = count + 1;
        }

        ret_counts
    }
}


pub async fn _testable_build_map_async(ratings_uris: &[&str]) -> (FxHashMap<i32, UserMapEntry>, usize) {

    let (map, longest_history)  = build_map_async(ratings_uris).await;

    (map, longest_history)
}

/// Given the URIs for the ratings files, creates a hashmap with keys of 1-based user_ids
/// and values of `UserMapEntry` where the vectors in `UserMapEntry` are sorted
/// ascending by timestamp.
pub async fn build_map_async(ratings_uris: &[&str]) -> (FxHashMap<i32, UserMapEntry>, usize) {
    let semaphore = Arc::new(Semaphore::new(4));
    let mut tasks = Vec::new();

    for ratings_uri in ratings_uris {
        let _path = Path::new(ratings_uri);
        let absolute_path = std::fs::canonicalize(_path)
            .expect("Failed to canonicalize path: File does not exist or access denied");

        let sem = semaphore.clone();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let mut local_map: FxHashMap<i32, UserMapEntry> = FxHashMap::default();

            let (storage, path) = util::parse_uri(&absolute_path.to_str().expect("Path is not valid UTF-8"));
            let meta = storage.head(&path).await.expect("Failed to get file metadata");
            let reader = ParquetObjectReader::new(storage, meta.location).with_file_size(meta.size);

            let builder = ParquetRecordBatchStreamBuilder::new(reader)
                .await
                .expect("Failed to create Parquet stream builder");

            let mut stream = builder.build().expect("Failed to build stream");

            while let Some(batch_result) = stream.next().await {
                let batch = batch_result.expect("Failed to read RecordBatch");

                let user_ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                let movie_ids = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
                let ratings = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
                let timestamps = batch.column(3).as_any().downcast_ref::<Int64Array>().unwrap();

                for i in 0..batch.num_rows() {
                    let u_id = user_ids.value(i);

                    let entry = local_map.entry(u_id).or_insert_with(|| UserMapEntry {
                        movie_ids: Vec::with_capacity(100),
                        ratings: Vec::with_capacity(100),
                        timestamps: Vec::with_capacity(100),
                    });

                    entry.movie_ids.push(movie_ids.value(i));
                    entry.ratings.push(ratings.value(i));
                    entry.timestamps.push(timestamps.value(i));
                }
            }

            local_map
        });

        tasks.push(task);
    }

    let results = join_all(tasks).await;

    let mut user_history_map: FxHashMap<i32, UserMapEntry> = FxHashMap::default();

    for res in results {
        let local_map = res.expect("A background task panicked");

        for (uid, mut local_entry) in local_map {
            let master_entry = user_history_map.entry(uid).or_insert_with(|| UserMapEntry {
                movie_ids: Vec::new(),
                ratings: Vec::new(),
                timestamps: Vec::new(),
            });

            master_entry.movie_ids.append(&mut local_entry.movie_ids);
            master_entry.ratings.append(&mut local_entry.ratings);
            master_entry.timestamps.append(&mut local_entry.timestamps);
        }
    }

    let mut max_len: usize = 0;

    for entry in user_history_map.values_mut() {
        let mut indices: Vec<usize> = (0..entry.timestamps.len()).collect();
        indices.sort_unstable_by_key(|&i| entry.timestamps[i]);

        entry.movie_ids = indices.iter().map(|&i| entry.movie_ids[i]).collect();
        entry.ratings = indices.iter().map(|&i| entry.ratings[i]).collect();
        entry.timestamps = indices.iter().map(|&i| entry.timestamps[i]).collect();

        let z = entry.timestamps.len();
        if z > max_len {
            max_len = z;
        }
    }

    (user_history_map, max_len)
}

/// Given the hashmap of user histories and a `max_history` length to extract,
/// creates dense flattened arrays of length `(n_users + 1) * max_history` indexed
/// directly by `user_id as usize`, where row `0` is the permanent padded sentinel row
/// and rows `1..=n_users` hold each user's up to `max_history` most recent interactions.
fn prepare_user_data(
    lookup: &FxHashMap<i32, UserMapEntry>,
    max_history: usize,
    pad_value: i32,
    ts_pad_value: i64,
) -> (Vec<i32>, Vec<i32>, Vec<i64>, usize) {
    let max_uid = lookup.keys().copied().max().unwrap_or(0);
    let n_users = if max_uid > 0 { max_uid as usize } else { 0 };
    let total_rows = n_users + 1;

    // Pre-allocate flattened 2D arrays of shape (n_users + 1, max_history).
    // Row 0 (and any user_id in 1..=n_users absent from `lookup`) stays filled with pad_value / ts_pad_value.
    let mut movie_ids: Vec<i32> = vec![pad_value; total_rows * max_history];
    let mut ratings: Vec<i32> = vec![pad_value; total_rows * max_history];
    let mut timestamps: Vec<i64> = vec![ts_pad_value; total_rows * max_history];

    for (&uid, entry) in lookup.iter() {
        assert!(
            uid >= 1,
            "Expected 1-based user_id >= 1 (index 0 is reserved for padded sentinel), found user_id={}",
            uid
        );
        let user_idx = uid as usize;

        let history_len = entry.movie_ids.len();
        let actual_history = min(history_len, max_history);

        if actual_history > 0 {
            let start_idx = history_len - actual_history;
            let offset = user_idx * max_history;

            movie_ids[offset..offset + actual_history]
                .copy_from_slice(&entry.movie_ids[start_idx..]);
            ratings[offset..offset + actual_history]
                .copy_from_slice(&entry.ratings[start_idx..]);
            timestamps[offset..offset + actual_history]
                .copy_from_slice(&entry.timestamps[start_idx..]);
        }
    }

    (movie_ids, ratings, timestamps, n_users)
}

/// Builds the dense `UserHistory` from the given list of URIs for the ratings parquet files.
pub async fn build_user_history(ratings_uris: &[&str], max_history: usize) -> UserHistory {
    let (map, longest_history) = build_map_async(ratings_uris).await;

    println!(
        "map length={}, longest history read has length={}",
        map.len(),
        longest_history
    );

    let pad_value: i32 = -1;
    let ts_pad_value: i64 = 2524608000; // year 2050
    let (movie_ids, ratings, timestamps, n_users) =
        prepare_user_data(&map, max_history, pad_value, ts_pad_value);

    UserHistory {
        movie_ids,
        ratings,
        timestamps,
        n_users,
        max_history,
        pad_value,
        ts_pad_value,
    }
}
