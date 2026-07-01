use rustc_hash::FxHashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::runtime::Runtime;
use object_store::ObjectStoreExt;
use arrow_array::{Int32Array, Int64Array};
use futures::stream::StreamExt;
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use std::cmp::min;
use futures::future::join_all;
use crate::util;

pub struct UserMapEntry {
    pub movie_ids: Vec<i32>,
    pub ratings: Vec<i32>,
    pub timestamps: Vec<i64>,
}

pub struct UserHistory {
    user_ids : Vec<i32>,
    movie_ids : Vec<i32>,
    ratings : Vec<i32>,
    timestamps : Vec<i64>,
    max_history : usize,
    pad_value : i32,
    ts_pad_value : i64
}

impl UserHistory {
    /*
     * Retrieves up to `requested_max_hist` movies and ratings for given users where
     * the movie's timestamp is strictly less than the provided target timestamp.
     * Shape of returned data is conceptually (user_ids.len(), requested_max_hist),
     * but flattened into a 1D Vec for performance.   so resulting vector lengths are user_ids.len() * requested_max_history
     */
    pub fn get_history_before_timestamp(
        &self,
        user_ids: Vec<i32>,
        timestamps: Vec<i64>,
        requested_max_hist: usize
    ) -> (Vec<i32>, Vec<i32>) {

        let n_requests = user_ids.len();

        // Pre-allocate the output arrays filled entirely with the pad_value.
        // This instantly replaces the need for np.full(...)
        let mut ret_movie_ids = vec![self.pad_value; n_requests * requested_max_hist];
        let mut ret_ratings = vec![self.pad_value; n_requests * requested_max_hist];

        // Iterate through each requested user and their target timestamp
        for (i, (&target_uid, &target_ts)) in user_ids.iter().zip(timestamps.iter()).enumerate() {

            // np.searchsorted -> Rust's native binary_search
            if let Ok(user_idx) = self.user_ids.binary_search(&target_uid) {

                // Calculate where this specific user's data starts in our flattened arrays
                let start_idx = user_idx * self.max_history;
                let end_idx = start_idx + self.max_history;

                // Create lightweight slices (views) into the data
                let user_movies = &self.movie_ids[start_idx..end_idx];
                let user_ratings = &self.ratings[start_idx..end_idx];
                let user_timestamps = &self.timestamps[start_idx..end_idx];

                let mut count = 0;
                let out_offset = i * requested_max_hist;

                //  Filter and extract.
                // Because Rust is compiled, this loop is vastly faster than NumPy's mask operations.
                for j in 0..self.max_history {
                    let ts = user_timestamps[j];

                    // Stop early if we hit padding or if we've collected enough history
                    if ts == self.ts_pad_value || count >= requested_max_hist {
                        break;
                    }

                    // Apply the timestamp mask condition
                    if ts < target_ts {
                        ret_movie_ids[out_offset + count] = user_movies[j];
                        ret_ratings[out_offset + count] = user_ratings[j];
                        count += 1;
                    }
                }
            }
            // If binary_search returns Err(), the user wasn't found.
            // We do nothing, leaving that row in the output safely filled with pad_values.
        }

        (ret_movie_ids, ret_ratings)
    }
}


pub fn _testable_build_map_async(ratings_uris: &[&String]) -> (FxHashMap<i32, UserMapEntry>, usize) {
    let rt = Runtime::new().unwrap();

    let (map, longest_history)  = rt.block_on(build_map_async(ratings_uris));

    (map, longest_history)
}

async fn build_map_async(ratings_uris: &[&String]) -> (FxHashMap<i32, UserMapEntry>, usize) {

    // use semaphore to provent memore swap thrashing from too may threads
    let semaphore = Arc::new(Semaphore::new(4));

    let mut tasks = Vec::new();

    // 2. The Map Phase: Spawn a concurrent task for each file
    for file_uri in ratings_uris {
        // We must clone the string so the background task owns the data
        let uri = file_uri.to_string();

        let sem = semaphore.clone();

        let task = tokio::spawn(async move {

            let _permit = sem.acquire_owned().await.unwrap();

            // Each task gets its own independent HashMap (no locking needed!)
            let mut local_map: FxHashMap<i32, UserMapEntry> = FxHashMap::default();

            let (storage, path) = util::parse_uri(&uri);
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

            // Return the populated map from this specific task
            local_map
        });

        tasks.push(task);
    }

    // 3. Wait for all files to be downloaded and parsed CONCURRENTLY
    let results = join_all(tasks).await;

    // 4. The Reduce Phase: Merge all local maps into the master map
    let mut master_map: FxHashMap<i32, UserMapEntry> = FxHashMap::default();

    for res in results {
        let local_map = res.expect("A background task panicked");

        for (uid, mut local_entry) in local_map {
            let master_entry = master_map.entry(uid).or_insert_with(|| UserMapEntry {
                movie_ids: Vec::new(),
                ratings: Vec::new(),
                timestamps: Vec::new(),
            });

            // .append() is highly optimized in Rust for moving data between vectors
            master_entry.movie_ids.append(&mut local_entry.movie_ids);
            master_entry.ratings.append(&mut local_entry.ratings);
            master_entry.timestamps.append(&mut local_entry.timestamps);
        }
    }

    // 5. Post-Processing: Sort parallel arrays by timestamp
    let mut max_len: usize = 0;

    for entry in master_map.values_mut() {
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

    (master_map, max_len)
}

fn prepare_user_data(
    lookup: &FxHashMap<i32, UserMapEntry>,
    max_history: usize,
    pad_value: i32,
    ts_pad_value: i64,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i64>) {
    let n_users = lookup.len();

    // Sort entries by user_id first.
    // This allows us to fill the final arrays in the correct sorted order,
    // avoiding the need to sort the giant buffers later.
    let mut items: Vec<(&i32, &UserMapEntry)> = lookup.iter().collect();
    items.sort_by_key(|(uid, _)| **uid);

    // Pre-allocate flattened 2D arrays.
    // Memory layout: [user0_col0, user0_col1, ... user1_col0, user1_col1, ...]
    let mut user_ids : Vec<i32> = Vec::with_capacity(n_users);
    let mut movie_ids: Vec<i32> = vec![pad_value; n_users * max_history];
    let mut ratings: Vec<i32> = vec![pad_value; n_users * max_history];
    let mut timestamps: Vec<i64> = vec![ts_pad_value; n_users * max_history];

    for (i, (&uid, entry)) in items.into_iter().enumerate() {
        user_ids.push(uid);

        let history_len = entry.movie_ids.len();
        let actual_history = min(history_len, max_history);

        // Calculate the starting index of the most recent history in the user's data
        let start_idx = history_len - actual_history;

        // Calculate the row offset in our flattened array
        let offset = i * max_history;

        // Efficiently copy the slice using memcpy
        movie_ids[offset..offset + actual_history]
            .copy_from_slice(&entry.movie_ids[start_idx..]);

        ratings[offset..offset + actual_history]
            .copy_from_slice(&entry.ratings[start_idx..]);

        timestamps[offset..offset + actual_history]
            .copy_from_slice(&entry.timestamps[start_idx..]);
    }

    (user_ids, movie_ids, ratings, timestamps)
}


/// build the UserHistory from the given list of uris for the ratings parquet files
/// # Arguments
/// * `ratings_uris` -  vector of uris for ratings parquet files for building UserHistory hashmap
pub fn build_user_history(ratings_uris: &[&String], max_history: usize) -> UserHistory {
    // Because PyO3 functions are synchronous by default, we spin up a
    // Tokio runtime to execute our highly concurrent async disk/network I/O.
    // The code can begin processing data while it is still loading because the loading onto CPU is usually a bottleneck.
    let rt = Runtime::new().unwrap();

    // Block the thread until the map is built and sorted
    let (map, longest_history)  = rt.block_on(build_map_async(&ratings_uris));

    print!("longest history read has length={}", longest_history);

    let pad_value : i32 = -1;
    let ts_pad_value: i64 = 2524608000; //year 2050
    let (user_ids, movie_ids, ratings, timestamps)
        = prepare_user_data(&map, max_history, pad_value, ts_pad_value);

    UserHistory { user_ids, movie_ids, ratings, timestamps, max_history, pad_value, ts_pad_value}
}
