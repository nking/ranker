
use arrow_array::{Int32Array, Int64Array, FixedSizeListArray};
use tokio::runtime::Runtime;
use object_store::ObjectStoreExt;
use futures::stream::StreamExt;
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use arrow_schema::DataType;
use crate::util;

//NOTE: unlike the python version, the rust version uses implicit user indices for the movie_ids
// and timestamp rows that are 1 through num_users.  the python datastructures use 0 through num_users - 1
// movie_ids and timestamps are shape  (num_users * num_movies)
pub struct RecommendedMovies {
    pub movie_ids : Vec<i32>,
    pub timestamps : Vec<i64>,
    num_users: usize,
    pad_value : i32,
    ts_pad_value : i64,
}

impl RecommendedMovies {
    /*
    given array of user_ids, return top_k recommended movies for user that they haven't seen before time=timestamp.
    the unseen movies have been moved to front of array and retain their respective original order which is by decreasing similarity score.

    Arguments:
    * user_id: input array of shape (None,), e.g. np.array([2,4])
    * timestamp: timestamp representing current time.  any recommendations with timestamps > timestamp are yet unseen.
    * top_k: number of top unseen recommendations to return

    Returns:
     top k of movie recommendations unseen by user_id.  shape returned is (len(user_id_, top_k)

     */
    pub fn get_unseen_movies(
        &self,
        user_ids: Vec<i32>,
        timestamps: Vec<i64>,
        top_k: usize
    ) -> Vec<i32> {
        // Derive num_movies dynamically from the flat array length
        let num_movies = self.movie_ids.len() / (self.num_users);

        if top_k > num_movies {
            // Equivalent to raising ValueError
            panic!("top_k ({}) must be smaller than the number of recommendations per user ({})", top_k, num_movies);
        }

        let n_requests = user_ids.len();
        let mut ret_movie_ids = vec![self.pad_value; n_requests * top_k];

        for (i, (&target_uid, &target_ts)) in user_ids.iter().zip(timestamps.iter()).enumerate() {

            // Assuming 1-based user_ids mapping to 0-based indices based on your previous code
            let user_idx = (target_uid - 1) as usize;

            // Locate the user's data chunk in the flattened 1D array
            let start_idx = user_idx * num_movies;
            let end_idx = start_idx + num_movies;

            let user_movies = &self.movie_ids[start_idx..end_idx];
            let user_timestamps = &self.timestamps[start_idx..end_idx];

            let out_offset = i * top_k;
            let mut count = 0;

            // PASS 1: Collect "Unseen" movies first (timestamps > target_ts)
            // This is the equivalent of argsort putting ~mask (0s) at the front
            for j in 0..num_movies {
                if count >= top_k { break; }

                if user_timestamps[j] > target_ts {
                    ret_movie_ids[out_offset + count] = user_movies[j];
                    count += 1;
                }
            }

            // PASS 2: If we haven't reached top_k yet, fill the rest with "Seen" movies
            // This is the equivalent of argsort putting ~mask (1s) at the back, preserving their order
            if count < top_k {
                for j in 0..num_movies {
                    if count >= top_k { break; }

                    if user_timestamps[j] <= target_ts {
                        ret_movie_ids[out_offset + count] = user_movies[j];
                        count += 1;
                    }
                }
            }
        }

        ret_movie_ids
    }
}


/// Efficiently reads complementary movies and timestamps files into flattened 2D arrays.
/// Returns a tuple of (movie_ids_flat, timestamps_flat) each matching the shape (num_users, num_movies)
pub async fn read_recommendation_files(
    movies_uri: &str,
    timestamps_uri: &str,
    num_users: usize,
    num_movies: usize,
    pad_value: i32,
    ts_pad_value : i64
) -> (Vec<i32>, Vec<i64>) {

    let movies_uri = movies_uri.to_string();
    let timestamps_uri = timestamps_uri.to_string();

    let movies_task = tokio::spawn(async move {
        let mut movies_data = vec![pad_value; num_users * num_movies]; // Pre-allocate with a pad value

        let (storage, path) = util::parse_uri(&movies_uri);
        let meta = storage.head(&path).await.expect("Failed to get movies metadata");
        let reader = ParquetObjectReader::new(storage, meta.location).with_file_size(meta.size);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.unwrap();
        let mut stream = builder.build().unwrap();

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result.expect("Failed to read movies RecordBatch");

            let user_ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            // Downcast the nested List column
            let movie_lists = batch.column(1).as_any().downcast_ref::<FixedSizeListArray>().unwrap();

            for i in 0..batch.num_rows() {
                let uid = user_ids.value(i);
                // Map 1-based user_id directly to a 0-based array index
                let user_idx = (uid - 1) as usize;
                let offset = user_idx * num_movies;

                // Extract the inner primitive array from the List row
                let list_element = movie_lists.value(i);
                let movie_values = list_element.as_any().downcast_ref::<Int32Array>().unwrap();

                // Blast the nested array data directly into our pre-allocated matrix slice via memcpy
                movies_data[offset..offset + num_movies].copy_from_slice(movie_values.values());
            }
        }
        movies_data
    });

    // Task 2: Process the Timestamps file concurrently
    let timestamps_task = tokio::spawn(async move {
        let mut timestamps_data = vec![ts_pad_value; num_users * num_movies];

        let (storage, path) = util::parse_uri(&timestamps_uri);
        let meta = storage.head(&path).await.expect("Failed to get timestamps metadata");
        let reader = ParquetObjectReader::new(storage, meta.location).with_file_size(meta.size);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.unwrap();
        let mut stream = builder.build().unwrap();

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result.expect("Failed to read timestamps RecordBatch");

            let user_ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let ts_lists = batch.column(1).as_any().downcast_ref::<FixedSizeListArray>().unwrap();

            for i in 0..batch.num_rows() {
                let uid = user_ids.value(i);
                let user_idx = (uid - 1) as usize;
                let offset = user_idx * num_movies;

                let list_element = ts_lists.value(i);
                let ts_values = list_element.as_any().downcast_ref::<Int64Array>().unwrap();

                timestamps_data[offset..offset + num_movies].copy_from_slice(ts_values.values());
            }
        }
        timestamps_data
    });

    // Await both tasks together. They run completely in parallel!
    let (movies_res, timestamps_res) = tokio::join!(movies_task, timestamps_task);

    (
        movies_res.expect("Movies processing panicked"),
        timestamps_res.expect("Timestamps processing panicked"),
    )
}

fn read_num_movies(movies_uri: &String) -> usize {
    let rt = Runtime::new().unwrap();

    rt.block_on(async {
        let (storage, path) = util::parse_uri(movies_uri);
        let meta = storage.head(&path).await.expect("Failed to get movies metadata");
        let reader = ParquetObjectReader::new(storage, meta.location).with_file_size(meta.size);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.unwrap();

        let schema = builder.schema();

        // 3. Fetch the field by name and inspect its Arrow DataType
        match schema.field_with_name("movie_ids").unwrap().data_type() {
            DataType::FixedSizeList(_inner_field, size) => *size as usize,
            _ => panic!("Expected movie_ids to be a FixedSizeList schema! Check your PyArrow writer."),
        }
    })
}

/// build the RecommendedMovies from the given list of uris for the movie recommendation parquet files
/// # Arguments
/// * `ratings_uris` -  vector of uris for ratings parquet files for building UserHistory hashmap
pub fn build_recommended_movies(num_users:usize, movie_rec_file_uri:&String, movie_rec_ts_file_uri:&String) -> RecommendedMovies {

    let pad_value : i32 = -1;
    let ts_pad_value: i64 = 2524608000; //year 2050

    let num_movies = read_num_movies(&movie_rec_file_uri);

    // Because PyO3 functions are synchronous by default, we spin up a
    // Tokio runtime to execute our highly concurrent async disk/network I/O.
    // The code can begin processing data while it is still loading because the loading onto CPU is usually a bottleneck.
    let rt = Runtime::new().unwrap();

    // Block the thread until the map is built and sorted
    let (movie_ids, timestamps)  = rt.block_on(read_recommendation_files(
        &movie_rec_file_uri, &movie_rec_ts_file_uri, num_users, num_movies, pad_value, ts_pad_value));

    RecommendedMovies { movie_ids, timestamps, num_users, pad_value, ts_pad_value}

}