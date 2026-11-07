use arrow_array::{FixedSizeListArray, Int32Array, Int64Array};
use arrow_schema::DataType;
use futures::stream::StreamExt;
use object_store::ObjectStoreExt;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use tokio::runtime::Runtime;
use crate::util;

pub struct RecommendedMovies {
    pub movie_ids: Vec<i32>,
    pub timestamps: Vec<i64>,
    pub num_users: usize,
    pub pad_value: i32,
    pub ts_pad_value: i64,
}

impl RecommendedMovies {
    /// Returns up to `top_k` unseen recommended movies (`timestamp > target_ts`) per user,
    /// ordered by decreasing similarity score, padded at the right with `self.pad_value` (-1).
    pub fn get_unseen_movies(
        &self,
        user_ids: &[i32],
        timestamps: &[i64],
        top_k: usize,
    ) -> Vec<i32> {
        let num_movies = self.movie_ids.len() / self.num_users;

        if top_k > num_movies {
            panic!(
                "top_k ({}) must be smaller than the number of recommendations per user ({})",
                top_k, num_movies
            );
        }

        let n_requests = user_ids.len();
        let mut ret_movie_ids = vec![self.pad_value; n_requests * top_k];

        for (i, (&target_uid, &target_ts)) in user_ids.iter().zip(timestamps.iter()).enumerate() {
            if target_uid < 1 || (target_uid as usize) > self.num_users {
                continue;
            }
            let user_idx = (target_uid - 1) as usize;

            let start_idx = user_idx * num_movies;
            let end_idx = start_idx + num_movies;

            let user_movies = &self.movie_ids[start_idx..end_idx];
            let user_timestamps = &self.timestamps[start_idx..end_idx];

            let out_offset = i * top_k;
            let mut count = 0;

            for j in 0..num_movies {
                if count >= top_k {
                    break;
                }
                if user_timestamps[j] > target_ts {
                    ret_movie_ids[out_offset + count] = user_movies[j];
                    count += 1;
                }
            }
        }

        ret_movie_ids
    }

    /// Returns up to `worst_unseen_k` unseen recommended movies (`timestamp > target_ts`) per user,
    /// starting from the end of the recommendation list (lowest similarity score first),
    /// padded at the right with `self.pad_value` (-1).
    pub fn get_lowest_scoring_unseen_movies(
        &self,
        user_ids: &[i32],
        timestamps: &[i64],
        worst_unseen_k: usize,
    ) -> Vec<i32> {
        let num_movies = self.movie_ids.len() / self.num_users;

        if worst_unseen_k > num_movies {
            panic!(
                "worst_unseen_k ({}) must be <= recommendations per user ({})",
                worst_unseen_k, num_movies
            );
        }

        let n_requests = user_ids.len();
        let mut ret_movie_ids = vec![self.pad_value; n_requests * worst_unseen_k];

        for (i, (&target_uid, &target_ts)) in user_ids.iter().zip(timestamps.iter()).enumerate() {
            if target_uid < 1 || (target_uid as usize) > self.num_users {
                continue;
            }
            let user_idx = (target_uid - 1) as usize;

            let start_idx = user_idx * num_movies;
            let end_idx = start_idx + num_movies;

            let user_movies = &self.movie_ids[start_idx..end_idx];
            let user_timestamps = &self.timestamps[start_idx..end_idx];

            let out_offset = i * worst_unseen_k;
            let mut count = 0;

            for j in (0..num_movies).rev() {
                if count >= worst_unseen_k {
                    break;
                }
                if user_timestamps[j] > target_ts {
                    ret_movie_ids[out_offset + count] = user_movies[j];
                    count += 1;
                }
            }
        }

        ret_movie_ids
    }
}

pub async fn read_recommendation_files(
    movies_uri: &str,
    timestamps_uri: &str,
    num_users: usize,
    num_movies: usize,
    pad_value: i32,
    ts_pad_value: i64,
) -> (Vec<i32>, Vec<i64>) {
    let movies_uri = movies_uri.to_string();
    let timestamps_uri = timestamps_uri.to_string();

    let movies_task = tokio::spawn(async move {
        let mut movies_data = vec![pad_value; num_users * num_movies];

        let (storage, path) = util::parse_uri(&movies_uri);
        let meta = storage.head(&path).await.expect("Failed to get movies metadata");
        let reader = ParquetObjectReader::new(storage, meta.location).with_file_size(meta.size);
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await.unwrap();
        let mut stream = builder.build().unwrap();

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result.expect("Failed to read movies RecordBatch");

            let user_ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let movie_lists = batch.column(1).as_any().downcast_ref::<FixedSizeListArray>().unwrap();

            for i in 0..batch.num_rows() {
                let uid = user_ids.value(i);
                let user_idx = (uid - 1) as usize;
                let offset = user_idx * num_movies;

                let list_element = movie_lists.value(i);
                let movie_values = list_element.as_any().downcast_ref::<Int32Array>().unwrap();

                movies_data[offset..offset + num_movies].copy_from_slice(movie_values.values());
            }
        }
        movies_data
    });

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

        match schema.field_with_name("movie_ids").unwrap().data_type() {
            DataType::FixedSizeList(_inner_field, size) => *size as usize,
            _ => panic!("Expected movie_ids to be a FixedSizeList schema! Check your PyArrow writer."),
        }
    })
}

pub fn build_recommended_movies(
    num_users: usize,
    movie_rec_file_uri: &String,
    movie_rec_ts_file_uri: &String,
) -> RecommendedMovies {
    let pad_value: i32 = -1;
    let ts_pad_value: i64 = 2524608000; // year 2050

    let num_movies = read_num_movies(movie_rec_file_uri);
    let rt = Runtime::new().unwrap();

    let (movie_ids, timestamps) = rt.block_on(read_recommendation_files(
        movie_rec_file_uri,
        movie_rec_ts_file_uri,
        num_users,
        num_movies,
        pad_value,
        ts_pad_value,
    ));

    RecommendedMovies {
        movie_ids,
        timestamps,
        num_users,
        pad_value,
        ts_pad_value,
    }
}