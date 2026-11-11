use std::path::PathBuf;

/*
for the re-ranker training, we will write the parquet file :
    user_id, movie_id, rating, timestamp, candidate_ids, enhanced user_embedding, enhanced candidate
the movie_id is in the candidate_ids.
*/

// Assuming your UserRequest is accessible here
pub mod helper {
    // Tell Rust to literally include the code from helper.rs here
    include!("helper.rs");
    include!("helper_users.rs");
}
use inference_engine::app_config::AppConfig;
use inference_engine::orchestrator::Orchestrator;
use helper::{get_train_val_test_liked_uris, DataSize};
use crate::helper::{get_bin_dir, get_config_json_uri};

struct Harness {
    orchestrator: Orchestrator,
    ratings_positive_uri: String,
    output_file_path: String,
    q_ver: usize,
    r_ver: usize
}

impl Harness {
    async fn new(training: bool, use_full: bool) -> Self {

        println!("\n[SETUP]: Initializing test resource...");

        let q_ver = 1;
        let r_ver = 1;

        let config_path = get_config_json_uri().to_string();
        let config = AppConfig::load_from_file(&config_path)
            .expect("Could not load config json file");

        let query_uri = config.query_uri.clone();
        let ranker_uri = config.ranker_uri.clone();
        let ranker_n_local_devices = config.ranker_n_local_devices;
        let top_k = config.top_k;
        let user_db_path: PathBuf = config.user_db_path.clone();
        let persisted_index_path : PathBuf = config.persisted_index_path.clone();
        let _params_json_uri : String = config.params_json_path;

        let movie_embeddings_uri : String = config.movie_embeddings_path;

        let datasize = if use_full {DataSize::Full} else {DataSize::Tiny3};
        // only use train history vecause are using this to train the reranker model
        let ratings_map = get_train_val_test_liked_uris(datasize, false);
        let mut ratings_uris: Vec<&str> = vec![
            ratings_map.get("train_liked").unwrap(),
            ratings_map.get("train_3").unwrap(),
            ratings_map.get("train_disliked").unwrap(),
        ];
        if training {
            ratings_uris.push(ratings_map.get("val_liked").unwrap());
            ratings_uris.push(ratings_map.get("val_3").unwrap());
            ratings_uris.push(ratings_map.get("val_disliked").unwrap());
        }

        let orchestrator = Orchestrator::new(
            query_uri,
            ranker_uri,
            config.query_saved_models_uri,
            config.ranker_saved_models_uri,
            config.ranker_serving_is_batched,
            &movie_embeddings_uri,
            ratings_uris,
            ranker_n_local_devices,
            top_k,
            persisted_index_path,
            user_db_path
        ).await.unwrap();

        let output_base_dir = get_bin_dir().unwrap().to_string_lossy().into_owned();

        let type_str = if training { "training" } else { "validation" };
        let type_dir = if use_full {""} else {"tiny3/"};

        let parquet_output_path = format!(
            "{}/{}reranker_{}_parquet_q{:02}_r{:02}.parquet",
            output_base_dir.trim_end_matches('/'),
            type_dir,
            type_str,
            q_ver,
            r_ver
        );

        let ratings_positive_uri = if training {ratings_map.get("train_liked").unwrap().clone()}
            else {ratings_map.get("val_liked").unwrap().clone()};

        Self {
            orchestrator: orchestrator,
            ratings_positive_uri: ratings_positive_uri,
            output_file_path: parquet_output_path,
            q_ver: q_ver,
            r_ver: r_ver
        }
    }
}

// --- TEARDOWN LOGIC ---
impl Drop for Harness {
    fn drop(&mut self) {
        println!("[TEARDOWN]");
    }
}

#[cfg(test)]
mod write_reranker_parquet {
    use polars::prelude::{col, LazyFrame};
    // Bring everything from the outer scope (TestHarness, helper functions, etc.) into the test module
    use super::*;

    //bring the gRPC trait into scope so its methods (.predict) are visible
    use inference_engine::pb::recommender_service_server::RecommenderService;
    use inference_engine::pb::{RankedMoviesAndRepr, UsersRequest};
    use tonic::{Request, Response};
    use inference_engine::tensorflow::serving::model_spec::VersionChoice;
    use crate::helper::{load_and_concat_parquet};

    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;
    use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;


    #[tokio::test(flavor = "multi_thread")]
    async fn test_write_parquet_train_full() -> Result<(), Box<dyn std::error::Error>> {
        run_write_parquet(true, true).await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_write_parquet_validation_full() -> Result<(), Box<dyn std::error::Error>> {
        run_write_parquet(false, true).await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_write_parquet_train_tiny3() -> Result<(), Box<dyn std::error::Error>> {
        run_write_parquet(true, false).await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_write_parquet_validation_tiny3() -> Result<(), Box<dyn std::error::Error>> {
        run_write_parquet(false, false).await
    }

    async fn run_write_parquet(training: bool, use_full: bool) -> Result<(), Box<dyn std::error::Error>> {
        /*
        for the re-ranker training, we will write into the parquet file :
            user_id, movie_id, rating, timestamp, candidate_ids, candidate_scores,
            user_embedding, candidate_embeddings
        the movie_id is also in the candidate_ids because it is a positive used in training and
        the movie embedding is needed for it..
        */

        // Setup runs here
        let harness = Harness::new(training, use_full).await;

        let top_k = harness.orchestrator.top_k;
        let query_model_version = Some(VersionChoice::Version(harness.q_ver as i64));
        let ranker_model_version = Some(VersionChoice::Version(harness.r_ver as i64));

        let ranker_metadata = harness
            .orchestrator
            .get_or_fetch_ranker_metadata(ranker_model_version.clone())
            .expect("error fetching ranker metadata");
        let _num_candidates = ranker_metadata.num_candidates;
        let out_features = ranker_metadata.out_features;

        let ranker_batch_size = harness
            .orchestrator
            .get_ranker_batch_size(ranker_model_version.clone())
            .expect("error while reading ranker metadata");

        let ratings_uri_str: &[&str] = &[&harness.ratings_positive_uri];

        let pos_df: LazyFrame =
            load_and_concat_parquet(ratings_uri_str).expect("error reading ratings into df");
        let pos_df = pos_df
            .select([
                col("user_id"),
                col("movie_id"),
                col("rating"),
                col("timestamp"),
            ])
            .collect()?;

        let user_ca = pos_df.column("user_id")?.i32()?;
        let movie_ca = pos_df.column("movie_id")?.i32()?;
        let rating_ca = pos_df.column("rating")?.i32()?;
        let time_ca = pos_df.column("timestamp")?.i64()?;

        let user_ids: Vec<i32> = user_ca.into_no_null_iter().collect();
        let movie_ids: Vec<i32> = movie_ca.into_no_null_iter().collect();
        let ratings: Vec<i32> = rating_ca.into_no_null_iter().collect();
        let timestamps: Vec<i64> = time_ca.into_no_null_iter().collect();

        // 1. Create the Arrow schema and Parquet writer for `parquet_output_path`
        if let Some(parent) = Path::new(&harness.output_file_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let cand_id_item_field = Arc::new(Field::new("item", DataType::Int32, false));
        let float_item_field = Arc::new(Field::new("item", DataType::Float32, false));
        let cand_emb_inner_field = Arc::new(Field::new(
            "item",
            DataType::FixedSizeList(float_item_field.clone(), out_features as i32),
            false,
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int32, false),
            Field::new("movie_id", DataType::Int32, false),
            Field::new("rating", DataType::Int32, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new(
                "candidate_ids",
                DataType::FixedSizeList(cand_id_item_field.clone(), top_k as i32),
                false,
            ),
            Field::new(
                "candidate_scores",
                DataType::FixedSizeList(float_item_field.clone(), top_k as i32),
                false,
            ),
            Field::new(
                "user_embedding",
                DataType::FixedSizeList(float_item_field.clone(), out_features as i32),
                false,
            ),
            Field::new(
                "candidate_embeddings",
                DataType::FixedSizeList(cand_emb_inner_field.clone(), top_k as i32),
                false,
            ),
        ]));

        let output_file = File::create(&harness.output_file_path)?;
        let writer_props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut parquet_writer =
            ArrowWriter::try_new(output_file, schema.clone(), Some(writer_props))?;

        // 2. Zip user_ids, timestamps, movie_ids, and ratings in batches of `ranker_batch_size`
        for (((chunk_users, chunk_timestamps), chunk_movies), chunk_ratings) in user_ids
            .chunks(ranker_batch_size)
            .zip(timestamps.chunks(ranker_batch_size))
            .zip(movie_ids.chunks(ranker_batch_size))
            .zip(ratings.chunks(ranker_batch_size))
        {
            let tonic_request: Request<UsersRequest> = harness
                .orchestrator
                .get_users_request(
                    chunk_users,
                    chunk_timestamps,
                    query_model_version.clone(),
                    ranker_model_version.clone(),
                )
                .await?;
            let users_request: UsersRequest = tonic_request.into_inner();
            //users_request.k = Some(top_k as u32);

            let predict_req = tonic::Request::new(users_request);
            let results: Result<Response<RankedMoviesAndRepr>, tonic::Status> =
                harness.orchestrator.predict_diagnostic(predict_req).await;
            assert!(results.is_ok(), "Prediction failed: {:?}", results.err());
            let rmar = results
                .expect("Error receiving response from ranker diagnostic request")
                .into_inner();

            // 3. Assert that chunk_users matches rmar.user_ids in exact order
            assert_eq!(chunk_users, rmar.user_ids.as_slice());

            let n_chunk = chunk_users.len();
            debug_assert_eq!(rmar.movie_ids.len(), n_chunk * top_k);
            debug_assert_eq!(rmar.scores.len(), n_chunk * top_k);
            debug_assert_eq!(rmar.user_reprs.len(), n_chunk * out_features);
            debug_assert_eq!(rmar.cand_reprs.len(), n_chunk * top_k * out_features);

            // 4. Build Arrow arrays for this batch and append to the Parquet file
            let candidate_ids_list = FixedSizeListArray::try_new(
                cand_id_item_field.clone(),
                top_k as i32,
                Arc::new(Int32Array::from(rmar.movie_ids)),
                None,
            )?;

            let candidate_scores_list = FixedSizeListArray::try_new(
                float_item_field.clone(),
                top_k as i32,
                Arc::new(Float32Array::from(rmar.scores)),
                None,
            )?;

            let user_emb_list = FixedSizeListArray::try_new(
                float_item_field.clone(),
                out_features as i32,
                Arc::new(Float32Array::from(rmar.user_reprs)),
                None,
            )?;

            let cand_emb_flat_vecs = FixedSizeListArray::try_new(
                float_item_field.clone(),
                out_features as i32,
                Arc::new(Float32Array::from(rmar.cand_reprs)),
                None,
            )?;
            let candidate_emb_list = FixedSizeListArray::try_new(
                cand_emb_inner_field.clone(),
                top_k as i32,
                Arc::new(cand_emb_flat_vecs),
                None,
            )?;

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(chunk_users.to_vec())) as ArrayRef,
                    Arc::new(Int32Array::from(chunk_movies.to_vec())) as ArrayRef,
                    Arc::new(Int32Array::from(chunk_ratings.to_vec())) as ArrayRef,
                    Arc::new(Int64Array::from(chunk_timestamps.to_vec())) as ArrayRef,
                    Arc::new(candidate_ids_list) as ArrayRef,
                    Arc::new(candidate_scores_list) as ArrayRef,
                    Arc::new(user_emb_list) as ArrayRef,
                    Arc::new(candidate_emb_list) as ArrayRef,
                ],
            )?;

            parquet_writer.write(&batch)?;
        }

        // 5. Flush and close the Parquet file
        parquet_writer.close()?;
        println!("Wrote reranker training parquet to: {}", harness.output_file_path);

        // 6. Verify the written Parquet file can be read, row count matches, and spot-check rows
        assert_training_parquet_valid(
            &harness.output_file_path,
            &user_ids,
            &movie_ids,
            &ratings,
            &timestamps,
            top_k,
            out_features,
        )?;

        Ok(())
    }

    fn assert_training_parquet_valid(
        parquet_path: &str,
        expected_user_ids: &[i32],
        expected_movie_ids: &[i32],
        expected_ratings: &[i32],
        expected_timestamps: &[i64],
        top_k: usize,
        out_features: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(parquet_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut total_rows = 0usize;

        for batch_result in reader {
            let batch: RecordBatch = batch_result?;
            let num_rows = batch.num_rows();
            assert!(num_rows > 0, "Expected non-empty RecordBatch");
            assert_eq!(batch.num_columns(), 8, "Expected 8 columns in training Parquet");

            let user_col = batch
                .column_by_name("user_id")
                .expect("missing user_id")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("user_id must be Int32Array");

            let movie_col = batch
                .column_by_name("movie_id")
                .expect("missing movie_id")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("movie_id must be Int32Array");

            let rating_col = batch
                .column_by_name("rating")
                .expect("missing rating")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("rating must be Int32Array");

            let ts_col = batch
                .column_by_name("timestamp")
                .expect("missing timestamp")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("timestamp must be Int64Array");

            let cand_ids_col = batch
                .column_by_name("candidate_ids")
                .expect("missing candidate_ids")
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("candidate_ids must be FixedSizeListArray");

            let cand_scores_col = batch
                .column_by_name("candidate_scores")
                .expect("missing candidate_scores")
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("candidate_scores must be FixedSizeListArray");

            let user_emb_col = batch
                .column_by_name("user_embedding")
                .expect("missing user_embedding")
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("user_embedding must be FixedSizeListArray");

            let cand_embs_col = batch
                .column_by_name("candidate_embeddings")
                .expect("missing candidate_embeddings")
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("candidate_embeddings must be FixedSizeListArray");

            // Spot-check the first, middle, and last row of each batch
            let check_indices = [0, num_rows / 2, num_rows - 1];
            for &row_idx in &check_indices {
                let global_idx = total_rows + row_idx;

                // 1. Verify scalar columns match original inputs
                assert_eq!(user_col.value(row_idx), expected_user_ids[global_idx]);
                assert_eq!(movie_col.value(row_idx), expected_movie_ids[global_idx]);
                assert_eq!(rating_col.value(row_idx), expected_ratings[global_idx]);
                assert_eq!(ts_col.value(row_idx), expected_timestamps[global_idx]);

                // 2. Verify candidate_ids type and length (top_k)
                let row_cand_ids_arr = cand_ids_col.value(row_idx);
                let row_cand_ids = row_cand_ids_arr
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("candidate_ids inner elements must be Int32Array");
                assert_eq!(row_cand_ids.len(), top_k);

                // 3. Verify candidate_scores type and length (top_k)
                let row_cand_scores_arr = cand_scores_col.value(row_idx);
                let row_cand_scores = row_cand_scores_arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("candidate_scores inner elements must be Float32Array");
                assert_eq!(row_cand_scores.len(), top_k);
                assert!(
                    row_cand_scores.iter().all(|v| v.map_or(false, |x| x.is_finite())),
                    "candidate_scores contains non-finite values at row {}",
                    global_idx
                );

                // 4. Verify user_embedding type and length (out_features)
                let row_user_emb_arr = user_emb_col.value(row_idx);
                let row_user_emb = row_user_emb_arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("user_embedding inner elements must be Float32Array");
                assert_eq!(row_user_emb.len(), out_features);
                assert!(
                    row_user_emb.iter().all(|v| v.map_or(false, |x| x.is_finite())),
                    "user_embedding contains non-finite values at row {}",
                    global_idx
                );

                // 5. Verify nested candidate_embeddings shape [top_k, out_features]
                let row_cand_embs_outer = cand_embs_col.value(row_idx);
                let row_cand_embs_list = row_cand_embs_outer
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .expect("candidate_embeddings outer element must be FixedSizeListArray");
                assert_eq!(row_cand_embs_list.len(), top_k);

                for k_idx in 0..top_k {
                    let inner_emb_arr = row_cand_embs_list.value(k_idx);
                    let inner_emb = inner_emb_arr
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("candidate_embeddings inner vector must be Float32Array");
                    assert_eq!(inner_emb.len(), out_features);
                    assert!(
                        inner_emb.iter().all(|v| v.map_or(false, |x| x.is_finite())),
                        "candidate_embeddings[{}][{}] contains non-finite values",
                        global_idx,
                        k_idx
                    );
                }
            }

            total_rows += num_rows;
        }

        assert_eq!(
            total_rows,
            expected_user_ids.len(),
            "Total rows in Parquet ({}) did not match expected user count ({})",
            total_rows,
            expected_user_ids.len()
        );

        Ok(())
    }
}