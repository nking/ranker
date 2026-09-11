
#[cfg(test)]
mod graph_builder_tests {

    // In src/test/rust/integration_test.rs

    // Import the functions/structs you want to test from your main code
    //
    // To run:
    //   cd src/main/rust
    //   cargo test

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    use arrow_array::{Int32Array, Int64Array};
    use std::fs::File;
    use std::path::Path;
    use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, ParquetRecordBatchReader};
    //use parquet::arrow::ParquetRecordBatchStreamBuilder;
    //use tokio::runtime::Runtime;

    use helper::{get_train_val_test_liked_uris, DataSize, get_python_path};
    use inference_engine::embeddings_util::{read_movie_embeddings, read_user_embeddings, get_user_embeddings};
    use inference_engine::graph_builder::{build_enriched_padded_supergraph, create_fake_padded_super_batch, JraphGraph};
    use inference_engine::user_history::{build_user_history, UserHistory};
    use crate::graph_builder_tests::helper::{assert_slices_nearly_equal, get_embeddings_uris};

    use safetensors::SafeTensors;
    use std::fs;
    use std::process::Command;

    #[test]
    pub fn test_create_fake_batch() {

        // and is a test of build_padded_super_graph

        let batch_size : usize = 3;
        let max_history: usize = 4;
        let num_candidates: usize = 5;
        let user_id_range : (usize, usize) = (1, 6040);
        let movie_id_range : (usize, usize) = (6041, 6041+3883);

        let n_local_devices : usize = 1;

        let (user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris();

        let ranker_batch_size : usize = batch_size;
        
        let padded_super_graph : JraphGraph  = create_fake_padded_super_batch(batch_size,
            ranker_batch_size,
            max_history, num_candidates, user_id_range,
            movie_id_range, n_local_devices,
            &user_embeddings_uri, &movie_embeddings_uri
        );

        print!("padded_super_graph={:?}", padded_super_graph);

        // compare to python version used in training:
        let output_path = "../../../bin/expected_fake_graph.safetensors";

        let user_id_range = serde_json::to_string(&user_id_range)
            .expect("Failed to serialize user_id_range");
        let movie_id_range = serde_json::to_string(&movie_id_range)
            .expect("Failed to serialize movie_id_range");
        let user_embeddings_uri = user_embeddings_uri.replace("parquet", "array_record");
        let movie_embeddings_uri = movie_embeddings_uri.replace("parquet", "array_record");

        // get the conda venv:
        let python_bin = get_python_path();

        let status = Command::new(&python_bin)
            .arg("../../../src/test/python/movie_lens_ranker/write_fake_paddedsupergraph.py")
            .arg("--output_path").arg(output_path)
            .arg("--user_embeddings_uri").arg(user_embeddings_uri)
            .arg("--movie_embeddings_uri").arg(movie_embeddings_uri)
            .arg("--max_history").arg(max_history.to_string())
            .arg("--batch_size").arg(batch_size.to_string())
            .arg("--num_candidates").arg(num_candidates.to_string())
            .arg("--user_id_range").arg(user_id_range)
            .arg("--movie_id_range").arg(movie_id_range)
            .arg("--n_local_devices").arg(n_local_devices.to_string())
            .status()
            .expect("Failed to execute Python script");

        assert!(status.success());

        // Read output file
        let buffer = fs::read(output_path).expect("Failed to read safetensors file");
        let tensors = SafeTensors::deserialize(&buffer).expect("Failed to parse safetensors");

        // Helper closure to pull i32 slices
        let get_i32_vec = |name: &str| -> Vec<i32> {
            let tensor = tensors.tensor(name).unwrap();
            // Convert raw byte slice to i32 slice safely
            tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect()
        };

        // Helper for 32-bit float vectors
        let get_f32_vec = |name: &str| -> Vec<f32> {
            let tensor = tensors.tensor(name).unwrap();
            tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect()
        };

        let get_bool_vec = |name: &str| -> Vec<bool> {
            let tensor = tensors.tensor(name).unwrap();
            tensor
                .data()
                .iter()
                .map(|&byte| byte != 0)
                .collect()
        };

        let expected_n_node: Vec<i32> = get_i32_vec("n_node");
        let expected_n_edge: Vec<i32> = get_i32_vec("n_edge");
        let expected_senders: Vec<i32> = get_i32_vec("senders");
        let expected_receivers: Vec<i32> = get_i32_vec("receivers");
        let expected_edge_features: Vec<i32> = get_i32_vec("edge_features");
        let expected_node_ids: Vec<i32> = get_i32_vec("node_ids");
        let expected_node_labels: Vec<i32> = get_i32_vec("node_label");
        let expected_node_types: Vec<i32> = get_i32_vec("node_type");

        let expected_node_embeddings: Vec<f32> = get_f32_vec("embeddings");
        let expected_candidate_mask : Vec<bool> = get_bool_vec("candidate_mask");

        assert_eq!(padded_super_graph.n_node, expected_n_node);
        assert_eq!(padded_super_graph.n_edge, expected_n_edge);
        assert_eq!(padded_super_graph.senders, expected_senders);
        assert_eq!(padded_super_graph.receivers, expected_receivers);
        assert_eq!(padded_super_graph.edge_features, expected_edge_features);
        assert_eq!(padded_super_graph.node_ids, expected_node_ids);
        assert_eq!(padded_super_graph.node_labels, expected_node_labels);
        assert_eq!(padded_super_graph.node_types, expected_node_types);
        assert_eq!(padded_super_graph.candidate_mask, expected_candidate_mask);

        assert_slices_nearly_equal(&padded_super_graph.node_embeddings, &expected_node_embeddings, 1E-6);

    }

    fn read_user_ratings(ratings_uri: &str, read_rows: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i64>) {
        let file = File::open(ratings_uri).expect("Failed to open the parquet file");

        // Build the reader without projection.
        // It will read all columns into the batch, and we will pick the ones we want.
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader: ParquetRecordBatchReader = builder.build().unwrap();

        let mut user_ids : Vec<i32> = Vec::new();
        let mut movie_ids : Vec<i32> = Vec::new();
        let mut ratings : Vec<i32> = Vec::new();
        let mut timestamps : Vec<i64> = Vec::new();

        let mut current_row_index: usize = 0;
        let num_rows = read_rows.len();
        let mut count = 0;

        // Iterate through record batches
        for maybe_batch in reader {
            let batch = maybe_batch.unwrap();
            let batch_size = batch.num_rows();

            //println!("Column 0 data type: {:?}\n", batch.column(0).data_type());

            // Check if our target rows fall into this batch
            for &target_i32 in read_rows {
                let target = target_i32 as usize; // Cast to usize for math/comparisons

                if target >= current_row_index && target < current_row_index + batch_size {
                    let local_idx = target - current_row_index;

                    // Extract Column 0 (User ID - Int32)
                    let col0 = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    user_ids.push(col0.value(local_idx));

                    let col1 = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
                    movie_ids.push(col1.value(local_idx));

                    let col2 = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
                    ratings.push(col2.value(local_idx));

                    // Extract Column 3 (Timestamp - Int64)
                    // Note: It is index 3 because we didn't project/remove columns 1 and 2!
                    let col3 = batch.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
                    timestamps.push(col3.value(local_idx));

                    count += 1;

                    if count == num_rows {
                        break;
                    }
                }
            }

            if count == num_rows {
                break;
            }
            current_row_index += batch_size;
        }

        println!("User IDs: {:?}", user_ids);
        println!("Timestamps: {:?}", timestamps);

        (user_ids, movie_ids, ratings, timestamps)
    }

    #[tokio::test]
    pub async fn test_create_inference_batch() {

        // to compare results to python test_ranker.py method test_create_inference_batch()

        let max_history : usize = 4;
        let batch_size : usize = 2;
        let num_candidates : usize = 5;
        let jax_n_local_devices : usize = 1;

        let (user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris();

        assert_file_exists(&user_embeddings_uri);
        assert_file_exists(&movie_embeddings_uri);

        let (movie_embeddings_catalog, num_movies, embed_len) = read_movie_embeddings(
            &movie_embeddings_uri);

        let (user_embeddings_catalog, num_users, _embed_len) = read_user_embeddings(
            &user_embeddings_uri);

        let ratings_map = get_train_val_test_liked_uris(DataSize::Tiny3, false);

        let ratings_uri = ratings_map.get("train_liked").unwrap();
        let rows : Vec<i32> = vec![3, 4];

        let (user_ids, movie_ids, ratings, timestamps) = read_user_ratings(&ratings_uri, &rows);

        assert!(&6040 >= &user_ids[0] && &1 <= &user_ids[0]);
        assert!(&6040 >= &user_ids[1] && &1 <= &user_ids[1]);

        // storing the items as references
        let ratings_uris: Vec<&str> = vec![&ratings_uri];

        let user_history : UserHistory = build_user_history(&ratings_uris, 2048).await;

        let user_ids = vec![user_ids[0], user_ids[1]];
        let movie_ids = vec![movie_ids[0], movie_ids[1]];
        let ratings = vec![ratings[0], ratings[1]];
        let timestamps = vec![timestamps[0], timestamps[1]];
        let candidate_ids : Vec<i32> = vec![
            6610, 6252, 9083, 6564, 6584,
            9477, 6941, 6948, 8475, 6356];

        // for embeddings, the graph build is given the user_embedding because it may have been
        // recently built with updated timestamp, age, etc
        // and graph builder performs look-ups for chosen user_history and candidate_ids
        //let user_emb : Vec<f32> = vec![];
        let _rows = rows.len();
        let _cols = embed_len;
        let _rng = rand::thread_rng();
        
        // Generate 32 (2 * 16) random numbers
        let user_embeddings : Vec<f32> = get_user_embeddings(&user_ids, &user_embeddings_catalog, embed_len);

        let labels: Vec<i32> = vec![1; candidate_ids.len()];

        let padded_super_graph : JraphGraph = build_enriched_padded_supergraph(
            user_ids.len(),
            &user_ids,
            &timestamps,
            &candidate_ids,
            &labels,
            &user_history,
            max_history,
            num_users, num_movies, embed_len, &movie_embeddings_catalog, &user_embeddings,
            jax_n_local_devices);

        print!("graph={:?}\n", padded_super_graph);


        // =================================================================
        // get the expected graph arrays from the python code: =============
        let output_path = "../../../bin/expected_graph.safetensors";

        let candidate_ids_str = serde_json::to_string(&candidate_ids)
            .expect("Failed to serialize candidate_ids");

        let in_batch: Vec<(i32, i32, i32, i64)> = user_ids.into_iter()
            .zip(movie_ids)
            .zip(ratings)
            .zip(timestamps)
            .map(|(((a, b), c), d)| (a, b, c, d))
            .collect();

        let in_batch_str = serde_json::to_string(&in_batch)
            .expect("Failed to serialize in_batch");

        // get the conda venv:
        let python_bin = get_python_path();

        let ratings_uri = ratings_uri.replace("parquet", "array_record");
        let user_embeddings_uri = user_embeddings_uri.replace("parquet", "array_record");
        let movie_embeddings_uri = movie_embeddings_uri.replace("parquet", "array_record");

        let status = Command::new(&python_bin)
            .arg("../../../src/test/python/movie_lens_ranker/write_paddedsupergraph.py")
            .arg("--output_path").arg(output_path)
            .arg("--in_batch").arg(in_batch_str)
            .arg("--ratings_uri").arg(ratings_uri)
            .arg("--user_embeddings_uri").arg(user_embeddings_uri)
            .arg("--movie_embeddings_uri").arg(movie_embeddings_uri)
            .arg("--max_history").arg(max_history.to_string())
            .arg("--batch_size").arg(batch_size.to_string())
            .arg("--num_candidates").arg(num_candidates.to_string())
            .arg("--jax_n_local_devices").arg(jax_n_local_devices.to_string())
            .arg("--candidate_ids").arg(candidate_ids_str)
            .status()
            .expect("Failed to execute Python script");

        assert!(status.success());

        // Read output file
        let buffer = fs::read(output_path).expect("Failed to read safetensors file");
        let tensors = SafeTensors::deserialize(&buffer).expect("Failed to parse safetensors");

        // Helper closure to pull i32 slices
        let get_i32_vec = |name: &str| -> Vec<i32> {
            let tensor = tensors.tensor(name).unwrap();
            // Convert raw byte slice to i32 slice safely
            tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect()
        };

        // Helper for 32-bit float vectors
        let get_f32_vec = |name: &str| -> Vec<f32> {
            let tensor = tensors.tensor(name).unwrap();
            tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect()
        };

        let get_bool_vec = |name: &str| -> Vec<bool> {
            let tensor = tensors.tensor(name).unwrap();
            tensor
                .data()
                .iter()
                .map(|&byte| byte != 0)
                .collect()
        };

        let expected_n_node: Vec<i32> = get_i32_vec("n_node");
        let expected_n_edge: Vec<i32> = get_i32_vec("n_edge");
        let expected_senders: Vec<i32> = get_i32_vec("senders");
        let expected_receivers: Vec<i32> = get_i32_vec("receivers");
        let expected_edge_features: Vec<i32> = get_i32_vec("edge_features");
        let expected_node_ids: Vec<i32> = get_i32_vec("node_ids");
        let expected_node_labels: Vec<i32> = get_i32_vec("node_label");
        let expected_node_types: Vec<i32> = get_i32_vec("node_type");

        let expected_node_embeddings: Vec<f32> = get_f32_vec("embeddings");
        let expected_candidate_mask : Vec<bool> = get_bool_vec("candidate_mask");

        assert_eq!(padded_super_graph.n_node, expected_n_node);
        assert_eq!(padded_super_graph.n_edge, expected_n_edge);
        assert_eq!(padded_super_graph.senders, expected_senders);
        assert_eq!(padded_super_graph.receivers, expected_receivers);
        assert_eq!(padded_super_graph.edge_features, expected_edge_features);
        assert_eq!(padded_super_graph.node_ids, expected_node_ids);
        assert_eq!(padded_super_graph.node_labels, expected_node_labels);
        assert_eq!(padded_super_graph.node_types, expected_node_types);
        assert_eq!(padded_super_graph.candidate_mask, expected_candidate_mask);

        assert_slices_nearly_equal(&padded_super_graph.node_embeddings, &expected_node_embeddings, 1E-6);

    }

    fn assert_file_exists(file_uri: &String) {
        let file_path = Path::new(file_uri);
        if !file_path.exists() {
            panic!("The path {} does not exist.", file_uri);
        }
    }
}