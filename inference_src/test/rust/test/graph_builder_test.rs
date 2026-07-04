
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
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use std::fs::File;
    use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, ParquetRecordBatchReader};
    //use parquet::arrow::ParquetRecordBatchStreamBuilder;
    //use tokio::runtime::Runtime;

    use helper::{get_train_val_test_liked_uris, DataSize};

    use inference_engine::graph_builder::{build_enriched_padded_supergraph, create_fake_padded_super_batch, JraphGraph};
    use inference_engine::user_history::{build_user_history, UserHistory};
    use inference_engine::util;

    #[test]
    pub fn test_create_fake_batch() {

        // and is a test of build_padded_super_graph

        let batch_size = 3;
        let max_history = 4;
        let num_candidates = 5;
        let user_id_range = (1, 10);
        let movie_id_range = (1, 10);
        let n_local_devices = 1;

        let padded_super_graph : JraphGraph  = create_fake_padded_super_batch(batch_size,
            max_history, num_candidates, user_id_range,
            movie_id_range, n_local_devices);

        print!("padded_super_graph={:?}", padded_super_graph);

        let expected_n_node : Vec<i32> = vec![7,  8,  9, 40,  0];
        let expected_n_edge : Vec<i32> = vec![6,  7,  8, 43,  0];

        let expected_senders : Vec<i32> = vec![1,  0,  0,  0,  0,  0,  8,  9,  7,  7,  7,  7,  7, 16, 17, 18, 15,
            15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24];
        let expected_receivers : Vec<i32> = vec![0,  2,  3,  4,  5,  6,  7,  7, 10, 11, 12, 13, 14, 15, 15, 15, 19,
            20, 21, 22, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24];
        let expected_edge_features : Vec<i32> = vec![3, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 3, 4, 3, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let expected_node_ids : Vec<i32> = vec![1, 2, 1, 6, 7, 8, 9, 2, 3, 4, 2, 6, 7, 8, 9, 3, 4, 5, 6, 3, 6, 7,
            8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let expected_node_labels : Vec<i32> = vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let expected_node_types : Vec<i32> = vec![0, 1, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2,
            2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let expected_candidate_mask : Vec<bool> = vec![
            false, false,  true,  true,  true,  true, true, false, false,
            false, true,  true,  true,  true,  true, false, false, false,
            false, true,  true,  true,  true,  true, false, false, false,
            false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false,
            false];


        assert_eq!(padded_super_graph.n_node, expected_n_node);
        assert_eq!(padded_super_graph.n_edge, expected_n_edge);
        assert_eq!(padded_super_graph.senders, expected_senders);
        assert_eq!(padded_super_graph.receivers, expected_receivers);
        assert_eq!(padded_super_graph.edge_features, expected_edge_features);
        assert_eq!(padded_super_graph.node_ids, expected_node_ids);
        assert_eq!(padded_super_graph.node_labels, expected_node_labels);
        assert_eq!(padded_super_graph.node_types, expected_node_types);
        assert_eq!(padded_super_graph.candidate_mask, expected_candidate_mask);

        /*
        EXPECTED from python:
        padded_super_graph_1=
G       raphsTuple(nodes={'candidate_mask': array([false, false,  true,  true,  true,  true,  true, false, false,
       false,  true,  true,  true,  true,  true, false, false, false,
       false,  true,  true,  true,  true,  true, false, false, false,
       false, false, false, false, false, false, false, false, false,
       false, false, false, false, false, false, false, false, false,
       false, false, false, false, false, false, false, false, false,
       false, false, false, false, false, false, false, false, false,
       false]), 'ids': array([1, 2, 1, 6, 7, 8, 9, 2, 3, 4, 2, 6, 7, 8, 9, 3, 4, 5, 6, 3, 6, 7,
       8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       'label': array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       'type': array([0, 1, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2,
       2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=int32)}, edges={
      'rating': array([3, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 3, 4, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
       receivers=array([ 0,  2,  3,  4,  5,  6,  7,  7, 10, 11, 12, 13, 14, 15, 15, 15, 19,
       20, 21, 22, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
       24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
       24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]),
       senders=array([ 1,  0,  0,  0,  0,  0,  8,  9,  7,  7,  7,  7,  7, 16, 17, 18, 15,
       15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
       24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
       24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]),
       globals=None,
       n_node=array([ 7,  8,  9, 40,  0]),
       n_edge=array([ 6,  7,  8, 43,  0]))

         */

    }

    fn read_user_timestamps(ratings_uri: &str, read_rows: &[i32]) -> (Vec<i32>, Vec<i64>) {
        let file = File::open(ratings_uri).expect("Failed to open the parquet file");

        // 1. Build the reader without projection.
        // It will read all columns into the batch, and we will pick the ones we want.
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader: ParquetRecordBatchReader = builder.build().unwrap();

        let mut user_ids = Vec::new();
        let mut timestamps = Vec::new();

        let mut current_row_index: usize = 0;
        let num_rows = read_rows.len();
        let mut count = 0;

        // Iterate through record batches
        for maybe_batch in reader {
            let batch = maybe_batch.unwrap();
            let batch_size = batch.num_rows();

            // Check if our target rows fall into this batch
            for &target_i32 in read_rows {
                let target = target_i32 as usize; // Cast to usize for math/comparisons

                if target >= current_row_index && target < current_row_index + batch_size {
                    let local_idx = target - current_row_index;

                    // Extract Column 0 (User ID - Int32)
                    let col0 = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    user_ids.push(col0.value(local_idx));

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

        (user_ids, timestamps)
    }

    #[test]
    pub fn test_create_inference_batch() {

        // to compare results to python test_ranker.py method test_create_inference_batch()

        let max_history = 4;
        let batch_size = 2;
        let num_candidates = 5;
        let n_local_devices = 1;

        let ratings_map = get_train_val_test_liked_uris(DataSize::Tiny, false);

        let r = ratings_map.get("train_liked").unwrap();
        let rows : Vec<i32> = vec![3, 4];

        let (user_ids, timestamps) = read_user_timestamps(&r, &rows);

        // storing the items as references
        let ratings_uris: Vec<&str> = vec![&r];

        let user_history : UserHistory = build_user_history(&ratings_uris, 2048);

        /*
        candidate_ids = np.array([
            [6610, 6252, 9083, 6564, 6584],
            [9477, 6941, 6948, 8475, 6356]
        ])
         */
        let candidate_ids : Vec<i32> = vec![6610, 6252, 9083, 6564, 6584, 9477, 6941, 6948, 8475, 6356];

        //labels aren't used in inference.  a value of -1 can help distinguish that is isn't used.
        let labels: Vec<i32> = vec![1; candidate_ids.len()];

        let padded_super_graph : JraphGraph = build_enriched_padded_supergraph(&user_ids, &timestamps,
            &candidate_ids, &labels, &user_history, max_history,
            n_local_devices);

        print!("graph={:?}", padded_super_graph);

        let expected_n_node : Vec<i32> = vec![9,  9, 46,  0];
        let expected_n_edge : Vec<i32> = vec![ 8,  8, 48,  0];

        let expected_senders : Vec<i32> = vec![
            1,  2,  3,  0,  0,  0,  0,  0, 10, 11, 12,  9,  9,  9,  9,  9, 18,
            18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
            18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
            18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18
        ];
        let expected_receivers : Vec<i32> = vec![
            0,  0,  0,  4,  5,  6,  7,  8,  9,  9,  9, 13, 14, 15, 16, 17, 18,
            18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
            18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
            18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18
        ];
        let expected_edge_features : Vec<i32> = vec![
            4, 5, 4, 0, 0, 0, 0, 0, 4, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ];
        let expected_node_ids : Vec<i32> = vec![
            6040, 6888, 6630, 8356, 6610, 6252, 9083, 6564, 6584, 6040, 6888,
            6630, 8356, 9477, 6941, 6948, 8475, 6356,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0
        ];
        let expected_node_labels : Vec<i32> = vec![
            0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ];
        let expected_node_types : Vec<i32> = vec![
            0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ];
        let expected_candidate_mask : Vec<bool> = vec![
            false, false, false, false,  true,  true,  true,  true,  true,
            false, false, false, false,  true,  true,  true,  true,  true,
            false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false,
            false
        ];


        assert_eq!(padded_super_graph.n_node, expected_n_node);
        assert_eq!(padded_super_graph.n_edge, expected_n_edge);
        assert_eq!(padded_super_graph.senders, expected_senders);
        assert_eq!(padded_super_graph.receivers, expected_receivers);
        assert_eq!(padded_super_graph.edge_features, expected_edge_features);
        assert_eq!(padded_super_graph.node_ids, expected_node_ids);
        assert_eq!(padded_super_graph.node_labels, expected_node_labels);
        assert_eq!(padded_super_graph.node_types, expected_node_types);
        assert_eq!(padded_super_graph.candidate_mask, expected_candidate_mask);

        /*
        from python:
         GraphsTuple(nodes={'candidate_mask': array([false, false, false, false,  true,  true,  true,  true,  true,
       false, false, false, false,  true,  true,  true,  true,  true,
       false, false, false, false, false, false, false, false, false,
       false, false, false, false, false, false, false, false, false,
       false, false, false, false, false, false, false, false, false,
       false, false, false, false, false, false, false, false, false,
       false, false, false, false, false, false, false, false, false,
       false]),
       'ids': array([6040, 6888, 6630, 8356, 6610, 6252, 9083, 6564, 6584, 6040, 6888,
       6630, 8356, 9477, 6941, 6948, 8475, 6356,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0]),
          'label': array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=int32),
      'type': array([0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=int32)},
      edges={'rating': array([4, 5, 4, 0, 0, 0, 0, 0, 4, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      dtype=int32)},
      receivers=array([ 0,  0,  0,  4,  5,  6,  7,  8,  9,  9,  9, 13, 14, 15, 16, 17, 18,
       18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
       18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
       18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]),
       senders=array([ 1,  2,  3,  0,  0,  0,  0,  0, 10, 11, 12,  9,  9,  9,  9,  9, 18,
       18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
       18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
       18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]),
       globals=None,
       n_node=array([ 9,  9, 46,  0]),
       n_edge=array([ 8,  8, 48,  0]))

         */

    }
}