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
    use helper::{get_train_val_test_liked_uris, DataSize};

    use prep_inputs_for_graphranker::graph_builder::{create_fake_padded_super_batch, JraphGraph};

    #[test]
    pub fn test_create_fake_batch() {

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
        let expected_node_labels : Vec<f32> = vec![0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
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


        assert_eq!(padded_super_graph.n_node.len(), expected_n_node.len());
        assert_eq!(padded_super_graph.n_edge.len(), expected_n_edge.len());




        let zz = 2;
        /*
        padded_super_graph=JraphGraph { n_node: [7, 8, 9, 40, 0], n_edge: [6, 7, 8, 43, 0],
        senders: [1, 0, 0, 0, 0, 0, 8, 9, 7, 7, 7, 7, 7, 16, 17, 18, 15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24],
        receivers: [0, 2, 3, 4, 5, 6, 7, 7, 10, 11, 12, 13, 14, 15, 15, 15, 19, 20, 21, 22, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24],
        edge_features: [3, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        node_ids: [1, 2, 1, 6, 7, 8, 9, 2, 3, 4, 2, 6, 7, 8, 9, 3, 4, 5, 6, 3, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        node_labels: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        node_types: [0, 1, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        candidate_mask: [false, false, true, true, true, true, true, false, false, false, true, true, true, true, true, false, false, false, false, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false] }

         */
        /*
        EXPECTED from python:
        padded_super_graph_1=
GraphsTuple(nodes={'candidate_mask': array([False, False,  True,  True,  True,  True,  True, False, False,
       False,  True,  True,  True,  True,  True, False, False, False,
       False,  True,  True,  True,  True,  True, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False]), 'ids': array([1, 2, 1, 6, 7, 8, 9, 2, 3, 4, 2, 6, 7, 8, 9, 3, 4, 5, 6, 3, 6, 7,
       8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       'label': array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
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

}