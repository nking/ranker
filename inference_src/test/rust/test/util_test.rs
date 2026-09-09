#[cfg(test)]
mod util_tests {
    use std::collections::HashSet;
    //use std::error::Error;
    use inference_engine::util::{calc_number_jax_graph_components, ceiling_search, sort_in_place_by_desc_scores};
    use crate::util_tests::helper::assert_slices_nearly_equal;

    //use super::*;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    #[test]
    pub fn test_sort() {
        let ids = vec![10, 20, 30, 40];
        let scores = vec![0.15, 0.92, 0.45, 0.88];
        let mut sorted_ids = ids.clone();
        let mut sorted_scores = scores.clone();
        sort_in_place_by_desc_scores(&mut sorted_ids, &mut sorted_scores, 4);

        let mut set = HashSet::new();
        let mut last_score: f32 = 2.0;

        for i in 0..ids.len() {
            let id = &sorted_ids[i];
            let score = &sorted_scores[i];

            assert!(*score <= last_score);
            last_score = *score;

            let mut found : i32 = -1;
            for j in 0..ids.len() {
                if &ids[j] == id {
                    found = j as i32;
                    break;
                }
            }
            assert!(scores[found as usize] == *score);
            set.insert(id);
        }
        assert_eq!(set.len(), ids.len());
    }

    #[test]
    pub fn test_sort2() {
        let mut ids = vec![10, 20, 30, 40];
        let mut scores = vec![0.95, 0.92, 0.45, 0.88];

        let expected_ids = vec![10, 20, 40, 30];
        let expected_scores = vec![0.95, 0.92, 0.88, 0.45];
        sort_in_place_by_desc_scores(&mut ids, &mut scores, 2);

        assert_slices_nearly_equal(&scores, &expected_scores, 1E-6);

        for i in 0..ids.len() {
            assert_eq!(ids[i], expected_ids[i]);
        }
        assert_eq!(expected_ids.len(), ids.len());
    }

    #[test]
    pub fn test_calc_number_jax_graph_components() {
        let max_history : usize = 40;
        let num_candidates : usize = 50;
        let batch_size : usize = 1;
        let num_local_devices : usize = 1;
        let (max_nodes,  max_edges, max_graphs) =
            calc_number_jax_graph_components(batch_size, max_history, num_candidates, num_local_devices);
        assert_eq!(max_nodes, 128);
        assert_eq!(max_edges, 128);
        assert_eq!(max_graphs, 3);
    }


    #[test]
    pub fn test_ceiling_search() {
        let ids: Vec<i64> = vec![1, 2, 3, 3, 4];
        let srchs : Vec<i64> = vec![3, 4, 9, 0];
        let expected : Vec<usize> = vec![3, 4, 5, 0];
        for i in 0..srchs.len() {
            let idx = ceiling_search(&ids, srchs[i]);
            assert_eq!(expected[i], idx);
        }
    }
}