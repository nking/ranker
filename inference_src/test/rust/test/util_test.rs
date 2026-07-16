#[cfg(test)]
mod util_tests {
    use std::collections::HashSet;
    use std::error::Error;
    use inference_engine::util::sort_by_scores;
    use super::*;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    #[test]
    pub fn test_sort() {
        let ids = vec![10, 20, 30, 40];
        let scores = vec![0.15, 0.92, 0.45, 0.88];
        let (sorted_ids, sorted_scores) = sort_by_scores(&ids, &scores);

        assert_eq!(ids.len(), sorted_ids.len());
        assert_eq!(scores.len(), sorted_scores.len());

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
}