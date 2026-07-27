#[cfg(test)]
mod calc_metrics_tests {
    use std::collections::HashSet;
    //use std::error::Error;
    use inference_engine::util::{calc_number_jax_graph_components, sort_by_scores};
    //use super::*;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    use helper::{get_project_dir, get_model_param_json_uri};
    use std::path::PathBuf;
    use std::fs::File;
    use std::collections::HashMap;
    use serde_json::Value;
    use inference_engine::calc_metrics::{Evaluator};

    #[test]
    pub fn test_calc_metrics() {

        let mut file_patterns : Option<PathBuf> = get_project_dir();
        if let Some(ref mut p) = file_patterns {
            p.push("src/test/resources/data/src/test/resources/data/ratings_test*.parquet");
        }

        let params_json_uri = get_model_param_json_uri();
        let file = File::open(params_json_uri).unwrap();
        let reader = BufReader::new(file);
        let dict: HashMap<String, Value> = serde_json::from_reader(reader).unwrap();
        let max_history = dict.get("max_history")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_candidates = dict.get("num_candidates")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_catalog_users = dict.get("num_catalog_users")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let top_k = 20;

        let evaluator = Evaluator::new(top_k=top_k, num_candidates=num_candidates,
            n_draws = 3, rating_threshold = 3);

        match evaluator.evaluate(file_patterns) {
            Ok(stats) => {
                println!("=== Evaluation Results (K=20) ===");
                println!("{:<12} | {:<8} | {:<8} | {:<8} | {:<8}", "Metric", "Mean", "Median", "MAD", "MAD-Std");
                println!("{:-<56}", "");
                println!("{:<12} | {:.4}   | {:.4}   | {:.4}   | {:.4}", "NDCG", stats.mean.ndcg, stats.median.ndcg, stats.mad.ndcg, stats.mad_std.ndcg);
                println!("{:<12} | {:.4}   | {:.4}   | {:.4}   | {:.4}", "MRR", stats.mean.mrr, stats.median.mrr, stats.mad.mrr, stats.mad_std.mrr);
                println!("{:<12} | {:.4}   | {:.4}   | {:.4}   | {:.4}", "Recall", stats.mean.recall, stats.median.recall, stats.mad.recall, stats.mad_std.recall);
                println!("{:<12} | {:.4}   | {:.4}   | {:.4}   | {:.4}", "Precision", stats.mean.precision, stats.median.precision, stats.mad.precision, stats.mad_std.precision);
                println!("{:<12} | {:.4}   | {:.4}   | {:.4}   | {:.4}", "F1-Score", stats.mean.f1, stats.median.f1, stats.mad.f1, stats.mad_std.f1);
            }
            Err(e) => eprintln!("Evaluation Failed: {}", e),
        }

    }
}