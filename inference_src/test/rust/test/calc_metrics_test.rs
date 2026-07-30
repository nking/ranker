#[cfg(test)]
mod calc_metrics_tests {
    use tokio::sync::oneshot;
    use tokio::task::JoinHandle;
    use std::error::Error;

    struct TestServerGuard {
        tx_shutdown: Option<oneshot::Sender<()>>,
        server_handle: Option<JoinHandle<()>>,
    }

    impl Drop for TestServerGuard {
        fn drop(&mut self) {
            // Trigger the shutdown signal
            if let Some(tx) = self.tx_shutdown.take() {
                let _ = tx.send(());
            }

            // Safely block and join the background server thread using block_in_place
            if let Some(handle) = self.server_handle.take() {
                let _ = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(handle)
                });
            }
        }
    }

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    use helper::{get_project_dir, get_model_param_json_uri};
    use std::path::PathBuf;
    use std::fs::File;
    use std::collections::{HashMap};
    use std::io::BufReader;
    use serde_json::Value;
    use inference_engine::app_config::AppConfig;
    use inference_engine::app_runner::AppRunner;
    use inference_engine::calc_metrics::{Evaluator, MetricStats};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    pub async fn test_calc_test_metrics() {

        // ======= setup server ======================

        let params_json_uri = get_model_param_json_uri();
        let file = File::open(params_json_uri).unwrap();
        let reader = BufReader::new(file);
        let dict: HashMap<String, Value> = serde_json::from_reader(reader).unwrap();
        let _max_history = dict.get("max_history")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_candidates = dict.get("num_candidates")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let _num_catalog_users = dict.get("num_catalog_users")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let config_path = "./config/default.json";
        let config = AppConfig::load_from_file(config_path).unwrap();

        let top_k = config.top_k;

        //let user_db_path = &config.user_db_path;

        let runner = AppRunner::new(config.to_owned());

        let (tx_shutdown, rx_shutdown) = oneshot::channel::<()>();
        let (tx_addr, rx_addr) = oneshot::channel::<std::net::SocketAddr>();
        // Spawn the server in a background Tokio task
        let server_handle = tokio::spawn(async move {
            let shutdown_future = async {
                rx_shutdown.await.ok();
            };
            runner.run(shutdown_future, Some(tx_addr)).await.expect("Server crashed");
        });
        let addr = rx_addr.await.expect("Failed to receive server address");
        // Initialize the RAII Guard.
        // It will automatically trigger shutdown and join if the test finishes or panics.
        let _server_guard = TestServerGuard {
            tx_shutdown: Some(tx_shutdown),
            server_handle: Some(server_handle),
        };

        let endpoint = format!("http://{}", addr);

        // ===== run tests ==========

        if true {
            run_evaluator(top_k, num_candidates, endpoint).await;
        }



        // server is shutdown by the guard when this method is out of scope

    }

    async fn run_evaluator(top_k: usize, num_candidates: usize,  endpoint: String) {
        let n_draws = 10;
        let mut test_file_patterns: Option<PathBuf> = get_project_dir();
        if let Some(ref mut p) = test_file_patterns {
            p.push("src/test/resources/data/ratings_test*.parquet");
        }

        let evaluator = Evaluator::new(top_k, num_candidates, n_draws,
            endpoint).await;

        match evaluator {
            Ok(evaluator) => {
                let results : Result<MetricStats, Box<dyn Error>>
                    = evaluator.evaluate(test_file_patterns.unwrap().to_str().unwrap()).await;
                match results {
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
            Err(e) => eprintln!("Evaluator construction Failed: {}", e),
        }
        /*
       == Evaluation Results (K=20) ===
       Metric       | Mean     | Median   | MAD      | MAD-Std
       --------------------------------------------------------
       NDCG         | 0.6857   | 0.6912   | 0.1242   | 0.1842
       MRR          | 0.8465   | 0.9500   | 0.0500   | 0.0741
       Recall       | 0.4768   | 0.4666   | 0.0457   | 0.0678
       Precision    | 0.6459   | 0.6600   | 0.1400   | 0.2076
       F1-Score     | 0.5273   | 0.5485   | 0.0463   | 0.0687
       */
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    pub async fn test_polarized_metrics() {
        /*
        given a file of :
        user_id,total_pos_ratings,count_Action,count_Adventure,count_Animation,count_Children,
        count_Comedy,count_Crime,count_Documentary,count_Drama,count_Fantasy,count_Film-Noir,
        count_Horror,count_Musical,count_Mystery,count_Romance,count_Sci-Fi,count_Thriller,
        count_War,count_Western,ratio_Action,ratio_Adventure,ratio_Animation,ratio_Children,
        ratio_Comedy,ratio_Crime,ratio_Documentary,ratio_Drama,ratio_Fantasy,ratio_Film-Noir,
        ratio_Horror,ratio_Musical,ratio_Mystery,ratio_Romance,ratio_Sci-Fi,ratio_Thriller,
        ratio_War,ratio_Western,max_genre_ratio,max_genre_count

        get user_ids, make requests for recommendations
        then
         */
    }
}