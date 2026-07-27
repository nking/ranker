use glob::glob;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::fs::File;

#[derive(Debug, Clone)]
pub struct Interaction {
    pub movie_id: usize,
    pub rating: usize,
}

#[derive(Debug, Default, Clone)]
pub struct Metrics {
    pub ndcg: f64,
    pub mrr: f64,
    pub recall: f64,
    pub precision: f64,
    pub f1: f64,
}

#[derive(Debug)]
pub struct MetricStats {
    pub mean: Metrics,
    pub median: Metrics,
    pub mad: Metrics,
    pub mad_std: Metrics,
}

pub struct Evaluator {
    pub top_k: usize,
    pub num_candidates: usize,
    pub n_draws: usize,
    pub rating_threshold: usize,
}

impl Evaluator {
    pub fn new(top_k: usize, num_candidates: usize, n_draws: usize) -> Self {
        Self {
            top_k,
            num_candidates,
            n_draws,
            rating_threshold: 3,
        }
    }

    /// -----------------------------------------------------------------------
    /// STUB: Inference Endpoint Hook
    /// -----------------------------------------------------------------------
    /// You will fill this out with your gRPC or JSON-RPC implementation.
    /// It takes a user_id and a list of candidate movie_ids, and returns the
    /// top K movie_ids sorted descending by score.
    fn get_top_k_scored_candidates(&self, _user_id: usize, candidate_ids: &[usize]) -> Vec<usize> {
        // TODO: Implement gRPC/JSON-RPC request here.
        // Simulated response for now: just return the first K items.
        candidate_ids.iter().take(self.top_k).copied().collect()
    }

    /// Reads all parquet files matching the pattern using the native parquet crate
    fn load_data(&self, file_pattern: &str) -> Result<HashMap<usize, Vec<Interaction>>, Box<dyn std::error::Error>> {

        let mut ratings_history: HashMap<usize, Vec<Interaction>> = HashMap::new();

        for entry in glob(file_pattern)? {
            let path = entry?;
            let file = File::open(&path)?;
            let reader = SerializedFileReader::new(file)?;

            // Iterate over all rows in the parquet file
            for row_result in reader.get_row_iter(None)? {
                let row = row_result?;

                // Assuming schema order: user_id (0), movie_id (1), rating (2), timestamp (3)
                // Parquet usually maps large integers to INT64 (`get_long`)
                let uid = row.get_int(0)? as usize;
                let mid = row.get_int(1)? as usize;
                let rat = row.get_int(2)? as usize;
                //let ts = row.get_long(3)? as usize;

                ratings_history
                    .entry(uid)
                    .or_default()
                    .push(Interaction {
                        movie_id: mid,
                        rating: rat,
                    });
            }
        }
        Ok(ratings_history)
    }

    /// Core metric calculation for a single ranked list
    fn compute_metrics_at_k(&self, ranked_candidates: &[&Interaction], total_positives: usize) -> Metrics {
        let n = ranked_candidates.len().min(self.top_k);
        let mut hits = 0;
        let mut dcg = 0.0;
        let mut mrr = 0.0;

        for i in 0..n {
            if ranked_candidates[i].rating > self.rating_threshold {
                if hits == 0 {
                    mrr = 1.0 / (i + 1) as f64;
                }
                hits += 1;
                dcg += 1.0 / ((i + 2) as f64).log2();
            }
        }

        let ideal_hits = total_positives.min(self.top_k);
        let mut idcg = 0.0;
        for i in 0..ideal_hits {
            idcg += 1.0 / ((i + 2) as f64).log2();
        }

        let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };
        let precision = hits as f64 / self.top_k as f64;
        let recall = if total_positives > 0 { hits as f64 / total_positives as f64 } else { 0.0 };

        let f1 = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        Metrics { ndcg, mrr, recall, precision, f1 }
    }

    /// Runs the Monte Carlo evaluation process
    pub fn evaluate(&self, file_pattern: &str) -> Result<MetricStats, Box<dyn std::error::Error>> {
        let ratings_history = self.load_data(file_pattern)?;
        let mut rng = thread_rng();
        let mut user_averaged_metrics: Vec<Metrics> = Vec::with_capacity(ratings_history.len());

        for (user_id, history) in ratings_history.into_iter() {
            if history.len() < self.num_candidates {
                continue;
            }

            let mut user_draw_totals = Metrics::default();
            let mut valid_draws = 0;

            for _ in 0..self.n_draws {
                // Draw num_candidates
                let candidates: Vec<&Interaction> = history
                    .choose_multiple(&mut rng, self.num_candidates)
                    .collect();

                let total_positives_in_draw = candidates
                    .iter()
                    .filter(|c| c.rating > self.rating_threshold)
                    .count();

                if total_positives_in_draw == 0 {
                    continue;
                }

                //  Extract IDs and pass to the Inference method
                let candidate_ids: Vec<usize> = candidates.iter().map(|c| c.movie_id).collect();
                let ranked_movie_ids = self.get_top_k_scored_candidates(user_id, &candidate_ids);

                //TODO: this could be improved by making a candidates hashmap for the search ("find")
                // Map the returned IDs back to the ground truth Interactions
                let mut ranked_interactions = Vec::with_capacity(self.top_k);
                for mid in ranked_movie_ids.iter().take(self.top_k) {
                    if let Some(&interaction) = candidates.iter().find(|c| c.movie_id == *mid) {
                        ranked_interactions.push(interaction);
                    }
                }

                // Calculate metrics
                let draw_metrics = self.compute_metrics_at_k(&ranked_interactions, total_positives_in_draw);

                user_draw_totals.ndcg += draw_metrics.ndcg;
                user_draw_totals.mrr += draw_metrics.mrr;
                user_draw_totals.recall += draw_metrics.recall;
                user_draw_totals.precision += draw_metrics.precision;
                user_draw_totals.f1 += draw_metrics.f1;
                valid_draws += 1;
            }

            // Average metrics over valid draws for this specific user
            if valid_draws > 0 {
                let v = valid_draws as f64;
                user_averaged_metrics.push(Metrics {
                    ndcg: user_draw_totals.ndcg / v,
                    mrr: user_draw_totals.mrr / v,
                    recall: user_draw_totals.recall / v,
                    precision: user_draw_totals.precision / v,
                    f1: user_draw_totals.f1 / v,
                });
            }
        }

        Ok(self.calculate_global_stats(user_averaged_metrics))
    }

    /// Aggregates per-user averages into global Mean, Median, MAD, and MAD-Stdev
    fn calculate_global_stats(&self, metrics: Vec<Metrics>) -> MetricStats {
        let n = metrics.len() as f64;
        if n == 0.0 {
            panic!("No valid users found for evaluation.");
        }

        // --- Helper functions for statistics ---
        fn get_median(values: &mut [f64]) -> f64 {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = values.len() / 2;
            if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            }
        }

        fn get_mad(values: &[f64], median: f64) -> f64 {
            let mut deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
            get_median(&mut deviations)
        }
        // ---------------------------------------

        // Mean
        let mut sum = Metrics::default();
        for m in &metrics {
            sum.ndcg += m.ndcg; sum.mrr += m.mrr; sum.recall += m.recall; sum.precision += m.precision; sum.f1 += m.f1;
        }
        let mean = Metrics {
            ndcg: sum.ndcg / n, mrr: sum.mrr / n, recall: sum.recall / n, precision: sum.precision / n, f1: sum.f1 / n,
        };

        // Extract individual vectors for median & MAD calculations
        let mut ndcg_vals: Vec<f64> = metrics.iter().map(|m| m.ndcg).collect();
        let mut mrr_vals: Vec<f64> = metrics.iter().map(|m| m.mrr).collect();
        let mut recall_vals: Vec<f64> = metrics.iter().map(|m| m.recall).collect();
        let mut precision_vals: Vec<f64> = metrics.iter().map(|m| m.precision).collect();
        let mut f1_vals: Vec<f64> = metrics.iter().map(|m| m.f1).collect();

        // Median
        let median = Metrics {
            ndcg: get_median(&mut ndcg_vals),
            mrr: get_median(&mut mrr_vals),
            recall: get_median(&mut recall_vals),
            precision: get_median(&mut precision_vals),
            f1: get_median(&mut f1_vals),
        };

        // MAD (Median Absolute Deviation)
        let mad = Metrics {
            ndcg: get_mad(&ndcg_vals, median.ndcg),
            mrr: get_mad(&mrr_vals, median.mrr),
            recall: get_mad(&recall_vals, median.recall),
            precision: get_mad(&precision_vals, median.precision),
            f1: get_mad(&f1_vals, median.f1),
        };

        // MAD-based Standard Deviation (MAD * 1.4826)
        let scale = 1.4826;
        let mad_std = Metrics {
            ndcg: mad.ndcg * scale,
            mrr: mad.mrr * scale,
            recall: mad.recall * scale,
            precision: mad.precision * scale,
            f1: mad.f1 * scale,
        };

        MetricStats { mean, median, mad, mad_std }
    }
}