use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;
use clap::Parser;
use plotters::prelude::{SVGBackend, ChartBuilder, LineSeries, PathElement, RGBColor, BLACK, WHITE, Color, Rectangle, IntoDrawingArea, IntoFont};
use polars::prelude::*;
//use plotters::prelude::*;
use serde::Serialize;
use serde_json::{json, Value};

pub mod helper {
    // Tell Rust to literally include the code from helper.rs here
    include!("helper.rs");
    include!("helper_users.rs");
}
use helper::{ensure_plotters_font_registered};

/// Command-line arguments for comparing two post-training analysis runs
#[derive(Parser, Debug, Clone)]
#[command(name = "compare-post-training-runs")]
#[command(about = "Compares summary JSONs and runs paired t-tests & Wilcoxon signed-rank tests on Parquet outputs")]
pub struct CompareArgs {
    /// Path to Baseline (Run A) summary_output_dir (e.g. post_training_analysis_q01_r01)
    #[arg(long)]
    pub summary_dir_a: PathBuf,

    /// Path to Baseline (Run A) parquet_output_dir (e.g. parquet_metrics_q01_r01)
    #[arg(long)]
    pub parquet_dir_a: PathBuf,

    /// Path to Candidate (Run B) summary_output_dir (e.g. post_training_analysis_q02_r02)
    #[arg(long)]
    pub summary_dir_b: PathBuf,

    /// Path to Candidate (Run B) parquet_output_dir (e.g. parquet_metrics_q02_r02)
    #[arg(long)]
    pub parquet_dir_b: PathBuf,

    /// Directory where summary_metrics.json and summary_conclusions.json will be written
    #[arg(long)]
    pub output_dir: PathBuf,

    /// Optional readable name for Run A (defaults to summary_dir_a folder name)
    #[arg(long)]
    pub label_a: Option<String>,

    /// Optional readable name for Run B (defaults to summary_dir_b folder name)
    #[arg(long)]
    pub label_b: Option<String>,

    /// Significance threshold alpha for paired t-test and Wilcoxon test
    #[arg(long, default_value_t = 0.05)]
    pub alpha: f64,
}

// ==============================================================================
// CONFIGURATION: METRIC POLARITY & PARQUET SCHEMAS
// ==============================================================================

struct ParquetTestSpec {
    prefix: &'static str,
    join_key: &'static str,
    columns: &'static [&'static str],
}

const PARQUET_TEST_SPECS: &[ParquetTestSpec] = &[
    ParquetTestSpec {
        prefix: "stratified_metrics_top_",
        join_key: "user_id",
        columns: &[
            "recall_global",
            "precision_global",
            "ndcg_global",
            "rr_global",
            "recall_tier_0",
            "recall_tier_1",
            "recall_tier_2",
            "precision_tier_0",
            "precision_tier_1",
            "precision_tier_2",
            "ndcg_tier_0",
            "ndcg_tier_1",
            "ndcg_tier_2",
            "rr_tier_0",
            "rr_tier_1",
            "rr_tier_2",
        ],
    },
    ParquetTestSpec {
        prefix: "popularity_bias_k_",
        join_key: "user_id",
        columns: &["rec_mean_pop", "pop_bias_delta"],
    },
    ParquetTestSpec {
        prefix: "diversity_metrics_k_",
        join_key: "user_id",
        columns: &["interlist_diversity", "intralist_diversity"],
    },
    ParquetTestSpec {
        prefix: "hubness_user_stats_k_",
        join_key: "user_id",
        columns: &["avg_slate_nk", "avg_slate_sim_to_centroid"],
    },
    ParquetTestSpec {
        prefix: "hubness_item_stats_k_",
        join_key: "movie_id",
        columns: &["retrieval_count", "sim_to_user_centroid"],
    },
    ParquetTestSpec {
        prefix: "contextual_hubness_user_stats_k_",
        join_key: "user_id",
        columns: &["avg_slate_nk", "avg_slate_anisotropy"],
    },
    ParquetTestSpec {
        prefix: "contextual_hubness_item_stats_k_",
        join_key: "movie_id",
        columns: &["retrieval_count", "avg_sim_to_user"],
    },
    ParquetTestSpec {
        prefix: "rank_agreement_k_",
        join_key: "user_id",
        columns: &["overlap_at_k", "spearman_rho"],
    },
    ParquetTestSpec {
        prefix: "item_frequencies_top_",
        join_key: "movie_id",
        columns: &["retrieval_count"],
    },
];

/// Returns +1 if higher is better, -1 if lower is better, 0 if neutral/diagnostic.
fn infer_polarity(metric_name: &str) -> i32 {
    let lower = metric_name.to_lowercase();
    if lower.contains("random")
        || lower.contains("n_samples")
        || lower.contains("hist_mean")
        || lower.contains("n_unique_entities")
        || lower.contains("n_multi_occurrence_entities")
    {
        return 0;
    }

    let positive_keys = [
        "recall",
        "precision",
        "ndcg",
        "mrr",
        "rr_global",
        "rr_tier",
        "coverage",
        "bottom_80_share",
        "interlist_diversity",
        "intralist_diversity",
        "mean_resultant_length",
        "vmf_kappa",
        "pct_high_fidelity",
    ];
    for k in positive_keys {
        if lower.contains(k) {
            return 1;
        }
    }

    let negative_keys = [
        "gini",
        "top_10_share",
        "pop_bias_delta",
        "pop_bias_rec_mean",
        "rec_mean_pop",
        "hubness_skewness",
        "hubness_max_nk",
        "hubness_centroid_corr",
        "hub_exposure_mean_nk",
        "hub_exposure_mean_sim",
        "avg_slate_nk",
        "avg_slate_sim_to_centroid",
        "ctx_hubness_skewness",
        "ctx_hubness_max_nk",
        "ctx_hubness_mean_sim",
        "ctx_hubness_centroid_corr",
        "ctx_exposure_mean_nk",
        "ctx_exposure_mean_anisotropy",
        "avg_slate_anisotropy",
        "avg_sim_to_user",
        "sim_to_user_centroid",
        "spherical_variance",
        "mean_angle_deg",
        "p90_angle_deg",
        "robust_mad_angle_deg",
        "pct_context_polysemous",
    ];
    for k in negative_keys {
        if lower.contains(k) {
            return -1;
        }
    }

    0
}

#[inline]
fn clean_f64(val: f64) -> Option<f64> {
    if val.is_finite() {
        Some(val)
    } else {
        None
    }
}

// ==============================================================================
// PURE-RUST STATISTICAL DISTRIBUTIONS (Student's t & Normal/Wilcoxon)
// ==============================================================================

/// Complementary error function erfc(x) via Abramowitz & Stegun / Chebyshev approximation
fn erfc_approx(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587
                                        + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Lanczos approximation for ln(Gamma(z))
fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * z).sin()).ln() - ln_gamma(1.0 - z);
    }
    let z_adj = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (z_adj + (i as f64));
    }
    let t = z_adj + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z_adj + 0.5) * t.ln() - t + x.ln()
}

/// Lentz's continued fraction algorithm for the Incomplete Beta Function
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa2 = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta function I_x(a, b)
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Two-tailed p-value for Student's t-distribution with `df` degrees of freedom
fn student_t_two_tailed_pvalue(t_stat: f64, df: f64) -> f64 {
    if df <= 0.0 || !t_stat.is_finite() {
        return 1.0;
    }
    let x = df / (df + t_stat * t_stat);
    regularized_incomplete_beta(x, 0.5 * df, 0.5).clamp(0.0, 1.0)
}

/// Inverse two-tailed Student's t critical value via bisection
fn student_t_critical_two_tailed(alpha: f64, df: f64) -> f64 {
    let mut low = 0.0;
    let mut high = 1000.0;
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        let p = student_t_two_tailed_pvalue(mid, df);
        if p > alpha {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// Wilcoxon signed-rank test (with tie-rank averaging and tie variance correction, `wilcox` zero handling)
fn wilcoxon_signed_rank_test(diffs: &[f64]) -> (f64, f64, usize) {
    // Exclude zero differences (|d| <= 1e-15)
    let mut nz: Vec<(f64, f64)> = diffs
        .iter()
        .copied()
        .filter(|d| d.is_finite() && d.abs() > 1e-15)
        .map(|d| (d.abs(), d.signum()))
        .collect();

    let n_r = nz.len();
    if n_r == 0 {
        return (0.0, 1.0, 0);
    }

    nz.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut w_plus = 0.0;
    let mut w_minus = 0.0;
    let mut tie_correction_sum = 0.0;

    let mut i = 0;
    while i < n_r {
        let mut j = i + 1;
        while j < n_r && (nz[j].0 - nz[i].0).abs() <= 1e-15 {
            j += 1;
        }
        // 1-based average rank for tied elements in [i, j)
        let avg_rank = 0.5 * (((i + 1) + j) as f64);
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_correction_sum += t * t * t - t;
        }
        for k in i..j {
            if nz[k].1 > 0.0 {
                w_plus += avg_rank;
            } else {
                w_minus += avg_rank;
            }
        }
        i = j;
    }

    let w_stat = w_plus.min(w_minus);
    let n_f = n_r as f64;
    let mean_w = n_f * (n_f + 1.0) / 4.0;
    let var_w = (n_f * (n_f + 1.0) * (2.0 * n_f + 1.0)) / 24.0 - (tie_correction_sum / 48.0);

    if var_w <= 0.0 {
        return (w_stat, 1.0, n_r);
    }

    let z = (w_stat - mean_w) / var_w.sqrt();
    let p_val = erfc_approx(z.abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0);

    (w_stat, p_val, n_r)
}

// ==============================================================================
// PART 1: SUMMARY JSON COMPARISON
// ==============================================================================

#[derive(Serialize, Debug, Clone)]
pub struct ScalarMetricComparison {
    pub baseline_a: f64,
    pub candidate_b: f64,
    pub abs_diff: Option<f64>,
    pub rel_diff_pct: Option<f64>,
    pub polarity: i32,
    pub status: String,
}

fn compare_scalar_maps(
    map_a: &serde_json::Map<String, Value>,
    map_b: &serde_json::Map<String, Value>,
) -> BTreeMap<String, ScalarMetricComparison> {
    let mut out = BTreeMap::new();
    let all_keys: BTreeSet<&String> = map_a.keys().chain(map_b.keys()).collect();

    for key in all_keys {
        let Some(val_a) = map_a.get(key).and_then(|v| v.as_f64()).and_then(clean_f64) else {
            continue;
        };
        let Some(val_b) = map_b.get(key).and_then(|v| v.as_f64()).and_then(clean_f64) else {
            continue;
        };

        let abs_diff = val_b - val_a;
        let rel_diff_pct = if val_a.abs() > 1e-12 {
            clean_f64(((val_b - val_a) / val_a.abs()) * 100.0)
        } else {
            None
        };
        let polarity = infer_polarity(key);

        let status = if abs_diff.abs() <= 1e-9 {
            "unchanged".to_string()
        } else if polarity == 1 {
            if abs_diff > 0.0 { "improved" } else { "degraded" }.to_string()
        } else if polarity == -1 {
            if abs_diff < 0.0 { "improved" } else { "degraded" }.to_string()
        } else if abs_diff > 0.0 {
            "increased".to_string()
        } else {
            "decreased".to_string()
        };

        out.insert(
            key.clone(),
            ScalarMetricComparison {
                baseline_a: val_a,
                candidate_b: val_b,
                abs_diff: clean_f64(abs_diff),
                rel_diff_pct,
                polarity,
                status,
            },
        );
    }

    out
}

fn list_files_with_ext(dir: &Path, ext: &str) -> Result<BTreeMap<String, PathBuf>, Box<dyn Error + Send + Sync>> {
    let mut map = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(ext) {
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                map.insert(name.to_string(), path);
            }
        }
    }
    Ok(map)
}

pub fn compare_summary_dirs(
    summary_dir_a: &Path,
    summary_dir_b: &Path,
) -> Result<(BTreeMap<String, Value>, BTreeMap<String, Value>), Box<dyn Error + Send + Sync>> {
    let files_a = list_files_with_ext(summary_dir_a, "json")?;
    let files_b = list_files_with_ext(summary_dir_b, "json")?;

    let mut summary_diffs = BTreeMap::new();
    let mut run_conclusions = BTreeMap::new();

    for (fname, path_a) in &files_a {
        // Skip consolidated rollup files generated by post_training_analysis or prior comparisons
        if fname == "summary_metrics.json" {
            continue;
        }
        let Some(path_b) = files_b.get(fname) else {
            continue;
        };

        let val_a: Value = serde_json::from_reader(BufReader::new(File::open(path_a)?))?;
        let val_b: Value = serde_json::from_reader(BufReader::new(File::open(path_b)?))?;

        // Lorenz curves are top-level JSON arrays of 100 floats
        if fname.starts_with("lorenz_curve_") {
            if let (Some(arr_a), Some(arr_b)) = (val_a.as_array(), val_b.as_array()) {
                if arr_a.len() == arr_b.len() && !arr_a.is_empty() {
                    let diffs: Vec<f64> = arr_a
                        .iter()
                        .zip(arr_b.iter())
                        .filter_map(|(a, b)| Some(b.as_f64()? - a.as_f64()?))
                        .collect();
                    let mean_shift = diffs.iter().sum::<f64>() / (diffs.len() as f64);
                    let max_pos = diffs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let max_neg = diffs.iter().copied().fold(f64::INFINITY, f64::min);

                    summary_diffs.insert(
                        fname.clone(),
                        json!({
                            "lorenz_mean_shift": clean_f64(mean_shift),
                            "lorenz_max_positive_shift": clean_f64(max_pos),
                            "lorenz_max_negative_shift": clean_f64(max_neg),
                        }),
                    );
                }
            }
            continue;
        }

        let (Some(obj_a), Some(obj_b)) = (val_a.as_object(), val_b.as_object()) else {
            continue;
        };

        // Case 1: Top-level "metrics" and "automated_conclusions" (contextual_hubness, rank_agreement)
        if let (Some(m_a), Some(m_b)) = (
            obj_a.get("metrics").and_then(|v| v.as_object()),
            obj_b.get("metrics").and_then(|v| v.as_object()),
        ) {
            let cmp = compare_scalar_maps(m_a, m_b);
            summary_diffs.insert(fname.clone(), json!({ "default": cmp }));
            run_conclusions.insert(
                fname.clone(),
                json!({
                    "baseline_a": obj_a.get("automated_conclusions").cloned().unwrap_or_else(|| json!([])),
                    "candidate_b": obj_b.get("automated_conclusions").cloned().unwrap_or_else(|| json!([])),
                }),
            );
        }
        // Case 2: cross_metric_analysis.json or summary_conclusions.json
        else if obj_a.contains_key("cross_metric_conclusions") {
            run_conclusions.insert(
                fname.clone(),
                json!({
                    "baseline_a": obj_a.get("cross_metric_conclusions").cloned().unwrap_or_else(|| json!([])),
                    "candidate_b": obj_b.get("cross_metric_conclusions").cloned().unwrap_or_else(|| json!([])),
                }),
            );
        } else if fname == "summary_conclusions.json" {
            if obj_a.contains_key("executive_summary") || obj_b.contains_key("executive_summary") {
                run_conclusions.insert(
                    fname.clone(),
                    json!({
                        "baseline_a": obj_a.get("executive_summary").cloned().unwrap_or_else(|| json!([])),
                        "candidate_b": obj_b.get("executive_summary").cloned().unwrap_or_else(|| json!([])),
                    }),
                );
            }
        }
        // Case 3: Tag-keyed blocks (retrieval_ann_100, retrieval_ann_20, ranker_20, cold_start_*, gatv2_spherical_fidelity)
        else {
            let mut file_diff = BTreeMap::new();
            let mut file_concl = BTreeMap::new();

            for (tag, block_a_val) in obj_a {
                let Some(block_b_val) = obj_b.get(tag) else {
                    continue;
                };
                if let (Some(block_a), Some(block_b)) = (block_a_val.as_object(), block_b_val.as_object()) {
                    if let (Some(m_a), Some(m_b)) = (
                        block_a.get("metrics").and_then(|v| v.as_object()),
                        block_b.get("metrics").and_then(|v| v.as_object()),
                    ) {
                        file_diff.insert(tag.clone(), compare_scalar_maps(m_a, m_b));
                    } else {
                        // Direct scalar map inside the tag block (e.g. gatv2_spherical_fidelity.json)
                        let direct_cmp = compare_scalar_maps(block_a, block_b);
                        if !direct_cmp.is_empty() {
                            file_diff.insert(tag.clone(), direct_cmp);
                        }
                    }
                    if block_a.contains_key("automated_conclusions") || block_b.contains_key("automated_conclusions") {
                        file_concl.insert(
                            tag.clone(),
                            json!({
                                "baseline_a": block_a.get("automated_conclusions").cloned().unwrap_or_else(|| json!([])),
                                "candidate_b": block_b.get("automated_conclusions").cloned().unwrap_or_else(|| json!([])),
                            }),
                        );
                    }
                }
            }
            if !file_diff.is_empty() {
                summary_diffs.insert(fname.clone(), serde_json::to_value(file_diff)?);
            }
            if !file_concl.is_empty() {
                run_conclusions.insert(fname.clone(), serde_json::to_value(file_concl)?);
            }
        }
    }

    Ok((summary_diffs, run_conclusions))
}

// ==============================================================================
// PART 2: PAIRED STATISTICAL TESTS ON PARQUET FILES (Polars 0.54.4)
// ==============================================================================

#[derive(Serialize, Debug, Clone)]
pub struct TestStatisticResult {
    pub statistic: Option<f64>,
    pub p_value: Option<f64>,
    pub significant_at_alpha: bool,
}

#[derive(Serialize, Debug, Clone)]
pub struct PairedMetricStats {
    pub n_pairs: usize,
    pub n_non_zero_diffs: usize,
    pub mean_baseline_a: Option<f64>,
    pub std_baseline_a: Option<f64>,
    pub median_baseline_a: Option<f64>,
    pub mean_candidate_b: Option<f64>,
    pub std_candidate_b: Option<f64>,
    pub median_candidate_b: Option<f64>,
    pub mean_paired_diff: Option<f64>,
    pub median_paired_diff: Option<f64>,
    pub rel_diff_pct: Option<f64>,
    pub ci_95_diff: [Option<f64>; 2],
    pub cohens_dz: Option<f64>,
    pub paired_t_test: TestStatisticResult,
    pub wilcoxon_signed_rank: TestStatisticResult,
    pub significant_both_tests: bool,
    pub polarity: i32,
    pub direction: String,
}

fn compute_median(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return 0.0;
    }
    let mut sorted = slice.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn compute_mean_and_std(slice: &[f64]) -> (f64, f64) {
    let n = slice.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = slice.iter().sum::<f64>() / (n as f64);
    if n < 2 {
        return (mean, 0.0);
    }
    let var = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / ((n - 1) as f64);
    (mean, var.sqrt())
}

pub fn run_paired_stats_on_series(
    s_a: &Series,
    s_b: &Series,
    metric_name: &str,
    alpha: f64,
) -> PolarsResult<Option<PairedMetricStats>> {
    let ca_a = s_a.cast(&DataType::Float64)?;
    let ca_b = s_b.cast(&DataType::Float64)?;
    let f_a = ca_a.f64()?;
    let f_b = ca_b.f64()?;

    let mut valid_a = Vec::with_capacity(f_a.len());
    let mut valid_b = Vec::with_capacity(f_b.len());
    let mut diffs = Vec::with_capacity(f_a.len());

    for (opt_a, opt_b) in f_a.iter().zip(f_b.iter()) {
        if let (Some(va), Some(vb)) = (opt_a, opt_b) {
            if va.is_finite() && vb.is_finite() {
                valid_a.push(va);
                valid_b.push(vb);
                diffs.push(vb - va);
            }
        }
    }

    let n_pairs = valid_a.len();
    if n_pairs < 2 {
        return Ok(None);
    }

    let (mean_a, std_a) = compute_mean_and_std(&valid_a);
    let (mean_b, std_b) = compute_mean_and_std(&valid_b);
    let (mean_diff, std_diff) = compute_mean_and_std(&diffs);

    let median_a = compute_median(&valid_a);
    let median_b = compute_median(&valid_b);
    let median_diff = compute_median(&diffs);

    let rel_diff_pct = if mean_a.abs() > 1e-12 {
        clean_f64(((mean_b - mean_a) / mean_a.abs()) * 100.0)
    } else {
        None
    };

    let (w_stat, w_pval, n_non_zero) = wilcoxon_signed_rank_test(&diffs);

    let (t_stat, t_pval, cohens_dz, ci_low, ci_high) = if n_non_zero == 0 || std_diff <= 1e-15 {
        (0.0, 1.0, 0.0, mean_diff, mean_diff)
    } else {
        let n_f = n_pairs as f64;
        let df = n_f - 1.0;
        let se = std_diff / n_f.sqrt();
        let t = mean_diff / se;
        let p = student_t_two_tailed_pvalue(t, df);
        let t_crit = student_t_critical_two_tailed(alpha, df);
        let dz = mean_diff / std_diff;
        (t, p, dz, mean_diff - t_crit * se, mean_diff + t_crit * se)
    };

    let polarity = infer_polarity(metric_name);
    let is_sig_t = t_pval < alpha;
    let is_sig_w = w_pval < alpha;
    let is_sig_both = is_sig_t && is_sig_w;

    let direction = if mean_diff.abs() <= 1e-12 {
        "identical".to_string()
    } else if polarity == 1 {
        if mean_diff > 0.0 { "improved" } else { "degraded" }.to_string()
    } else if polarity == -1 {
        if mean_diff < 0.0 { "improved" } else { "degraded" }.to_string()
    } else if mean_diff > 0.0 {
        "increased".to_string()
    } else {
        "decreased".to_string()
    };

    Ok(Some(PairedMetricStats {
        n_pairs,
        n_non_zero_diffs: n_non_zero,
        mean_baseline_a: clean_f64(mean_a),
        std_baseline_a: clean_f64(std_a),
        median_baseline_a: clean_f64(median_a),
        mean_candidate_b: clean_f64(mean_b),
        std_candidate_b: clean_f64(std_b),
        median_candidate_b: clean_f64(median_b),
        mean_paired_diff: clean_f64(mean_diff),
        median_paired_diff: clean_f64(median_diff),
        rel_diff_pct,
        ci_95_diff: [clean_f64(ci_low), clean_f64(ci_high)],
        cohens_dz: clean_f64(cohens_dz),
        paired_t_test: TestStatisticResult {
            statistic: clean_f64(t_stat),
            p_value: clean_f64(t_pval),
            significant_at_alpha: is_sig_t,
        },
        wilcoxon_signed_rank: TestStatisticResult {
            statistic: clean_f64(w_stat),
            p_value: clean_f64(w_pval),
            significant_at_alpha: is_sig_w,
        },
        significant_both_tests: is_sig_both,
        polarity,
        direction,
    }))
}

#[derive(Serialize, Debug, Clone)]
pub struct ParquetFileComparison {
    pub join_key: String,
    pub n_matched_entities: usize,
    pub global: BTreeMap<String, PairedMetricStats>,
    pub by_tier: BTreeMap<String, BTreeMap<String, PairedMetricStats>>,
}

pub fn compare_parquet_dirs(
    parquet_dir_a: &Path,
    parquet_dir_b: &Path,
    alpha: f64,
) -> Result<BTreeMap<String, ParquetFileComparison>, Box<dyn Error + Send + Sync>> {
    let files_a = list_files_with_ext(parquet_dir_a, "parquet")?;
    let files_b = list_files_with_ext(parquet_dir_b, "parquet")?;

    let mut results = BTreeMap::new();

    for (fname, path_a) in &files_a {
        let Some(path_b) = files_b.get(fname) else {
            continue;
        };
        let Some(spec) = PARQUET_TEST_SPECS.iter().find(|s| fname.starts_with(s.prefix)) else {
            continue;
        };

        let df_a = ParquetReader::new(File::open(path_a)?).finish()?;
        let df_b = ParquetReader::new(File::open(path_b)?).finish()?;

        let join_key = spec.join_key;
        let schema_a = df_a.schema();
        let schema_b = df_b.schema();

        if !schema_a.contains(join_key) || !schema_b.contains(join_key) {
            continue;
        }

        // Explicitly alias Run B's metric columns to `{col}_b` so there are no suffix collisions
        let mut b_select_exprs = vec![col(join_key).cast(DataType::Int64)];
        for &c in spec.columns {
            if schema_b.contains(c) {
                let expr = if c == "rr_global" {
                    col(c).cast(DataType::Float64).fill_null(lit(0.0)).alias(format!("{}_b", c))
                } else {
                    col(c).cast(DataType::Float64).alias(format!("{}_b", c))
                };
                b_select_exprs.push(expr);
            }
        }

        let mut a_with_cols = vec![col(join_key).cast(DataType::Int64)];
        if schema_a.contains("rr_global") {
            a_with_cols.push(col("rr_global").cast(DataType::Float64).fill_null(lit(0.0)));
        }

        let joined_df = df_a
            .lazy()
            .with_columns(a_with_cols)
            .inner_join(
                df_b.lazy().select(b_select_exprs),
                col(join_key),
                col(join_key),
            )
            .collect()?;

        let mut global_map = BTreeMap::new();
        for &c in spec.columns {
            let col_b_name = format!("{}_b", c);
            if let (Ok(col_a), Ok(col_b)) = (joined_df.column(c), joined_df.column(&col_b_name)) {
                let s_a = col_a.as_materialized_series();
                let s_b = col_b.as_materialized_series();
                if let Some(st) = run_paired_stats_on_series(s_a, s_b, c, alpha)? {
                    global_map.insert(c.to_string(), st);
                }
            }
        }

        // Stratified paired tests by user_tier or movie_tier if present
        let mut by_tier_map = BTreeMap::new();
        let tier_col_opt = if joined_df.schema().contains("user_tier") {
            Some("user_tier")
        } else if joined_df.schema().contains("movie_tier") {
            Some("movie_tier")
        } else {
            None
        };

        if let Some(tier_col) = tier_col_opt {
            let tier_series = joined_df.column(tier_col)?.as_materialized_series().cast(&DataType::Int32)?;
            let tier_ca = tier_series.i32()?;

            for t in 0..=2 {
                let mask = tier_ca.equal(t);
                let tier_df = joined_df.filter(&mask)?;
                if tier_df.height() < 2 {
                    continue;
                }

                let mut tier_metrics = BTreeMap::new();
                for &c in spec.columns {
                    let col_b_name = format!("{}_b", c);
                    if let (Ok(col_a), Ok(col_b)) = (tier_df.column(c), tier_df.column(&col_b_name)) {
                        let s_a = col_a.as_materialized_series();
                        let s_b = col_b.as_materialized_series();
                        if let Some(st) = run_paired_stats_on_series(s_a, s_b, c, alpha)? {
                            tier_metrics.insert(c.to_string(), st);
                        }
                    }
                }
                if !tier_metrics.is_empty() {
                    by_tier_map.insert(format!("{}_{}", tier_col, t), tier_metrics);
                }
            }
        }

        results.insert(
            fname.clone(),
            ParquetFileComparison {
                join_key: join_key.to_string(),
                n_matched_entities: joined_df.height(),
                global: global_map,
                by_tier: by_tier_map,
            },
        );
    }

    Ok(results)
}

// ==============================================================================
// PART 3: AUTOMATED COMPARATIVE CONCLUSIONS & BUSINESS VALUE SCORECARD
// ==============================================================================

#[derive(Serialize, Debug, Clone)]
pub struct BusinessDimensionSummary {
    pub dimension_id: String,
    pub dimension_title: String,
    pub business_kpis_linked: Vec<String>,
    pub winning_model: String,
    pub confidence_level: String,
    pub baseline_a_score: f64,
    pub candidate_b_score: f64,
    pub baseline_a_evidence: Vec<String>,
    pub candidate_b_evidence: Vec<String>,
    pub business_recommendation: String,
}

struct DimensionAccumulator {
    id: &'static str,
    title: &'static str,
    kpis: Vec<&'static str>,
    score_a: f64,
    score_b: f64,
    sig_both_wins_a: usize,
    sig_both_wins_b: usize,
    evidence_a: Vec<String>,
    evidence_b: Vec<String>,
}

impl DimensionAccumulator {
    fn new(id: &'static str, title: &'static str, kpis: Vec<&'static str>) -> Self {
        Self {
            id,
            title,
            kpis,
            score_a: 0.0,
            score_b: 0.0,
            sig_both_wins_a: 0,
            sig_both_wins_b: 0,
            evidence_a: Vec::new(),
            evidence_b: Vec::new(),
        }
    }

    fn eval_parquet_stat(
        &mut self,
        context_label: &str,
        metric_label: &str,
        stat: &PairedMetricStats,
        weight: f64,
        label_a: &str,
        label_b: &str,
    ) {
        if stat.polarity == 0 {
            return;
        }
        let mean_a = stat.mean_baseline_a.unwrap_or(0.0);
        let mean_b = stat.mean_candidate_b.unwrap_or(0.0);
        let rel_pct = stat.rel_diff_pct.unwrap_or(0.0);
        let sig_t = stat.paired_t_test.significant_at_alpha;
        let sig_w = stat.wilcoxon_signed_rank.significant_at_alpha;

        let (pts, tag) = if stat.significant_both_tests {
            (3.0 * weight, "Confirmed (t & Wilcoxon p<alpha)")
        } else if sig_t || sig_w {
            (1.0 * weight, "Marginal (1-test p<alpha)")
        } else {
            return;
        };

        if stat.direction == "improved" {
            self.score_b += pts;
            if stat.significant_both_tests {
                self.sig_both_wins_b += 1;
            }
            self.evidence_b.push(format!(
                "[{}] {} {} beat {} ({:.4} vs {:.4}, {:+.2}%, {})",
                context_label, metric_label, label_b, label_a, mean_b, mean_a, rel_pct, tag
            ));
        } else if stat.direction == "degraded" {
            self.score_a += pts;
            if stat.significant_both_tests {
                self.sig_both_wins_a += 1;
            }
            let a_over_b_pct = if mean_b.abs() > 1e-12 {
                ((mean_a - mean_b) / mean_b.abs()) * 100.0
            } else {
                -rel_pct
            };
            self.evidence_a.push(format!(
                "[{}] {} {} beat {} ({:.4} vs {:.4}, {:+.2}% in favor of {}, {})",
                context_label, metric_label, label_a, label_b, mean_a, mean_b, a_over_b_pct, label_a, tag
            ));
        }
    }

    fn eval_summary_scalar(
        &mut self,
        context_label: &str,
        metric_label: &str,
        summary_diffs: &BTreeMap<String, Value>,
        file_name: &str,
        block_name: &str,
        metric_key: &str,
        weight: f64,
        min_rel_pct: f64,
        min_abs_diff: f64,
        label_a: &str,
        label_b: &str,
    ) {
        let Some(metric_obj) = summary_diffs
            .get(file_name)
            .and_then(|f| f.get(block_name))
            .and_then(|b| b.get(metric_key))
        else {
            return;
        };

        let val_a = metric_obj.get("baseline_a").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let val_b = metric_obj.get("candidate_b").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let abs_diff = metric_obj.get("abs_diff").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let rel_pct = metric_obj.get("rel_diff_pct").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let status = metric_obj.get("status").and_then(|v| v.as_str()).unwrap_or("unchanged");

        if rel_pct.abs() < min_rel_pct || abs_diff.abs() < min_abs_diff {
            return;
        }

        let pts = 1.5 * weight;
        if status == "improved" {
            self.score_b += pts;
            self.evidence_b.push(format!(
                "[{}] {} {} beat {} ({:.4} vs {:.4}, {:+.2}%, Global Catalog Shift)",
                context_label, metric_label, label_b, label_a, val_b, val_a, rel_pct
            ));
        } else if status == "degraded" {
            self.score_a += pts;
            let a_over_b_pct = if val_b.abs() > 1e-12 {
                ((val_a - val_b) / val_b.abs()) * 100.0
            } else {
                -rel_pct
            };
            self.evidence_a.push(format!(
                "[{}] {} {} beat {} ({:.4} vs {:.4}, {:+.2}% in favor of {}, Global Catalog Shift)",
                context_label, metric_label, label_a, label_b, val_a, val_b, a_over_b_pct, label_a
            ));
        }
    }

    fn finalize(self, label_a: &str, label_b: &str) -> BusinessDimensionSummary {
        let diff = self.score_b - self.score_a;
        let (winning_model, confidence_level) = if diff.abs() < 0.75 {
            (
                "TIE / EQUIVALENT".to_string(),
                "NEUTRAL (No Statistically Meaningful Advantage)".to_string(),
            )
        } else if diff > 0.0 {
            let conf = if self.sig_both_wins_b > 0 && diff >= 3.0 {
                "HIGH (Confirmed by Paired Parametric & Non-Parametric Tests)"
            } else {
                "MODERATE (Supported by Directional / Summary Shifts)"
            };
            (label_b.to_string(), conf.to_string())
        } else {
            let conf = if self.sig_both_wins_a > 0 && diff <= -3.0 {
                "HIGH (Confirmed by Paired Parametric & Non-Parametric Tests)"
            } else {
                "MODERATE (Supported by Directional / Summary Shifts)"
            };
            (label_a.to_string(), conf.to_string())
        };

        let kpis_vec: Vec<String> = self.kpis.iter().map(|&s| s.to_string()).collect();
        let kpi_joined = kpis_vec.join(", ");

        let business_recommendation = if winning_model == "TIE / EQUIVALENT" {
            format!(
                "Both {} and {} perform equivalently for {} ({}). Choose based on other business priorities.",
                label_a, label_b, self.title, kpi_joined
            )
        } else {
            let top_ev = if winning_model == label_b {
                self.evidence_b.first().cloned().unwrap_or_default()
            } else {
                self.evidence_a.first().cloned().unwrap_or_default()
            };
            format!(
                "Select {} when prioritizing {} (impacts: {}). Key driver: {}",
                winning_model, self.title, kpi_joined, top_ev
            )
        };

        BusinessDimensionSummary {
            dimension_id: self.id.to_string(),
            dimension_title: self.title.to_string(),
            business_kpis_linked: kpis_vec,
            winning_model,
            confidence_level,
            baseline_a_score: (self.score_a * 100.0).round() / 100.0,
            candidate_b_score: (self.score_b * 100.0).round() / 100.0,
            baseline_a_evidence: self.evidence_a,
            candidate_b_evidence: self.evidence_b,
            business_recommendation,
        }
    }
}

fn evaluate_business_value_scorecard(
    label_a: &str,
    label_b: &str,
    parquet_stats: &BTreeMap<String, ParquetFileComparison>,
    summary_diffs: &BTreeMap<String, Value>,
) -> (Vec<BusinessDimensionSummary>, BTreeMap<String, String>, Vec<String>) {
    // -------------------------------------------------------------------------
    // 1. Top-K Slate Conversion & Immediate Engagement (CTR & Session Velocity)
    // -------------------------------------------------------------------------
    let mut dim_conversion = DimensionAccumulator::new(
        "top_k_conversion_and_engagement",
        "Top-K Slate Conversion & Immediate User Engagement",
        vec![
            "Click-Through Rate (CTR)",
            "Immediate Session Conversion",
            "Time-to-First-Click (MRR Surfacing Speed)",
            "Mainstream / Power-User Satisfaction",
        ],
    );

    if let Some(rnk) = parquet_stats.get("stratified_metrics_top_20_ranker_20.parquet") {
        if let Some(st) = rnk.global.get("ndcg_global") {
            dim_conversion.eval_parquet_stat("Ranker@20", "Global NDCG@20", st, 1.5, label_a, label_b);
        }
        if let Some(st) = rnk.global.get("rr_global") {
            dim_conversion.eval_parquet_stat("Ranker@20", "Global MRR@20", st, 1.5, label_a, label_b);
        }
        if let Some(st) = rnk.global.get("precision_global") {
            dim_conversion.eval_parquet_stat("Ranker@20", "Global Precision@20", st, 1.2, label_a, label_b);
        }
        if let Some(st) = rnk.global.get("recall_global") {
            dim_conversion.eval_parquet_stat("Ranker@20", "Global Recall@20", st, 1.2, label_a, label_b);
        }
        if let Some(st) = rnk.global.get("ndcg_tier_0") {
            dim_conversion.eval_parquet_stat("Ranker@20", "Head Movie (Tier 0) NDCG@20", st, 1.0, label_a, label_b);
        }
        if let Some(st) = rnk.global.get("recall_tier_0") {
            dim_conversion.eval_parquet_stat("Ranker@20", "Head Movie (Tier 0) Recall@20", st, 1.0, label_a, label_b);
        }
        if let Some(u0) = rnk.by_tier.get("user_tier_0") {
            if let Some(st) = u0.get("ndcg_global") {
                dim_conversion.eval_parquet_stat("Ranker@20 Power Users (Tier 0)", "NDCG@20", st, 1.0, label_a, label_b);
            }
            if let Some(st) = u0.get("rr_global") {
                dim_conversion.eval_parquet_stat("Ranker@20 Power Users (Tier 0)", "MRR@20", st, 1.0, label_a, label_b);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Stage-1 Candidate Funnel Breadth & Recall Ceiling
    // -------------------------------------------------------------------------
    let mut dim_funnel = DimensionAccumulator::new(
        "stage1_candidate_funnel_recall",
        "Stage-1 ANN Candidate Funnel Breadth & Recall Ceiling",
        vec![
            "Downstream Re-Ranking Recall Ceiling (@100)",
            "Candidate Pool Quality",
            "Stage-1 ANN Filtering Efficiency",
        ],
    );

    if let Some(ret100) = parquet_stats.get("stratified_metrics_top_100_retrieval_ann_100.parquet") {
        if let Some(st) = ret100.global.get("recall_global") {
            dim_funnel.eval_parquet_stat("Retriever@100", "Global Recall@100", st, 1.5, label_a, label_b);
        }
        if let Some(st) = ret100.global.get("ndcg_global") {
            dim_funnel.eval_parquet_stat("Retriever@100", "Global NDCG@100", st, 1.2, label_a, label_b);
        }
        if let Some(st) = ret100.global.get("rr_global") {
            dim_funnel.eval_parquet_stat("Retriever@100", "Global MRR@100", st, 1.0, label_a, label_b);
        }
        if let Some(st) = ret100.global.get("recall_tier_2") {
            dim_funnel.eval_parquet_stat("Retriever@100", "Long-Tail Movie (Tier 2) Recall@100", st, 1.0, label_a, label_b);
        }
    }
    if let Some(ret20) = parquet_stats.get("stratified_metrics_top_20_retrieval_ann_20.parquet") {
        if let Some(st) = ret20.global.get("recall_global") {
            dim_funnel.eval_parquet_stat("Retriever@20", "Global Recall@20", st, 1.0, label_a, label_b);
        }
        if let Some(st) = ret20.global.get("ndcg_global") {
            dim_funnel.eval_parquet_stat("Retriever@20", "Global NDCG@20", st, 1.0, label_a, label_b);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Long-Tail Catalog Discovery, Inventory Utilization & Supplier Fairness
    // -------------------------------------------------------------------------
    let mut dim_long_tail = DimensionAccumulator::new(
        "long_tail_discovery_and_catalog_equity",
        "Long-Tail Catalog Discovery, Inventory Utilization & Supplier Fairness",
        vec![
            "Catalog Coverage (% Active Inventory Surfaced)",
            "Long-Tail & Torso Content ROI",
            "Impression Equity (Lower Gini / Higher Bottom-80% Share)",
            "Reduced Popularity Echo-Chamber Bias",
        ],
    );

    if let Some(rnk) = parquet_stats.get("stratified_metrics_top_20_ranker_20.parquet") {
        if let Some(st) = rnk.global.get("ndcg_tier_2") {
            dim_long_tail.eval_parquet_stat("Ranker@20", "Long-Tail Movie (Tier 2) NDCG@20", st, 1.5, label_a, label_b);
        }
        if let Some(st) = rnk.global.get("recall_tier_2") {
            dim_long_tail.eval_parquet_stat("Ranker@20", "Long-Tail Movie (Tier 2) Recall@20", st, 1.5, label_a, label_b);
        }
        if let Some(st) = rnk.global.get("ndcg_tier_1") {
            dim_long_tail.eval_parquet_stat("Ranker@20", "Torso Movie (Tier 1) NDCG@20", st, 1.2, label_a, label_b);
        }
        if let Some(st) = rnk.global.get("recall_tier_1") {
            dim_long_tail.eval_parquet_stat("Ranker@20", "Torso Movie (Tier 1) Recall@20", st, 1.2, label_a, label_b);
        }
    }
    if let Some(pop) = parquet_stats.get("popularity_bias_k_20_ranker_20.parquet") {
        if let Some(st) = pop.global.get("pop_bias_delta") {
            dim_long_tail.eval_parquet_stat("Ranker@20", "Popularity Bias Delta (Lower=Less Bias)", st, 1.0, label_a, label_b);
        }
    }
    dim_long_tail.eval_summary_scalar(
        "Ranker@20",
        "Full Catalog Coverage@20",
        summary_diffs,
        "coverage.json",
        "ranker_20",
        "coverage_at_20_full_ranker_20",
        1.2,
        2.0,
        0.01,
        label_a,
        label_b,
    );
    dim_long_tail.eval_summary_scalar(
        "Ranker@20",
        "Catalog Exposure Gini@20 (Lower=More Equitable)",
        summary_diffs,
        "coverage.json",
        "ranker_20",
        "gini_at_20_full_ranker_20",
        1.2,
        1.5,
        0.01,
        label_a,
        label_b,
    );
    dim_long_tail.eval_summary_scalar(
        "Ranker@20",
        "Bottom-80% Catalog Impression Share@20",
        summary_diffs,
        "coverage.json",
        "ranker_20",
        "lorenz_at_20_bottom_80_share_ranker_20",
        1.2,
        2.0,
        0.005,
        label_a,
        label_b,
    );

    // -------------------------------------------------------------------------
    // 4. Cross-User Personalization, Slate Genre Variety & Light-User Retention
    // -------------------------------------------------------------------------
    let mut dim_personalization = DimensionAccumulator::new(
        "personalization_diversity_and_cohort_retention",
        "Cross-User Personalization, Slate Genre Variety & Light-User Retention",
        vec![
            "Long-Term User Retention (Anti-Churn)",
            "Light / Niche User Satisfaction (User Tier 2)",
            "Intra-Slate Genre Breadth (Reduced Slate Fatigue)",
            "Cross-User Slate Uniqueness (Personalization)",
        ],
    );

    if let Some(div20) = parquet_stats.get("diversity_metrics_k_20_ranker_20.parquet") {
        if let Some(st) = div20.global.get("interlist_diversity") {
            dim_personalization.eval_parquet_stat("Ranker@20", "Inter-List Diversity (Personalization)", st, 1.5, label_a, label_b);
        }
        if let Some(st) = div20.global.get("intralist_diversity") {
            dim_personalization.eval_parquet_stat("Ranker@20", "Intra-List Diversity (Genre Breadth)", st, 1.5, label_a, label_b);
        }
    }
    if let Some(div100) = parquet_stats.get("diversity_metrics_k_100_retrieval_ann_100.parquet") {
        if let Some(st) = div100.global.get("intralist_diversity") {
            dim_personalization.eval_parquet_stat("Retriever@100", "Intra-List Diversity@100", st, 1.0, label_a, label_b);
        }
        if let Some(st) = div100.global.get("interlist_diversity") {
            dim_personalization.eval_parquet_stat("Retriever@100", "Inter-List Diversity@100", st, 1.0, label_a, label_b);
        }
    }
    if let Some(rnk) = parquet_stats.get("stratified_metrics_top_20_ranker_20.parquet") {
        if let Some(u2) = rnk.by_tier.get("user_tier_2") {
            if let Some(st) = u2.get("ndcg_global") {
                dim_personalization.eval_parquet_stat("Ranker@20 Light Users (Tier 2)", "NDCG@20", st, 1.3, label_a, label_b);
            }
            if let Some(st) = u2.get("recall_global") {
                dim_personalization.eval_parquet_stat("Ranker@20 Light Users (Tier 2)", "Recall@20", st, 1.2, label_a, label_b);
            }
        }
        if let Some(u1) = rnk.by_tier.get("user_tier_1") {
            if let Some(st) = u1.get("ndcg_global") {
                dim_personalization.eval_parquet_stat("Ranker@20 Mid-Tail Users (Tier 1)", "NDCG@20", st, 1.0, label_a, label_b);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5. Cold-Start Onboarding & New Catalog Launch Resilience
    // -------------------------------------------------------------------------
    let mut dim_cold_start = DimensionAccumulator::new(
        "cold_start_onboarding_and_launch_velocity",
        "New User Onboarding & New Catalog Launch Cold-Start Resilience",
        vec![
            "Day-1 New User Conversion (Zero History)",
            "New Release / Unwatched Title Launch Velocity (OOV Items)",
            "Demographic & Content Metadata Generalization",
        ],
    );

    if let Some(csm_rnk) = parquet_stats.get("stratified_metrics_top_20_ranker_20_cold_start_movies.parquet") {
        if let Some(st) = csm_rnk.global.get("ndcg_global") {
            dim_cold_start.eval_parquet_stat("Cold-Start Movies Ranker@20", "NDCG@20", st, 1.5, label_a, label_b);
        }
        if let Some(st) = csm_rnk.global.get("recall_global") {
            dim_cold_start.eval_parquet_stat("Cold-Start Movies Ranker@20", "Recall@20", st, 1.3, label_a, label_b);
        }
        if let Some(st) = csm_rnk.global.get("rr_global") {
            dim_cold_start.eval_parquet_stat("Cold-Start Movies Ranker@20", "MRR@20", st, 1.0, label_a, label_b);
        }
    }
    if let Some(csm_ret) = parquet_stats.get("stratified_metrics_top_100_retrieval_ann_100_cold_start_movies.parquet") {
        if let Some(st) = csm_ret.global.get("recall_global") {
            dim_cold_start.eval_parquet_stat("Cold-Start Movies Retriever@100", "Recall@100", st, 1.2, label_a, label_b);
        }
    }
    if let Some(csu_rnk) = parquet_stats.get("stratified_metrics_top_20_ranker_20_cold_start_users.parquet") {
        if let Some(st) = csu_rnk.global.get("ndcg_global") {
            dim_cold_start.eval_parquet_stat("Cold-Start Users Ranker@20", "NDCG@20", st, 1.5, label_a, label_b);
        }
        if let Some(st) = csu_rnk.global.get("recall_global") {
            dim_cold_start.eval_parquet_stat("Cold-Start Users Ranker@20", "Recall@20", st, 1.3, label_a, label_b);
        }
        if let Some(st) = csu_rnk.global.get("rr_global") {
            dim_cold_start.eval_parquet_stat("Cold-Start Users Ranker@20", "MRR@20", st, 1.0, label_a, label_b);
        }
    }
    if let Some(csu_ret) = parquet_stats.get("stratified_metrics_top_100_retrieval_ann_100_cold_start_users.parquet") {
        if let Some(st) = csu_ret.global.get("recall_global") {
            dim_cold_start.eval_parquet_stat("Cold-Start Users Retriever@100", "Recall@100", st, 1.2, label_a, label_b);
        }
    }

    // -------------------------------------------------------------------------
    // 6. Latent Space Health, Anti-Oversmoothing & Hub Suppression
    // -------------------------------------------------------------------------
    let mut dim_geometry = DimensionAccumulator::new(
        "latent_geometry_and_anti_hubness",
        "Latent Space Health, Anti-Oversmoothing & Hub Suppression",
        vec![
            "Algorithmic Stability & Resistance to Degenerate Hub Loops",
            "GNN Subgraph Expressivity (Low Anisotropy / Anti-Oversmoothing)",
            "Spherical Representation Fidelity Across Contexts",
        ],
    );

    if let Some(ctx_u) = parquet_stats.get("contextual_hubness_user_stats_k_20_ranker_contextual_k_20.parquet") {
        if let Some(st) = ctx_u.global.get("avg_slate_anisotropy") {
            dim_geometry.eval_parquet_stat("GATv2 Contextual User Stats@20", "Slate Anisotropy (Lower=Less Oversmoothing)", st, 1.3, label_a, label_b);
        }
        if let Some(st) = ctx_u.global.get("avg_slate_nk") {
            dim_geometry.eval_parquet_stat("GATv2 Contextual User Stats@20", "Hub Exposure N_k (Lower=Less Hub Dominance)", st, 1.0, label_a, label_b);
        }
    }
    if let Some(hub_u) = parquet_stats.get("hubness_user_stats_k_20_ranker_20.parquet") {
        if let Some(st) = hub_u.global.get("avg_slate_sim_to_centroid") {
            dim_geometry.eval_parquet_stat("Ranker@20 Hubness", "Slate Proximity to Global Centroid (Lower=Better)", st, 1.0, label_a, label_b);
        }
    }
    dim_geometry.eval_summary_scalar(
        "GATv2 Contextual Hubness",
        "Contextual k-NN Skewness",
        summary_diffs,
        "contextual_hubness_k_20.json",
        "default",
        "ctx_hubness_skewness_ranker_contextual_k_20",
        1.2,
        5.0,
        0.20,
        label_a,
        label_b,
    );
    dim_geometry.eval_summary_scalar(
        "GATv2 Movie Spherical Fidelity",
        "Mean Resultant Length R_bar (Higher=More Stable)",
        summary_diffs,
        "gatv2_spherical_fidelity.json",
        "GATv2_spherical_fidelity_movie",
        "mean_resultant_length_avg",
        1.2,
        1.5,
        0.01,
        label_a,
        label_b,
    );
    dim_geometry.eval_summary_scalar(
        "GATv2 User Spherical Fidelity",
        "Mean Resultant Length R_bar (Higher=More Stable)",
        summary_diffs,
        "gatv2_spherical_fidelity.json",
        "GATv2_spherical_fidelity_user",
        "mean_resultant_length_avg",
        1.2,
        1.5,
        0.01,
        label_a,
        label_b,
    );

    let scorecard = vec![
        dim_conversion.finalize(label_a, label_b),
        dim_funnel.finalize(label_a, label_b),
        dim_long_tail.finalize(label_a, label_b),
        dim_personalization.finalize(label_a, label_b),
        dim_cold_start.finalize(label_a, label_b),
        dim_geometry.finalize(label_a, label_b),
    ];

    let mut best_by_goal = BTreeMap::new();
    let mut decision_matrix = Vec::new();

    for dim in &scorecard {
        best_by_goal.insert(
            dim.dimension_id.clone(),
            format!(
                "{} (Score: {}={:.1} vs {}={:.1})",
                dim.winning_model, label_a, dim.baseline_a_score, label_b, dim.candidate_b_score
            ),
        );
        decision_matrix.push(dim.business_recommendation.clone());
    }

    (scorecard, best_by_goal, decision_matrix)
}

pub fn synthesize_comparative_conclusions(
    label_a: &str,
    label_b: &str,
    parquet_stats: &BTreeMap<String, ParquetFileComparison>,
    summary_diffs: &BTreeMap<String, Value>,
    run_conclusions: &BTreeMap<String, Value>,
    alpha: f64,
) -> Value {
    let mut significant_improvements = Vec::new();
    let mut significant_regressions = Vec::new();
    let mut neutral_or_noise = Vec::new();
    let mut executive_highlights = Vec::new();

    for (fname, fdata) in parquet_stats {
        for (m_name, m_stat) in &fdata.global {
            let mean_a = m_stat.mean_baseline_a.unwrap_or(0.0);
            let mean_b = m_stat.mean_candidate_b.unwrap_or(0.0);
            let rel_str = m_stat
                .rel_diff_pct
                .map(|p| format!("{:+.2}%", p))
                .unwrap_or_else(|| "N/A".to_string());
            let t_p = m_stat.paired_t_test.p_value.unwrap_or(1.0);
            let w_p = m_stat.wilcoxon_signed_rank.p_value.unwrap_or(1.0);
            let dz = m_stat.cohens_dz.unwrap_or(0.0);
            let sig_t = m_stat.paired_t_test.significant_at_alpha;
            let sig_w = m_stat.wilcoxon_signed_rank.significant_at_alpha;

            let stat_str = format!(
                "[{} -> {}] {}: {:.4} vs {}: {:.4} ({}, n={}, t-test p={:.3e}, Wilcoxon p={:.3e}, Cohen's dz={:+.3})",
                fname, m_name, label_a, mean_a, label_b, mean_b, rel_str, m_stat.n_pairs, t_p, w_p, dz
            );

            if sig_t && sig_w {
                if m_stat.direction == "improved" {
                    significant_improvements.push(format!(
                        "CONFIRMED IMPROVEMENT (Both Tests p<{:.2}): {}",
                        alpha, stat_str
                    ));
                } else if m_stat.direction == "degraded" {
                    significant_regressions.push(format!(
                        "CONFIRMED REGRESSION (Both Tests p<{:.2}): {}",
                        alpha, stat_str
                    ));
                } else {
                    neutral_or_noise.push(format!(
                        "SIGNIFICANT SHIFT (Diagnostic Metric, p<{:.2}): {}",
                        alpha, stat_str
                    ));
                }
            } else if sig_t || sig_w {
                let which_test = if sig_t { "Paired t-test" } else { "Wilcoxon signed-rank" };
                if m_stat.direction == "improved" {
                    significant_improvements.push(format!(
                        "MARGINAL/SKEWED IMPROVEMENT ({} only p<{:.2}): {}",
                        which_test, alpha, stat_str
                    ));
                } else if m_stat.direction == "degraded" {
                    significant_regressions.push(format!(
                        "MARGINAL/SKEWED REGRESSION ({} only p<{:.2}): {}",
                        which_test, alpha, stat_str
                    ));
                } else {
                    neutral_or_noise.push(format!(
                        "MARGINAL SHIFT ({} only p<{:.2}): {}",
                        which_test, alpha, stat_str
                    ));
                }
            } else {
                neutral_or_noise.push(format!(
                    "NO STATISTICALLY SIGNIFICANT DIFFERENCE (p>={:.2}): {}",
                    alpha, stat_str
                ));
            }
        }
    }

    // Generate stage-level executive highlights for stratified_metrics files
    for (p_name, p_data) in parquet_stats {
        if !p_name.starts_with("stratified_metrics_top_") {
            continue;
        }
        let stage_label = if p_name.contains("cold_start_movies") {
            format!("COLD-START MOVIES [{}]", p_name)
        } else if p_name.contains("cold_start_users") {
            format!("COLD-START USERS [{}]", p_name)
        } else if p_name.contains("ranker_") {
            format!("WARM RANKER [{}]", p_name)
        } else {
            format!("WARM RETRIEVER [{}]", p_name)
        };

        if let (Some(ndcg), Some(recall)) = (p_data.global.get("ndcg_global"), p_data.global.get("recall_global")) {
            let ndcg_pct = ndcg.rel_diff_pct.unwrap_or(0.0);
            let rec_pct = recall.rel_diff_pct.unwrap_or(0.0);
            let mrr_pct = p_data
                .global
                .get("rr_global")
                .and_then(|m| m.rel_diff_pct)
                .unwrap_or(0.0);

            let verdict = if ndcg.significant_both_tests || recall.significant_both_tests {
                "STATISTICALLY SIGNIFICANT"
            } else {
                "STATISTICALLY INDISTINGUISHABLE (NOISE)"
            };

            executive_highlights.push(format!(
                "{} ({}): {} vs {} -> NDCG delta: {:+.2}% (t-p={:.2e}, W-p={:.2e}), Recall delta: {:+.2}% (t-p={:.2e}, W-p={:.2e}), MRR delta: {:+.2}%.",
                stage_label,
                verdict,
                label_b,
                label_a,
                ndcg_pct,
                ndcg.paired_t_test.p_value.unwrap_or(1.0),
                ndcg.wilcoxon_signed_rank.p_value.unwrap_or(1.0),
                rec_pct,
                recall.paired_t_test.p_value.unwrap_or(1.0),
                recall.wilcoxon_signed_rank.p_value.unwrap_or(1.0),
                mrr_pct
            ));
        }
    }

    // Evaluate the 6 Business Value Dimensions & Decision Matrix
    let (business_value_scorecard, best_model_by_business_goal, executive_business_decision_matrix) =
        evaluate_business_value_scorecard(label_a, label_b, parquet_stats, summary_diffs);

    let a_dim_wins: Vec<&str> = business_value_scorecard
        .iter()
        .filter(|d| d.winning_model == label_a)
        .map(|d| d.dimension_title.as_str())
        .collect();
    let b_dim_wins: Vec<&str> = business_value_scorecard
        .iter()
        .filter(|d| d.winning_model == label_b)
        .map(|d| d.dimension_title.as_str())
        .collect();

    let n_imp = significant_improvements.len();
    let n_reg = significant_regressions.len();
    let overall_verdict = if n_imp > 0 && n_reg == 0 {
        format!(
            "RECOMMEND PROMOTING {}: Achieved {} statistically significant improvements with zero significant regressions across paired tests (Wins {} business pillars: {}).",
            label_b,
            n_imp,
            b_dim_wins.len(),
            if b_dim_wins.is_empty() { "All Tied".to_string() } else { b_dim_wins.join("; ") }
        )
    } else if n_reg > 0 && n_imp == 0 {
        format!(
            "KEEP BASELINE {}: {} introduced {} statistically significant regressions with zero significant improvements ({} leads across {} business pillars: {}).",
            label_a,
            label_b,
            n_reg,
            label_a,
            a_dim_wins.len(),
            if a_dim_wins.is_empty() { "All Tied".to_string() } else { a_dim_wins.join("; ") }
        )
    } else if n_imp > 0 && n_reg > 0 {
        let a_summary = if a_dim_wins.is_empty() {
            "None".to_string()
        } else {
            a_dim_wins.join("; ")
        };
        let b_summary = if b_dim_wins.is_empty() {
            "None".to_string()
        } else {
            b_dim_wins.join("; ")
        };
        format!(
            "TRADEOFF DETECTED BETWEEN {} AND {}: {} yielded {} significant improvements alongside {} significant regressions. {} is best for [{}], whereas {} is best for [{}]. See `business_value_scorecard` to align model selection with target business KPIs.",
            label_a, label_b, label_b, n_imp, n_reg, label_b, b_summary, label_a, a_summary
        )
    } else {
        format!(
            "NEUTRAL / EQUIVALENT MODELS: Neither {} nor {} showed statistically significant differences at alpha={:.2}.",
            label_a, label_b, alpha
        )
    };

    json!({
        "comparison_metadata": {
            "baseline_model_a": label_a,
            "candidate_model_b": label_b,
            "significance_alpha": alpha,
            "overall_verdict": overall_verdict,
        },
        "best_model_by_business_goal": best_model_by_business_goal,
        "executive_business_decision_matrix": executive_business_decision_matrix,
        "business_value_scorecard": business_value_scorecard,
        "executive_highlights": executive_highlights,
        "statistically_significant_improvements_in_b": significant_improvements,
        "statistically_significant_regressions_in_b": significant_regressions,
        "neutral_or_diagnostic_shifts": neutral_or_noise,
        "per_run_automated_conclusions": run_conclusions,
    })
}

/// Native Rust Graphics Rendering using `plotters`
pub fn generate_comparison_graphics(
    parquet_dir_a: &Path,
    parquet_dir_b: &Path,
    summary_dir_a: &Path,
    summary_dir_b: &Path,
    output_dir: &Path,
    label_a: &str,
    label_b: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {

    ensure_plotters_font_registered();

    fs::create_dir_all(output_dir)?;

    // 1. Render Lorenz Curve Comparison
    let l_a_path = summary_dir_a.join("lorenz_curve_top_20_ranker_20.json");
    let l_b_path = summary_dir_b.join("lorenz_curve_top_20_ranker_20.json");
    if l_a_path.is_file() && l_b_path.is_file() {
        let curve_a: Vec<f64> = serde_json::from_reader(BufReader::new(File::open(l_a_path)?))?;
        let curve_b: Vec<f64> = serde_json::from_reader(BufReader::new(File::open(l_b_path)?))?;

        let file_path = output_dir.join("lorenz_curve_comparison.png");
        let root = SVGBackend::new(&file_path, (800, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Catalog Popularity Concentration (Lorenz Curve)", ("sans-serif", 22).into_font())
            .margin(25)
            .x_label_area_size(45)
            .y_label_area_size(45)
            .build_cartesian_2d(0f64..100f64, 0f64..1f64)?;

        chart
            .configure_mesh()
            .x_desc("Cumulative Catalog Items (%)")
            .y_desc("Cumulative Recommendations Share (%)")
            .draw()?;

        let n = curve_a.len().max(1) as f64;
        let pts_a: Vec<(f64, f64)> = curve_a
            .iter()
            .enumerate()
            .map(|(i, &y)| ((i as f64 + 1.0) * 100.0 / n, y))
            .collect();
        let pts_b: Vec<(f64, f64)> = curve_b
            .iter()
            .enumerate()
            .map(|(i, &y)| ((i as f64 + 1.0) * 100.0 / n, y))
            .collect();

        chart
            .draw_series(LineSeries::new(pts_a, &RGBColor(46, 139, 87)))?
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(46, 139, 87)))
            .label(label_a);

        chart
            .draw_series(LineSeries::new(pts_b, &RGBColor(220, 20, 60)))?
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(220, 20, 60)))
            .label(label_b);

        chart
            .draw_series(LineSeries::new(vec![(0.0, 0.0), (100.0, 1.0)], &RGBColor(128, 128, 128)))?
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(128, 128, 128)))
            .label("Line of Perfect Equality");

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.85))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
    }

    // 2. Load Parquet Data for Stratified & Delta Analysis
    let strat_file = "stratified_metrics_top_20_ranker_20.parquet";
    let path_a = parquet_dir_a.join(strat_file);
    let path_b = parquet_dir_b.join(strat_file);

    if path_a.is_file() && path_b.is_file() {
        let df_a = ParquetReader::new(File::open(&path_a)?).finish()?;
        let df_b = ParquetReader::new(File::open(&path_b)?).finish()?;

        if df_a.schema().contains("user_id") && df_b.schema().contains("user_id") {
            let joined = df_a
                .lazy()
                .select([
                    col("user_id").cast(DataType::Int64),
                    col("user_tier"),
                    col("ndcg_global"),
                    col("recall_global"),
                ])
                .inner_join(
                    df_b.lazy().select([
                        col("user_id").cast(DataType::Int64),
                        col("ndcg_global").alias("ndcg_global_b"),
                        col("recall_global").alias("recall_global_b"),
                    ]),
                    col("user_id"),
                    col("user_id"),
                )
                .collect()?;

            if joined.height() > 0 {
                // Render User Tier Grouped Bar Chart
                if joined.schema().contains("user_tier") {
                    let file_path = output_dir.join("user_tier_stratification.png");
                    let root = SVGBackend::new(&file_path, (900, 500)).into_drawing_area();
                    root.fill(&WHITE)?;

                    let mut mean_ndcg_a = [0.0; 3];
                    let mut mean_ndcg_b = [0.0; 3];
                    let tier_series = joined.clone().column("user_tier")?.as_materialized_series().cast(&DataType::Int32)?;
                    let tier_ca = tier_series.i32()?;
                    let tmp = joined.column("ndcg_global")?.as_materialized_series().cast(&DataType::Float64).expect("error extracting column ndcg_global");
                    let ndcg_a_ca = tmp.f64()?;
                    let tmp = joined.column("ndcg_global_b")?.as_materialized_series().cast(&DataType::Float64).expect("error extracting column ndcg_global_b");
                    let ndcg_b_ca = tmp.f64()?;

                    let mut counts = [0; 3];
                    for (t_opt, (na_opt, nb_opt)) in tier_ca.iter().zip(ndcg_a_ca.iter().zip(ndcg_b_ca.iter())) {
                        if let (Some(t), Some(na), Some(nb)) = (t_opt, na_opt, nb_opt) {
                            if t >= 0 && t <= 2 {
                                let idx = t as usize;
                                mean_ndcg_a[idx] += na;
                                mean_ndcg_b[idx] += nb;
                                counts[idx] += 1;
                            }
                        }
                    }
                    for i in 0..3 {
                        if counts[i] > 0 {
                            mean_ndcg_a[i] /= counts[i] as f64;
                            mean_ndcg_b[i] /= counts[i] as f64;
                        }
                    }

                    let mut chart = ChartBuilder::on(&root)
                        .caption(format!("Performance Stratification by User Tier: {} vs {}", label_b, label_a), ("sans-serif", 20).into_font())
                        .margin(25)
                        .x_label_area_size(40)
                        .y_label_area_size(40)
                        .build_cartesian_2d(0f64..3f64, 0f64..1f64)?;

                    // Tier 0 = Power Users (top 20%), Tier 1 = Mid-Tail (middle 60%), Tier 2 = Light Users (bottom 20%)
                    chart.configure_mesh().x_labels(3).x_label_formatter(&|v| match *v as i32 {
                        0 => "Tier 0 (Power)".to_string(),
                        1 => "Tier 1 (Mid-Tail)".to_string(),
                        _ => "Tier 2 (Light)".to_string(),
                    }).y_desc("Mean NDCG@20").draw()?;

                    let width = 0.35;
                    chart.draw_series(
                        (0..3).map(|i| Rectangle::new([(i as f64 + 0.15, 0.0), (i as f64 + 0.15 + width, mean_ndcg_a[i])], RGBColor(46, 139, 87).filled()))
                    )?.legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(46, 139, 87))).label(label_a);

                    chart.draw_series(
                        (0..3).map(|i| Rectangle::new([(i as f64 + 0.15 + width, 0.0), (i as f64 + 0.15 + width * 2.0, mean_ndcg_b[i])], RGBColor(220, 20, 60).filled()))
                    )?.legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(220, 20, 60))).label(label_b);

                    chart.configure_series_labels().background_style(&WHITE.mix(0.85)).border_style(&BLACK).draw()?;
                    root.present()?;
                }
            }
        }
    }

    println!("[Graphics] Native Plotters charts successfully rendered to: {:?}", output_dir);
    Ok(())
}

pub fn generate_comparison_graphics_python(
    parquet_dir_a: &Path,
    parquet_dir_b: &Path,
    summary_dir_a: &Path,
    summary_dir_b: &Path,
    output_dir: &Path,
    label_a: &str,
    label_b: &str,
) {
    let script_content = format!(
        r#"
import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

parquet_a = Path(r"{0}")
parquet_b = Path(r"{1}")
summary_a = Path(r"{2}")
summary_b = Path(r"{3}")
out_dir = Path(r"{4}")
label_a = r"{5}"
label_b = r"{6}"

out_dir.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# 1. Paired Delta Distributions
strat_file = "stratified_metrics_top_20_ranker_20.parquet"
p_a = parquet_a / strat_file
p_b = parquet_b / strat_file

if p_a.is_file() and p_b.is_file():
    df_a = pl.read_parquet(p_a)
    df_b = pl.read_parquet(p_b)
    if "user_id" in df_a.columns and "user_id" in df_b.columns:
        joined = df_a.join(df_b, on="user_id", how="inner", suffix="_b")
        if joined.height > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            metrics = [
                ("ndcg_global", "ndcg_global_b", "NDCG@20 Paired Delta (Candidate - Baseline)"),
                ("recall_global", "recall_global_b", "Recall@20 Paired Delta (Candidate - Baseline)"),
            ]
            for idx, (ca, cb, title) in enumerate(metrics):
                if ca in joined.columns and cb in joined.columns:
                    diffs = (joined[cb].cast(pl.Float64) - joined[ca].cast(pl.Float64)).to_numpy()
                    valid = diffs[np.isfinite(diffs)]
                    m_diff = float(np.mean(valid))
                    sns.histplot(valid, ax=axes[idx], kde=True, color="royalblue", bins=30)
                    axes[idx].axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Zero Delta")
                    axes[idx].axvline(m_diff, color="darkorange", linestyle="-", linewidth=2, label=f"Mean: {{m_diff:+.4f}}")
                    axes[idx].set_title(title, fontsize=12, fontweight="bold")
                    axes[idx].set_xlabel("Delta Value", fontsize=10)
                    axes[idx].set_ylabel("User Count", fontsize=10)
                    axes[idx].legend(frameon=True, fontsize=9)
            plt.suptitle(f"Paired Per-User Delta Distribution: {{label_b}} vs {{label_a}}", fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()
            plt.savefig(out_dir / "paired_deltas_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()

        # 2. User Tier Stratification (Tier 0 = Power, Tier 1 = Mid-Tail, Tier 2 = Light)
        if "user_tier" in joined.columns:
            tier_names = {{0: "Tier 0 (Power)", 1: "Tier 1 (Mid-Tail)", 2: "Tier 2 (Light)"}}
            rows = []
            for t_val in (0, 1, 2):
                sub = joined.filter(pl.col("user_tier") == t_val)
                if sub.height > 0:
                    for m_col, m_lbl in [("ndcg_global", "NDCG@20"), ("recall_global", "Recall@20")]:
                        mA = sub[m_col].cast(pl.Float64).to_numpy()
                        mB_vals = sub[f"{{m_col}}_b"].cast(pl.Float64).to_numpy()
                        rows.append({{ "Tier": tier_names.get(t_val, f"Tier {{t_val}}"), "Metric": m_lbl, "Model": label_a, "Value": float(np.nanmean(mA)) }})
                        rows.append({{ "Tier": tier_names.get(t_val, f"Tier {{t_val}}"), "Metric": m_lbl, "Model": label_b, "Value": float(np.nanmean(mB_vals)) }})
            if rows:
                import pandas as pd
                df_plot = pd.DataFrame(rows)
                plt.figure(figsize=(10, 5))
                sns.barplot(data=df_plot, x="Tier", y="Value", hue="Model", palette="Set2")
                plt.title(f"Performance Stratification by User Tier: {{label_b}} vs {{label_a}}", fontsize=13, fontweight="bold")
                plt.ylabel("Metric Score", fontsize=11)
                plt.xlabel("User Interaction Tier", fontsize=11)
                plt.legend(title="Model Run", frameon=True)
                plt.tight_layout()
                plt.savefig(out_dir / "user_tier_stratification.png", dpi=300, bbox_inches="tight")
                plt.close()

# 3. Lorenz Curve Comparison
l_a = summary_a / "lorenz_curve_top_20_ranker_20.json"
l_b = summary_b / "lorenz_curve_top_20_ranker_20.json"
if l_a.is_file() and l_b.is_file():
    with open(l_a, "r", encoding="utf-8") as fa, open(l_b, "r", encoding="utf-8") as fb:
        c_a = json.load(fa)
        c_b = json.load(fb)
    if isinstance(c_a, list) and isinstance(c_b, list):
        plt.figure(figsize=(6, 6))
        x_vals = np.linspace(0, 100, len(c_a))
        plt.plot(x_vals, [0] + c_a[:-1] if len(c_a) > 0 else c_a, label=label_a, linewidth=2.5, color="forestgreen")
        plt.plot(x_vals, [0] + c_b[:-1] if len(c_b) > 0 else c_b, label=label_b, linewidth=2.5, color="crimson")
        plt.plot([0, 100], [0, 1], linestyle="--", color="gray", label="Line of Perfect Equality", linewidth=1.5)
        plt.title("Catalog Popularity Concentration (Lorenz Curve)", fontsize=13, fontweight="bold")
        plt.xlabel("Cumulative Catalog Items (%)", fontsize=11)
        plt.ylabel("Cumulative Recommendations Share (%)", fontsize=11)
        plt.legend(frameon=True, loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / "lorenz_curve_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

print("[Graphics] Comparison charts rendered successfully.")
"#,
        parquet_dir_a.display(),
        parquet_dir_b.display(),
        summary_dir_a.display(),
        summary_dir_b.display(),
        output_dir.display(),
        label_a,
        label_b,
    );

    let script_path = output_dir.join("generate_plots.py");
    if let Ok(mut file) = File::create(&script_path) {
        use std::io::Write;
        let _ = file.write_all(script_content.as_bytes());
        let _ = Command::new("python3").arg(&script_path).status();
    }
}

// ==============================================================================
// PUBLIC ORCHESTRATOR & CLI / TEST ENTRYPOINTS
// ==============================================================================

pub fn compare_post_training_runs(args: &CompareArgs) -> Result<(), Box<dyn Error + Send + Sync>> {
    for (name, path) in [
        ("summary_dir_a", &args.summary_dir_a),
        ("parquet_dir_a", &args.parquet_dir_a),
        ("summary_dir_b", &args.summary_dir_b),
        ("parquet_dir_b", &args.parquet_dir_b),
    ] {
        if !path.is_dir() {
            return Err(format!("Directory for {} does not exist: {:?}", name, path).into());
        }
    }

    fs::create_dir_all(&args.output_dir)?;

    let label_a = args.label_a.clone().unwrap_or_else(|| {
        args.summary_dir_a
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("Model_A")
            .to_string()
    });
    let label_b = args.label_b.clone().unwrap_or_else(|| {
        args.summary_dir_b
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("Model_B")
            .to_string()
    });

    println!("[1/4] Comparing JSON summaries: {} vs {}...", label_a, label_b);
    let (summary_diffs, run_conclusions) =
        compare_summary_dirs(&args.summary_dir_a, &args.summary_dir_b)?;

    println!("[2/4] Running paired t-tests and Wilcoxon signed-rank tests on Parquet files...");
    let parquet_stats = compare_parquet_dirs(&args.parquet_dir_a, &args.parquet_dir_b, args.alpha)?;

    println!("[3/4] Synthesizing comparative conclusions & business value scorecard...");
    let conclusions_payload = synthesize_comparative_conclusions(
        &label_a,
        &label_b,
        &parquet_stats,
        &summary_diffs,
        &run_conclusions,
        args.alpha,
    );

    println!("[4/4] Generating publication-ready comparison graphics...");
    if let Err(e) = generate_comparison_graphics(
        &args.parquet_dir_a,
        &args.parquet_dir_b,
        &args.summary_dir_a,
        &args.summary_dir_b,
        &args.output_dir,
        &label_a,
        &label_b,
    ) {
        eprintln!("[Graphics] Warning: Failed to render comparison graphics: {}", e);
    }

    let metrics_payload = json!({
        "metadata": {
            "baseline_a": {
                "label": label_a,
                "summary_dir": args.summary_dir_a,
                "parquet_dir": args.parquet_dir_a,
            },
            "candidate_b": {
                "label": label_b,
                "summary_dir": args.summary_dir_b,
                "parquet_dir": args.parquet_dir_b,
            },
            "alpha": args.alpha,
        },
        "summary_json_comparisons": summary_diffs,
        "paired_parquet_significance_tests": parquet_stats,
    });

    let metrics_out_path = args.output_dir.join("summary_metrics.json");
    let conclusions_out_path = args.output_dir.join("summary_conclusions.json");

    serde_json::to_writer_pretty(File::create(&metrics_out_path)?, &metrics_payload)?;
    serde_json::to_writer_pretty(File::create(&conclusions_out_path)?, &conclusions_payload)?;

    println!("Wrote detailed metrics & statistical tests to: {:?}", metrics_out_path);
    println!("Wrote comparative conclusions to:              {:?}", conclusions_out_path);

    if let Some(verdict) = conclusions_payload
        .get("comparison_metadata")
        .and_then(|m| m.get("overall_verdict"))
        .and_then(|v| v.as_str())
    {
        println!("\n=== OVERALL VERDICT ===\n{}", verdict);
    }

    if let Some(matrix) = conclusions_payload
        .get("executive_business_decision_matrix")
        .and_then(|v| v.as_array())
    {
        println!("\n=== BEST MODEL BY OVERALL & BUSINESS VALUE ===");
        for line in matrix {
            if let Some(s) = line.as_str() {
                println!("  * {}", s);
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = CompareArgs::parse();
    compare_post_training_runs(&args)
}

#[cfg(test)]
mod compare_post_training_analysis {
    use super::*;

    #[test]
    fn test_compare_two_runs() -> Result<(), Box<dyn Error + Send + Sync>> {
        let q_ver_a: usize = 1;
        let r_ver_a: usize = 2;
        let q_ver_b: usize = 1;
        let r_ver_b: usize = 3;

        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../");

        let base_dir_a = project_root.join("TMP02");
        let base_dir_b = project_root.join("TMP03");

        let target_dir = project_root.join("bin");
        let output_dir = target_dir.join(
            format!(
                "comparison_q{:02}_r{:02}_vs_q{:02}_r{:02}",
                q_ver_a, r_ver_a, q_ver_b, r_ver_b
            )
        );

        println!("output_dir: {:?}", output_dir);

        let args = CompareArgs {
            summary_dir_a: base_dir_a.join(format!("post_training_analysis_q{:02}_r{:02}", q_ver_a, r_ver_a)),
            parquet_dir_a: base_dir_a.join(format!("parquet_metrics_q{:02}_r{:02}", q_ver_a, r_ver_a)),
            summary_dir_b: base_dir_b.join(format!("post_training_analysis_q{:02}_r{:02}", q_ver_b, r_ver_b)),
            parquet_dir_b: base_dir_b.join(format!("parquet_metrics_q{:02}_r{:02}", q_ver_b, r_ver_b)),
            output_dir: output_dir.clone(),
            label_a: Some(format!("q{:02}_r{:02}", q_ver_a, r_ver_a)),
            label_b: Some(format!("q{:02}_r{:02}", q_ver_b, r_ver_b)),
            alpha: 0.05,
        };

        // Only execute if both directories exist on disk
        if args.summary_dir_a.is_dir() && args.summary_dir_b.is_dir() {
            compare_post_training_runs(&args)?;
        } else {
            panic!(
                "Skipping test_compare_two_runs: {:?} or {:?} not yet generated.",
                args.summary_dir_a, args.summary_dir_b
            );
        }

        // Verify both output JSON files were created and Model 1 (q01_r01) is kept over Model 2 (q02_r02)
        let conclusions_path = output_dir.join("summary_conclusions.json");
        let metrics_path = output_dir.join("summary_metrics.json");
        assert!(conclusions_path.is_file(), "summary_conclusions.json should exist");
        assert!(metrics_path.is_file(), "summary_metrics.json should exist");

        let conclusions_json: Value =
            serde_json::from_reader(BufReader::new(File::open(&conclusions_path)?))?;
        let verdict = conclusions_json["comparison_metadata"]["overall_verdict"]
            .as_str()
            .unwrap_or("");

        println!("Test Verdict: {}", verdict);

        Ok(())
    }

    #[test]
    fn test_test_data() -> Result<(), Box<dyn Error + Send + Sync>> {
        let q_ver_a: usize = 1;
        let r_ver_a: usize = 1;
        let q_ver_b: usize = 2;
        let r_ver_b: usize = 2;

        let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../");

        let base_dir = project_root.join("inference_src/test/rust/resources/post_training_comparison_test");

        let target_dir = project_root.join("inference_src/main/rust/target/debug");
        let output_dir = target_dir.join(
            format!(
                "comparison_q{:02}_r{:02}_vs_q{:02}_r{:02}",
                q_ver_a, r_ver_a, q_ver_b, r_ver_b
            )
        );


        let args = CompareArgs {
            summary_dir_a: base_dir.join(format!("post_training_analysis_q{:02}_r{:02}", q_ver_a, r_ver_a)),
            parquet_dir_a: base_dir.join(format!("parquet_metrics_q{:02}_r{:02}", q_ver_a, r_ver_a)),
            summary_dir_b: base_dir.join(format!("post_training_analysis_q{:02}_r{:02}", q_ver_b, r_ver_b)),
            parquet_dir_b: base_dir.join(format!("parquet_metrics_q{:02}_r{:02}", q_ver_b, r_ver_b)),
            output_dir: output_dir.clone(),
            label_a: Some(format!("q{:02}_r{:02}", q_ver_a, r_ver_a)),
            label_b: Some(format!("q{:02}_r{:02}", q_ver_b, r_ver_b)),
            alpha: 0.05,
        };

        // Only execute if both directories exist on disk
        if args.summary_dir_a.is_dir() && args.summary_dir_b.is_dir() {
            compare_post_training_runs(&args)?;
        } else {
            println!(
                "Skipping test_compare_two_runs: {:?} or {:?} not yet generated.",
                args.summary_dir_a, args.summary_dir_b
            );
        }

        // Verify both output JSON files were created and Model 1 (q01_r01) is kept over Model 2 (q02_r02)
        let conclusions_path = output_dir.join("summary_conclusions.json");
        let metrics_path = output_dir.join("summary_metrics.json");
        assert!(conclusions_path.is_file(), "summary_conclusions.json should exist");
        assert!(metrics_path.is_file(), "summary_metrics.json should exist");

        let conclusions_json: Value =
            serde_json::from_reader(BufReader::new(File::open(&conclusions_path)?))?;
        let verdict = conclusions_json["comparison_metadata"]["overall_verdict"]
            .as_str()
            .unwrap_or("");

        println!("Test Verdict: {}", verdict);
        assert!(
            verdict.contains("KEEP BASELINE q01_r01"),
            "Expected Baseline Model 1 to clearly beat Candidate Model 2, got: {}",
            verdict
        );

        Ok(())
    }
}
