use std::path::PathBuf;

/*
==========================================================================
a look at what an ANN search using the TwoTowerDNN user embeddings can
produce in terms of the number of tail movies it will return for a user
who likes tail distribution movies.

not finished with this one yet

==========================================================================
*/

pub mod helper {
    include!("helper.rs");
    include!("helper_users.rs");
}
use inference_engine::app_config::AppConfig;
use inference_engine::orchestrator::Orchestrator;
use helper::{get_train_val_test_liked_uris, DataSize};
use crate::helper::get_config_json_uri;

use polars::prelude::{ DataFrame, IntoLazy};

struct TestHarness {
    orchestrator: Orchestrator,
    train_uri: String,
    val_uri: String,
    test_uri: String,
    movie_tiers_path: String
}

impl TestHarness {
    async fn new() -> Self {
        println!("\n[SETUP]: Initializing test resource...");
        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let query_uri = config.query_uri.clone();
        let ranker_uri = config.ranker_uri.clone();
        let ranker_n_local_devices = config.ranker_n_local_devices;
        let top_k = config.top_k;
        let user_db_path: PathBuf = config.user_db_path.clone();
        let persisted_index_path : PathBuf = config.persisted_index_path.clone();
        let _params_json_uri : String = config.params_json_path;

        let movie_embeddings_uri : String = config.movie_embeddings_path;

        // this won't be used
        let ratings_map = get_train_val_test_liked_uris(DataSize::Tiny3, false);
        let ratings_uris: Vec<&str> = vec![
            ratings_map.get("train_liked").unwrap(),
            ratings_map.get("train_3").unwrap(),
            ratings_map.get("train_disliked").unwrap(),
            ratings_map.get("val_liked").unwrap(),
            ratings_map.get("val_3").unwrap(),
            ratings_map.get("val_disliked").unwrap(),
        ];
        let full_ratings_map = get_train_val_test_liked_uris(DataSize::Full, false);

        let orchestrator = Orchestrator::new(
            query_uri,
            ranker_uri,
            config.query_metadata_uri,
            config.ranker_metadata_uri,
            &movie_embeddings_uri,
            ratings_uris,
            ranker_n_local_devices,
            top_k,
            persisted_index_path,
            user_db_path
        ).await.unwrap();

        /*
        let ratings_uris: Vec<String> = vec![
            ratings_map.get("train_liked").unwrap().clone(),
            ratings_map.get("train_3").unwrap().clone(),
            ratings_map.get("train_disliked").unwrap().clone(),
            ratings_map.get("val_liked").unwrap().clone(),
            ratings_map.get("val_3").unwrap().clone(),
            ratings_map.get("val_disliked").unwrap().clone(),
        ];*/

        Self {
            orchestrator: orchestrator,
            train_uri: full_ratings_map.get("train_liked").unwrap().clone(),
            val_uri: full_ratings_map.get("val_liked").unwrap().clone(),
            test_uri: full_ratings_map.get("test_liked").unwrap().clone(),
            movie_tiers_path: config.movie_tiers_path
        }
    }
}

// --- TEARDOWN LOGIC ---
impl Drop for TestHarness {
    fn drop(&mut self) {
        println!("[TEARDOWN]");
    }
}

#[cfg(test)]
mod num_candidates_tests {
    use std::collections::{HashMap, HashSet};
    use polars::df;
    use polars::prelude::{col, JoinArgs, JoinType, lit};
    // Bring everything from the outer scope (TestHarness, helper functions, etc.) into the test module
    use super::*;

    //bring the gRPC trait into scope so its methods (.predict) are visible
    use inference_engine::pb::{ApproxNearestNeighborsResponse, UsersRequest};
    use tonic::{Request};
    use inference_engine::movie_tiers::load_from_file;
    use crate::helper::{get_unique_user_and_first_timestamp, get_user_movie_tier_map, load_and_concat_parquet};

    #[tokio::test(flavor = "multi_thread")]
    async fn test_orchestrator() -> Result<(), Box<dyn std::error::Error>>{

        /*
        gets the tail users present in all 3 datasets train, val, and test and looks at how large
        the ANN search k must be to return a significant number of tier==2 movies such that
        after ranked, the top_k could possibly have tier==2 recommendations in them.
         */

        // Setup runs here
        let harness = TestHarness::new().await;

        let movie_tiers : HashMap<i32, i32> = load_from_file(&harness.movie_tiers_path)?;

        let ranker_batch_size = harness.orchestrator.get_ranker_model_metadata().batch_size;

        let common_user_ids : HashSet<i32> = get_intersection_of_user_ids_tier_2(&harness.train_uri, &harness.val_uri,
            &harness.test_uri, &movie_tiers)?;

        let ks: Vec<i32> = [70, 100]
            .into_iter()
            .chain((200..2100).step_by(100).map(|x| x as i32))
            .collect();

        // hashmap key= train_k, val_k, test_k, value=avg number of unwatched movie_tier=2 returned in k ANN
        let mut ds_k_counts: HashMap<String, usize> = HashMap::new();
        for &k in &ks {
            ds_k_counts.insert(format!("train_{:05}", k), 0);
            ds_k_counts.insert(format!("val_{:05}", k), 0);
            ds_k_counts.insert(format!("test_{:05}", k), 0);
        }

        for (i, ratings_file_path) in vec![&harness.train_uri, &harness.val_uri, &harness.test_uri].iter().enumerate() {

            let dataset_name = match i {
                0 => "train",
                1 => "val",
                _ => "test",
            };

            let (_movie_tier_vec_map, user_ids, timestamps) : (Vec<HashMap<i32, HashSet<i32>>>,Vec<i32>, Vec<i64> )
                = get_user_datastructures(&[&ratings_file_path], &movie_tiers)?;

            let movie_tier_2_set : HashSet<i32> = movie_tiers
                .iter()
                .filter(|&(_, &tier)| tier == 2)
                .map(|(&movie_id, _)| movie_id)
                .collect();

            // keep only the common_user_ids
            let (user_ids, timestamps): (Vec<i32>, Vec<i64>) = user_ids
                .into_iter()
                .zip(timestamps)
                .filter(|(uid, _)| common_user_ids.contains(uid))
                .unzip();

            for (chunk_users, chunk_timestamps) in user_ids.chunks(ranker_batch_size).zip(timestamps.chunks(ranker_batch_size)) {

                let tonic_request: Request<UsersRequest>  =
                    harness.orchestrator.get_users_request(chunk_users, chunk_timestamps).await?;
                let mut users_request : UsersRequest = tonic_request.into_inner();

                for  &k in &ks {

                    let key = format!("{}_{:05}", dataset_name, k);

                    users_request.k = Some(k as u32);

                    let ann_reqs = Request::new(users_request.clone());
                    let ann_res: ApproxNearestNeighborsResponse = harness.orchestrator._approx_nearest_neighbors(ann_reqs).await?.into_inner();

                    // Extract the generated fields from the new protobuf response message
                    //let user_ids = ann_res.user_ids;
                    //let n_users = user_ids.len();
                    // length: n_users * k
                    let ann_movie_ids = ann_res.candidate_ids;

                    // how many are in tier=2 movies?
                    for movie_candidates in ann_movie_ids.chunks_exact(k as usize) {
                        let intersection_count = movie_candidates
                            .iter()
                            .filter(|&movie_id| movie_tier_2_set.contains(movie_id))
                            .count();

                        // Use entry API to update in place safely and idiomatically
                        *ds_k_counts.entry(key.clone()).or_insert(0) += intersection_count;
                    }

                }

            }
        }

        let mut keys: Vec<&String> = ds_k_counts.keys().collect();
        keys.sort();

        let n = common_user_ids.len() as f32;

        for key in keys {
            if let Some((_dataset_name, k)) = parse_key(key) {
                if let Some(val) = ds_k_counts.get(key) {
                    let v : f32 = *val as f32 / n;
                    println!("{}: {:.3}  : {:.3}", key, v, v/(k as f32));
                }
            }
        }

        Ok(())

        // Teardown automatically runs here when `_harness` goes out of scope at the end of the test function
    }

    fn parse_key(key: &str) -> Option<(&str, u32)> {
        let (dataset_name, k_str) = key.rsplit_once('_')?;
        let k = k_str.parse::<u32>().ok()?;
        Some((dataset_name, k))
    }

    fn get_intersection_of_user_ids_tier_2(train_path: &String, val_path: &String,
        test_path: &String, movie_tier_map_ref : &HashMap<i32, i32>) -> Result<HashSet<i32>, Box<dyn std::error::Error>> {

        let df_train = load_and_concat_parquet(&[train_path])?;
        let df_val = load_and_concat_parquet(&[val_path])?;
        let df_test = load_and_concat_parquet(&[test_path])?;

        // read movie_tiers into a df to join with train, val, test
        let movie_ids: Vec<i32> = movie_tier_map_ref.keys().copied().collect();
        let tiers: Vec<i32> = movie_tier_map_ref.values().copied().collect();
        let tier_df = df!["movie_id" => movie_ids, "movie_tier" => tiers]?
            .lazy();

        // join and filter for movie_tier==2
        let df_train = df_train.join(tier_df.clone(), [col("movie_id")],
                [col("movie_id")], JoinArgs::new(JoinType::Inner), )
            .filter(col("movie_tier").eq(lit(2))) // Added filter for tier 2
            .select([col("user_id"), col("movie_id"), col("movie_tier")])
            .collect()?;

        let df_val = df_val.join(tier_df.clone(), [col("movie_id")],
            [col("movie_id")], JoinArgs::new(JoinType::Inner), )
            .filter(col("movie_tier").eq(lit(2))) // Added filter for tier 2
            .select([col("user_id"), col("movie_id"), col("movie_tier")])
            .collect()?;

        let df_test = df_test.join(tier_df, [col("movie_id")],
            [col("movie_id")], JoinArgs::new(JoinType::Inner), )
            .filter(col("movie_tier").eq(lit(2))) // Added filter for tier 2
            .select([col("user_id"), col("movie_id"), col("movie_tier")])
            .collect()?;

        let extract_users = |df: &DataFrame| -> Option<HashSet<i32>> {
            let ca = df.column("user_id").ok()?.i32().ok()?;
            Some(ca.iter().flatten().collect())
        };

        let set_train = extract_users(&df_train).ok_or("error")?;
        let set_val = extract_users(&df_val).ok_or("error")?;
        let set_test = extract_users(&df_test).ok_or("error")?;

        // Find the multi-way intersection
        let mut intersection_set = set_train;
        intersection_set.retain(|user_id| set_val.contains(user_id) && set_test.contains(user_id));

        Ok(intersection_set)
    }

    pub fn get_user_datastructures(
        ratings_uris: &[&str],
        movie_tiers: &HashMap<i32, i32>
    ) -> Result<(Vec<HashMap<i32, HashSet<i32>>>, Vec<i32>, Vec<i64>), Box<dyn std::error::Error>> {

        let df_gt = load_and_concat_parquet(ratings_uris)?;

        let movie_tier_maps_vec = get_user_movie_tier_map(df_gt.clone(), movie_tiers)?;

        let (user_ids, timestamps) = get_unique_user_and_first_timestamp(df_gt)?;

        Ok((movie_tier_maps_vec, user_ids, timestamps))
    }

}