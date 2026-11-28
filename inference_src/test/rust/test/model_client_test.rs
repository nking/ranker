#[cfg(test)]
mod client_tests {
    use std::collections::HashMap;
    use std::error::Error;
    use std::path::{PathBuf};
    use std::sync::Arc;
    use polars::prelude::{col, LazyFrame};
    use rand::distributions::{Distribution, Uniform};
    use serde_json::Value;
    use inference_engine::app_config::AppConfig;
    use inference_engine::hard_negative_sampling::{HardNegativeSamplingTransform, SampledCandidates, SamplingStrategy};
    use inference_engine::model_client::{CandidateModelClient, QueryModelClient, RankerModelClient, RerankerModelClient};
    use inference_engine::movie_db::MovieDb;
    use inference_engine::ranker_graph_builder::{create_fake_padded_super_batch, JraphGraph};
    use inference_engine::tensorflow::serving::model_spec::VersionChoice;
    //use super::*;
    // Assuming your UserRequest is accessible here
    use inference_engine::pb::{UsersRequest, MoviesRequest};
    use inference_engine::query_model_metadata::QueryModelMetadata;
    use inference_engine::ranker_model_metadata::RankerModelMetadata;
    use inference_engine::recommended_movies::{build_recommended_movies};
    use inference_engine::user_history::{build_user_history, UserHistory};
    use inference_engine::reranker_prompt_formatter::{from_movie_db_and_yaml_files};
    use reranker_prompt::reranker_prompt_formatter::FormattedBatch;

    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
        include!("helper_users.rs");
    }
    use crate::client_tests::helper::{get_config_json_uri, get_embeddings_metadata_uris,
        get_embeddings_uris, get_project_dir, get_tiny_config_json_uri,
        get_train_val_test_liked_uris, DataSize, get_recommended_movies_uris,
        load_and_concat_parquet};

    #[tokio::test]
    async fn test_query_model_connection() {

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let q_ver = 1;
        let r_ver = 1;

        let query_metadata_uri = format!(
            "{}/{}/assets.extra/{}",
            config.query_models_path.trim_end_matches('/'), q_ver, "hyperparameters.json");

        let query_metadata = QueryModelMetadata::load_from_file(&query_metadata_uri).unwrap();

        let client = QueryModelClient::new(config.query_uri.clone(), config.query_models_path.clone()).await;

        let mock_request = UsersRequest {
            user_ids: vec![42],
            genders: vec!["M".to_string()],
            occupations: vec![10],
            ages: vec![25],
            timestamps: vec![1620000000],
            n_users: 1,
            k : None,
            query_model_version:Some(q_ver),
            ranker_model_version:Some(r_ver)
        };

        let query_model_version_choice = Some(VersionChoice::Version(1));

        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>
            = client.get_users_embeddings(mock_request, query_model_version_choice).await;

        assert!(result.is_ok(), "Failed to get user embedding: {:?}", result.err());

        let embedding = result.unwrap();
        assert_eq!(query_metadata.embed_len, embedding.len());

        // Assert the expected dimension length (e.g., 128)
        assert!(!embedding.is_empty(), "User Embedding vector is empty!");

        let latest_version = client.get_latest_model_version().await.expect("Couldn't get latest model version");

        let expected_latest_version: i64 = std::fs::read_dir(&config.query_models_path)
            .expect("Failed to read model repository directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                // Ignore files; only keep directories
                if !entry.file_type().ok()?.is_dir() {
                    return None;
                }
                // Parse directory name into i64 (ignores non-numeric directory names)
                entry.file_name().to_str()?.parse::<i64>().ok()
            })
            .max()
            .expect("No valid numeric version subdirectories found");

        assert_eq!(latest_version, expected_latest_version);

        let q_version_choice = Some(VersionChoice::Version(q_ver));
        let metadata = client.get_or_fetch_metadata(q_version_choice).await;
        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();

        let r = client.update_latest_model_version().await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_candidate_model_connection() {

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let c_ver = 1;
        let model_version_choice = Some(VersionChoice::Version(c_ver));

        let client = CandidateModelClient::new(config.candidate_uri,
            config.candidate_models_path).await;

        let metadata = client.get_or_fetch_metadata(model_version_choice.clone()).await;
        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();

        let mock_request = MoviesRequest {
            movie_ids: vec![6041],
            genres: vec!["Action|Sci-Fi".to_string()],
            n_movies : 1,
            candidate_model_version:Some(1),
        };


        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>
            = client.get_movies_embeddings(mock_request, model_version_choice.clone()).await;

        assert!(result.is_ok(), "Failed to get movie embedding: {:?}", result.err());

        let embedding = result.unwrap();
        assert_eq!(metadata.embed_len, embedding.len());

        // Assert the expected dimension length (e.g., 128)
        assert!(!embedding.is_empty(), "Movie Embedding vector is empty!");

        let latest_version = client.get_latest_model_version().await.expect("Couldn't get latest model version");

        let expected_latest_version: i64 = std::fs::read_dir(client.get_models_path())
            .expect("Failed to read model repository directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                // Ignore files; only keep directories
                if !entry.file_type().ok()?.is_dir() {
                    return None;
                }
                // Parse directory name into i64 (ignores non-numeric directory names)
                entry.file_name().to_str()?.parse::<i64>().ok()
            })
            .max()
            .expect("No valid numeric version subdirectories found");

        assert_eq!(latest_version, expected_latest_version);

        let r = client.update_latest_model_version().await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_ranker_model_connection() -> Result<(), Box<dyn std::error::Error>>{

        let ver = 1;
        let model_version_choice = Some(VersionChoice::Version(ver));

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let filename = if config.ranker_serving_is_batched {
            "metadata_batch.json"
        } else {
            "metadata_single.json"
        };
        let ranker_metadata_uri = format!(
            "{}/{}/assets.extra/{}",
            config.ranker_models_path.trim_end_matches('/'), ver, filename);

        // the default config is for the batch ranker model, so change to the single inference model:
        let single_uri = ranker_metadata_uri.replace("batch", "single");

        let ranker_metadata = RankerModelMetadata::load_from_file(&single_uri).unwrap();

        // instantiated for not using the batch configuration:
        let client = RankerModelClient::new(config.ranker_uri.clone(),
            config.ranker_models_path.clone(), false).await;

        let _top_k = config.ranker_top_k;
        let _user_db_path: PathBuf = config.user_db_path.clone();
        let _persisted_index_path : PathBuf = config.persisted_index_path.clone();

        let max_history = ranker_metadata.max_history;
        let num_candidates = ranker_metadata.num_candidates;
        let _num_catalog_users = ranker_metadata.num_catalog_users;

        // the placeholder mode used:
        //"signature_name": "serving_batch", "batch_size": 256, "max_history": 60, "num_candidates": 60, "max_nodes": 31040, "max_edges": 30784, "max_graphs": 258, "embed_len": 16
        //    the range of movie_id_range must be >= (num_history + num_candidates + 1)
        let batch_size : usize = ranker_metadata.batch_size;
        let n_users = 1;
        assert_eq!(batch_size, 1);
        let user_id_range = (1, 10);
        let movie_id_range = (6041, 6040 + (max_history + num_candidates + 2));
        let n_local_devices = 1;


        let (user_embeddings_uri, movie_embeddings_uri) = get_embeddings_uris(ver as i32);

        let padded_super_graph : JraphGraph  = create_fake_padded_super_batch(
            n_users,
            batch_size,
            max_history, num_candidates, user_id_range,
            movie_id_range, n_local_devices,
            &user_embeddings_uri, &movie_embeddings_uri
        );

        let (user_emb_metadata_uri, _movie_emb_metadata_uri) : (String, String)
            = get_embeddings_metadata_uris(ver as i32);

        let json_content = tokio::fs::read_to_string(&user_emb_metadata_uri).await?;
        let dict: HashMap<String, Value> = serde_json::from_str(&json_content)?;
        let embed_len = dict.get("embed_dim").and_then(|v| v.as_u64()).unwrap();

        let query_model_version = Some(VersionChoice::Version(1));

        // If the docker container isn't running, or the model isn't loaded,
        // this will fail and print the gRPC status error.
        let result : Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>
            = client.get_candidate_ranks(padded_super_graph, embed_len as usize, query_model_version.clone()).await;

        assert!(result.is_ok(), "Failed to get ranks: {:?}", result.err());

        let ranks: Vec<f32> = result.unwrap();
        // the batch_size * num_candidates are the scored values, all else are dummy scores, piled up at end
        let _ranks : &[f32] = &ranks[0..batch_size * num_candidates];
        assert_eq!(_ranks.len(), batch_size * num_candidates);

        // ============================================================================================
        // switch to a batch client to test diagnostics

        let ranker_metadata = RankerModelMetadata::load_from_file(&ranker_metadata_uri).unwrap();
        // instantiated for not using the batch configuration:
        let client = RankerModelClient::new(config.ranker_uri.clone(),
            config.ranker_models_path.clone(), true).await;

        let ranker_batch_size : usize = ranker_metadata.batch_size;
        let out_features = ranker_metadata.out_features;

        let padded_super_graph : JraphGraph  = create_fake_padded_super_batch(
            n_users,
            ranker_batch_size,
            max_history, num_candidates, user_id_range,
            movie_id_range, n_local_devices,
            &user_embeddings_uri, &movie_embeddings_uri
        );

        let result = client.get_candidate_diagnostics(
            padded_super_graph, embed_len as usize, query_model_version.clone()).await;
        assert!(result.is_ok(), "Failed to get ranks: {:?}", result.err());
        let (mut outputs, mut user_reprs, mut cand_reprs) = result
            .expect("error with request or response for ranker diagnostics");

        outputs.truncate(batch_size * num_candidates);
        user_reprs.truncate(batch_size * out_features);
        cand_reprs.truncate(batch_size * num_candidates * out_features);

        let latest_version = client.get_latest_model_version().await.expect("Couldn't get latest model version");

        let model_dir = get_project_dir()
            .expect("Failed to get project directory")
            .join("src/test/resources/model_repositories/saved_model_formats/cross-encoder/graph-ranker/");

        let expected_latest_version: i64 = std::fs::read_dir(&model_dir)
            .expect("Failed to read model repository directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                // Ignore files; only keep directories
                if !entry.file_type().ok()?.is_dir() {
                    return None;
                }
                // Parse directory name into i64 (ignores non-numeric directory names)
                entry.file_name().to_str()?.parse::<i64>().ok()
            })
            .max()
            .expect("No valid numeric version subdirectories found");

        assert_eq!(latest_version, expected_latest_version);

        let metadata = client.get_or_fetch_metadata(model_version_choice.clone()).await;
        assert!(metadata.is_ok());

        let r = client.update_latest_model_version().await;
        assert!(r.is_ok());

        Ok(())

    }

    #[tokio::test]
    async fn test_reranker_model_connection() -> Result<(), Box<dyn std::error::Error>> {
        let ver = 1;
        let version_option = Some(ver);

        let config_path = get_config_json_uri();
        let config = AppConfig::load_from_file(&config_path).unwrap();

        let client = RerankerModelClient::new(
            config.reranker_uri.clone(),
            config.reranker_models_path.clone(),
        ).await;

        let latest_version = client
            .get_latest_model_version()
            .await
            .expect("Couldn't get latest model version");

        let expected_latest_version: i64 = std::fs::read_dir(&config.reranker_models_path)
            .expect("Failed to read model repository directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if !entry.file_type().ok()?.is_dir() {
                    return None;
                }
                entry.file_name().to_str()?.parse::<i64>().ok()
            })
            .max()
            .expect("No valid numeric version subdirectories found");

        assert_eq!(latest_version, expected_latest_version);

        let metadata = client.get_or_fetch_metadata(version_option).await;
        assert!(metadata.is_ok());

        let r = client.update_latest_model_version().await;
        assert!(r.is_ok());

        let metadata = client
            .get_or_fetch_metadata(version_option)
            .await
            .expect("Failed to fetch reranker metadata");

        let models_path = client.get_models_path();
        assert_eq!(models_path, config.reranker_models_path);

        let r = client.is_model_ready(version_option).await;
        assert!(r.is_ok());

        // Load Hugging Face fast tokenizer from `<models_path>/<ver>/tokenizer.json`
        //let version_dir = Path::new(&models_path).join(ver.to_string());
        let tokenizer = client.get_or_fetch_tokenizer(version_option).await
            .expect("Error getting tokenizer");
        let placeholder_token_id = tokenizer
            .token_to_id(&metadata.emb_placeholder_token)
            .expect("emb_placeholder_token not found as a single special token in tokenizer.json")
            as i64;

        //===================================================================================
        //  test for get_reranked_scores
        //===================================================================================

        let q_ver = 1;
        let tiny_config_path = get_tiny_config_json_uri().to_string();
        let tiny_config = AppConfig::load_from_file(&tiny_config_path)
            .expect("Could not load tiny config json file");
        let ratings_map = get_train_val_test_liked_uris(DataSize::Full, false);
        let max_history = 2048;
        let ratings_uris_all: Vec<&str> = vec![
            ratings_map.get("train_liked").unwrap(),
            ratings_map.get("train_3").unwrap(),
            ratings_map.get("train_disliked").unwrap(),
            ratings_map.get("val_liked").unwrap(),
            ratings_map.get("val_3").unwrap(),
            ratings_map.get("val_disliked").unwrap(),
        ];
        let user_history: UserHistory = build_user_history(&ratings_uris_all, max_history).await;

        let ratings_uris_disliked: Vec<&str> = vec![
            ratings_map.get("train_disliked").unwrap(),
            ratings_map.get("val_disliked").unwrap(),
        ];
        let user_history_disliked: UserHistory =
            build_user_history(&ratings_uris_disliked, max_history).await;

        let (movies_rec_uri, movies_rec_ts_uri) = get_recommended_movies_uris(q_ver);
        let recommended_movies =
            build_recommended_movies(metadata.num_catalog_users, &movies_rec_uri, &movies_rec_ts_uri);

        let (query_yaml_path, passages_yaml_path) = client
            .get_prompt_yaml_paths(version_option)
            .await
            .expect("Error getting prompt yaml paths");

        let movie_db = MovieDb::new(&tiny_config.movies_path).unwrap();

        let hard_negative_sampling = HardNegativeSamplingTransform::new(
            Arc::new(user_history.clone()),
            Arc::new(user_history_disliked),
            movie_db.get_all_movie_ids(),
            recommended_movies,
            metadata.num_candidates,
            SamplingStrategy::WorstRecommended,
        ).expect("Error constructing HardNegativeSamplingTransform");

        let prompt_formatter = from_movie_db_and_yaml_files(
            &movie_db,
            &tiny_config.movie_tiers_path,
            &query_yaml_path,
            &passages_yaml_path,
            metadata.emb_placeholder_token.clone(),
            metadata.max_prompt_history,
            metadata.num_candidates,
        )
            .expect("Error creating prompt formatter");

        let seed = 42;
        let preserve_positive = true;
        // Inference mode keeps candidate order deterministic (no shuffling, no completions)
        let training = false;

        let pos_test_df: LazyFrame =
            load_and_concat_parquet(&[&ratings_map.get("test_liked").unwrap().clone()])
                .expect("error reading test ratings into df");
        let pos_df = pos_test_df
            .select([
                col("user_id"),
                col("movie_id"),
                col("rating"),
                col("timestamp"),
            ])
            .collect()?;
        let user_ids: Vec<i32> = pos_df.column("user_id")?.i32()?.into_no_null_iter().collect();
        let movie_ids: Vec<i32> = pos_df.column("movie_id")?.i32()?.into_no_null_iter().collect();
        let _ratings: Vec<i32> = pos_df.column("rating")?.i32()?.into_no_null_iter().collect();
        let timestamps: Vec<i64> = pos_df.column("timestamp")?.i64()?.into_no_null_iter().collect();

        let emb_dim = metadata.ranker_emb_dim;
        let mut rng = rand::thread_rng();
        let dist = Uniform::new_inclusive(-1.0_f32, 1.0_f32);

        // Process 2 small batches (use a small batch size for local Qwen ONNX VRAM/RAM limits)
        let batch_size = 4;
        let mut total_scored_rows = 0usize;

        for b in 0..2 {
            let i0 = b * batch_size;
            let i1 = (i0 + batch_size).min(user_ids.len());
            if i0 >= i1 {
                break;
            }
            let cur_batch_size = i1 - i0;
            let row_indices: Vec<u64> = (i0 as u64..i1 as u64).collect();

            let (hist_movie_ids, hist_ratings, hist_timestamps) = user_history
                .get_all_history_before_timestamp(
                    &user_ids[i0..i1],
                    &timestamps[i0..i1],
                    metadata.max_prompt_history,
                    true,
                );

            let sampled_candidates: SampledCandidates = hard_negative_sampling
                .sample_candidates_with_pos(
                    &user_ids[i0..i1],
                    &movie_ids[i0..i1],
                    &timestamps[i0..i1],
                    seed,
                    false,
                )
                .expect("Error during HardNegativeSamplingTransform.sample_candidates_with_pos");

            let candidate_ids_batch: Vec<i32> = sampled_candidates
                .candidate_ids
                .iter()
                .flatten()
                .copied()
                .collect();

            // Synthetic scores and embeddings: positive candidate at slot 0 matches user_emb
            let mut candidate_scores_batch: Vec<f32> =
                Vec::with_capacity(cur_batch_size * metadata.num_candidates);
            let mut user_embeddings_batch: Vec<f32> = Vec::with_capacity(cur_batch_size * emb_dim);
            let mut candidate_embeddings_batch: Vec<f32> =
                Vec::with_capacity(cur_batch_size * metadata.num_candidates * emb_dim);

            for _ in 0..cur_batch_size {
                candidate_scores_batch.push(100.0);

                let user_emb: Vec<f32> = dist.sample_iter(&mut rng).take(emb_dim).collect();
                user_embeddings_batch.extend_from_slice(&user_emb);
                candidate_embeddings_batch.extend_from_slice(&user_emb);

                for _ in 1..metadata.num_candidates {
                    candidate_scores_batch.push(10.0);
                    candidate_embeddings_batch.extend(dist.sample_iter(&mut rng).take(emb_dim));
                }
            }

            let formatted_prompts_batch: FormattedBatch = prompt_formatter
                .format_batch_from_padded_i32(
                    &user_ids[i0..i1],
                    &movie_ids[i0..i1],
                    &timestamps[i0..i1],
                    &hist_movie_ids,
                    &hist_ratings,
                    &hist_timestamps,
                    metadata.max_prompt_history,
                    user_history.pad_value,
                    &candidate_ids_batch,
                    &candidate_scores_batch,
                    &user_embeddings_batch,
                    &candidate_embeddings_batch,
                    &row_indices,
                    seed,
                    training,
                    preserve_positive,
                );

            let kept_batch_size = formatted_prompts_batch.kept_indices.len();
            if kept_batch_size == 0 {
                continue;
            }

            // Tokenize kept prompts in parallel and pack into flat [kept_batch_size, seq_len] i64 buffers
            let prompt_slices: Vec<&str> = formatted_prompts_batch
                .prompts
                .iter()
                .map(String::as_str)
                .collect();
            let encodings = tokenizer
                .encode_batch(prompt_slices, false)
                .map_err(|e| -> Box<dyn std::error::Error> { e })?;

            let seq_len = encodings[0].len();
            let expected_placeholders = 1 + formatted_prompts_batch.effective_top_k;

            let mut input_ids = Vec::with_capacity(kept_batch_size * seq_len);
            let mut attention_mask = Vec::with_capacity(kept_batch_size * seq_len);

            for enc in &encodings {
                assert_eq!(enc.len(), seq_len, "Batch padding produced uneven sequence lengths");
                let row_ids = enc.get_ids();
                let row_mask = enc.get_attention_mask();

                let mut placeholder_count = 0usize;
                for (&id, &m) in row_ids.iter().zip(row_mask.iter()) {
                    let id_i64 = id as i64;
                    if id_i64 == placeholder_token_id {
                        placeholder_count += 1;
                    }
                    input_ids.push(id_i64);
                    attention_mask.push(m as i64);
                }
                assert_eq!(
                    placeholder_count, expected_placeholders,
                    "Expected {} `{}` tokens per prompt, found {}",
                    expected_placeholders, metadata.emb_placeholder_token, placeholder_count
                );
            }

            let ranker_embs_opt = metadata
                .include_ranker_embeddings
                .then_some(formatted_prompts_batch.ranker_embs.as_slice());

            let scores = client
                .get_reranked_scores(
                    &input_ids,
                    &attention_mask,
                    ranker_embs_opt,
                    kept_batch_size,
                    seq_len,
                    version_option,
                )
                .await
                .map_err(|e| -> Box<dyn std::error::Error> { e })?;

            assert_eq!(
                scores.len(),
                kept_batch_size * metadata.num_candidates,
                "Unexpected reranker output length"
            );
            assert!(
                scores.iter().all(|s| s.is_finite()),
                "Reranker returned NaN or Inf scores"
            );
            assert!(
                scores.iter().all(|&s| s <= 1e-4),
                "Expected candidate log-probabilities <= 0, got max={}",
                scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            );

            println!(
                "Batch {b}: kept={kept_batch_size}, seq_len={seq_len}, row 0 log_probs={:?}",
                &scores[..metadata.num_candidates]
            );
            total_scored_rows += kept_batch_size;
        }

        assert!(
            total_scored_rows > 0,
            "Expected at least one non-empty-history row to be scored"
        );

        Ok(())
    }
}