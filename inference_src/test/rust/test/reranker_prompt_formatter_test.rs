use rustc_hash::FxHashMap;
use inference_engine::reranker_prompt_formatter::{
    format_batch_with_user_history, write_movielens_genres, write_movielens_title,
    write_query_time, write_relative_delta, MovieCatalog, PassagesYaml, QueryYaml,
    RerankerPromptFormatter,
};
use inference_engine::user_history::UserHistory;

fn make_test_formatter(top_k: usize) -> RerankerPromptFormatter {
    let mut catalog_map = FxHashMap::default();
    catalog_map.insert(1, ("Matrix, The (1999)".to_string(), "Action|Sci-Fi".to_string()));
    catalog_map.insert(2, ("Toy Story (1995)".to_string(), "Animation|Children".to_string()));
    catalog_map.insert(3, ("Story, A (2001)".to_string(), "Drama".to_string()));
    catalog_map.insert(4, ("Godfather, The (1972)".to_string(), "Crime|Drama".to_string()));

    let mut tiers = FxHashMap::default();
    tiers.insert(1, 0);
    tiers.insert(2, 0);
    tiers.insert(3, 1);
    tiers.insert(4, 2);

    let query_yaml = QueryYaml {
        system_template: "Rank candidates {{ first_label }} to {{ last_label }}.".to_string(),
        template: "{{ system_prompt }}\nTime: {{ query_time }} {{ user_emb }}\nHistory:\n{{ history_block }}\nCandidates:\n{{ candidates_block }}".to_string(),
        completion_template: "{{ ranked_labels | join(\",\") }}".to_string(),
    };
    let passages_yaml = PassagesYaml {
        history_template: "- {{ title }} [{{ genres }}] rating={{ rating }} ({{ recency }})".to_string(),
        candidate_template: "{{ label }}. {{ cand_emb }} {{ title }} [{{ genres }}]".to_string(),
    };

    RerankerPromptFormatter::from_loaded(
        std::sync::Arc::new(MovieCatalog::from_map(catalog_map)),
        tiers,
        query_yaml,
        passages_yaml,
        "<|vision_pad|>",
        4,
        top_k,
    )
        .expect("Failed to construct test formatter")
}

#[test]
fn test_string_and_time_formatters() {
    let mut out = String::new();
    write_movielens_title("Matrix, The (1999)", &mut out);
    assert_eq!(out, "The Matrix (1999)");

    out.clear();
    write_movielens_genres("Action|Sci-Fi|Thriller", &mut out);
    assert_eq!(out, "Action, Sci-Fi, Thriller");

    // 2000-10-14 23:00:00 UTC -> Fall, Sat Night (Oct)
    out.clear();
    write_query_time(971_564_400, &mut out);
    assert_eq!(out, "Fall, Sat Night (Oct)");

    out.clear();
    write_relative_delta(971_564_400, 971_560_800, &mut out);
    assert_eq!(out, "1h ago");
}

#[test]
fn test_format_batch_with_rust_user_history_and_training_mode() {
    let fmt = make_test_formatter(3);

    // Dense UserHistory with 2 catalog users (row 0 = sentinel, row 1 = user 1, row 2 = user 2)
    let pad = -1;
    let ts_pad = 2_524_608_000i64;
    let max_hist = 4usize;
    let uh = UserHistory {
        // user 0: padded; user 1: watched movie 1 at t=971_560_800; user 2: empty history
        movie_ids: vec![
            pad, pad, pad, pad,
            1,   pad, pad, pad,
            pad, pad, pad, pad,
        ],
        ratings: vec![
            pad, pad, pad, pad,
            5,   pad, pad, pad,
            pad, pad, pad, pad,
        ],
        timestamps: vec![
            ts_pad, ts_pad, ts_pad, ts_pad,
            971_560_800, ts_pad, ts_pad, ts_pad,
            ts_pad, ts_pad, ts_pad, ts_pad,
        ],
        n_users: 2,
        max_history: max_hist,
        pad_value: pad,
        ts_pad_value: ts_pad,
    };

    let user_ids = [1, 2];
    let movie_ids = [4, 2];
    let timestamps = [971_564_400, 971_564_400];
    // 4 candidates (> top_k=3), positive movie_id=4 at slot 3
    let candidate_ids = [1, 2, 3, 4, 1, 2, 3, 4];
    let candidate_scores = [0.9, 0.8, 0.7, 0.1, 0.5, 0.4, 0.3, 0.2];
    let user_embs = [1.0, 2.0, 9.0, 9.0];
    let cand_embs = [
        10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    // 1. Inference bridge with `UserHistory`
    let inf_out = format_batch_with_user_history(
        &fmt,
        &uh,
        &user_ids,
        &movie_ids,
        &timestamps,
        &candidate_ids,
        &candidate_scores,
        &user_embs,
        &cand_embs,
        false,
    );
    assert_eq!(inf_out.kept_indices, vec![0]); // user 2 skipped (empty history)
    assert!(inf_out.prompts[0].contains("The Matrix (1999) [Action, Sci-Fi] rating=5 (1h ago)"));
    assert!(inf_out.prompts[0].contains("Fall, Sat Night (Oct)"));
    assert!(inf_out.prompts[0].ends_with('\n'));
    assert!(inf_out.completions[0].is_none());
    assert_eq!(inf_out.candidate_ids, vec![1, 2, 3]);
    assert_eq!(inf_out.pos_tiers, vec![2]);
    assert_eq!(
        inf_out.ranker_embs,
        vec![1.0, 2.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]
    );

    // 2. Training path (same padded slices, training=true -> positive swap + shuffle + completion)
    let (h_mids, h_rats, h_ts) = uh.get_all_history_before_timestamp(&user_ids, &timestamps, 4, true);
    let tr_out = fmt.format_batch_from_padded_i32(
        &user_ids,
        &movie_ids,
        &timestamps,
        &h_mids,
        &h_rats,
        &h_ts,
        4,
        pad,
        &candidate_ids,
        &candidate_scores,
        &user_embs,
        &cand_embs,
        &[0, 1],
        42,
        true,
        true,
    );
    assert_eq!(tr_out.kept_indices, vec![0]);
    assert!(tr_out.candidate_ids.contains(&4));
    assert_eq!(tr_out.label_rels.iter().sum::<i32>(), 1);
    assert!(tr_out.completions[0].is_some());
}