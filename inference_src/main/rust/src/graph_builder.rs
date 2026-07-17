use std::cmp::min;
use crate::embeddings_util::{read_movie_embeddings, read_user_embeddings};
use crate::util;

fn build_graph_arrays(
    user_id: i32,
    n_real_history: usize,
    candidate_ids: &[i32],
    history_ratings: &[i32],
    history_movie_ids: &[i32],
    labels: &[i32],
    user_embedding: &[f32],
    movie_embeddings_catalog : &[f32],
    num_catalog_users : usize,
    embed_len : usize,
) -> (
    Vec<i32>,  // senders
    Vec<i32>,  // receivers
    Vec<i32>,  // edge_features
    Vec<i32>,  // node_ids
    Vec<i32>,  // node_labels
    Vec<i32>,  // node_types
    Vec<f32>,  //node_embeddings
    Vec<bool>, // candidate_mask
    usize,     // total_nodes
    usize,     // total_edges
) {
    let num_candidates = candidate_ids.len();
    let total_nodes = 1 + n_real_history + num_candidates;
    let total_edges = n_real_history + num_candidates;

    //print!("n_real_history: {}, total_nodes : {}, total_edges : {}", n_real_history, total_nodes, total_edges);

    // Senders, Receivers, Edge Features
    // vec![val; len] is highly optimized in Rust and avoids uninitialized memory risks.
    let mut senders = vec![0; total_edges];
    let mut receivers = vec![0; total_edges];
    let mut edge_features: Vec<i32> = vec![0; total_edges];

    // History -> User (Inward)
    for i in 0..n_real_history {
        senders[i] = (i + 1) as i32;
    }
    // Note: receivers[0..n_real_history] is already correctly initialized to 0
    edge_features[0..n_real_history].copy_from_slice(&history_ratings[0..n_real_history]);

    // User -> Candidates (Outward)
    for i in 0..num_candidates {
        receivers[n_real_history + i] = (1 + n_real_history + i) as i32;
    }
    // Note: senders[n_real_history..] is already 0, and edge_features[n_real_history..] is already 0.0

    // Nodes (User + History + Candidates)
    let mut node_ids: Vec<i32> = vec![0; total_nodes];
    node_ids[0] = user_id;
    node_ids[1..1 + n_real_history].copy_from_slice(&history_movie_ids[0..n_real_history]);
    node_ids[1 + n_real_history..].copy_from_slice(candidate_ids);

    let mut node_embeddings: Vec<f32> = vec![0.0; total_nodes * embed_len];
    // copy in the user embedding
    node_embeddings[0..embed_len].copy_from_slice(&user_embedding[0..embed_len]);
    let mut emb_off = embed_len;
    //copy in the movie embeddings for user history
    for i in 0..n_real_history {
        let movie_id = &history_movie_ids[i];
        let m_idx = (movie_id - 1) as usize - num_catalog_users;
        let i00 = m_idx * embed_len;
        node_embeddings[emb_off..(emb_off + embed_len)].copy_from_slice(&movie_embeddings_catalog[i00..(i00 + embed_len)]);
        emb_off += embed_len;
    }

    // copy in the movie embeddings for the candidates
    for i in 0..candidate_ids.len() {
        let movie_id = &candidate_ids[i];
        let m_idx = (movie_id - 1) as usize - num_catalog_users;
        let i00 = m_idx * embed_len;
        node_embeddings[emb_off..(emb_off + embed_len)].copy_from_slice(&movie_embeddings_catalog[i00..(i00 + embed_len)]);
        emb_off += embed_len;
    }

    // Labels
    let mut node_labels: Vec<i32> = vec![0; total_nodes];
    // node_labels[0..1+n_real_history] is already 0.0
    node_labels[1 + n_real_history..].copy_from_slice(labels);

    // Types & Masks
    let mut node_types = vec![0; total_nodes]; // 0 is User type
    node_types[1..1 + n_real_history].fill(1); // 1 is History type
    node_types[1 + n_real_history..].fill(2);  // 2 is Candidate type

    let mut candidate_mask = vec![false; total_nodes];
    candidate_mask[1 + n_real_history..].fill(true);

    (
        senders,
        receivers,
        edge_features,
        node_ids,
        node_labels,
        node_types,
        node_embeddings,
        candidate_mask,
        total_nodes,
        total_edges,
    )
}

#[derive(Debug)]
// A struct to hold your final GraphTuple equivalent
pub struct JraphGraph {
    pub n_node: Vec<i32>,
    pub n_edge: Vec<i32>,
    pub senders: Vec<i32>,
    pub receivers: Vec<i32>,
    pub edge_features: Vec<i32>,
    pub node_ids: Vec<i32>,
    pub node_labels: Vec<i32>,
    pub node_types: Vec<i32>,
    pub node_embeddings : Vec<f32>,
    pub candidate_mask: Vec<bool>,
}

#[allow(unused_variables)]
pub fn build_padded_super_graph(
    user_ids: &[i32],
    history_movie_ids: &[i32],
    history_ratings: &[i32],
    history_lengths: &[usize],
    candidate_ids: &[i32],
    labels: &[i32],
    num_candidates: usize,
    max_history: usize,

    num_catalog_users : usize,
    num_catalog_movies : usize,
    embed_len : usize,
    movie_embeddings_catalog : &[f32],
    user_embeddings : &[f32],

    n_local_devices: usize,
) -> JraphGraph {
    let batch_size = user_ids.len();

    let (max_nodes, max_edges, max_graphs) = util::calc_number_jax_graph_components(
        batch_size, max_history, num_candidates, n_local_devices,
    );

    // PRE-ALLOCATE THE SUPER GRAPH (Replaces pad_features concatenations)
    // By filling with defaults (0, 0.0, false), the padding data is implicitly created.
    let mut n_node_padded: Vec<i32> = vec![0; max_graphs];
    let mut n_edge_padded: Vec<i32> = vec![0; max_graphs];

    let mut senders_padded = vec![0; max_edges];
    let mut receivers_padded = vec![0; max_edges];
    let mut edge_features_padded: Vec<i32> = vec![0; max_edges];

    let mut node_ids_padded = vec![0; max_nodes];
    let mut node_labels_padded = vec![0; max_nodes];
    let mut node_types_padded = vec![0; max_nodes];
    let mut candidate_mask_padded = vec![false; max_nodes];

    let mut node_embeddings_padded = vec![0.0; max_nodes * embed_len];

    // Tracking offsets for where to write the next graph's data
    let mut current_node_offset = 0;
    let mut current_edge_offset = 0;

    for i in 0..batch_size {
        let c_offset = i * num_candidates;
        let h_offset = i * max_history;

        // Cast to i64/f64 if build_graph_arrays expects them
        let c_ids: Vec<i32> = candidate_ids[c_offset..c_offset + num_candidates].iter().map(|&x| x).collect();
        let lbls: Vec<i32> = labels[c_offset..c_offset + num_candidates].iter().map(|&x| x).collect();

        let h_rats: Vec<i32> = history_ratings[h_offset..h_offset + max_history].iter().map(|&x| x).collect();
        let h_m_ids: Vec<i32> = history_movie_ids[h_offset..h_offset + max_history].iter().map(|&x| x).collect();

        let u_id: i32 = user_ids[i];
        let n_hist: usize = history_lengths[i];

        let u_emb = &user_embeddings[i*embed_len..(i+1)*embed_len];

        // Generate the subgraph
        let (senders, receivers, edge_features, node_ids, node_labels,
            node_types, node_embeddings, candidate_mask, total_nodes, total_edges) =
            build_graph_arrays(u_id, n_hist, &c_ids, &h_rats, &h_m_ids, &lbls,
                &u_emb,&movie_embeddings_catalog, num_catalog_users, embed_len);

        //WRITE METADATA
        n_node_padded[i] = total_nodes as i32;
        n_edge_padded[i] = total_edges as i32;

        let n_end = current_node_offset + total_nodes;
        let e_end = current_edge_offset + total_edges;

        // copy into super graph
        node_ids_padded[current_node_offset..n_end].copy_from_slice(&node_ids);
        node_labels_padded[current_node_offset..n_end].copy_from_slice(&node_labels);
        node_types_padded[current_node_offset..n_end].copy_from_slice(&node_types);
        candidate_mask_padded[current_node_offset..n_end].copy_from_slice(&candidate_mask);
        edge_features_padded[current_edge_offset..e_end].copy_from_slice(&edge_features);

        let i0 = current_node_offset * embed_len;
        let i1 = i0 + (total_nodes * embed_len);
        node_embeddings_padded[i0 .. i1].copy_from_slice(&node_embeddings);

        // ... continue with senders/receivers vector offset math ...

        // APPLY VECTOR OFFSET TO SENDERS/RECEIVERS
        // Replaces the Python: np.concatenate([g.senders for g in graphs]) + repeated_offsets
        for j in 0..total_edges {
            senders_padded[current_edge_offset + j] = senders[j] + (current_node_offset as i32);
            receivers_padded[current_edge_offset + j] = receivers[j] + (current_node_offset as i32);
        }

        current_node_offset += total_nodes;
        current_edge_offset += total_edges;
    }

    //  VALIDATE PADDING BOUNDARIES
    if max_nodes < current_node_offset || max_edges < current_edge_offset || max_graphs < batch_size {
        panic!("Graph too large for padding. max_nodes: {}, used: {}", max_nodes, current_node_offset);
    }

    let pad_n_node = max_nodes - current_node_offset;
    let pad_n_edge = max_edges - current_edge_offset;

    // 6. APPLY PADDING METADATA TO THE "DUMMY" GRAPH (Index = batch_size)
    // The empty graphs after the dummy graph remain 0 because of our initial vec![0; max]
    n_node_padded[batch_size] = pad_n_node as i32;
    n_edge_padded[batch_size] = pad_n_edge as i32;

    // 7. POINT PADDING EDGES TO THE FIRST PADDING NODE
    // Replaces Python: np.full(pad_n_edge, total_nodes, dtype=np.int32)
    // This ensures padded edges don't point to real node 0 and corrupt data.
    let padding_node_index = current_node_offset as i32;
    for j in current_edge_offset..max_edges {
        senders_padded[j] = padding_node_index;
        receivers_padded[j] = padding_node_index;
    }

    JraphGraph {
        n_node: n_node_padded,
        n_edge: n_edge_padded,
        senders: senders_padded,
        receivers: receivers_padded,
        edge_features: edge_features_padded,
        node_ids: node_ids_padded,
        node_labels: node_labels_padded,
        node_types: node_types_padded,
        node_embeddings: node_embeddings_padded,
        candidate_mask: candidate_mask_padded,
    }
}


///
///
/// # Arguments
///
/// * `batch_size`:
/// * `max_history`:
/// * `num_candidates`:
/// * `user_id_range`: a tuple of (start_user_id, end_user_id) where the range should be as large as
//         batch_size
/// * `movie_id_range`: a tuple of (start_movie_id, end_movie_id) where the range should be as large as
//         batch_size + max_history + 1 + num_candidates
///
/// returns: ()
///
/// # Examples
///
/// ```
///
/// ```
pub fn create_fake_padded_super_batch(batch_size: usize,
    max_history: usize,
    num_candidates: usize,
    user_id_range: (usize, usize),
    movie_id_range: (usize, usize),
    n_local_devices: usize,
    user_embeddings_uri : &String,
    movie_embeddings_uri : &String
    ) -> JraphGraph {
    if (movie_id_range.1 - movie_id_range.0 + 1) < (num_candidates + max_history + 1) {
        panic!("the range of movie_id_range must be >= (num_history + num_candidates + 1)")
    }

    let (user_embeddings_catalog, num_users, embed_len) = read_user_embeddings(&user_embeddings_uri);
    let (movie_embeddings_catalog, num_movies, _) = read_movie_embeddings(&movie_embeddings_uri);


    let mut user_ids: Vec<i32> = vec![0; batch_size];
    let mut movie_ids: Vec<i32> = vec![0; batch_size];
    let mut ratings: Vec<i32> = vec![0; batch_size];
    let mut history_lengths: Vec<usize> = vec![0; batch_size];
    let mut history_movie_ids: Vec<i32> = vec![-1; batch_size * max_history];
    let mut history_ratings: Vec<i32> = vec![-1; batch_size * max_history];
    let mut candidate_ids: Vec<i32> = vec![0; batch_size * num_candidates];
    let mut labels: Vec<i32> = vec![0; batch_size * num_candidates];

    let mut user_embeddings = vec![0.0; batch_size * embed_len];

    for i in 0..batch_size {
        user_ids[i] = (user_id_range.0 + i) as i32;
        movie_ids[i] = (movie_id_range.0 + i) as i32;
        ratings[i] = 4 + ((i as i32) % 2);
        history_lengths[i] = min(i + 1, max_history);
        for j in 0..history_lengths[i] {
            let idx = i * max_history + j;
            history_movie_ids[idx] = (movie_id_range.0 + i + 1 + j) as i32;
            history_ratings[idx] = (3 + i % 2 + j % 2) as i32;
        }
        for j in 0..num_candidates {
            let idx = i * num_candidates + j;
            candidate_ids[idx] = (movie_id_range.0 + max_history + j) as i32
        }
        // choose the positive candidate to be at index 0
        candidate_ids[i * num_candidates] = movie_ids[i];
        labels[i * num_candidates] = 1;

        let idx = (user_ids[i] -1) as usize;
        let idx_offset = idx * embed_len;
        let out_offset = i * embed_len;
        user_embeddings[out_offset..(out_offset + embed_len)].copy_from_slice(
            &user_embeddings_catalog[idx_offset..(idx_offset + embed_len)]);
    }

    let padded_super_graph = build_padded_super_graph(
            &user_ids,
            &history_movie_ids,
            &history_ratings,
            &history_lengths,
            &candidate_ids,
            &labels,
            num_candidates,
            max_history,
            num_users,
            num_movies,
            embed_len,
            &movie_embeddings_catalog,
            &user_embeddings,
            n_local_devices);

        padded_super_graph
    }
    pub fn build_enriched_padded_supergraph(
        user_ids: &[i32],
        timestamps: &[i64],
        candidate_ids: &[i32],
        labels: &[i32],
        user_history: &crate::user_history::UserHistory,
        max_history: usize,
        num_users : usize,
        num_movies : usize,
        embed_len: usize,
        movie_embeddings_catalog : &[f32],
        user_embeddings : &[f32],
        n_local_devices: usize) -> JraphGraph {

        let batch_size = user_ids.len();
        let num_candidates = candidate_ids.len() / batch_size;

        // get max_history lengths of most recent histories of user_ids, but only if before timestamp.
        // empty elements are represented by user_history.pad_value
        let (history_movie_ids, history_ratings) = user_history.get_history_before_timestamp(
            user_ids, timestamps, max_history,
        );

        let history_lengths = util::get_non_padded_lengths_of_flattened_arrays(
            batch_size, max_history, &history_movie_ids, user_history.pad_value);

        let padded_super_graph = build_padded_super_graph(
            &user_ids,
            &history_movie_ids,
            &history_ratings,
            &history_lengths,
            &candidate_ids,
            &labels,
            num_candidates,
            max_history,
            num_users,
            num_movies,
            embed_len,
            &movie_embeddings_catalog,
            &user_embeddings,
            n_local_devices);

        padded_super_graph
    }