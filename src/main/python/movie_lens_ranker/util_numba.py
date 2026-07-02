from typing import Tuple

import numpy as np
import numba

@numba.njit(parallel=True, cache=True)
def row_wise_intersect(arr1:np.ndarray, arr2:np.ndarray, pad_value:int) -> np.ndarray:
    """
    perform row-wise intersection of arr1 with arr2.
    The runtime complexity of this function is O(num_rows * num_cols_arr1 * num_cols_arr2).
    :param arr1: rows of values in which pad_value are at highest index if present at all
    :param arr2: rows of values in which pad_value are at highest index if present at all
    :param pad_value: value representing empty value
    :return: the resulting intersection of rows of arr1 with rows of arr2, with order in arr1 preserved
    """
    
    n_rows, n1_cols = arr1.shape
    n2_rows = arr2.shape[1]
    
    if n_rows != arr2.shape[0]:
        raise ValueError('arr1 and arr2 must have same first dimension lengths (shape[0]')
    
    # Pre-allocate output filled with -1 padding
    result = np.full((n_rows, n1_cols), pad_value, dtype=arr1.dtype)
    
    # parallel=True will distribute these rows across all CPU cores
    for i in numba.prange(n_rows):
        write_idx = 0
        for j in range(n1_cols):
            val1 = arr1[i, j]
            if val1 == pad_value:
                break  # Assuming padded -1s are at the end
            
            # Linear scan check against arr2 row
            for k in range(n2_rows):
                val2 = arr2[i, k]
                if val2 == pad_value:
                    break
                if val1 == val2:
                    result[i, write_idx] = val1
                    write_idx += 1
                    break
    
    return result

@numba.njit(parallel=True, cache=True)
def row_wise_sortedset_subtract(arr1:np.ndarray, arr2:np.ndarray, pad_value:int) -> np.ndarray:
    """
    perform row-wise subtraction of arr1 with arr2.
        The runtime complexity of this function is O(num_rows * num_cols_arr1 * num_cols_arr2).
    :param arr1:
    :param arr2:
    :param pad_value:
    :return:
    """
    n_rows, n1_cols = arr1.shape
    n2_cols = arr2.shape[1]
    
    if n_rows != arr2.shape[0]:
        raise ValueError('arr1 and arr2 must have same first dimension lengths (shape[0]')
    
    result = np.full((n_rows, n1_cols), -1, dtype=arr1.dtype)
    
    for i in numba.prange(n_rows):
        write_idx = 0
        for j in range(n1_cols):
            val1 = arr1[i, j]
            if val1 == pad_value:
                break
            
            # Check if val1 exists in arr2's row
            exists_in_arr2 = False
            for k in range(n2_cols):
                if arr2[i, k] == pad_value:
                    break
                if val1 == arr2[i, k]:
                    exists_in_arr2 = True
                    break
            
            # Subtraction logic: Keep if it does NOT exist in arr2
            if not exists_in_arr2:
                result[i, write_idx] = val1
                write_idx += 1
    
    return result

@numba.njit(parallel=True, cache=True)
def build_negative_pool(arr1:np.ndarray, arr2:np.ndarray, arr3:np.ndarray,
        arr4:np.ndarray, target1:int, target2:int, target3:int, num_negatives:int,
        pad_value:int, seed:int) -> np.ndarray:
    """
    given arrays of negatives of type1, 2, 3 and 4, combina them using target1, target2,
    target3, number of samples drawn from arr1, arr2, and arr3 respectively along with the
    remainder from arr4 to fill an array of shape (arr1.shape[0], num_negatives).
    arr1, arr2, arr3, and arr4 all have same number of rows, but can have different numbers of columns.
    For best results though, hold their shapes fixed over all invocatios to avoic recompiling the
    compute graph for the method.
    
    runtime complexity is O( n_rows * max(n_cols))
    
    :param arr1:
        note that the pad_values should be in the highest indices per row if present.
    :param arr2:
        note that the pad_values should be in the highest indices per row if present.
    :param arr3:
        note that the pad_values should be in the highest indices per row if present.
    :param arr4:
        note that the pad_values should be in the highest indices per row if present.
    :param target1: number of samples to draw from non-pad_values of each row of arr1.
    :param target2:number of samples to draw from non-pad_values of each row of arr2
    :param target3: number of samples to draw from non-pad_values of each row of arr3
    :param num_negatives: the number of columns per row in the resulting combined array.
    :param pad_value: represents an "empty" value in the arrays arr1, arr2, arr3, arr4
    :return: the combined results
    """
    
    num_users = arr1.shape[0]
    
    # Pre-allocate the final pool with -1 padding
    pool = np.full((num_users, num_negatives), pad_value, dtype=arr1.dtype)
    
    buff1 = np.empty((num_users, arr1.shape[1]), dtype=arr1.dtype)
    buff2 = np.empty((num_users, arr2.shape[1]), dtype=arr2.dtype)
    buff3 = np.empty((num_users, arr3.shape[1]), dtype=arr3.dtype)
    buff4 = np.empty((num_users, arr4.shape[1]), dtype=arr4.dtype)
    
    # Distribute rows across CPU cores
    for i in numba.prange(num_users):
        
        np.random.seed(seed + i)
        
        write_idx = 0
        
        # --- PROCESS ARRAY 1 ---
        write_idx = _extract_shuffle_and_append(arr1[i], target1, pool[i],
            write_idx, num_negatives, pad_value=pad_value, buffer_row=buff1[i])
        
        # --- PROCESS ARRAY 2 ---
        write_idx = _extract_shuffle_and_append(arr2[i], target2, pool[i],
            write_idx, num_negatives, pad_value=pad_value, buffer_row=buff2[i])
        
        # --- PROCESS ARRAY 3 ---
        write_idx = _extract_shuffle_and_append(arr3[i], target3, pool[i],
            write_idx, num_negatives, pad_value=pad_value, buffer_row=buff3[i])
        
        # --- PROCESS ARRAY 4 (Filler) ---
        # For the filler, we can take up to the remaining slots available in the pool
        remainder_needed = num_negatives - write_idx
        if remainder_needed > 0:
            _extract_shuffle_and_append(arr4[i], remainder_needed, pool[i],
                write_idx, num_negatives, pad_value=pad_value, buffer_row=buff4[i])
    
    return pool

@numba.njit
def _extract_shuffle_and_append(source_row:np.ndarray, max_take:int, pool_row:np.ndarray,
        write_idx:int, num_negatives:int, pad_value:int, buffer_row:np.ndarray):
    """
    
    :param source_row:
    :param max_take:
    :param pool_row:
    :param write_idx:
    :param num_negatives:
    :param pad_value:
    :param buffer_row: input array of length source_row.shape[0] to be used as a buffer
    :return:
    """
    
    """Helper function to process a single row section without sorting hacks."""
    
    # Gather non-padded elements into a temporary workspace
    # (Pre-allocating to row size to avoid dynamic list overhead in Numba)
    #beause pad_value are at highest indices if present at all, we can find
    # the first pad_value and copy up to that point into new buffer
    
    count = 0
    for val in source_row:
        if val == pad_value:
            break  # Stop early if we hit padding
        count += 1
    buffer_row[0:count] = source_row[0:count].copy()
    
    if count == 0:
        return write_idx
    
    # In-place Fisher-Yates Shuffle (Blazing fast O(N) shuffle)
    for j in range(count - 1, 0, -1):
        k = np.random.randint(0, j + 1)
        # Swap elements
        tmp = buffer_row[j]
        buffer_row[j] = buffer_row[k]
        buffer_row[k] = tmp
    
    available_space = num_negatives - write_idx
    items_to_take = min(max_take, count, available_space)
    
    if items_to_take > 0:
        # LLVM (Numba's compiler) converts this slice into a raw memory block move
        pool_row[write_idx: write_idx + items_to_take] = buffer_row[0: items_to_take]
        write_idx += items_to_take
    
    return write_idx

@numba.njit(parallel=True, cache=True)
def generate_type_4_negatives(all_movie_ids:np.ndarray, exclude:np.ndarray, n_negs:int, pad_value:int,
    seed:int):
    """
    create the type 4 negatives "easy negatives" = movie catalog - watch history using rejection sampling
    :param all_movie_ids: an array of the entire movie catalog as integers
    :param exclude: a 2D array of watch history to exclude from the movie_ids.  the shape is (num_users in batch, fixed_max_history).
    :param n_negs: the number of negative samples to generate per row.
    :param pad_value: represents an "empty" value in the array exclude
    :return: a 2D array of shape (exclude.shape[0], n_negs) holding a random sampling of n_negs number of negative samples per row
    after excluding the exclude contents.
    """
    n_users = exclude.shape[0]
    n_movies = len(all_movie_ids)
    
    #NOTE: it is more efficient to not use a numba set to track whether a candidate was chosen, because n_negs is small,
    # especially compared to n_movies and a set has the overhead of object creation, hashing lookups and memory bucket lookups.
    
    # Pre-allocate exactly the size we need (No surplus guesswork)
    pool = np.empty((n_users, n_negs), dtype=all_movie_ids.dtype)
    
    for i in numba.prange(n_users):
        
        np.random.seed(seed + i)
        
        count = 0
        
        # Keep drawing until we successfully find n_negs valid movies
        while count < n_negs:
            # Draw a random index
            idx = np.random.randint(0, n_movies)
            candidate = all_movie_ids[idx]
            
            # Collision Check against exclude history
            is_forbidden = False
            for j in range(exclude.shape[1]):
                val = exclude[i, j]
                if val == pad_value:
                    break  # Stop checking early if we hit padding!
                if val == candidate:
                    is_forbidden = True
                    break
            
            # Ensure uniqueness within the negatives pool itself
            if not is_forbidden:
                already_in_pool = False
                for k in range(count):
                    if pool[i, k] == candidate:
                        already_in_pool = True
                        break
                
                # If it passes all checks, add to pool
                if not already_in_pool:
                    pool[i, count] = candidate
                    count += 1
    
    return pool

@numba.njit(parallel=True, cache=True)
def simultaneous_shuffle(candidates:np.ndarray, labels:np.ndarray, seed:int):
    """
    Performs an in-place, row-wise Fisher-Yates shuffle on both arrays
    simultaneously to maintain the alignment of candidates and labels.
    Runtime complexity is O(nu_rows * num_cols).
    """
    num_users = candidates.shape[0]
    num_cols = candidates.shape[1]
    
    if labels.shape[0] != num_users or labels.shape[1] != num_cols:
        raise ValueError('candidates and labels must have same shapes')
    
    for i in numba.prange(num_users):
        
        np.random.seed(seed + i)
        
        # Fisher-Yates shuffle across the columns of the current row
        row_cand = candidates[i]
        row_label = labels[i]
        for j in range(num_cols - 1, 0, -1):
            k = np.random.randint(0, j + 1)
            
            # Swap the candidate IDs
            tmp_c = row_cand[j]
            row_cand[j] = row_cand[k]
            row_cand[k] = tmp_c
            
            # Swap the exact same indices in the labels array
            tmp_l = row_label[j]
            row_label[j] = row_label[k]
            row_label[k] = tmp_l
    
    return candidates, labels

@numba.njit
def build_graph_arrays(
        user_id: int,
        n_real_history: int,
        candidate_ids: np.ndarray,
        history_ratings: np.ndarray,
        history_movie_ids: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    
    :param user_id:
    :param n_real_history:
    :param candidate_ids: length is num_candidates
         Note that candidate_ids is guaranteed to have all read movie_ids.
         Note that candidate_ids and labels have been shuffled so that the target positive movie_id is not always
         at index 0.
    :param history_ratings: length is max_history
    :param history_movie_ids: length is max_history
    :param labels: length is num_candidates
         Note that candidate_ids and labels have been shuffled so that the target positive movie_id is not always
         at index 0.
        labels are all 0 with exception of being 1 at the index where candidate_ids has the target positive movie_id.
    :return:
    """
    
    n_candidates = len(candidate_ids)
    total_nodes = 1 + n_real_history + n_candidates
    total_edges = n_real_history + n_candidates
    
    # Pre-allocate output arrays to avoid concatenation overhead where possible
    senders = np.empty(total_edges, dtype=np.int32)
    receivers = np.empty(total_edges, dtype=np.int32)
    edge_features = np.empty(total_edges, dtype=np.int32)
    
    # History -> User (Inward)
    for i in range(n_real_history):
        senders[i] = i + 1
    receivers[:n_real_history] = 0
    edge_features[:n_real_history] = history_ratings[:n_real_history]
    
    # User -> Candidates (Outward)
    for i in range(n_candidates):
        idx = n_real_history + i
        receivers[idx] = 1 + n_real_history + i
    edge_features[n_real_history:n_real_history + n_candidates] = 0
    senders[n_real_history:n_real_history + n_candidates] = 0

    # Nodes (User + History + Candidates)
    node_ids = np.empty(total_nodes, dtype=np.int64)
    node_ids[0] = user_id
    node_ids[1:1 + n_real_history] = history_movie_ids[:n_real_history]
    node_ids[1 + n_real_history:] = candidate_ids
    
    # Labels
    node_labels = np.empty(total_nodes, dtype=np.int32)
    node_labels[0] = 0
    node_labels[1:1 + n_real_history] = 0
    node_labels[1 + n_real_history:] = labels
    
    # Types & Masks.  0=target positive, 1=real_history, 2=candidate or negative
    node_types = np.empty(total_nodes, dtype=np.int32)
    node_types[0] = 0
    node_types[1:1 + n_real_history] = 1
    node_types[1 + n_real_history:] = 2
    
    candidate_mask = np.zeros(total_nodes, dtype=np.bool_)
    candidate_mask[1 + n_real_history:] = True
    
    return (senders, receivers, edge_features, node_ids,
        node_labels, node_types, candidate_mask, total_nodes, total_edges)