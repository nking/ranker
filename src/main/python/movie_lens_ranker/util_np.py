import numpy as np

def shuffle_and_slice(arr:np.ndarray, pad_value:int=-1, max_take=None):
    """
    Randomly shuffles valid elements to the front of each row.
    runtime complexity is O(n1*n2*log(n2)) where n1 = arr.shape[0] and n2 = arr.shape[1]
    :param arr: 2D array
    :param pad_value: values represeting "empty"
    :param max_take: the maximum number elements to take from each row.
    :return: a matrix of shape(n1, max_take) that might have pad_value at largest indicies in the
    rows if not enough non-pad values were available.
    """
    
    # Create a random weight matrix
    rand_weights = np.random.rand(*arr.shape)
    
    # Force -1s to the end by giving them an artificially high weight
    rand_weights[arr == pad_value] = 2.0
    
    # Sort indices along the rows
    sort_idx = np.argsort(rand_weights, axis=1)
    
    # Gather the elements in their new random order
    shuffled = np.take_along_axis(arr, sort_idx, axis=1)
    
    # Slice off the maximum requested
    if max_take is not None:
        return shuffled[:, :max_take]
    
    return shuffled

def push_invalid_right(arr:np.ndarray, pad_value:int=-1):
    """
    Pushes all pad_values to the far right side of the array
    :param arr: 1 2D array
    :param pad_value: the value to shoft to ends of rows if found.
    :return: matrix in which the elements in each row have been shifted to the end of the
     array such that the pad_values are at highest indices. The order of non-pad_values is maintained.
    """
    valid_mask = (arr != -1)
    
    # ~valid_mask makes valid items 0 (False) and invalid items 1 (True).
    # argsort puts 0s before 1s. kind='stable' preserves the order of the valid items.
    sort_idx = np.argsort(~valid_mask, axis=1, kind='stable')
    return np.take_along_axis(arr, sort_idx, axis=1)