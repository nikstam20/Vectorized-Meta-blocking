import torch
import time

def block_purging(X_gpu, smoothing_factor=1.025, device='cuda:0'):
    """
    Function to perform block purging on the input sparse tensor.
    
    Parameters:
    - X_gpu: Input sparse tensor (in COO format) on which block purging is to be performed.
    - smoothing_factor: The factor used for adjusting the result vector.
    - device: The device ('cuda' or 'cpu') where the tensor resides.

    Returns:
    - X_gpu: The purged sparse tensor after removing rows with negative values in the result vector.
    - block_purging_time: Time taken for the block purging process.
    """
    
    block_purging_start = time.time()

    # Sum of rows
    row_sums = torch.sparse.sum(X_gpu, dim=1)
    dense_row_sums = row_sums.to_dense()

    # Sort the rows based on the sum
    sorted_rows, sorted_rows_indices = torch.sort(dense_row_sums, descending=True)
    index_dict = {element.item(): index for index, element in enumerate(sorted_rows_indices)}

    # Coalesce the sparse tensor
    X_gpu = X_gpu.coalesce()

    # Convert the index dictionary to a tensor for GPU-based mapping
    index_dict_tensor = torch.tensor(list(index_dict.values()), device=device)

    # Mapping function for sorting indices
    def mapping_function(x):
        return torch.gather(index_dict_tensor, 0, x)

    # Vectorized mapping function to apply to indices
    vectorized_mapping_function = torch.vmap(mapping_function)
    mapped_indices = vectorized_mapping_function(X_gpu.indices()[0])

    # Update indices with mapped ones
    X_gpu.indices()[0] = mapped_indices

    # Create the new sparse tensor with updated indices
    X_gpu = torch.sparse_coo_tensor(X_gpu.indices(), X_gpu.values(), X_gpu.size(), device=device)
    X_gpu = X_gpu.coalesce()

    # Block-purging
    indices = X_gpu._indices()
    values = X_gpu._values()
    size = X_gpu.size()

    # Sum over each row
    s = torch.sparse.sum(X_gpu, dim=1).to_dense()

    # Compute s * (s - 1)
    comps = s * (s - 1)

    # Compute cumulative sum for both s and comps (reversed)
    s_reversed = torch.flip(s, dims=[0])
    s_cumsum_reversed = torch.cumsum(s_reversed, dim=0)
    s = torch.flip(s_cumsum_reversed, dims=[0])

    s_up = torch.roll(s, shifts=-1)
    s_up[-1] = 0  # Set last element to zero to avoid shifting issues

    comps_reversed = torch.flip(comps, dims=[0])
    comps_cumsum_reversed = torch.cumsum(comps_reversed, dim=0)
    comps = torch.flip(comps_cumsum_reversed, dims=[0])

    comps_up = torch.roll(comps, shifts=-1)
    comps_up[-1] = 0  # Set last element to zero to avoid shifting issues

    # Calculate the result vector
    result_vector = s_up * comps - smoothing_factor * s * comps_up

    # Find negative indices in the result vector
    negative_indices = (result_vector < 0).nonzero(as_tuple=True)[0]

    if len(negative_indices) > 0:
        first_negative_position = negative_indices[0].item()
        first_negative_value = result_vector[first_negative_position].item()
    else:
        first_negative_position = -1  # No negative values case

    # Keep rows whose indices are greater than or equal to the first negative position
    if first_negative_position > -1:
        rows_to_keep = indices[0] > first_negative_position - 1
        new_indices = indices[:, rows_to_keep]
        new_values = values[rows_to_keep]

        # Create a new sparse tensor with the remaining rows
        X_gpu = torch.sparse_coo_tensor(new_indices, new_values, size=size)

    # End the timer
    block_purging_end = time.time()
    block_purging_time = block_purging_end - block_purging_start
    print(f"Block Purging finished in: {block_purging_time} seconds")
    
    del row_sums, dense_row_sums, sorted_rows, sorted_rows_indices, index_dict, index_dict_tensor, indices, values, rows_to_keep, new_indices, new_values

    return X_gpu, block_purging_time

# Example usage:
# X_gpu, purging_time = block_purging(X_gpu, smoothing_factor=1.025, device='cuda:0')
