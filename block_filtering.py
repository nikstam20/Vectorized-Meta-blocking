import torch
import time

def block_filtering(X_gpu, device='cuda:0'):
    """
    Function to perform block filtering on the input sparse tensor.
    
    Parameters:
    - X_gpu: Input sparse tensor (in COO format) to be filtered.
    - device: The device ('cuda' or 'cpu') where the tensor resides.

    Returns:
    - step_function_result: The filtered sparse tensor after applying the step function.
    - block_filtering_time: Time taken for the block filtering process.
    """
    
    block_filtering_start = time.time()

    # Sum along columns
    sum_columns = torch.sparse.sum(X_gpu, dim=0)
    dense_sum_columns = sum_columns.to_dense()

    # Round the result of half the column sums
    rounded_result = torch.round(dense_sum_columns * 0.5)
    X_gpu = X_gpu.coalesce()

    # Function to compute cumulative counts
    def cumulative_count_vector(input_vector):
        unique_numbers, counts = torch.unique(input_vector, return_counts=True)
        cumulative_dict = {num: 0 for num in unique_numbers.tolist()}

        def get_cumulative_count(num):
            nonlocal cumulative_dict
            result = cumulative_dict[num]
            cumulative_dict[num] += 1
            return result
        
        result_vector = torch.tensor(list(map(get_cumulative_count, input_vector.tolist())))
        return result_vector

    # # Print current GPU memory (optional)
    # print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    # Coalesce the sparse tensor and compute cumulative sums over the column indices
    X_gpu = X_gpu.coalesce()
    colindxs = X_gpu.indices()[1]
    cumsums = cumulative_count_vector(colindxs).to(device)
    values = X_gpu.values()
    values += cumsums
    X_gpu._values().set_(values)
    X_gpu = X_gpu.coalesce()

    # Apply the step function based on thresholds
    indices = X_gpu.indices()
    values = X_gpu.values()
    thresholds = rounded_result[indices[1]]
    result_values = torch.where(values > thresholds, torch.tensor(1, device=device), torch.tensor(0, device=device))

    # Create the step function result tensor
    step_function_result = torch.sparse_coo_tensor(indices, result_values, size=X_gpu.size())

    # Function to remove explicit zeros from the sparse tensor
    def remove_explicit_zeros(input_sparse_tensor):
        indices = input_sparse_tensor._indices()
        values = input_sparse_tensor._values()
        non_zero_mask = values != 0
        non_zero_indices = indices[:, non_zero_mask]
        non_zero_values = values[non_zero_mask]
        non_zero_result = torch.sparse_coo_tensor(non_zero_indices, non_zero_values, size=input_sparse_tensor.size())
        return non_zero_result

    # Remove explicit zeros and coalesce the final result
    step_function_result = remove_explicit_zeros(step_function_result)
    step_function_result = step_function_result.coalesce()

    # End the timer and calculate the filtering time
    block_filtering_end = time.time()
    block_filtering_time = block_filtering_end - block_filtering_start

    print(f"Block Filtering finished in: {block_filtering_time} seconds")

    # Clear unnecessary variables
    del rounded_result, colindxs, cumsums, values, indices, thresholds, result_values, X_gpu

    return step_function_result, block_filtering_time

# Example usage:
# step_function_result, filtering_time = block_filtering(X_gpu, device='cuda:0')
