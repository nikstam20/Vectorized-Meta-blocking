import torch
import time

def edge_pruning(step_function_result, device='cuda:0', k_threshold=None):
    """
    Function to perform edge pruning on the input sparse tensor.

    Parameters:
    - step_function_result: The input sparse tensor after block filtering.
    - device: The device ('cuda' or 'cpu') where the tensor resides.
    - k_threshold: Threshold for selecting top-k values, defaults to the number of values in the tensor.

    Returns:
    - topk_sparse_matrix: The pruned sparse tensor with top-k edges.
    - edge_pruning_time: Time taken for the edge pruning process.
    """

    edge_pruning_start = time.time()

    # Transpose the input tensor
    X_transposed = step_function_result.t().to(dtype=torch.float16)
    X_transposed = X_transposed.coalesce()

    if k_threshold is None:
        k_threshold = len(step_function_result.values())

    # Sum along columns and compute the batch size
    sum_columns = torch.sparse.sum(step_function_result, dim=0)
    dense_sum_columns = sum_columns.to_dense().float()
    batch_size = round(16384 * 1.5 / torch.mean(dense_sum_columns, dim=0).item())
    
    del step_function_result  # Clear memory

    batch_indices = None
    batch_values = None
    batch_counter = 0

    for i in range(0, X_transposed.size()[0], batch_size):
        X_transposed = X_transposed.coalesce()

        row_indices = X_transposed.indices()[1]
        column_indices = X_transposed.indices()[0]
        mask = torch.logical_and(i <= column_indices, column_indices < i + batch_size)

        selected_columns = column_indices[mask]
        selected_rows = row_indices[mask]
        values_batch = X_transposed.values()[mask]
        selected_indices = torch.stack((selected_rows, selected_columns))

        batch_step_function_result = torch.sparse_coo_tensor(
            indices=selected_indices,
            values=values_batch,
            size=(X_transposed.size()[1], X_transposed.size()[0]),
            device=X_transposed.device,
            dtype=torch.float16
        )

        # Perform matrix multiplication
        X_transposed = X_transposed.to(dtype=torch.half)
        batch_step_function_result = batch_step_function_result.to(dtype=torch.half)
        batch_result = torch.sparse.mm(X_transposed, batch_step_function_result).to(dtype=torch.int8)

        batch_row_indices = batch_result._indices()[0].to(dtype=torch.int32)
        batch_column_indices = batch_result._indices()[1].to(dtype=torch.int32)
        batch_values_c = batch_result._values().clone()

        del batch_result  # Clear memory

        # Mask and filter relevant columns
        batch_mask = torch.logical_and(
            torch.logical_and(i <= batch_column_indices, batch_column_indices < i + batch_size),
            batch_row_indices > batch_column_indices
        )

        batch_selected_columns = batch_column_indices[batch_mask]
        batch_selected_rows = batch_row_indices[batch_mask]
        batch_values_batch = batch_values_c[batch_mask].to(dtype=torch.int8)
        batch_selected_indices = torch.stack((batch_selected_rows, batch_selected_columns)).to(dtype=torch.int32)

        # Calculate modified values based on log term
        common_blocks = batch_values_batch
        log_term_i = torch.log10(X_transposed.size(1) / dense_sum_columns[batch_selected_indices[0]])
        log_term_j = torch.log10(X_transposed.size(1) / dense_sum_columns[batch_selected_indices[1]])
        modified_values = common_blocks * log_term_i * log_term_j

        if len(modified_values) > k_threshold:
            topk_batch_values, topk_batch_flat_indices = modified_values.topk(k=k_threshold)
            batch_indices = (batch_selected_indices[:, topk_batch_flat_indices]
                             if batch_indices is None else torch.cat((batch_indices, batch_selected_indices[:, topk_batch_flat_indices]), dim=1))
            batch_values = (topk_batch_values
                            if batch_values is None else torch.cat((batch_values, topk_batch_values), dim=0))
            batch_values, topk_indices = batch_values.topk(k=k_threshold)
            batch_indices = batch_indices[:, topk_indices]
        else:
            batch_indices = (batch_selected_indices
                             if batch_indices is None else torch.cat((batch_indices, batch_selected_indices), dim=1))
            batch_values = (modified_values
                            if batch_values is None else torch.cat((batch_values, modified_values), dim=0))
            batch_values, topk_indices = batch_values.topk(k=k_threshold)
            batch_indices = batch_indices[:, topk_indices]

        del common_blocks, log_term_i, log_term_j, modified_values, batch_selected_indices

    # Final top-k values and indices
    final_topk_values, topk_indices = batch_values.topk(k=k_threshold)
    final_topk_indices = batch_indices[:, topk_indices]

    # Create sparse tensor from top-k values and indices
    topk_sparse_matrix = torch.sparse_coo_tensor(
        indices=final_topk_indices,
        values=final_topk_values,
        size=(X_transposed.size()[0], X_transposed.size()[0]),
        device=device
    ).coalesce()

    edge_pruning_end = time.time()
    edge_pruning_time = edge_pruning_end - edge_pruning_start

    print(f"Edge Pruning finished in: {edge_pruning_time} seconds")

    return topk_sparse_matrix, edge_pruning_time

# Example usage:
# topk_sparse_matrix, pruning_time = edge_pruning(step_function_result, device='cuda:0', k_threshold=10000)
