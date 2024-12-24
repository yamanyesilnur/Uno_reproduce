import torch

def voxelize(
    frame,
    mlp,
    t_index,
    x_low=-100,
    y_low=-100,
    x_high=100,
    y_high=100,
    grid_width=0.15,
    grid_length=0.15,
    n_features=128,
):    
    # Encode the input frame using MLP to get feature representation
    
    normalized_t_index = (t_index / t_index.max()).unsqueeze(-1)
   
    frame = torch.cat((frame, normalized_t_index), dim=1)

    encoded_frame = mlp(frame) #mlp(frame)  # (N, F)
    
    # Compute grid indices for each point in the x and y directions
    grid_ids_x = ((frame[:, 0] - x_low) / grid_width).to(torch.long)  # (N,) With Original filtered_frame coordinates because for the ray casting 
    grid_ids_y = ((frame[:, 1] - y_low) / grid_length).to(torch.long)  # (N,)

    # Filter out points that fall outside the defined grid bounds
    max_grid_x = int((x_high - x_low) / grid_width) + 1
    max_grid_y = int((y_high - y_low) / grid_length) + 1

    valid_mask = (
        (grid_ids_x >= 0) & (grid_ids_x < max_grid_x) &
        (grid_ids_y >= 0) & (grid_ids_y < max_grid_y)
    )

    grid_ids_x = grid_ids_x[valid_mask]
    grid_ids_y = grid_ids_y[valid_mask]
    encoded_frame = encoded_frame[valid_mask]  # Filtered (N_valid, F)

    # Create a flattened index tensor to map each point to a unique 2D grid cell index
    flat_grid_ids = grid_ids_y * max_grid_x + grid_ids_x  # (N_valid,)

    # Initialize the grid to accumulate features
    grid = torch.zeros((n_features, max_grid_y * max_grid_x), dtype=torch.float32, device=frame.device)

    # Scatter-add the features of points into their respective flattened grid cells
    grid = grid.scatter_add(1, flat_grid_ids.unsqueeze(0).expand(n_features, -1), encoded_frame.T)

    # Reshape to (n_features, max_grid_y, max_grid_x) to restore the 2D grid shape
    grid = grid.view(n_features, max_grid_y, max_grid_x)

    return grid

def custom_collate_fn(batch):
    
    past_xyz_points_batch, past_t_index_batch, occupied_points_batch, unoccupied_points_batch = zip(*batch)
    return (
        list(past_xyz_points_batch),
        list(past_t_index_batch),
        torch.stack(occupied_points_batch, dim=0),
        torch.stack(unoccupied_points_batch, dim=0),
    )