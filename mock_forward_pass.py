import numpy as np
import torch

from model import UnO

def log_model_summary(model: torch.nn.Module, max_depth: int = 4, root_name="model"):
    counts_by_depth: dict[int, dict[str, int]] = {depth: {} for depth in range(max_depth)}
    for name, param in model.named_parameters():
        tokens = name.split(".")
        param_count = param.numel()
        for depth in range(max_depth):
            counts = counts_by_depth[depth]
            prefix = ".".join([root_name] + tokens[:depth])
            if prefix not in counts:
                counts[prefix] = 0
            counts[prefix] += param_count

    total = counts_by_depth[0][root_name]
    assert total == 16400315, "This is the number of params from the official codebase. Remove this line if you are trying to make changes"
    print("Model summary:")
    for depth in range(max_depth):
        print(f"Depth {depth}")
        for name, count in counts_by_depth[depth].items():
            print(f"{name} {count / total * 100:.2f}% {count}")


config = {
            "pc_range": [-100, -100, -3, 100, 100, 3],
            "voxel_size": 0.15,
            "n_input": 6,
            "n_output": 6,
            "ray_step": 0.1,
            "n_ray_points": 15,
            "n_query_points": 900_000,
            "scale": 1,
}
kwargs = {
'x_low' : -100,
'y_low' : -100,
'x_high' : 100,
'y_high' : 100,
'grid_width' : 0.15,
'grid_length' : 0.15,
}

[past_xyz_points, past_t_index, occupied_points, unoccupied_points] = np.load('data/past_xyz_points.npy'), np.load('data/past_t_index.npy'), np.load('data/occupied_points.npy'), np.load('data/unoccupied_points.npy')

print('Past xyz points shape',past_xyz_points.shape)
print('Past t index shape',past_t_index.shape)
print('Occupied points shape',occupied_points.shape)
print('Unoccupied points shape',unoccupied_points.shape)
print()

### Have to run everything on GPU because MSDA is not implemented on CPU
network = UnO().cuda()

log_model_summary(network)

# Forward pass
batch_size = 1
lidar_sweeps = [[torch.rand((100_000, 5)).cuda() for _ in range(6)] for _ in range(batch_size)] # outer list over batch, inner list over timesteps. Tensor is shape (num_points, (x, y, z, intensity, t))
query_points = torch.rand((batch_size, 10_000, 4)).cuda() # (batch_size, num_query_points, 4). Last dim is (x, y, z, t)
occupied_output = network(lidar_sweeps,  query_points)

