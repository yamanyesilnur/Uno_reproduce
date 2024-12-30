import numpy as np
import torch

from model import Network

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

[past_xyz_points, past_t_index, occupied_points, unoccupied_points] = np.load('past_xyz_points.npy'), np.load('past_t_index.npy'), np.load('occupied_points.npy'), np.load('unoccupied_points.npy')

print('Past xyz points shape',past_xyz_points.shape)
print('Past t index shape',past_t_index.shape)
print('Occupied points shape',occupied_points.shape)
print('Unoccupied points shape',unoccupied_points.shape)
print()

### Have to run everything on GPU because MSDA is not implemented on CPU
network = Network(kwargs).cuda()

# Forward pass
past_xyz_points = torch.from_numpy(past_xyz_points).float().cuda()
past_t_index = torch.from_numpy(past_t_index).float().cuda()
occupied_points = torch.from_numpy(occupied_points).float().cuda()
unoccupied_points = torch.from_numpy(unoccupied_points).float().cuda()

occupied_output = network([past_xyz_points], [past_t_index], occupied_points.unsqueeze(0))
unoccuied_output = network([past_xyz_points], [past_t_index], unoccupied_points.unsqueeze(0))
print('Predictions for occupied shape',occupied_output.shape)
print('Predictions for unoccupied shape',unoccuied_output.shape)

