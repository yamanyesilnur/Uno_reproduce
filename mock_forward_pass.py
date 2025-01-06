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


network = UnO().cuda()

log_model_summary(network, max_depth=1)

# Forward pass
batch_size = 1
lidar_sweeps = [[torch.rand((100_000, 5)).cuda() for _ in range(6)] for _ in range(batch_size)] # outer list over batch, inner list over timesteps. Tensor is shape (num_points, (x, y, z, intensity, t))
query_points = torch.rand((batch_size, 10_000, 4)).cuda() # (batch_size, num_query_points, 4). Last dim is (x, y, z, t)
occupied_output = network(lidar_sweeps,  query_points)

print(f"Input query points shape: {query_points.shape}")
print(f"Output shape: {occupied_output.shape}")

