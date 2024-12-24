import torch
import lightning as L
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader

from model import Network
from model_lightning import LitNetwork
from data import nuScenesDataset
from utils import custom_collate_fn

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
nusc = NuScenes(version="v1.0-trainval", dataroot="/datasets/nuscenes/", verbose=False)
train_dataset = nuScenesDataset(nusc, 'train', config)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
valid_dataset = nuScenesDataset(nusc, 'val', config)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

network = Network(kwargs)
lit_network = LitNetwork(network)

trainer = L.Trainer(default_root_dir="logs", accelerator="gpu", devices="auto", strategy="ddp")
trainer.fit(lit_network, train_loader, valid_loader)
