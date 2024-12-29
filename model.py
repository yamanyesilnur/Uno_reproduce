import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ops.modules.ms_deform_attn import MSDeformAttn
from torchvision.ops import SqueezeExcitation

from dynamic_conv import Dynamic_conv2d
from utils import voxelize


class PointCloudMLP(nn.Module):  
    def __init__(self, input_size=4, hidden_size=128, output_size=128) -> None:
        super(PointCloudMLP, self).__init__()
        # 2 layer hidden 128
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        

    def forward(self, x):

        x = self.relu(self.linear1(x))
        
        x = self.linear2(x)
        
        

        return x


class ConvolutionalStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalStem, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        sh = x.shape
        x = self.relu(self.bn1(self.conv1(x)))
        assert x.shape[2] == math.ceil(sh[2] / 2) and x.shape[3] == math.ceil(sh[3] / 2)

        x = self.relu(self.bn2(self.conv2(x)))

        x = self.relu(self.bn3(self.conv3(x)))

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.stride = 2 if downsample else 1

        self.d_conv1 = Dynamic_conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )
        self.d_conv2 = Dynamic_conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.d_bn1 = nn.BatchNorm2d(out_channels)
        self.d_bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.se_layer = SqueezeExcitation(out_channels, 128)
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if self.downsample:
            self.conv_downsample = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=2
            )
            self.bn_downsample = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        res_x = x
        sh = x.shape

        x = self.bn1(self.conv1(x))
        x = self.relu(self.d_bn1(self.d_conv1(x)))

        assert (x.shape[2] == math.ceil(sh[2] / self.stride)) and x.shape[3] == math.ceil(sh[3] / self.stride)

        x = self.d_bn2(self.d_conv2(x))
        x = self.se_layer(x)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        assert (x.shape[2] == math.ceil(sh[2] / self.stride)) and x.shape[3] == math.ceil(sh[3] / self.stride)

        if self.downsample:
            res_x = self.bn_downsample(self.conv_downsample(res_x))
        x = x + res_x

        return x


class MSDeformAttnWrapper(nn.Module):
    def __init__(self, d_model=128, n_levels=3, n_heads=16, n_points=4):
        super(MSDeformAttnWrapper, self).__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # Initialize the MSDeformAttn module
        self.msda = MSDeformAttn(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )

    def forward(self, feature_map1, feature_map2, feature_map3):
        B, C, H1, W1 = feature_map1.shape
        _, _, H2, W2 = feature_map2.shape
        _, _, H3, W3 = feature_map3.shape

        # 1. Flatten and concatenate feature maps
        input_flatten = torch.cat(
            [
                feature_map1.flatten(start_dim=2).transpose(1, 2).contiguous(),
                feature_map2.flatten(start_dim=2).transpose(1, 2).contiguous(),
                feature_map3.flatten(start_dim=2).transpose(1, 2).contiguous(),
            ],
            dim=1,
        )

        # 2. Construct spatial shapes
        spatial_shapes = torch.tensor(
            [
                [H1, W1],
                [H2, W2],
                [H3, W3],
            ],
            dtype=torch.long,
            device=feature_map1.device,
        )

        # 3. Compute level start index
        level_start_index = torch.tensor(
            [0, H1 * W1, H1 * W1 + H2 * W2],
            dtype=torch.long,
            device=feature_map1.device,
        )

        # 4. Generate reference points
        reference_points = []
        for H, W in [(H1, W1), (H2, W2), (H3, W3)]:
            y_ref, x_ref = torch.meshgrid(
                torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=feature_map1.device),
                torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=feature_map1.device),
                indexing="ij",
            )
            ref_points = torch.stack([x_ref, y_ref], dim=-1).reshape(-1, 2)
            reference_points.append(ref_points)
        reference_points = (
            torch.cat(reference_points, dim=0).unsqueeze(0).repeat(B, 1, 1).unsqueeze(2)
        )

        # 5. Apply MSDA
        output = self.msda(
            query=input_flatten,
            reference_points=reference_points,
            input_flatten=input_flatten,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=None,
        )

        # 6. Reshape back to per-level feature maps
        output = output.split([H1 * W1, H2 * W2, H3 * W3], dim=1)
        output = [
            o.transpose(1, 2).contiguous().reshape(B, C, H, W)
            for o, (H, W) in zip(output, [(H1, W1), (H2, W2), (H3, W3)])
        ]
        return output


class Encoder(nn.Module):  # Parameter count: 13,866,176
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.lidar_stem = ConvolutionalStem(in_channels, hidden_channels)

        self.resblock0 = ResBlock(hidden_channels, hidden_channels, downsample=True)
        self.resblock1 = ResBlock(hidden_channels, hidden_channels)

        self.resblock2 = ResBlock(hidden_channels, hidden_channels, downsample=True)
        self.resblock3 = ResBlock(hidden_channels, hidden_channels)

        self.resblock4 = ResBlock(hidden_channels, hidden_channels, downsample=True)
        self.resblock5 = ResBlock(hidden_channels, hidden_channels)

        self.resblock6 = ResBlock(hidden_channels, hidden_channels)
        self.resblock7 = ResBlock(hidden_channels, hidden_channels)

        self.resblock8 = ResBlock(hidden_channels, hidden_channels)
        self.resblock9 = ResBlock(hidden_channels, out_channels)

        self.msda = MSDeformAttnWrapper(
            d_model=out_channels, n_levels=3, n_heads=16, n_points=4
        ).cuda()

    def forward(self, x):

        intermediate_feature_maps = []
        intermediate_feature_maps_after_msda = []

        x = self.lidar_stem(x)

        x = self.resblock0(x)
        x = self.resblock1(x)
        intermediate_feature_maps.append(x)

        x = self.resblock2(x)
        x = self.resblock3(x)
        intermediate_feature_maps.append(x)

        x = self.resblock4(x)
        x = self.resblock5(x)

        x = self.resblock6(x)
        x = self.resblock7(x)

        x = self.resblock8(x)
        x = self.resblock9(x)
        intermediate_feature_maps.append(x)

        intermediate_feature_maps_after_msda = self.msda(*intermediate_feature_maps)

        return intermediate_feature_maps_after_msda


def deconv3x3(in_channels, out_channels, kernel_size, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


class FPNFusion(torch.nn.Module):  # Parameter count: 262,784
    def __init__(self, in_channels_list, out_channels):
        super(FPNFusion, self).__init__()

        self.deconv1 = deconv3x3(
            in_channels_list[0], in_channels_list[1], kernel_size=3, stride=2, padding=1
        )
        self.deconv2 = deconv3x3(
            in_channels_list[1], in_channels_list[2], kernel_size=2, stride=2, padding=0
        )
        self.conv1 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=3, padding=1)
        self.mlp1 = nn.Conv2d(in_channels_list[0], in_channels_list[1], kernel_size=1)
        self.mlp2 = nn.Conv2d(in_channels_list[1], in_channels_list[2], kernel_size=1)
        self.mlp3 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)

    def forward(self, features):

        assert len(features) == 3

        fused_features = self.deconv1(self.mlp1(features[2]))
        assert fused_features.shape == features[1].shape
        fused_features = fused_features + self.mlp2(features[1])
        fused_features = self.conv1(fused_features)
        fused_features = self.deconv2(fused_features)
        assert fused_features.shape == features[0].shape
        fused_features = fused_features + self.mlp3(features[0])
        fused_features = self.conv2(fused_features)
        assert fused_features.shape == features[0].shape

        return fused_features


class OffsetPredictor(nn.Module):
    def __init__(
        self,
        x_low=-100,
        y_low=-100,
        x_high=100,
        y_high=150,
        grid_width=0.15,
        grid_length=0.15,
    ) -> None:
        super(OffsetPredictor, self).__init__()
        self.x_low = x_low
        self.y_low = y_low
        self.x_high = x_high
        self.y_high = y_high
        self.grid_width = grid_width
        self.grid_length = grid_length

        self.query_projector = torch.nn.Linear(4, 16)
        self.feature_projector = torch.nn.Linear(128, 16)
        # Two linear layer with res connection
        self.res_layer = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16)
        )
        self.project = nn.Linear(16, 2)
        # Initialize weights from N(0, 0.01)
        nn.init.normal_(self.project.weight, mean=0.0, std=0.01)
        # Set bias to zero
        nn.init.constant_(self.project.bias, 0.0)

    def forward(self, points, Z):

        B, F_, L_4, W_4 = Z.shape

        assert (
            points.shape[0] == B
        )  # Number of batches in points should be the same as in the feature maps

        #### Bilinear interpolation of (x,y) coordinates of q to get Z_q ####
        x, y = points[..., 0], points[..., 1]  # (B, n_points) both

        # From given coordinates change to map coordinates and divide by 4
        q_x = (x - self.x_low) / (4 * self.grid_width)  # (B, n_points)
        q_y = (y - self.y_low) / (4 * self.grid_length)  # (B, n_points)

        # Normalize for torch grid_smaple
        x_norm = (2 * q_x - W_4 + 1) / (W_4 - 1)  # (B, n_points, 1)
        y_norm = (2 * q_y - L_4 + 1) / (L_4 - 1)  # (B, n_points, 1)

        # Bilinear interpolation
        sampling_grid = torch.stack((y_norm, x_norm), dim=-1)  # (B, n_points, 2)
        sampling_grid = sampling_grid.unsqueeze(2)  # (B, n_points, 2, 1)
        sampling_grid = sampling_grid.cuda()

        interpolated_features = torch.nn.functional.grid_sample(
            Z, 
            sampling_grid, 
            mode="bilinear", 
            align_corners=True
        )  # This is same as in the Conv Occ Networks

        interpolated_features = interpolated_features.squeeze(-1).permute(
            0, 2, 1
        )  # (B, n_points, F)
        ############################################
        
        ######## Project and fuse ##################

        q_16 = self.query_projector(points)  # (B, n_points, 16) -- This is called in ConvOccNetworks positional encoding
        query_feat_16 = self.feature_projector(interpolated_features)  # (B, n_points, 16)

        fused_query = q_16 + query_feat_16 # (B, n_points, 16)
        fused_query = self.res_layer(fused_query) + fused_query  # (B, n_points, 16), Two layers with res connection
        
        offset = self.project(F.relu(fused_query))  # (B, n_points, 2)

        return offset, interpolated_features # (B, n_points, 2), (B, n_points, F)

    def after_offset_interpolate(self, r, Z):
        B, F_, L_4, W_4 = Z.shape

        # Same interpolator as above
        assert (
            r.shape[0] == B
        )  # Number of batches in points should be the same as in the feature maps

        x, y = r[..., 0], r[..., 1]

        q_x = (x - self.x_low) / (4 * self.grid_width)
        q_y = (y - self.y_low) / (4 * self.grid_length)

        x_norm = (2 * q_x - W_4 + 1) / (W_4 - 1)
        y_norm = (2 * q_y - L_4 + 1) / (L_4 - 1)

        sampling_grid = torch.stack((y_norm, x_norm), dim=-1)
        sampling_grid = sampling_grid.unsqueeze(2)
        sampling_grid = sampling_grid.cuda()

        interpolated_features = torch.nn.functional.grid_sample(
            Z,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # This is same as in the Conv Occ Networks
        interpolated_features = interpolated_features.squeeze(-1).permute(0, 2, 1)

        return interpolated_features
    
class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out, size_h=16) -> None:

        super().__init__()
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.relu = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization as in Conv Occ Networks
        # nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.relu(x))
        dx = self.fc_1(self.relu(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return dx + x_s
    
class Decoder(nn.Module): # 8,563
    def __init__(
        self,
        x_low=-100,
        y_low=-100,
        x_high=100,
        y_high=150,
        grid_width=0.15,
        grid_length=0.15,
    ) -> None:
        super().__init__()
        self.x_low = x_low
        self.y_low = y_low
        self.x_high = x_high
        self.y_high = y_high
        self.grid_width = grid_width
        self.grid_length = grid_length
        self.offset_predictor = OffsetPredictor()
        self.project_q = nn.Linear(4, 16)
        self.project_Z_f = nn.ModuleList([nn.Linear(128 * 2, 16) for _ in range(3)])
        self.res_blocks = nn.ModuleList([ResnetBlockFC(16, 16) for _ in range(3)])
        self.relu = nn.ReLU()
        self.predict_occupancy = nn.Linear(16, 1)

    def forward(self, q, Z):
        """
        args:
            q : query points (B, n_points, 4), 4 -> (x, y, z, t)
            Z : feature map output of encoder (B, F, L / 4, W / 4), L original feature map length, W original feature map width
        """
        offset, q_interpolated_features = self.offset_predictor(q, Z)
        r = q[..., :2] + offset
        r_interpolated_features = self.offset_predictor.after_offset_interpolate(r, Z)
        
        Z_f = torch.cat([q_interpolated_features, r_interpolated_features], dim=-1)
        
        q_16 = self.project_q(q)
        # Z_f_16 = self.project_Z_f(Z_f)
        
        fused_query = q_16 
        
        for i, res_block in enumerate(self.res_blocks): # 3 Resblocks, input of each block is the output of the previous block + Z_f_16
            
            fused_query = res_block(fused_query + self.project_Z_f[i](Z_f))
        
        occupancy = self.predict_occupancy(self.relu(fused_query))

        return occupancy        
        
        
class Network(nn.Module):
    def __init__(self, kwargs):
        super(Network, self).__init__()
        self.pc_mlp = PointCloudMLP()
        self.encoder = Encoder(128, 128, 128)
        self.fpn = FPNFusion([128, 128, 128], 128)
        self.x_low = kwargs['x_low']
        self.y_low = kwargs['y_low']
        self.x_high = kwargs['x_high']
        self.y_high = kwargs['y_high']
        self.grid_width = kwargs['grid_width']
        self.grid_length = kwargs['grid_length']
        self.decoder = Decoder(self.x_low, self.y_low, self.x_high, self.y_high, self.grid_width, self.grid_length)
        
    def forward(self, past_points_list, past_times_list, query_points):
        feature_maps = []
        assert len(past_points_list) == len(past_times_list)
        for past_xyz_points, past_t_index in zip(past_points_list, past_times_list):
            feature_maps.append(voxelize(past_xyz_points, self.pc_mlp ,past_t_index, self.x_low, self.y_low, self.x_high, self.y_high, self.grid_width, self.grid_length))
        feature_maps = self.encoder(torch.stack(feature_maps, dim=0))
        fused_feature_map = self.fpn(feature_maps)  
        occupancy = self.decoder(query_points, fused_feature_map)
        
        return occupancy      
        
            