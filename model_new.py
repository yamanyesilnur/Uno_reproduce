import math
import torch
from torch import nn
import sys
sys.path.append("/home/bagro/Uno_reproduce/Deformable-DETR/")
from models.ops.modules.ms_deform_attn import MSDeformAttn

class LayerNormReLU(nn.Module):
    """
    A ln layer wrapped up around the fast ln2d inference kernel.
    The training use regular nn layer norm implementation
    """

    def __init__(self, input_features, relu=False):
        super().__init__()
        self.ln_layer = nn.LayerNorm(input_features)
        self.use_relu = relu
        self.relu_layer = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, transpose: bool = False):
        out = self.ln_layer(x)
        if self.use_relu:
            out = self.relu_layer(out)
        if transpose:
            # TODO: revise the simple transpose for bs > 1
            out = out.T.contiguous()
        return out

def point_cloud_roi_mask(
    pts: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float | None = None,
    z_max: float | None = None,
) -> torch.Tensor:
    """Clips the point cloud to ROI defined by x_min, x_max, y_min, y_max, z_min, z_max.
    Points (N, ...) must be have x, y, z in the first 3 elements of the second dimension, where N is the nth point.

    Args:
        pts:
        x_min: _description_
        x_max: _description_
        y_min: _description_
        y_max: _description_
        z_min: _description_. Defaults to None.
        z_max: _description_. Defaults to None.

    Returns:
        _description_

    Raises:
        AssertionError: _description_
    """
    if (z_min is None) != (z_max is None):
        raise AssertionError("if z_min is supplied z_max must be supplied, and visa-versa")
    include_z = z_min is not None
    in_roi_mask = (
        (pts[..., 0] > x_min + 1e-2)
        & (pts[..., 0] < x_max - 1e-2)
        & (pts[..., 1] > y_min + 1e-2)
        & (pts[..., 1] < y_max - 1e-2)
    )
    if include_z:
        assert z_min is not None and z_max is not None
        in_roi_mask = in_roi_mask & (pts[..., 2] > z_min + 1e-6) & (pts[..., 2] < z_max - 1e-6)
    return in_roi_mask



class Voxelizer(nn.Module):
    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        step: float,
        z_step: float,
        n_out: int,
        num_sensors: int,
        num_sweeps: int,
        single_lidar=False,
        clip_outside_voxelizer_z_roi: bool = False,
        n_feat: int | None = None,
    ):
        super().__init__()
        self.single_lidar = single_lidar  # in case we use it to voxelize a single-lidar pointclouds
        self.lidar_range = [x_min, x_max, y_min, y_max, z_min, z_max]  # z: (-1.5, 5.5)
        self.z_step = z_step
        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = self.lidar_range
        self.clip_outside_voxelizer_z_roi = clip_outside_voxelizer_z_roi

        self.voxel_size = [step, step, self.z_step]
        self.num_sensors = num_sensors
        self.n_out = n_out
        self.num_sweeps = num_sweeps

        self.num_x = round((self.x_max - self.x_min) / step)
        self.num_y = round((self.y_max - self.y_min) / step)
        self.num_z = round((self.z_max - self.z_min) / self.z_step)
        self.depth = self.num_z

        n_input = 4
        self.n_input = n_input
        n_feat = 16 if n_feat is None else n_feat

        self.block = nn.Sequential(
            nn.Linear(n_input, n_feat),
            LayerNormReLU(n_feat, relu=True),
            nn.Linear(n_feat, n_feat),
        )
        self.norm = LayerNormReLU(n_feat)
        self.pos_embedding = nn.Embedding(self.num_z, n_feat)
        self.dense_block = nn.Sequential(
            nn.Linear(2 * n_feat, n_feat), LayerNormReLU(n_feat, relu=True), nn.Linear(n_feat, n_out)
        )
        self.dense_norm = LayerNormReLU(n_out)
        self.n_feat = n_feat

    def forward(self, lidar: list[list[torch.Tensor]]) -> torch.Tensor:
        # sweep features: (x, y, z, intensity, dt)

        for i, batch_lidar in enumerate(lidar):
            for j, lidar_t in enumerate(batch_lidar):
                in_roi_mask = point_cloud_roi_mask(
                    pts=lidar_t,
                    x_min=self.x_min,
                    x_max=self.x_max,
                    y_min=self.y_min,
                    y_max=self.y_max,
                    z_min=self.z_min if self.clip_outside_voxelizer_z_roi else None,
                    z_max=self.z_max if self.clip_outside_voxelizer_z_roi else None,
                )
                lidar[i][j] = lidar_t[in_roi_mask]

        batch_size = len(lidar)
        num_sweeps = len(lidar[0])
        assert num_sweeps == self.num_sweeps, f"Expected {self.num_sweeps}, got {num_sweeps}"

        assert lidar[0][0].shape[1] == 5

        x_size, y_size, z_size = self.voxel_size

        feats_list, coords_list = [], []
        for i, sweeps in enumerate(lidar):
            for j, sweep in enumerate(sweeps):  # noqa: B007
                x = (sweep[:, 0] - self.x_min) / x_size
                y = (self.y_max - sweep[:, 1]) / y_size
                z = (sweep[:, 2] - self.z_min) / z_size

                feat = torch.zeros(
                    (len(sweep), self.n_input),
                    dtype=sweep.dtype,
                    device=sweep.device,
                )
                feat[:, 0] = x - x.floor() - 0.5
                feat[:, 1] = y - y.floor() - 0.5
                feat[:, 2] = z - torch.clamp(z.floor(), min=0, max=self.num_z - 1) - 0.5
                feat[:, 3] = sweep[:, 4] 
                feats_list.append(feat)

                coord = torch.zeros((len(sweep), 4), dtype=torch.int64, device=sweep.device)
                coord[:, 0] = i
                coord[:, 1] = x.long()
                coord[:, 2] = y.long()
                coord[:, 3] = torch.clamp(z.long(), min=0, max=self.num_z - 1)
                coords_list.append(coord)

        feats = torch.cat(feats_list, 0)
        coords = torch.cat(coords_list, 0)
        feats = self.block(feats)

        coords = (
            coords[:, 0] * (self.num_x * self.num_y * self.num_z)
            + coords[:, 1] * self.num_y * self.num_z
            + coords[:, 2] * self.num_z
            + coords[:, 3]
        )
        coords, idcs = torch.unique(coords, return_inverse=True)

        buff = torch.zeros(coords.shape[0], feats.shape[1], device=feats.device, dtype=feats.dtype)
        buff.index_add_(0, idcs, feats)
        feats = buff

        buff = torch.zeros(coords.shape[0], 4, device=coords.device, dtype=coords.dtype)
        buff[:, 0] = torch.div(coords, self.num_x * self.num_y * self.num_z, rounding_mode="floor")
        buff[:, 1] = torch.div(
            coords % (self.num_x * self.num_y * self.num_z), self.num_y * self.num_z, rounding_mode="floor"
        )
        buff[:, 2] = torch.div(coords % (self.num_y * self.num_z), self.num_z, rounding_mode="floor")
        buff[:, 3] = coords % self.num_z
        coords = buff

        feats = self.norm(feats)

        batch_size = len(lidar)
        out_tensor = torch.zeros(
            (batch_size * self.num_y * self.num_x, self.n_out), device=feats.device, dtype=feats.dtype
        )
        bi = coords[:, 0]
        hi = coords[:, 2]
        wi = coords[:, 1]
        zi = coords[:, 3]
        pos = self.pos_embedding(zi)
        feats = torch.cat((feats, pos), 1)
        feats = self.dense_block(feats)
        out_tensor.index_add_(0, bi * self.num_x * self.num_y + hi * self.num_x + wi, feats)
        out_tensor = self.dense_norm(out_tensor)
        out = out_tensor.view(batch_size, -1, self.n_out).transpose(1, 2).contiguous().view(batch_size, self.n_out, self.num_y, self.num_x)
        return out

class Attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super().__init__()
        assert temperature % 3 == 1
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)
        else:
            hidden_planes = K
        self.fc1 = nn.Linear(in_planes, hidden_planes, bias=True)
        self.fc2 = nn.Linear(hidden_planes, K, bias=True)
        self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.int64))

    def update_temperature(self):
        assert isinstance(self.temperature, torch.Tensor)
        if self.temperature != 1:
            self.temperature -= 3

    def forward(self, x):
        x = x.mean((2, 3))
        x = self.fc1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = self.fc2(x).view(x.size(0), -1)
        return torch.nn.functional.softmax(x / self.temperature, -1)


class DynamicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        ratio=0.25,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        K=4,
        temperature=34,
        init_weight=True,
    ):
        super().__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.k = K
        self.attention = Attention2d(in_planes, ratio, self.k, temperature)

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.k):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, _, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.k, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(
            batch_size * self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size
        )
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = torch.nn.functional.conv2d(
                x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size,
            )
        else:
            output = torch.nn.functional.conv2d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size,
            )

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation module described in https://arxiv.org/abs/1709.01507"""

    def __init__(self, n_features, reduction=16):
        super().__init__()

        if n_features % reduction != 0:
            raise ValueError("n_features must be divisible by reduction (default = 16)")

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3))
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = x * y[:, :, None, None]
        return y


class SEDyBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        norm_type: str = "BN",
    ):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = DynamicConv2d(inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = build_norm_layer(norm_type, planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = DynamicConv2d(
            planes, planes, 3, stride=1, padding=dilation, dilation=dilation, groups=32, bias=False
        )
        self.norm2 = build_norm_layer(norm_type, planes)
        self.downsample = downsample
        self.stride = stride
        self.se_module = SqueezeExcitation(planes)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x.float())
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)

        out = self.se_module(out.float())
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def trunc_normal_(tensor, mean=0, std=1, a=-2, b=2):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = False
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def get_process_group(backend: str) -> torch.distributed.ProcessGroup:
    """Get group process for a specific backend"""
    group = None
    for k, v in torch.distributed.distributed_c10d._pg_map.items(): 
        if v[0] == backend:
            group = k
            break
    if not group:
        raise RuntimeError(f"Cannot find a process group for backend {backend}")
    return group

def sync_batchnorm(num_features: int, affine: bool = True, dim: int = 2) -> torch.nn.Module:
    """Implement SyncBatchNorm from torch"""
    m: torch.nn.Module
    if dim == 1:
        m = torch.nn.BatchNorm1d(num_features, affine=affine)
    elif dim == 2:
        m = torch.nn.BatchNorm2d(num_features, affine=affine)
    elif dim == 3:
        m = torch.nn.BatchNorm3d(num_features, affine=affine)

    pg = get_process_group(torch.distributed.Backend.NCCL)
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(m, pg)

def build_norm_layer(
    norm_type: str, num_features: int, channel_dim: int | None = None, affine: bool = True, dim: int = 2, group=4
) -> nn.Module:
    """Return norm layer according to the given norm type"""
    assert norm_type in ["BN", "GN", "SyncBN", "IN", "None"]
    assert dim in [1, 2, 3], "Norm layer dim must be one of [1, 2, 3]"
    if norm_type == "BN":
        if dim == 1:
            return nn.BatchNorm1d(num_features, affine=affine)
        elif dim == 2:
            return nn.BatchNorm2d(num_features, affine=affine)
        elif dim == 3:
            return nn.BatchNorm3d(num_features, affine=affine)
    elif norm_type == "GN":
        return nn.GroupNorm(group, num_features, affine=affine)
    elif norm_type == "SyncBN":
        if torch.distributed.is_initialized():
            return sync_batchnorm(num_features, affine=affine, dim=dim)
        else:
            print("distributed env is not initialized, fall back to batch norm")
            if dim == 1:
                return nn.BatchNorm1d(num_features, affine=affine)
            elif dim == 2:
                return nn.BatchNorm2d(num_features, affine=affine)
            elif dim == 3:
                return nn.BatchNorm3d(num_features, affine=affine)
    elif norm_type == "LN":
        assert channel_dim is not None, "Layer norm requires a channel dim argument"
    elif norm_type == "IN":
        if dim == 1:
            return nn.InstanceNorm1d(num_features, affine=affine)
        elif dim == 2:
            return nn.InstanceNorm2d(num_features, affine=affine)
        elif dim == 3:
            return nn.InstanceNorm3d(num_features, affine=affine)
    elif norm_type == "None":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown {norm_type = }")


class ResNet(nn.Module):
    """The conventional ResNet module"""

    def __init__(
        self,
        in_channels: int,
        groups_in_stem: int,
        stride_in_stem: int,
        channels_in_stem: tuple[int, ...],
        blocks_per_stage: tuple[int, ...],
        channels_per_stage: tuple[int, ...],
        strides_per_stage: tuple[int, ...],
        dilations_per_stage: tuple[int, ...],
        out_indices: tuple[int, ...],
        norm_type: str = "BN",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = groups_in_stem
        self.channels_in_stem = channels_in_stem
        self.stride_in_stem = stride_in_stem

        self.num_stages = len(blocks_per_stage)
        assert self.num_stages >= 1
        self.blocks_per_stage = blocks_per_stage

        self.strides = strides_per_stage
        self.dilations = dilations_per_stage
        self.channels = channels_per_stage
        assert len(self.strides) == len(self.dilations) == len(self.channels) == self.num_stages
        self.block = SEDyBasicBlock
        self.out_indices = [int(out_index) for out_index in out_indices]
        assert max(out_indices) < self.num_stages + 1
        self.norm_type = norm_type

        self.stem, self.stem_stride = self._make_stem_layer()
        res_layers = []
        inplanes = self.channels_in_stem[-1]
        for i, num_blocks in enumerate(blocks_per_stage):
            stride = self.strides[i]
            dilation = self.dilations[i]
            planes = self.channels[i]
            res_layer = self._make_res_layer(
                inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                norm_type=self.norm_type,
            )
            inplanes = planes * self.block.expansion
            res_layers.append(res_layer)
        self.res_layers = torch.nn.ModuleList(res_layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                with torch.no_grad():
                    trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stem_layer(self):
        """In conventional ResNet, the stem layer has output stride = 4
        here we remove the max pooling layer since BEV images may require higher resolution
        """
        layers = []
        out_channels = 0
        for i, channels in enumerate(self.channels_in_stem):
            num_groups = self.num_groups if i == 0 else 1
            in_channels = self.in_channels if i == 0 else out_channels
            out_channels = channels * num_groups if i == 0 else channels
            stride = self.stride_in_stem if i == 0 else 1

            layers.append(conv3x3(in_channels, out_channels, stride=stride, groups=num_groups))
            layers.append(build_norm_layer(self.norm_type, out_channels))
            layers.append(nn.ReLU(inplace=True))

        stem = nn.Sequential(*layers)

        return stem, self.stride_in_stem

    def _make_res_layer(
        self,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        norm_type="BN",
    ):
        downsample = None
        if stride != 1 or inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    inplanes, planes * self.block.expansion, stride=stride
                ),
                build_norm_layer(norm_type, planes * self.block.expansion),
            )

        layers = []
        layers.append(
            self.block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                norm_type=norm_type,
            )
        )
        inplanes = planes * self.block.expansion
        for _ in range(1, blocks):
            layers.append(self.block(inplanes=inplanes, planes=planes, stride=1, dilation=dilation, norm_type=norm_type))

        return nn.Sequential(*layers)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask: torch.Tensor | None = None):
        if mask is None:
            not_mask = x.new_ones((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool)
        else:
            not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class SpatialFusionLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_levels, n_heads, n_points, dropout=0.1):
        # self attention over different scales
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=False)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):
        # self attention
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src

class SpatialFusion(nn.Module):
    def __init__(
        self,
        n_levels,
        num_spatial_fusion_layers,
        d_model=256,
        d_ffn=1024,
        n_heads=8,
        n_points=4,
        dropout=0.1,
    ):
        super().__init__()
        self.pos_embed_func = PositionEmbeddingSine(d_model // 2)
        self.spatial_level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        self.spatial_fusion = nn.ModuleList(
            [
                SpatialFusionLayer(d_model, d_ffn, n_levels, n_heads, n_points, dropout)
                for _ in range(num_spatial_fusion_layers)
            ]
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.spatial_level_embed)

    def calc_reference_points(self, spatial_shapes, device: torch.device):
        reference_points_list = []
        for lvl in range(len(spatial_shapes)):  # noqa: N815,F841
            # (H, W)
            h_ = spatial_shapes[lvl][0]
            w_ = spatial_shapes[lvl][1]
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h_ - 0.5, h_, dtype=torch.float32, device=device),
                torch.linspace(0.5, w_ - 0.5, w_, dtype=torch.float32, device=device),
            )
            # (B, H*W)
            ref_y = ref_y.reshape(-1)[None] / h_
            ref_x = ref_x.reshape(-1)[None] / w_

            # (B, H*W, 2)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # (B, sum H*W, 2)
        reference_points = torch.cat(reference_points_list, 1)
        # (B, sum H*W, 1, 2) -> (B, sum H*W, #lvl, 2)
        reference_points = reference_points[:, :, None].repeat((1, 1, len(spatial_shapes), 1))
        return reference_points

    def forward(self, srcs: list[torch.Tensor]):
        """[summary]
        Args:
            srcs: List[Tensor] [B x C_i x H_i x W_i]
        """
        batch_size, _, _, _ = srcs[-1].shape

        src_flatten_list = []
        lvl_pos_embed_flatten_list = []
        spatial_shapes_list = [src.shape[-2:] for src in srcs]

        for lvl, src in enumerate(srcs):
            pos_embed = self.pos_embed_func(src)
            _, c, h, w = pos_embed.shape
            pos_embed = pos_embed.reshape(batch_size, c, h * w).transpose(1, 2).contiguous()
            pos_embed = pos_embed + self.spatial_level_embed[lvl].view(1, 1, -1)
            src = src.reshape(batch_size, -1, h * w).transpose(1, 2).contiguous()
            lvl_pos_embed_flatten_list.append(pos_embed)
            src_flatten_list.append(src)

        src_flatten = torch.cat(src_flatten_list, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten_list, 1)

        spatial_shapes_gpu = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=src_flatten.device)
        spatial_shapes_cpu = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device="cpu")

        level_start_index_gpu = torch.cat(
            (spatial_shapes_gpu.new_zeros((1,)), spatial_shapes_gpu.prod(1).cumsum(0)[:-1])
        )
        level_start_index_cpu = torch.cat(
            (spatial_shapes_cpu.new_zeros((1,)), spatial_shapes_cpu.prod(1).cumsum(0)[:-1])
        )

        reference_points = self.calc_reference_points(spatial_shapes_cpu, device=src_flatten.device).repeat(
            batch_size, 1, 1, 1
        )

        for i, spatial_fusion in enumerate(self.spatial_fusion):  # noqa: B007
            src_flatten = spatial_fusion(
                src_flatten, lvl_pos_embed_flatten, reference_points, spatial_shapes_gpu, level_start_index_gpu
            )  # (B*T, #ele, C)

        # unflatten src
        unflatten_src = []
        for i in range(len(level_start_index_cpu)):
            st = level_start_index_cpu[i].item()
            ed = level_start_index_cpu[i + 1].item() if i + 1 != len(level_start_index_cpu) else src_flatten.shape[1]
            unflatten_src_i = (
                src_flatten[:, st:ed]
                .transpose(1, 2)
                .reshape(batch_size, -1, spatial_shapes_cpu[i][0], spatial_shapes_cpu[i][1])
            )
            unflatten_src.append(unflatten_src_i)

        return unflatten_src


class AttnResNet(ResNet):
    """The ResNet module with Attention"""

    def __init__(
        self,
        in_channels: int,
        groups_in_stem: int,
        stride_in_stem: int,
        channels_in_stem: tuple[int, ...],
        blocks_per_stage: tuple[int, ...],
        channels_per_stage: tuple[int, ...],
        strides_per_stage: tuple[int, ...],
        dilations_per_stage: tuple[int, ...],
        out_indices: tuple[int, ...],
        norm_type: str = "BN",
        d_proj: int = 128,
    ):
        super().__init__(
            in_channels,
            groups_in_stem,
            stride_in_stem,
            channels_in_stem,
            blocks_per_stage,
            channels_per_stage,
            strides_per_stage,
            dilations_per_stage,
            out_indices,
            norm_type,
        )
        self.temporal_fusion = nn.ModuleList()
        self.input_proj = nn.ModuleList()
        self.output_proj = nn.ModuleList()
        for i in range(self.num_stages):
            self.input_proj.append(
                nn.Sequential(
                    conv1x1(self.channels[i] * self.block.expansion, d_proj),
                    nn.GroupNorm(32, d_proj),
                    nn.ReLU(),
                )
            )
            self.output_proj.append(
                nn.Sequential(
                    conv1x1(d_proj, self.channels[i] * self.block.expansion),
                    nn.GroupNorm(32, self.channels[i] * self.block.expansion),
                    nn.ReLU(),
                )
            )
        self.spatial_fusion = SpatialFusion(
            self.num_stages, num_spatial_fusion_layers=2, d_model=d_proj, d_ffn=4 * d_proj, dropout=0.1
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x (torch.Tensor batch_size x in_channels x input_height x input_width): the input images

        Returns:
            List[torch.Tensor]: for each stages in out_indices, there will be an output feature map
                tensor (torch.Tensor batch_size x hidden_dim_i x output_height_i x output_width_i) in outs
        """
        attn_feat = []
        x = self.stem(x)

        assert 0 not in self.out_indices
        for res_layer, input_proj in zip(self.res_layers, self.input_proj):
            x = res_layer(x)

            attn_feat.append(input_proj(x))

        attn_feat = self.spatial_fusion(attn_feat)

        outs = [attn_feat[i - 1] for i in self.out_indices]
        return outs

class FPN(nn.Module):
    """This is an lite implementation of FPN, which only output the highest
    resolution of the feature maps.
    """

    def __init__(self, in_channels: tuple[int, ...], out_channels: int, norm_type: str = "None"):
        super().__init__()
        assert len(in_channels) > 0
        # we require the order of in_channels to be low res->high res
        self.in_channels = in_channels[::-1]
        self.out_channels = out_channels

        self.convs_1x1 = nn.ModuleList()
        layer_with_bias = norm_type == "None"
        for in_channels_i in self.in_channels:
            self.convs_1x1.append(
                nn.Sequential(
                    conv1x1(in_channels_i, out_channels, bias=layer_with_bias),
                    build_norm_layer(norm_type, out_channels),
                )
            )
        # since the model only output the highest resolution, it only needs one conv3x3 layer
        self.convs_3x3 = nn.Sequential(
            conv3x3(out_channels, out_channels, bias=layer_with_bias),
            build_norm_layer(norm_type, out_channels),
        )
        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x (List[torch.Tensor]): the list of input tensor, from high res to low res, for the details check the
                forward function definition of the corresponding backbone

        Returns:
            out (torch.Tensor batch_size x out_channels x output_height x output_width): the output feature map with
                the highest resolution
        """
        assert len(x) == len(self.in_channels)
        out: torch.Tensor | None = None
        for i, conv1x1_i in enumerate(self.convs_1x1):
            j = len(self.in_channels) - i - 1
            # we require the order of x to be low res->high res
            lateral_output = conv1x1_i(x[j])
            if i == 0:
                out = lateral_output
            else:
                assert out is not None
                out = torch.nn.functional.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=False) + lateral_output

        assert out is not None
        final_conv_out: torch.Tensor = self.convs_3x3(out)
        return [final_conv_out]

class ResnetBlockFC(nn.Module):
    """Fully connected linear Resnet block, adatped from
    https://github.com/autonomousvision/convolutional_occupancy_networks

    Args:
        size_in: input dimension
        size_out: output dimension. If not specified, use size_in. Defaults to None.
        size_h: hiddent dimension. If not specified, use min(size_in, size_out). Defaults to None.
    """

    def __init__(
        self, size_in: int, size_out: int | None = None, size_h: int | None = None, norm_type: str = "None"
    ):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in, self.size_h, self.size_out = size_in, size_h, size_out

        self.fc_0 = nn.Linear(self.size_in, self.size_h)
        self.fc_1 = nn.Linear(self.size_h, self.size_out)
        self.actvn = nn.ReLU()

        self.norm_type = norm_type
        self.norm1 = build_norm_layer(norm_type, self.size_h, channel_dim=1, dim=1)
        self.norm2 = build_norm_layer(norm_type, self.size_out, channel_dim=1, dim=1)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape (B, C, self.size_in) or (B, self.size_in)

        Returns:
            Tensor with shape (B, C, size.size_out) or (B, self.size_out)
        """
        net = self.actvn(x)
        if self.norm_type != "None":
            if len(net.shape) == 3:
                net = self.norm1(net.transpose(1, 2)).transpose(1, 2)
            else:
                net = self.norm1(net)
        net = self.fc_0(net)
        dx = self.actvn(net)
        if self.norm_type != "None":
            if len(dx.shape) == 3:
                dx = self.norm2(dx.transpose(1, 2)).transpose(1, 2)
            else:
                dx = self.norm2(dx)
        dx = self.fc_1(dx)

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CONetResNet(nn.Module):
    """
    Implements the ResNet decoder from Convolutional Occupancy Networks.
    See https://drive.google.com/file/d/1a11A2HcD3qLaEyjZZDWAIg3d-uCE4S5o/view?usp=sharing
    for an architecture diagram
    """

    def __init__(self, n_blocks: int, feature_vector_dim: int, points_dim: int, hidden_size: int, norm_type: str = "None"):
        super().__init__()
        n_blocks = n_blocks
        assert n_blocks >= 0, "Can't have a negative number of blocks"
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        # self.feature_vector_dim = cfg.feature_vector_dim
        # self.points_dim
        self.resnet_blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size, norm_type=norm_type) for _ in range(n_blocks)]
        )
        self.linear_feature_vector = nn.ModuleList(
            [nn.Linear(feature_vector_dim, hidden_size) for _ in range(n_blocks)]
        )
        self.linear_points_encoder = nn.Linear(points_dim, hidden_size)

    def forward(self, feature_vector: torch.Tensor, points: torch.Tensor):
        """
        Args:
            feature_vector: Feature vectors of shape (B, L, F) or (B, F)
            points: Shape (B, L, F') or (B, L, F')
        Returns:
            net: Feature vector of shape (..., hidden_size)
        """
        assert feature_vector.shape[:-1] == points.shape[:-1]
        net = self.linear_points_encoder(points)
        for linear_feature_vector, resnet_block in zip(self.linear_feature_vector, self.resnet_blocks):
            net = net + linear_feature_vector(feature_vector)
            net = resnet_block(net)

        return net


def sample_planar_features(
    points: torch.Tensor, c: torch.Tensor, sample_mode: str = "bilinear", padding_mode: str = "border"
) -> torch.Tensor:
    """
    Interpolation on latent feature plane given query points.

    Args:
        points: Query points, points.shape = (B, Q, n_heads, 2) in order (y, x)
            It is assumed that the coordinates are normalized between [0, 1]
        c: Latent feature plane, c.shape = (B, F, Ry, Rx)

    Returns:
        A tensor of shape (B, Q, n_heads, F) of interpolated feature vectors from c at the
            points given by points
    """
    p = points.clone()
    vgrid = 2.0 * p - 1.0
    vgrid = torch.flip(vgrid, dims=[-1])
    c = torch.nn.functional.grid_sample(c, vgrid, align_corners=False, mode=sample_mode, padding_mode=padding_mode)
    c = torch.permute(c, (0, 2, 3, 1))
    return c



class Decoder(nn.Module):
    """Header implementing deformable attention as shown in this architecture diagram:
    https://drive.google.com/file/d/1FQYfVIaWAHvDjNqBlhA3BTdMIXIisaBE/view?usp=sharing

    Information of deformable convolution and attention can be found in these papers:
    https://arxiv.org/abs/2010.04159
    https://arxiv.org/abs/1703.06211

    Args:
        actor_classes: Actor classes present in the scenario
        decoder_cfg: Config for the occupancy/motion/modal_logits decoder
        attention_decoder_cfg: Config for the attention module
        offset_coords: Specifies which coordinates the offsets from the attention module are applied to.
            t = 0, y = 1, x = 2. E.g., [1, 2] specifies an offset in the y and x dimension, but not the t
            dimension
    """

    def __init__(
        self,
        attn_resnet_num_blocks: int = 1,
        attn_resnet_hidden_size: int = 16,
        attn_resnet_points_dim: int = 4,
        attn_resnet_feature_vector_dim: int = 128,
        decoder_resnet_num_blocks: int = 3,
        decoder_resnet_hidden_size: int = 16,
        decoder_resnet_points_dim: int = 4,
        decoder_resnet_feature_vector_dim: int = 256,
        time_scale_attention: bool = False,
    ):
        super().__init__()
        self.n_heads = 1
        # for jit purposes:
        self.offset_coords = [2, 3] # offset in x and y

        assert len(self.offset_coords) <= 3

        self.attn_interpolators = sample_planar_features
        self.attn_resnets = CONetResNet(
            n_blocks=attn_resnet_num_blocks,
            feature_vector_dim=attn_resnet_feature_vector_dim,
            points_dim=attn_resnet_points_dim,
            hidden_size=attn_resnet_hidden_size,
        )
        self.attn_decoder = nn.Linear(self.attn_resnets.hidden_size, 2)

        self.decoder_interpolators = sample_planar_features
        self.decoder_resnets = CONetResNet(
            n_blocks=decoder_resnet_num_blocks,
            feature_vector_dim=decoder_resnet_feature_vector_dim,
            points_dim=decoder_resnet_points_dim,
            hidden_size=decoder_resnet_hidden_size,
        )
        self.occ_decoder = nn.Linear(self.decoder_resnets.hidden_size, 1)

        self.actvn = torch.nn.functional.relu

    def forward(
        self,
        neck_fmaps: list[torch.Tensor],
        xyzt_points: torch.Tensor,
    ):
        c_lst = neck_fmaps
        assert c_lst is not None and len(c_lst) == 1
        c = c_lst[0]

        # flip so order is (t, z, y, x).
        query_points = xyzt_points  # (B, N, 4)
        query_points = torch.flip(query_points, dims=[-1])  # (t, z, y, x)

        batch_size = query_points.shape[0]
        num_query = query_points.shape[1]
        qry_pt_dims = query_points.shape[-1]
        dp_shape = (batch_size, num_query, self.n_heads, qry_pt_dims)
        query_points = query_points.unsqueeze(-2)  # (B, N, 1, 3)

        c_0: torch.Tensor | None = None
        delta_p: torch.Tensor | None = None
        assert self.n_heads == 1, "Not implemented for different number of heads"
        delta_p = torch.zeros(dp_shape, device=query_points.device, dtype=query_points.dtype)
        c_0 = self.attn_interpolators(query_points[:, :, :, -2:], c).squeeze(
            -2
        )  # interpolate at (y, x) only

        # (bi, num_query_points, feat_dim)
        pos = query_points[:, :, :, -2:].flip(dims=[-1]).flatten(1, 2)  # (x, y)
        pos = torch.concat((pos, torch.zeros_like(pos[:, :, 0:1])), dim=-1)

        net = self.actvn(self.attn_resnets(c_0, query_points.squeeze(-2)))  # (B, Q, hidden_size)
        attn_out = self.attn_decoder(net)  # List[(B, Q, len(self.offset_coords))]

        delta_p[:, :, :, self.offset_coords] = attn_out.unsqueeze(dim=-2)

        new_query_points = query_points[:, :, :, -1 * qry_pt_dims :] + delta_p  # (B, Q, n_heads, 3)


        c_1_n = self.decoder_interpolators(
            new_query_points[:, :, :, -2:], c
        )  # (B, Q, n_heads, c_dim)

        # aggregating interpolated features

        c = c_1_n.reshape(batch_size, num_query, -1)

        assert len(c.shape) == 3 and c.shape[:-1] == (batch_size, num_query)

        assert self.n_heads > 0 and c_0 is not None
        c = torch.concat((c, c_0), dim=-1)

        # agnostic to aggregation method: decide which points should be used in decoding
        points_decode_list = []
        points_decode_list.append(query_points.reshape((batch_size, num_query, -1)))
        assert len(points_decode_list) > 0
        points_decode = torch.concat(points_decode_list, dim=-1)

        c = c.reshape((batch_size, num_query, -1))

        net = self.actvn(self.decoder_resnets(c, points_decode))

        occ_out = self.occ_decoder(net)

        return occ_out

class UnO(nn.Module):
    def __init__(self):
        super().__init__()
        # Note: here im putting configs for av2
        self.voxelizer = Voxelizer(
            x_min=-100.0,
            x_max=150.0,
            y_min=-100.0,
            y_max=100.0,
            z_min=-2.5,
            z_max=12.5,
            step=0.15625,
            z_step=0.05,
            n_out=128,
            num_sensors=2,
            num_sweeps=6,
            single_lidar=False,
            clip_outside_voxelizer_z_roi=True,
        )
        self.encoder = AttnResNet(
            in_channels=128,
            groups_in_stem=1,
            blocks_per_stage=[2, 2, 6],
            channels_in_stem=[128, 128, 128],
            channels_per_stage=[64, 128, 256],
            strides_per_stage=[2, 2, 2],
            stride_in_stem=2,
            dilations_per_stage=[1, 1, 1],
            out_indices=[1, 2, 3],
            norm_type="BN", # TODO: change this to SyncBN if training on distributed
            d_proj=128,
        )
        self.fpn = FPN(
            in_channels=(128, 128, 128),
            out_channels=128,
            norm_type="BN",
        )
        self.decoder = Decoder()
        
    def forward(self, lidar_sweeps, query_points):
        x = self.voxelizer(lidar_sweeps)
        x = self.encoder(x) 
        neck_fmaps = self.fpn(x)
        return self.decoder(neck_fmaps, query_points)