import numpy as np
from pyquaternion import Quaternion
import torch
import time
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import train, val, test


class MyLidarPointCloud(LidarPointCloud):
    def get_ego_mask(self):
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5),
        )
        return ego_mask


class nuScenesDataset(Dataset):
    def __init__(self, nusc, nusc_split, kwargs):
        """
        Figure out a list of sample data tokens for training.
        """
        super(nuScenesDataset, self).__init__()

        self.nusc = nusc
        self.nusc_split = nusc_split
        self.nusc_root = self.nusc.dataroot

        self.pc_range = kwargs["pc_range"]
        self.voxel_size = kwargs["voxel_size"]
        self.grid_shape = [
            int((self.pc_range[5] - self.pc_range[2]) / self.voxel_size),
            int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size),
            int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size),
        ]

        # number of sweeps (every 1 sweep / 0.05s)
        self.n_input = kwargs["n_input"]
        # number of samples (every 10 sweeps / 0.5s)
        self.n_output = kwargs["n_output"]

        scenes = self.nusc.scene
        if self.nusc_split == "train":
            split_scenes = train
        elif self.nusc_split == "val":
            split_scenes = val
        else:
            split_scenes = test

        # list all sample data
        self.valid_index = []
        self.flip_flags = []
        self.scene_tokens = []
        self.sample_tokens = []
        self.sample_data_tokens = []
        self.timestamps = []
        for scene in scenes:
            # if scene["name"] not in split_scenes:
            #   continue
            scene_token = scene["token"]
            # location
            log = self.nusc.get("log", scene["log_token"])
            # flip x axis if in left-hand traffic (singapore)
            flip_flag = True if log["location"].startswith("singapore") else False
            #
            start_index = len(self.sample_tokens)
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            sample_token = first_sample["token"]
            i = 0
            while sample_token != "":
                self.flip_flags.append(flip_flag)
                self.scene_tokens.append(scene_token)
                self.sample_tokens.append(sample_token)
                sample = self.nusc.get("sample", sample_token)
                i += 1
                self.timestamps.append(sample["timestamp"])
                sample_data_token = sample["data"]["LIDAR_TOP"]

                self.sample_data_tokens.append(sample_data_token)
                sample_token = sample["next"]

            #
            end_index = len(self.sample_tokens)
            #
            valid_start_index = start_index + self.n_input  # (self.n_input // 10)
            valid_end_index = end_index - self.n_output
            self.valid_index += list(range(valid_start_index, valid_end_index))

        assert (
            len(self.sample_tokens)
            == len(self.scene_tokens)
            == len(self.flip_flags)
            == len(self.timestamps)
        )

        self.n_samples = len(self.valid_index)
        print(
            f"{self.nusc_split}: {self.n_samples} valid samples over {len(split_scenes)} scenes"
        )
        ############################# Ray Casting Parameters #############################
        self.n_query_points = kwargs["n_query_points"]
        self.n_ray_points = kwargs["n_ray_points"]
        self.ray_step = kwargs["ray_step"]
        self.scale = kwargs["scale"]
        ################################################################################

    def __len__(self):
        return self.n_samples

    def get_global_pose(self, sd_token, inverse=False):
        sd = self.nusc.get("sample_data", sd_token)
        sd_ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        sd_cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        if inverse is False:
            global_from_ego = transform_matrix(
                sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False
            )
            ego_from_sensor = transform_matrix(
                sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
            )
            pose = global_from_ego.dot(ego_from_sensor)
        else:
            sensor_from_ego = transform_matrix(
                sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True
            )
            ego_from_global = transform_matrix(
                sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True
            )
            pose = sensor_from_ego.dot(ego_from_global)

        return pose

    
    def unsupervised_labeling(self, future_xyz_points, future_t_index, sensor_origins):
        # Prepare steps and directions for rays cast
        """
        future_xyz_points  # (N, 3)
        future_t_index  # (N,)
        sensor_origins  # (n_output, 3)
        """
        # Prepare steps and directions for rays cast
        sensor_origins_br = sensor_origins[future_t_index.to(dtype=torch.int)] # (N,3)
        directions = future_xyz_points - sensor_origins_br
        norms = torch.norm(directions, dim=1, keepdim=True) 
        directions = directions / (norms + 1e-7) # Unit directions 
        
        # (N, 5) xyzt (steps = number of steps it takes from sensor orig to go to the corresponding pt) 
        directions_with_t_steps = torch.cat((directions, future_t_index.unsqueeze(-1) / future_t_index.max(), norms - 1e-7), dim=1) 
        
        return directions_with_t_steps

    def __getitem__(self, idx):

        ref_index = self.valid_index[idx]

        ref_sample_token = self.sample_tokens[ref_index]
        ref_sample_rec = self.nusc.get("sample", ref_sample_token)
        ref_scene_token = self.scene_tokens[ref_index]
        ref_timestamp = self.timestamps[ref_index]
        ref_sd_token = self.sample_data_tokens[ref_index]
        flip_flag = self.flip_flags[ref_index]

        # Keep Frame Lengths for visualization of frames voxelized by CUDA

        input_frame_lengths = []

        # reference coordinate frame
        ref_from_global = self.get_global_pose(ref_sd_token, inverse=True) # Used in transforming every past sweep coordinates into the current time sweep s coordinates

        # NOTE: getting input frames
        input_points_list = []
        input_tindex_list = []
        input_origin_list = []
        for i in range(self.n_input): # n_input is how many frames to go back in a sample 2 for 1s preds 6 for 3s
            index = ref_index - i
            # if this exists a valid target
            if self.scene_tokens[index] == ref_scene_token:
                curr_sd_token = self.sample_data_tokens[index]

                curr_sd = self.nusc.get("sample_data", curr_sd_token)

                # load the current lidar sweep
                curr_lidar_pc = MyLidarPointCloud.from_file(
                    f"{self.nusc_root}/{curr_sd['filename']}"
                )
                ego_mask = curr_lidar_pc.get_ego_mask()
                curr_lidar_pc.points = curr_lidar_pc.points[:, np.logical_not(ego_mask)]

                # transform from the current lidar frame to global and then to the reference lidar frame
                global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
                ref_from_curr = ref_from_global.dot(global_from_curr)
                curr_lidar_pc.transform(ref_from_curr)

                # NOTE: check if we are in Singapore (if so flip x)
                if flip_flag:
                    ref_from_curr[0, 3] *= -1
                    curr_lidar_pc.points[0] *= -1

                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)
                points_tf = np.array(curr_lidar_pc.points[:3].T, dtype=np.float32)

            else:  # filler
                print("came here to fill in nans in input point cloud")
                print(self.scene_tokens[index], ref_scene_index)
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                points_tf = np.full((0, 3), float("nan"), dtype=np.float32)

            # points
            input_points_list.append(points_tf)
            input_frame_lengths.append(points_tf)
            # origin
            input_origin_list.append(origin_tf)

            # timestamp index
            tindex = np.full(len(points_tf), i, dtype=np.float32)
            input_tindex_list.append(tindex)

        
        past_xyz_points = torch.from_numpy(np.concatenate(input_points_list))
        input_origin_tensor = torch.from_numpy(np.stack(input_origin_list))
        input_tindex_tensor = torch.from_numpy(np.concatenate(input_tindex_list))
        displacement = torch.from_numpy(input_origin_list[0] - input_origin_list[1])

        # NOTE: getting output frames
        output_origin_list = []
        output_points_list = []
        output_tindex_list = []
        output_labels_list = []
        for i in range(self.n_output):
            index = ref_index + i + 1
            # if this exists a valid target
            if self.scene_tokens[index] == ref_scene_token:
                curr_sd_token = self.sample_data_tokens[index]

                curr_sd = self.nusc.get("sample_data", curr_sd_token)

                # load the current lidar sweep
                curr_lidar_pc = MyLidarPointCloud.from_file(
                    f"{self.nusc_root}/{curr_sd['filename']}"
                )
                ego_mask = curr_lidar_pc.get_ego_mask()
                curr_lidar_pc.points = curr_lidar_pc.points[:, np.logical_not(ego_mask)]

                # transform from the current lidar frame to global and then to the reference lidar frame
                global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
                ref_from_curr = ref_from_global.dot(global_from_curr)
                curr_lidar_pc.transform(ref_from_curr)

                # NOTE: check if we are in Singapore (if so flip x)
                if flip_flag:
                    ref_from_curr[0, 3] *= -1
                    curr_lidar_pc.points[0] *= -1

                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)
                points_tf = np.array(curr_lidar_pc.points[:3].T, dtype=np.float32)
                # if self.nusc_split != "test":
                #     labels = self.load_fg_labels(curr_sd_token).astype(np.float32)[
                #         np.logical_not(ego_mask)
                #     ]
                # else:
                #     labels = np.full((len(points_tf),), -1, dtype=np.float32)
            else:  # filler
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                points_tf = np.full((0, 3), float("nan"), dtype=np.float32)
                # labels = np.full((len(points_tf),), -1, dtype=np.float32)

            #
            # assert len(labels) == len(points_tf)

            # origin
            output_origin_list.append(origin_tf)

            # points
            output_points_list.append(points_tf)

            # timestamp index
            tindex = np.full(len(points_tf), i, dtype=np.float32)
            output_tindex_list.append(tindex)

            

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))
        output_points_tensor = torch.from_numpy(np.concatenate(output_points_list))
        output_tindex_tensor = torch.from_numpy(np.concatenate(output_tindex_list))
        output_labels_tensor = output_labels_list #torch.from_numpy(np.concatenate(output_labels_list))
        
        ############################# Prepare Occupied and Unoccupied Points #############################
        
        directions_with_t_steps = self.unsupervised_labeling(output_points_tensor,output_tindex_tensor, output_origin_tensor)
        
        selected_ind = torch.randperm(len(directions_with_t_steps))[:self.n_query_points // self.n_ray_points] # Random indices for unoccupied n_query_points / 4
        binary_mask = torch.zeros(len(directions_with_t_steps), dtype=torch.bool)
        binary_mask[selected_ind] = True
        selected_directions = directions_with_t_steps[binary_mask] # (n_query_points / 4, 5)

        # sample n_ray_points unoccupied points on the ray for each chosen point
        random_factors = torch.rand(len(selected_directions),self.n_ray_points) # n_ray_points random factor for each dirn in selected dirns
        scaled_factors = random_factors * (selected_directions[:, -1:] - self.ray_step / self.scale) # (n_query_points / 4, 4), 0.5 adjusts how close the unoccupied points to actual points
        unit_dirns_expanded = selected_directions[:,:-2].unsqueeze(1) # (n_query_points / 4, 1, 3)
        scaled_factors = scaled_factors.unsqueeze(-1) # (n_query_points / 4, n_ray_points, 1)
        unoccupied_points = scaled_factors * unit_dirns_expanded # (n_query_points / 4, n_ray_points, 3) dim 1 = ray, dim 2 = point index on ray, dim 3 = xyz coordinates 

        # sample n_ray_points occupied points
        adjustments = torch.ones(len(selected_directions),self.n_ray_points)
        adjustments = adjustments * selected_directions[:,-1:]
        for i in range(self.n_ray_points):
            adjustments[:,i] += (self.ray_step / (self.n_ray_points) * (i + 1)) / self.scale
            

        occupied_points = unit_dirns_expanded * adjustments.unsqueeze(-1)
        
        t = selected_directions[:,-2].unsqueeze(1).expand(-1,self.n_ray_points).unsqueeze(-1)
        unoccupied_points = torch.cat((unoccupied_points,t), dim=-1)
        occupied_points= torch.cat((occupied_points, t), dim=-1)
        
        unoccupied_points = unoccupied_points.reshape(-1, unoccupied_points.shape[-1])
        occupied_points = occupied_points.reshape(-1, occupied_points.shape[-1]) 
        
        past_t_index = input_tindex_tensor / input_tindex_tensor.max() # Normalize time by dividing it to max
        
       
        return [past_xyz_points, past_t_index, occupied_points, unoccupied_points] 
            