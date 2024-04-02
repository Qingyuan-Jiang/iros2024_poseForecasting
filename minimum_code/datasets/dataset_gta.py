import os
from copy import deepcopy

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.gta_utils import crop_pcd_np
from utils.pcd_utils import np2pcd_xyzrgb


class DatasetGTA(Dataset):

    def __init__(self, mode, cfg):
        self.mode = mode
        self.scale = 1.0  # Transform from mm to m
        self.n_hist = cfg['dataset_specs']['n_hist']
        self.n_pred = cfg['dataset_specs']['n_pred']
        self.n_total = self.n_hist + self.n_pred
        self.n_joints = cfg['dataset_specs']['n_joints']
        self.step = cfg['dataset_specs']['step']
        self.num_scene_points = cfg['dataset_specs']['num_scene_points']
        self.num_scene_points_vis = cfg['dataset_specs']['num_scene_points_vis'] \
            if 'num_scene_points_vis' in cfg['dataset_specs'].keys() else 500000
        self.max_dist_from_human = cfg['dataset_specs']['max_dist_from_human'] * self.scale
        self.max_dist_from_human_vis = cfg['dataset_specs']['max_dist_from_human_vis'] * self.scale \
            if 'max_dist_from_human_vis' in cfg['dataset_specs'].keys() else 5 * self.scale

        self.root_idx = cfg['dataset_specs']['root_idx']
        self.left_hip_idx = cfg['dataset_specs']['left_hip_idx']
        self.right_hip_idx = cfg['dataset_specs']['right_hip_idx']

        self.use_raw_obs = cfg['dataset_specs']['use_raw_obs'] if 'use_raw_obs' in cfg[
            'dataset_specs'].keys() else False
        self.use_env_pcd = cfg['dataset_specs']['use_env_pcd'] if 'use_env_pcd' in cfg[
            'dataset_specs'].keys() else False
        self.use_map = cfg['dataset_specs']['use_map'] if 'use_map' in cfg['dataset_specs'].keys() else False
        self.use_traj_map = cfg['dataset_specs']['use_traj_map'] if 'use_traj_map' in cfg[
            'dataset_specs'].keys() else False
        self.use_pose_gt = cfg['dataset_specs']['use_pose_gt'] if 'use_pose_gt' in cfg[
            'dataset_specs'].keys() else False
        self.use_env_pcd_vis = cfg['dataset_specs']['use_env_pcd_vis'] if 'use_env_pcd_vis' in cfg[
            'dataset_specs'].keys() else False
        self.map_size = cfg['dataset_specs']['map_size'] if self.use_map else None
        self.voxel_size = self.max_dist_from_human / self.map_size if self.use_map else None
        self.z_min, self.z_max = cfg['dataset_specs']['z_min'], cfg['dataset_specs']['z_max']

        self.data_file = os.path.join(os.path.dirname(__file__), cfg['dataset_specs']['data_file'])
        # self.scene_split = {'train': ['r001', 'r002', 'r003', 'r006'],
        #                     'test': ['r010', 'r011', 'r013']}
        if cfg['dataset'] == 'GTA':
            self.scene_split = {'train': ['r001', 'r002', 'r003'],
                                'test': ['r012', 'r013']}
        elif cfg['dataset'] == 'Real_IM':
            self.scene_split = {'train': cfg['dataset_specs']['train'], 'test': cfg['dataset_specs']['test']}

        self.pose = {}
        self.points, self.colors = {}, {}
        self.idx2scene = {}
        self.hm_pose_a = {}
        self.pose_a_mat = {}
        self.traj_map_hist, self.traj_map_label = {}, {}
        self.occ_map = {}

        print('read original data file')
        for sub, seq in tqdm(enumerate(os.listdir(self.data_file))):
            if cfg['dataset'] == 'GTA':
                room = seq.split('_')[1]
                if room not in self.scene_split[mode]:
                    continue
            if cfg['dataset'] == 'Real_IM':
                room = seq.split('.')[0]
                if room not in self.scene_split[mode]:
                    continue
            data_tmp = np.load(f'{self.data_file}/{seq}', allow_pickle=True)
            self.pose[sub] = data_tmp['joints'] * self.scale
            self.idx2scene[sub] = seq[:-4]
            self.points[sub] = data_tmp['scene_points'] * self.scale
            self.colors[sub] = data_tmp['scene_colors'].astype(np.float32) / 255.0
            self.hm_pose_a[sub] = data_tmp['joints_a'] * self.scale
            self.pose_a_mat[sub] = data_tmp['pose_a_mat_list']
            self.traj_map_hist[sub] = data_tmp['traj_map_hist']
            self.traj_map_label[sub] = data_tmp['traj_map_label']
            self.occ_map[sub] = data_tmp['occ_map_list']

            # ########### for debug
            # if len(self.pose) > 0:
            #     break

        self.data = {}
        self.scene_point_idx = {}
        self.cont_idx = {}
        self.sdf_coord_idxs = {}
        self.pose_a_idx = {}
        k = 0
        self.min_num_scene, self.max_num_scene = np.inf, 0
        print("generating data idxs")
        for sub in tqdm(self.pose.keys()):
            room = self.idx2scene[sub].split('_')[1]
            seq_len = self.pose[sub].shape[0]
            idxs_frame = np.arange(0, seq_len - self.n_total + 1, self.step)
            for fidx in idxs_frame:
                self.data[k] = f'{sub}.{fidx}'
                k += 1

        print(f'seq length {self.n_total},in total {k} seqs')

    def __len__(self):
        return len(list(self.data.keys()))

    def __getitem__(self, idx):

        item_key = self.data[idx].split('.')
        sub = int(item_key[0])
        fidx = int(item_key[1])
        subj = self.idx2scene[sub]
        item_key = f"{subj}.{item_key[1]}"

        # Obtain human pose and human position (use spline4 with index 14 as representation).
        pose_a = torch.tensor(self.hm_pose_a[sub][fidx]).float()
        pose_hist = pose_a[:self.n_hist]
        pose_label = pose_a[self.n_hist:]
        pose_a_mat = torch.from_numpy(self.pose_a_mat[sub][fidx])

        env_info = {'pose_a_mat': pose_a_mat,
                    'item_key': item_key}
        pose_hist_info = {'pose': pose_hist}
        pose_label_info = {'pose': pose_label}

        # # Visualize the predicted poses with the environment point cloud.
        # # Comment out for testing.
        # from utils.vis_utils import vis_skeleton_pcd
        # from utils.pcd_utils import draw_frame
        # import open3d as o3d
        # line_set_hist = vis_skeleton_pcd(self.n_hist, self.n_joints, pose_hist.numpy(), color='r', dataset='GTA')
        # line_set_label = vis_skeleton_pcd(self.n_pred, self.n_joints, pose_label.numpy(), color='b', dataset='GTA')
        # points_full_np, colors_full_np = self.points[sub], self.colors[sub]
        # o3d.visualization.draw_geometries([line_set_hist, draw_frame()])
        # o3d.visualization.draw_geometries([line_set_hist, line_set_label, draw_frame()])

        if self.use_map:
            env_info['map'] = torch.from_numpy(self.occ_map[sub][fidx])

            # # Plot the occupancy map. Comment out for testing.
            # import matplotlib.pyplot as plt
            # plt.imshow(self.occ_map[sub][fidx].squeeze())
            # plt.show()

        if self.use_traj_map:
            pose_hist_info['traj_map'] = torch.from_numpy(self.traj_map_hist[sub][fidx])
            pose_label_info['traj_map'] = torch.from_numpy(self.traj_map_label[sub][fidx])

            # # Plot the occupancy map. Comment out for testing.
            # import matplotlib.pyplot as plt
            # plt.imshow(self.traj_map_hist[sub][fidx].sum(0).squeeze())
            # plt.show()

        if self.use_pose_gt:
            pose_hist_info['pose_gt'] = pose_label
            pose_label_info['pose_gt'] = pose_label

        if self.use_env_pcd:
            points_full_np, colors_full_np = self.points[sub], self.colors[sub]  # data_tmp['scene_points']
            if idx in self.scene_point_idx.keys():
                idx_cropped = self.scene_point_idx[idx]
            else:
                R, t = pose_a_mat.inverse()[:3, :3], pose_a_mat.inverse()[:3, 3].unsqueeze(dim=0)
                pos_a = pose_a_mat[:3, 3].unsqueeze(dim=0)
                # Filter out points that are too far away from the actor based on the bounding box.
                idx_coarse = crop_pcd_np(points_full_np, pos_a.numpy()[0], self.max_dist_from_human * 2)
                points_coarse_np = points_full_np[idx_coarse]

                points_coarse_a_np = (R @ points_coarse_np.T + t.T).T.numpy()
                idx_fine = crop_pcd_np(points_coarse_a_np, np.array([0, 0, 0]), self.max_dist_from_human)

                idx_cropped = idx_coarse[idx_fine]
                self.scene_point_idx[idx] = idx_coarse[idx_fine]

                self.min_num_scene = len(idx_fine) if self.min_num_scene > len(idx_fine) else self.min_num_scene
                self.max_num_scene = len(idx_fine) if self.max_num_scene < len(idx_fine) else self.max_num_scene
                # print(f"[Dataset] Add frame {idx:d} to points point idxs")
                # print(f"[Dataset] num of points points from {self.min_num_scene:d} to {self.max_num_scene:d}")

            # We need to sample a fixed number of points from it.
            n_pts = len(idx_cropped)
            if n_pts < self.num_scene_points:
                idx_sample = list(range(n_pts)) + np.random.choice(np.arange(n_pts),
                                                                   self.num_scene_points - n_pts).tolist()
            else:
                idx_sample = np.random.choice(np.arange(n_pts), self.num_scene_points, replace=False)
            idx_smp = idx_cropped[idx_sample]

            # Now we have environment point cloud around the actor.
            # We need to sample a fixed number of points from it.
            points_local, colors_local = points_full_np[idx_smp], colors_full_np[idx_smp]
            pcd_w_sampled = np2pcd_xyzrgb(np.hstack((points_local, colors_local)))
            # o3d.visualization.draw_geometries([pcd_w, draw_frame(origin=np.min(points_local, axis=0))])

            # Transform the points point cloud to the actor's coordinate system.
            pcd_a = deepcopy(pcd_w_sampled).transform(pose_a_mat.inverse().numpy())
            # o3d.visualization.draw_geometries([pcd_a, draw_frame(scale=self.scale)])
            # o3d.visualization.draw_geometries([pcd_w_sampled, draw_frame(t, Rotation.from_matrix(R).as_quat())])

            env_info['pcd.points'] = torch.from_numpy(np.asarray(pcd_a.points))
            env_info['pcd.colors'] = torch.from_numpy(np.asarray(pcd_a.colors))

        if self.use_env_pcd_vis:
            points_full_np, colors_full_np = self.points[sub], self.colors[sub]  # data_tmp['scene_points']
            if idx in self.scene_point_idx.keys():
                idx_cropped = self.scene_point_idx[idx]
            else:
                R, t = pose_a_mat.inverse()[:3, :3], pose_a_mat.inverse()[:3, 3].unsqueeze(dim=0)
                pos_a = pose_a_mat[:3, 3].unsqueeze(dim=0)
                # Filter out points that are too far away from the actor based on the bounding box.
                idx_coarse = crop_pcd_np(points_full_np, pos_a.numpy()[0], self.max_dist_from_human_vis * 4)
                points_coarse_np = points_full_np[idx_coarse]

                points_coarse_a_np = (R @ points_coarse_np.T + t.T).T.numpy()
                idx_fine = crop_pcd_np(points_coarse_a_np, np.array([0, 0, 0]), self.max_dist_from_human_vis * 2)

                idx_cropped = idx_coarse[idx_fine]
                self.scene_point_idx[idx] = idx_coarse[idx_fine]

            # We need to sample a fixed number of points from it.
            n_pts = len(idx_cropped)
            if n_pts < self.num_scene_points_vis:
                idx_sample = list(range(n_pts)) + np.random.choice(np.arange(n_pts),
                                                                   self.num_scene_points_vis - n_pts).tolist()
            else:
                idx_sample = np.random.choice(np.arange(n_pts), self.num_scene_points_vis, replace=False)
            idx_smp = idx_cropped[idx_sample]

            # Now we have environment point cloud around the actor.
            # We need to sample a fixed number of points from it.
            points_local_vis, colors_local_vis = points_full_np[idx_smp], colors_full_np[idx_smp]
            pcd_w_vis = np2pcd_xyzrgb(np.hstack((points_local_vis, colors_local_vis)))
            # o3d.visualization.draw_geometries([pcd_w, draw_frame(origin=np.min(points_local, axis=0))])

            # Transform the points point cloud to the actor's coordinate system.
            pcd_a_vis = deepcopy(pcd_w_vis).transform(pose_a_mat.inverse().numpy())
            # o3d.visualization.draw_geometries([pcd_a, draw_frame(scale=self.scale)])
            # o3d.visualization.draw_geometries([pcd_w_sampled, draw_frame(t, Rotation.from_matrix(R).as_quat())])

            env_info['pcd_vis.points'] = torch.from_numpy(np.asarray(pcd_a_vis.points))
            env_info['pcd_vis.colors'] = torch.from_numpy(np.asarray(pcd_a_vis.colors))

        # if self.use_raw_obs:
        #     env_info['color'] = torch.from_numpy(np.asarray(self.rgb[sub][fidx:fidx + self.n_total]))
        #     env_info['depth'] = torch.from_numpy(np.asarray(self.depth[sub][fidx:fidx + self.n_total]))
        hist = {'env': env_info, 'pose': pose_hist_info}
        label = {'env': env_info, 'pose': pose_label_info}
        return hist, label


if __name__ == '__main__':
    cfg_name = '../cfg/%s.yml' % 'real_Path_GRU'
    cfg = yaml.safe_load(open(cfg_name, 'r'))

    dataset = DatasetGTA('train', cfg)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    _ = next(iter(dataloader))
