from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from utils.gta_utils import map2traj
import torchvision
from utils.vis_utils import vis_skeleton_pcd
from utils.gta_utils import traj2map, get_relative_pose
from models.pathnet import UNet
import numpy as np


class PoseNet(torch.nn.Module):
    def __init__(self, in_feats_pose, in_feats_traj, out_feats, cfg):
        super(PoseNet, self).__init__()
        self.cfg = cfg
        self.n_hist = cfg['dataset_specs']['n_hist']
        self.n_pred = cfg['dataset_specs']['n_pred']
        self.n_joints = cfg['dataset_specs']['n_joints']
        self.n_feats_mlp = cfg['model_specs']['n_feats_mlp']
        self.n_feats_gru = cfg['model_specs']['n_feats_gru']
        self.root_idx = cfg['dataset_specs']['root_idx']

        # self.n_enc = self.n_feats_mlp
        n_inp_pose = self.n_joints * 3
        self.pose_enc = nn.Linear(n_inp_pose, self.n_feats_mlp)
        self.traj_enc = nn.Linear(2, self.n_feats_mlp)
        # self.pose_gru = nn.GRU(self.n_feats_mlp, self.n_feats_gru, batch_first=True)
        # self.traj_gru = nn.GRU(self.n_feats_mlp, self.n_feats_gru, batch_first=True)
        self.in_gru = nn.GRU(self.n_feats_mlp * 2, self.n_feats_gru, batch_first=True)
        self.out_enc = nn.Linear(self.n_joints * 3 + 2, self.n_feats_gru)
        self.out_gru = nn.GRUCell(self.n_feats_gru, self.n_feats_gru)
        self.out_mlp = nn.Linear(self.n_feats_gru, self.n_joints * 3)

    def forward(self, pose_raw, traj_pred):
        bs = pose_raw.shape[0]
        traj_raw = pose_raw[:, :, self.root_idx, :2]
        pose_rel = get_relative_pose(pose_raw, self.root_idx, self.n_joints)  # bs * nf * nj * 3
        pose_f = pose_rel.reshape(bs, self.n_hist, -1)
        traj_f = traj_raw.reshape(bs, self.n_hist, -1)
        pose_enc = self.pose_enc(pose_f).reshape(bs, self.n_hist, self.n_feats_mlp)
        traj_enc = self.traj_enc(traj_f).reshape(bs, self.n_hist, self.n_feats_mlp)
        # feat = pose_enc
        # pose_gru_enc = self.pose_gru(pose_enc)[1][0]
        # traj_gru_enc = self.traj_gru(traj_enc)[1][0]
        # feat = torch.cat((pose_gru_enc, traj_gru_enc), dim=1)
        feat_cat = torch.cat((pose_enc, traj_enc), dim=-1)
        feat = self.in_gru(feat_cat)[1][0]
        # feat1 = nn.Tanh(self.fc1(feat))
        # feat2 = nn.Tanh(self.fc2(feat1))
        # feat3 = self.fc3(feat2)
        # gru_dec = self.out_gru(feat, pose_gru_enc)
        # x = self.fc4(gru_dec)

        pose_pred = []
        pose_rel_last = pose_rel[:, -1]
        # traj_hist = pose_rel[:, :, self.root_idx, :2]
        # traj_full = torch.cat((traj_hist, traj), dim=1)
        feat_last_dec = feat
        for i in range(self.n_pred):
            feat_last_cat = torch.cat([pose_rel_last.reshape(bs, -1), traj_pred[:, i].reshape(bs, -1)], dim=-1)
            # feat_last_cat = pose_rel_last
            feat_last_enc = self.out_enc(feat_last_cat.reshape(bs, -1))
            # feat_last_cat = pose_gru_dec
            # feat_last_enc = self.out_enc(feat_last_cat)
            feat_last_dec = self.out_gru(feat_last_enc, feat_last_dec)
            pose_rel_last = self.out_mlp(feat_last_dec)
            pose_pred.append(pose_rel_last)

        pose_pred = torch.stack(pose_pred, dim=1)
        return pose_pred


class Path_GRU(torch.nn.Module):

    def __init__(self, cfg):
        super(Path_GRU, self).__init__()
        self.cfg = cfg
        self.n_hist = cfg['dataset_specs']['n_hist']
        self.n_pred = cfg['dataset_specs']['n_pred']
        self.n_joints = cfg['dataset_specs']['n_joints']
        self.n_feats_mlp = cfg['model_specs']['n_feats_mlp']
        self.n_feats_gru = cfg['model_specs']['n_feats_gru']
        self.test_mode = not (cfg['mode'] == 'train' and cfg['dataset_specs']['use_pose_gt'])
        self.train_pathnet = cfg['model_specs']['train_pathnet']
        self.train_posenet = cfg['model_specs']['train_posenet']
        self.traj_interp = cfg['model_specs']['traj_interp']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_inp_pose = self.n_joints * 3
        n_inp_traj = self.n_hist * 2
        n_out_pose = self.n_pred * self.n_joints * 3
        self.pathnet = UNet(1 + self.n_hist, self.n_pred, cfg)
        self.posenet = PoseNet(n_inp_pose, n_inp_traj, n_out_pose, cfg)
        self.pathnet.train() if self.train_pathnet else self.pathnet.eval()
        self.posenet.train() if self.train_posenet else self.posenet.eval()

        self.max_dist_from_human = cfg['dataset_specs']['max_dist_from_human']
        self.map_size = cfg['dataset_specs']['map_size']
        self.resol = 2 * self.max_dist_from_human / self.map_size
        self.root_idx = cfg['dataset_specs']['root_idx']

    def pred_pathnet(self, pose, occ_map, traj_map_hist):
        # Predict trajectory map first and then predict pose.
        traj_hist = pose[:, :, self.root_idx, :2]
        map_hist = torch.cat((occ_map, traj_map_hist), dim=1)
        traj_map_pred, bottleneck = self.pathnet(map_hist, traj_hist)
        traj_pred = map2traj(traj_map_pred, self.max_dist_from_human, self.resol, self.device)
        return traj_pred, traj_map_pred

    def pred_posenet(self, pose, traj):
        bs = pose.shape[0]

        traj_pred_3d_nj, pose_last_nf = self.prepare_pose_raw(pose, traj)

        pose_raw = torch.cat((pose,), dim=1)  # bs * (nf + np) * nj * 3
        # pose_raw = torch.cat((pose_rel, pose_last_nf), dim=1)  # bs * (nf + np) * nj * 3

        # Go through MLP for pose prediction.
        pose_pred_outp = self.posenet(pose_raw, traj)  # bs * np * nj * 3
        pose_pred_outp = pose_pred_outp.reshape(bs, self.n_pred, self.n_joints, 3)
        # pose_pred_rel = self.get_rel_pose(pose_pred_outp)  # Normalize the predicted pose.
        pose_pred_rel = pose_pred_outp

        # Add trajectory prediction to the relative pose.
        pose_pred = pose_pred_rel + traj_pred_3d_nj  # bs * np * nj * 3
        return pose_pred

    def prepare_pose_raw(self, pose, traj):
        # Predicted 3d trajectories in shape bs * np * 3.
        pos_root_last = pose[:, -1, self.root_idx, :]  # bs * 3
        pos_root_last_np = pos_root_last.unsqueeze(1).repeat((1, self.n_pred, 1))  # bs * np * 3
        traj_pred_3d = pos_root_last_np
        traj_pred_3d[:, :, :2] = traj
        traj_pred_3d_nj = traj_pred_3d.unsqueeze(2).repeat((1, 1, self.n_joints, 1))  # bs * np * nj * 3

        # Extract relative pose by subtracting the root joint.
        pose_rel = self.get_rel_pose(pose)  # bs * nf * nj * 3

        # Extend the last pose to the future and use as the raw pose.
        pose_last = pose_rel[:, -1, :, :]  # bs * nj * 3
        pose_last_nf = pose_last.unsqueeze(1).repeat((1, self.n_pred, 1, 1))  # bs * np * nj * 3
        return traj_pred_3d_nj, pose_last_nf

    def forward(self, poses_inp):
        # pred_info = {'pose': pose_pred, 'traj_map': traj_map_pred, 'traj': traj_raw}
        pred_info = {}
        pose = poses_inp['pose']['pose'].to(self.device)
        occ_map = poses_inp['env']['map'].to(self.device).float()
        traj_map_hist = poses_inp['pose']['traj_map'].to(self.device).float()

        if self.test_mode:
            traj_pred, traj_map_pred = self.pred_pathnet(pose, occ_map, traj_map_hist)
            if self.traj_interp:
                traj_pred_x, traj_pred_y = traj_pred[:, :, 0], traj_pred[:, :, 1]
                traj_pred_x_new, traj_pred_y_new = [], []
                for traj_x, traj_y in zip(traj_pred_x, traj_pred_y):
                    traj_x = traj_x.detach().cpu().numpy()
                    traj_y = traj_y.detach().cpu().numpy()
                    new_x = np.linspace(np.min(traj_x), np.max(traj_x), num=np.size(traj_x))
                    coefs = np.polyfit(traj_x, traj_y, 2)
                    new_y = np.polyval(coefs, new_x)
                    traj_pred_x_new.append(new_x)
                    traj_pred_y_new.append(new_y)
                traj_pred_x_new = torch.from_numpy(np.stack(traj_pred_x_new))
                traj_pred_y_new = torch.from_numpy(np.stack(traj_pred_y_new))
                traj_pred_interp = torch.stack((traj_pred_x_new, traj_pred_y_new), dim=-1)
                traj_pred_interp = traj_pred_interp.to(self.device).float()
                pose_pred = self.pred_posenet(pose, traj_pred_interp)
                pred_info = {'pose': pose_pred, 'traj_map': traj_map_pred, 'traj': traj_pred_interp}
            else:
                pose_pred = self.pred_posenet(pose, traj_pred)
                pred_info = {'pose': pose_pred, 'traj_map': traj_map_pred, 'traj': traj_pred}
        else:
            traj_gt = poses_inp['pose']['pose_gt'][:, :, self.root_idx, :2].to(self.device)
            if self.train_pathnet:
                traj_pred, traj_map_pred = self.pred_pathnet(pose, occ_map, traj_map_hist)
                pred_info['traj'] = traj_pred.float()
                pred_info['traj_map'] = traj_map_pred.float()
            else:
                pred_info['traj'] = traj_gt.float()
                resol = 2 * self.max_dist_from_human / self.map_size
                map_shape = (traj_gt.shape[0], self.n_pred, self.map_size, self.map_size, 1)
                pred_info['traj_map'] = traj2map(traj_gt.float(), self.max_dist_from_human, resol, map_shape,
                                                 device=None, binary=True).float()
                # pred_info['traj_map'] = traj_map_pred.float()

            if self.train_posenet:
                pose_pred = self.pred_posenet(pose, traj_gt)
                pred_info['pose'] = pose_pred.float()
            else:
                traj_pred_3d_nj, pose_last_nf = self.prepare_pose_raw(pose, traj_gt)
                pose_pred_raw = traj_pred_3d_nj + pose_last_nf
                pred_info['pose'] = pose_pred_raw.float()

        pred = {'env': poses_inp['env'], 'pose': pred_info}
        return pred

    def load_model(self, pathnet_path, posenet_path,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.pathnet.load_state_dict(torch.load(pathnet_path, map_location=device))
        self.posenet.load_state_dict(torch.load(posenet_path, map_location=device))

    def save_model(self, pathnet_path, posenet_path):
        torch.save(self.pathnet.state_dict(), pathnet_path)
        torch.save(self.posenet.state_dict(), posenet_path)

    def get_rel_pose(self, pose):
        return get_relative_pose(pose, self.root_idx, self.n_joints)
