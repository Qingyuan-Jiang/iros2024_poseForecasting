from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from utils.gta_utils import map2traj
import torchvision
from utils.vis_utils import vis_skeleton_pcd
from utils.gta_utils import traj2map


class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        super(UNet, self).__init__()
        features = cfg['model_specs']['n_feats_unet']
        self.init_features, self.in_channels, self.out_channels = features, in_channels, out_channels
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder4 = PoseForecastFPSDF._block(features * 4, features * 8, name="enc4")
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 4, features * 8, name="bottleneck")

        # self.upconv4 = nn.ConvTranspose2d(
        #     features * 16, features * 8, kernel_size=2, stride=2
        # )
        # self.decoder4 = PoseForecastFPSDF._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2, padding=0
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.cfg = cfg
        self.n_hist = cfg['dataset_specs']['n_hist']
        self.n_pred = cfg['dataset_specs']['n_pred']
        self.n_joints = cfg['dataset_specs']['n_joints']
        self.n_feats_mlp = cfg['model_specs']['n_feats_unet_mlp']
        self.fc1 = nn.Linear(features * 8 * 5 * 5 + self.n_hist * 2, features * 8 * 16)
        self.fc2 = nn.Linear(features * 8 * 16, features * 8 * 16)
        self.fc3 = nn.Linear(features * 8 * 16, features * 8 * 5 * 5)

    def forward(self, x, poses):
        # Input x shape: batch_size, n_frames, 18, 17, 4
        bs, n_inp, h, w, chn = x.shape
        x = x.permute((0, 2, 3, 1, 4)).reshape((bs, h, w, -1))  # bs, h, w, chn * n_inp
        x = x.permute((0, 3, 1, 2))  # bs, chn * n_inp, h, w

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        # enc4 = self.encoder4(self.pool3(enc3))
        feat_neck = self.pool3(enc3)

        bottleneck = self.bottleneck(feat_neck)
        bottleneck_flatten = bottleneck.reshape((bs, -1))
        poses_flatten = poses.reshape((bs, -1))
        feat = torch.cat((bottleneck_flatten, poses_flatten), dim=1)
        feat = F.relu(self.fc1(feat))
        feat = F.relu(self.fc2(feat))
        feat = self.fc3(feat).reshape((bs, -1, 5, 5))

        # dec4 = self.upconv4(bottleneck)
        # dec4 = torch.cat((dec4, enc4), dim=1)
        # dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(feat)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = self.conv(dec1)
        sig = nn.Sigmoid()
        x = sig(out.reshape((bs, self.out_channels, -1)))
        x = x.reshape((bs, self.out_channels, h, w))
        x = x.reshape((bs, h, w, chn, -1)).permute((0, 4, 1, 2, 3))
        return x, bottleneck

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class PoseNet(torch.nn.Module):
    def __init__(self, in_feats, n_feats, out_feats):
        super(PoseNet, self).__init__()
        self.fc1 = nn.Linear(in_feats, n_feats)
        self.fc2 = nn.Linear(n_feats, n_feats * 2)
        self.fc3 = nn.Linear(n_feats * 2, n_feats)
        self.fc4 = nn.Linear(n_feats, out_feats)

    def forward(self, pose_raw):
        bs = pose_raw.shape[0]
        pose_f = pose_raw.reshape(bs, -1)
        x = F.relu(self.fc1(pose_f))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Path_MLP(torch.nn.Module):

    def __init__(self, cfg):
        super(Path_MLP, self).__init__()
        self.cfg = cfg
        self.n_hist = cfg['dataset_specs']['n_hist']
        self.n_pred = cfg['dataset_specs']['n_pred']
        self.n_joints = cfg['dataset_specs']['n_joints']
        self.n_feats_mlp = cfg['model_specs']['n_feats_mlp']
        self.test_mode = not (cfg['mode'] == 'train' and cfg['dataset_specs']['use_pose_gt'])
        self.train_pathnet = cfg['model_specs']['train_pathnet']
        self.train_posenet = cfg['model_specs']['train_posenet']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_feats_inp = (self.n_hist + self.n_pred) * self.n_joints * 3
        self.pathnet = UNet(1 + self.n_hist, self.n_pred, cfg)
        self.posenet = PoseNet(n_feats_inp, self.n_feats_mlp, self.n_pred * self.n_joints * 3)
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
        pose_pred_raw = traj_pred_3d_nj + pose_last_nf

        pose_rel = self.get_rel_pose(pose)  # bs * nf * nj * 3

        pose_raw = torch.cat((pose_rel, pose_last_nf), dim=1)  # bs * (nf + np) * nj * 3
        # pose_raw = torch.cat((pose, pose_pred_raw), dim=1)  # bs * (nf + np) * nj * 3

        # Go through MLP for pose prediction.
        traj_pred_outp = self.posenet(pose_raw)  # bs * np * nj * 3
        traj_pred_outp = traj_pred_outp.reshape(bs, self.n_pred, self.n_joints, 3)
        traj_pred_rel = self.get_rel_pose(traj_pred_outp)  # Normalize the predicted pose.
        # traj_pred_rel = traj_pred_outp

        # Add trajectory prediction to the relative pose.
        pose_pred = traj_pred_rel + traj_pred_3d_nj  # bs * np * nj * 3
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

    def load_model(self, pathnet_path, posenet_path):
        self.pathnet.load_state_dict(torch.load(pathnet_path))
        self.posenet.load_state_dict(torch.load(posenet_path))

    def save_model(self, pathnet_path, posenet_path):
        torch.save(self.pathnet.state_dict(), pathnet_path)
        torch.save(self.posenet.state_dict(), posenet_path)

    def get_rel_pose(self, pose):
        pos_root = pose[:, :, self.root_idx, :]  # bs * nf * 3
        pos_root_nj = pos_root.unsqueeze(2).repeat((1, 1, self.n_joints, 1))  # bs * nf * nj * 3
        pose_rel = pose - pos_root_nj  # Relative poses. bs * nf * nj * 3
        return pose_rel


class PathNet_v2(nn.Module):

    def __init__(self, cfg):
        super(PathNet_v2, self).__init__()
        self.cfg = cfg
        self.n_hist = cfg['dataset_specs']['n_hist']
        self.n_pred = cfg['dataset_specs']['n_pred']
        self.n_joints = cfg['dataset_specs']['n_joints']
        self.n_feats_mlp = cfg['model_specs']['n_feats_mlp']
        self.fc1 = torch.nn.Linear(self.n_feats_mlp + self.n_hist * 2, 2 * self.n_feats_mlp)  # 5*5 from image dimension
        self.fc2 = nn.Linear(self.n_feats_mlp * 2, self.n_feats_mlp)
        self.fc3 = torch.nn.Linear(self.n_feats_mlp, self.n_hist * 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feat_extractor = torchvision.models.resnet18(pretrained=False)
        num_features = self.feat_extractor.fc.in_features
        self.feat_extractor.fc = nn.Linear(num_features, self.n_feats_mlp)

        self.root_idx = cfg['dataset_specs']['root_idx']

    def forward(self, poses_inp):
        pose = poses_inp['pose']['pose'].to(self.device)
        occ_map = poses_inp['env']['map'].to(self.device).float()
        bs = pose.shape[0]

        # Predict trajectory map first and then predict pose.
        occ_map_inp = occ_map[:, :, :, :, 0].repeat(1, 3, 1, 1)
        occ_map_feat = self.feat_extractor(occ_map_inp)
        traj_hist = pose[:, :, self.root_idx, :2]
        hist_feat = torch.cat((traj_hist.reshape(bs, -1), occ_map_feat), dim=1)
        x = F.relu(self.fc1(hist_feat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        traj_pred = x.reshape(bs, self.n_pred, 2)

        # # Predict poses.
        # pose_f = pose.reshape(bs, -1)
        pose_last = pose[:, -1, :, :]
        pos_root = pose_last[:, self.root_idx, :].unsqueeze(1).repeat((1, self.n_joints, 1))
        pos_root_nf = pos_root.unsqueeze(1).repeat((1, self.n_pred, 1, 1))
        pose_rel = pose_last - pos_root
        pose_rel_nf = pose_rel.unsqueeze(1).repeat((1, self.n_pred, 1, 1))

        traj_pred_3d = pos_root_nf
        traj_pred_3d[:, :, :, :2] = traj_pred.unsqueeze(2).repeat((1, 1, self.n_joints, 1))
        pose_pred_raw = pose_rel_nf + traj_pred_3d
        # x_cat = torch.cat((pose_f, traj.reshape(bs, -1)), dim=1)
        # x = F.relu(self.fc1(x_cat))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = x.reshape(bs, -1, self.n_joints, 3)

        # pose_pred = torch.rand(pose.shape).to(self.device)
        # pose_pred[:, :, self.root_idx, :2] = traj_pred
        pred_info = {'pose': pose_pred_raw, 'traj': traj_pred}
        pred = {'env': poses_inp['env'], 'pose': pred_info}
        return pred
