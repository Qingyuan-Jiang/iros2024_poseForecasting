import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_gta import DatasetGTA
from models.motion_pred import get_model
from utils.pcd_utils import np2pcd_xyzrgb, draw_frame
from utils.vis_utils import vis_skeleton_pcd, pad_with

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def displacement_error(pred_position, gt_position):
    """Compute the average and final displacement errors given a sequence of positions."""
    batch = gt_position.shape[0]
    ade = 0
    fde = 0
    for b in range(batch):
        gt_pos = gt_position[b]
        pred_pos = pred_position[b]
        # Initial position
        initial_position = gt_pos[0]
        final_position = gt_pos[-1]

        # Calculate total displacement
        total_displacement = 0
        for i in range(len(gt_pos)):
            dx = pred_pos[i][0] - gt_pos[i][0]
            dy = pred_pos[i][1] - gt_pos[i][1]
            total_displacement += (dx ** 2 + dy ** 2) ** 0.5

        # Average displacement
        average_displacement = total_displacement / len(gt_pos)

        # Final displacement error
        dx_final = pred_pos[-1][0] - final_position[0]
        dy_final = pred_pos[-1][1] - final_position[1]
        final_displacement_error = (dx_final ** 2 + dy_final ** 2) ** 0.5
        ade += average_displacement
        fde += final_displacement_error

    return ade / batch, fde / batch


@torch.no_grad()
def test(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cfg_name = 'cfg/%s.yml' % args.cfg
    cfg = yaml.safe_load(open(cfg_name, 'r'))
    cfg['mode'] = 'test' if not args.test_pose else 'train'
    cfg['dataset_specs']['use_env_pcd'] = args.vis_skeleton or cfg['dataset_specs']['use_env_pcd']
    cfg['model_specs']['traj_interp'] = False

    """model"""
    model = get_model(cfg)
    epoch = args.epoch
    weights_dir = os.getcwd() + f'/results/{args.cfg}_{args.exp}/models/'
    os.makedirs(weights_dir, exist_ok=True)
    pathnet_path = weights_dir + 'pathnet_{:03d}.pt'.format(epoch)
    posenet_path = weights_dir + 'posenet_{:03d}.pt'.format(epoch)
    model.load_model(pathnet_path, posenet_path, device)
    # model.load_state_dict(torch.load(cp_path))
    model.float().to(device).eval()
    print('loading model from checkpoint: %s' % pathnet_path)
    print('loading model from checkpoint: %s' % posenet_path)

    root_idx = cfg['dataset_specs']['root_idx']
    # root_joint_idx = 14
    # root_idx = [root_joint_idx * 3, root_joint_idx * 3 + 1, root_joint_idx * 3 + 2]

    bs = args.batch_size
    dataset = DatasetGTA('test', cfg)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=6)
    # dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)

    n_hist = cfg['dataset_specs']['n_hist']
    n_pred = cfg['dataset_specs']['n_pred']
    n_joints = cfg['dataset_specs']['n_joints']
    pose_err_list = []
    traj_err_list = []
    ade_list = []
    fde_list = []

    vis_dir = os.path.join(os.path.dirname(__file__), f'results/{args.cfg}_{args.exp}/vis/')
    os.makedirs(vis_dir, exist_ok=True)

    vis_traj_map = args.vis_traj_map
    vis_traj_map = vis_traj_map and cfg['loss']['use_traj_map']
    vis_skeleton = args.vis_skeleton

    size = 40
    map_mask = np.full((size, size), -1)
    center = (size // 2, size // 2)
    max_distance = np.sqrt(2 * (size // 2) ** 2)

    # Iterate over possible points in FoV
    for y in range(size):
        for x in range(size):
            # Determine the distance and angle from the center
            dx = x - center[1]
            dy = y - center[0]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance == 0:
                angle = 0  # Camera position
            else:
                angle = np.degrees(np.arcsin(abs(dx) / distance))

            # Check if the angle is within the 45 degrees from center
            if angle >= 43.5:
                map_mask[y, x] = 1

    map_mask[:, 0:20] = 1
    rows, cols = np.where(map_mask == -1)

    with torch.no_grad():
        for poses_inp, poses_label in tqdm(dataloader):
            bs = poses_inp['pose']['pose'].shape[0]
            if args.map == 'partial':
                poses_inp['env']['map'][:, :, rows, cols, :] = -1
            if args.map == 'unknown':
                poses_inp['env']['map'][:, :, :, :] = -1
            if args.map == 'def':
                pass
            poses_pred = model(poses_inp)

            """mpjpe error"""
            pose_hist = poses_inp['pose']['pose'].to(device)
            pose_pred = poses_pred['pose']['pose'].to(device)
            pose_label = poses_label['pose']['pose'].to(device)
            pose_idx = np.setdiff1d(np.arange(n_joints), root_idx)
            pose_pred_rel = pose_pred[:, :, pose_idx, :] - pose_pred[:, :, root_idx, :].unsqueeze(2).repeat(1, 1,
                                                                                                            n_joints - 1,
                                                                                                            1)
            pose_label_rel = pose_label[:, :, pose_idx, :] - pose_label[:, :, root_idx, :].unsqueeze(2).repeat(1, 1,
                                                                                                               n_joints - 1,
                                                                                                               1)
            pose_diff = pose_pred_rel - pose_label_rel
            pose_error = pose_diff.norm(dim=-1).mean(dim=-1).detach().cpu().numpy()
            pose_err_list.append(pose_error)

            traj_pred = pose_pred[:, :, root_idx, :2]
            traj_label = pose_label[:, :, root_idx, :2]
            path_error = torch.norm(traj_pred - traj_label, dim=-1).detach().cpu().numpy()
            traj_err_list.append(path_error)

            """ADE & FDE"""
            ade, fde = displacement_error(traj_pred, traj_label)
            ade_list.append(ade.detach().cpu().numpy())
            fde_list.append(fde.detach().cpu().numpy())

            # Visualize the predicted poses with the environment point cloud.
            for i in range(bs):
                if vis_traj_map:
                    map_env = poses_inp['env']['map'][i, 0, :, :, 0]
                    map_inp_i = poses_inp['pose']['traj_map'][i, :, :, :, 0]
                    map_pred_i = poses_pred['pose']['traj_map'][i, :, :, :, 0]
                    map_gt_i = poses_label['pose']['traj_map'][i, :, :, :, 0]
                    img_name = poses_inp['env']['item_key'][i]
                    map_inp_i_sum = map_inp_i.sum(dim=0).detach().cpu().numpy()
                    map_pred_i_sum = map_pred_i.sum(dim=0).detach().cpu().numpy()
                    map_gt_i_sum = map_gt_i.sum(dim=0).detach().cpu().numpy()

                    map_env_vis = np.pad(map_env, 2, pad_with, padder=1)
                    map_inp_i_sum_vis = np.pad(map_inp_i_sum, 2, pad_with, padder=1)
                    map_pred_i_sum_vis = np.pad(map_pred_i_sum, 2, pad_with, padder=1)
                    map_gt_i_sum_vis = np.pad(map_gt_i_sum, 2, pad_with, padder=1)

                    map_vis = np.hstack((map_env_vis, map_inp_i_sum_vis, map_pred_i_sum_vis, map_gt_i_sum_vis))
                    plt.imsave(vis_dir + f'traj_map_{img_name}.png', map_vis)
                    print(f'Saved {vis_dir + f"traj_map_{img_name}.png"}')

                if vis_skeleton:
                    points_local, colors_local = poses_inp['env']['pcd.points'][i], poses_inp['env']['pcd.colors'][i]
                    points_local, colors_local = points_local.detach().cpu().numpy(), colors_local.detach().cpu().numpy()

                    pose_hist_np = pose_hist[i].detach().cpu().numpy()
                    pose_pred_np = pose_pred[i].detach().cpu().numpy()
                    pose_label_np = pose_label[i].detach().cpu().numpy()

                    dataset_name = cfg['dataset']
                    vis_stride = 3
                    n_hist_vis, n_pred_vis = n_hist // vis_stride, n_pred // vis_stride
                    pose_hist_np_vis = pose_hist_np[::vis_stride]
                    pose_pred_np_vis = pose_pred_np[::vis_stride]
                    pose_label_np_vis = pose_label_np[::vis_stride]
                    line_set_hist = vis_skeleton_pcd(n_hist_vis, n_joints, pose_hist_np_vis, color='g', dataset=dataset_name)
                    line_set_pred = vis_skeleton_pcd(n_pred_vis, n_joints, pose_pred_np_vis, color='r', dataset=dataset_name)
                    line_set_label = vis_skeleton_pcd(n_pred_vis, n_joints, pose_label_np_vis, color='b', dataset=dataset_name)
                    line_sets = line_set_hist + line_set_pred + line_set_label

                    pcd = np2pcd_xyzrgb(np.hstack((points_local, colors_local)))
                    box = o3d.geometry.AxisAlignedBoundingBox([-10, -0.5, -3], [10, 10, 2])
                    # estimate radius for rolling ball
                    pcd_crop = pcd.crop(box)
                    o3d.visualization.draw_geometries([pcd_crop, draw_frame()] + line_sets)

    pose_err_list = np.concatenate(pose_err_list, axis=0)
    traj_err_list = np.concatenate(traj_err_list, axis=0)
    ade_list = np.array(ade_list)
    fde_list = np.array(fde_list)
    pose_mean2s_err = pose_err_list[:, :9].mean(axis=-1).mean(axis=-1)
    pose_mean3s_err = pose_err_list[:, :].mean(axis=-1).mean(axis=-1)
    pose_05s_err = (pose_err_list[:, 1].mean(axis=-1) + pose_err_list[:, 2].mean(axis=-1)) / 2
    pose_1s_err = pose_err_list[:, 4].mean(axis=-1)
    pose_15s_err = (pose_err_list[:, 7].mean(axis=-1) + pose_err_list[:, 8].mean(axis=-1)) / 2
    pose_2s_err = pose_err_list[:, 9].mean(axis=-1)
    pose_3s_err = pose_err_list[:, 14].mean(axis=-1)
    path_mean_err = traj_err_list[:, :9].mean(axis=-1).mean(axis=-1)
    path_05s_err = (traj_err_list[:, 1].mean(axis=-1) + traj_err_list[:, 2].mean(axis=-1)) / 2
    path_1s_err = traj_err_list[:, 4].mean(axis=-1)
    path_15s_err = (traj_err_list[:, 7].mean(axis=-1) + traj_err_list[:, 8].mean(axis=-1)) / 2
    path_2s_err = traj_err_list[:, 9].mean(axis=-1)
    path_3s_err = traj_err_list[:, 14].mean(axis=-1)
    print("Average path error: ", np.mean(traj_err_list, axis=0) * 1000)
    print("Average path 0.5s error: {:0.1f}".format(np.mean(path_05s_err) * 1000))
    print("Average path 1s error: {:0.1f}".format(np.mean(path_1s_err) * 1000))
    print("Average path 1.5s error: {:0.1f}".format(np.mean(path_15s_err) * 1000))
    print("Average path 2s error: {:0.1f}".format(np.mean(path_2s_err) * 1000))
    print("Average path 3s error: ", np.mean(path_3s_err) * 1000)
    print("Average path mean (2s) error: {:0.1f}".format(np.mean(path_mean_err) * 1000))
    print("Average path mean (3s) error: {:0.1f}".format(np.mean(ade_list) * 1000))
    print("Average MPJPE error: ", np.mean(pose_err_list, axis=0) * 1000)
    print("Average MPJPE 0.5s error: {:0.1f}".format(np.mean(pose_05s_err) * 1000))
    print("Average MPJPE 1s error: {:0.1f}".format(np.mean(pose_1s_err) * 1000))
    print("Average MPJPE 1.5s error: {:0.1f}".format(np.mean(pose_15s_err) * 1000))
    print("Average MPJPE 2s error: {:0.1f}".format(np.mean(pose_2s_err) * 1000))
    print("Average MPJPE 3s error: ", np.mean(pose_3s_err) * 1000)
    print("Average MPJPE mean (2s) error: {:0.1f}".format(np.mean(pose_mean2s_err) * 1000))
    print("Average MPJPE mean (3s) error: {:0.1f}".format(np.mean(pose_mean3s_err) * 1000))
    np.save(vis_dir + 'pose_mean_err.npy', pose_mean2s_err)
    np.save(vis_dir + 'pose_mean_err.npy', pose_mean3s_err)
    np.save(vis_dir + 'pose_1s_err.npy', pose_1s_err)
    np.save(vis_dir + 'pose_15s_err.npy', pose_15s_err)
    np.save(vis_dir + 'pose_2s_err.npy', pose_2s_err)
    np.save(vis_dir + 'pose_3s_err.npy', pose_3s_err)
    np.save(vis_dir + 'path_mean_err.npy', path_mean_err)
    np.save(vis_dir + 'path_1s_err.npy', path_1s_err)
    np.save(vis_dir + 'path_15s_err.npy', path_15s_err)
    np.save(vis_dir + 'path_2s_err.npy', path_2s_err)
    np.save(vis_dir + 'path_3s_err.npy', path_3s_err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='real_Path_GRU')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--test_pose', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=3600)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vis_traj_map', action='store_true', default=False)
    parser.add_argument('--vis_skeleton', action='store_true', default=False)
    parser.add_argument('--exp', default='exp', type=str, help='experiment name')
    parser.add_argument('--map', default='def', type=str, help='map visibility')
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    test(args)
