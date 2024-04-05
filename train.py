import os
import shutil

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataset_gta import DatasetGTA
from models.motion_pred import get_model
from utils.config import Config


def loss_traj_map_func(pred, label, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loss_fn = torch.nn.KLDivLoss(reduction="batchmean").to(device)
    loss_fn = torch.nn.BCELoss(weight=torch.tensor(cfg['loss']['weight_bce'])).to(device)
    # loss_fn = torch.nn.BCELoss().to(device)
    traj_map_pred = pred['pose']['traj_map'].to(device)
    traj_map_label = label['pose']['traj_map'].to(device)
    loss_traj_map = loss_fn(traj_map_pred, traj_map_label)
    return loss_traj_map


def loss_collision_func(pred, label, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traj_map_pred = pred['pose']['traj_map'].to(device)
    map = label['env']['map'].to(device)
    bs, nf, h, w, chn = traj_map_pred.shape
    map_ext = map.repeat(1, nf, 1, 1, 1)
    map_collision = torch.abs(map_ext)
    loss_collision = (map_collision * traj_map_pred).mean()
    return loss_collision


def loss_func(pred, label, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_idx = cfg['dataset_specs']['root_idx']
    loss_fn = torch.nn.MSELoss().to(device)

    # Loss components.
    use_pose = cfg['loss']['use_pose'] if 'use_pose' in cfg['loss'].keys() else True
    use_pose = use_pose and cfg['model_specs']['train_posenet'] if 'train_posenet' in cfg['model_specs'].keys() else use_pose
    use_traj = cfg['loss']['use_traj'] if 'use_traj' in cfg['loss'].keys() else False
    use_final_pos = cfg['loss']['use_final_pos'] if 'use_final_pos' in cfg['loss'].keys() else False
    use_traj_map = cfg['loss']['use_traj_map'] if 'use_traj_map' in cfg['loss'].keys() else False
    use_collision = cfg['loss']['use_collision'] if 'use_collision' in cfg['loss'].keys() else False

    loss = 0

    # Loss for skeleton poses for each frame.
    if use_pose:
        pred_pose, label_pose = pred['pose']['pose'].to(device), label['pose']['pose'].to(device)
        loss_pose = loss_fn(pred_pose, label_pose)
        loss = loss + loss_pose * cfg['loss']['weight_pose']

    # Loss of trajectory.
    if use_traj:
        traj_pred = pred['pose']['traj'].to(device)
        pose_label = label['pose']['pose'].to(device)
        traj_label = pose_label[:, :, root_idx, :2]
        loss_traj = loss_fn(traj_pred, traj_label) if use_traj else 0
        loss = loss + loss_traj * cfg['loss']['weight_traj']

    # Loss of position for the last frame.
    if use_final_pos:
        pred_traj = pred['pose']['traj'].to(device)
        label_pose = label['pose']['pose'].to(device)
        label_traj = label_pose[:, :, root_idx, :2]
        pos_pred, pos_label = pred_traj[:, -1, :2], label_traj[:, -1, :2]
        loss_pos = loss_fn(pos_pred, pos_label) if use_final_pos else 0
        loss = loss + loss_pos * cfg['loss']['weight_final_pos']

    # Loss of trajectory map.
    if use_traj_map:
        loss_traj_map = loss_traj_map_func(pred, label, cfg)
        loss = loss + loss_traj_map * cfg['loss']['weight_traj_map']

    # Loss of collision.
    if use_collision:
        loss_collision = loss_collision_func(pred, label, cfg)
        loss = loss + loss_collision * cfg['loss']['weight_collision']

    return loss


def test_model(net, testloader, loss_fn, cfg):
    loss_list = []
    net.eval()
    with torch.no_grad():
        for poses_inp, poses_label in testloader:
            poses_outp = net(poses_inp)
            loss = loss_fn(poses_outp, poses_label, cfg)
            loss_list.append(loss.detach().cpu().numpy())
    net.train()
    return np.average(loss_list)


def train_one_epoch(net, dataloader, optimizer, loss_fn, cfg):
    loss_list = []

    for j, (poses_inp, poses_label) in enumerate(tqdm(dataloader)):
        # print("Memory allocated:", torch.cuda.memory_allocated() / 1024 ** 2, "MB")
        optimizer.zero_grad()
        poses_outp = net(poses_inp)

        loss = loss_fn(poses_outp, poses_label, cfg)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.detach().cpu())
    return np.average(loss_list)


def main(args):
    # Load config.
    Config(f'{args.cfg}', test=args.test)
    cfg_name = 'cfg/%s.yml' % args.cfg
    cfg = yaml.safe_load(open(cfg_name, 'r'))
    cfg['model_specs']['train_pathnet'] = args.train_pathnet and cfg['model_specs']['train_pathnet']
    cfg['model_specs']['train_posenet'] = args.train_posenet and cfg['model_specs']['train_posenet']
    cfg['loss']['use_traj'] = args.train_pathnet and cfg['loss']['use_traj']
    cfg['loss']['use_traj_map'] = args.train_pathnet and cfg['loss']['use_traj_map']
    cfg['loss']['use_collision'] = args.train_pathnet and cfg['loss']['use_collision']
    cfg['loss']['use_final_pos'] = args.train_pathnet and cfg['loss']['use_final_pos']
    cfg['loss']['use_pose'] = args.train_posenet and cfg['loss']['use_pose']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pose_predictor = get_model(cfg).to(dtype=torch.float32, device=device)

    epoch = args.epoch
    start_epoch = args.start_epoch
    save_freq = cfg['save_freq']
    batch_size = args.batch_size
    lr = cfg['lr'] if args.lr == 0 else args.lr
    lr_step_size = cfg['lr_step_size']
    lr_gamma = cfg['lr_gamma']

    dataset = DatasetGTA('train', cfg)
    n_train_samp = int(0.8 * len(dataset))
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [n_train_samp, len(dataset) - n_train_samp])
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=6)
    # dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    params = [p for p in pose_predictor.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    weights_dir = os.getcwd() + f'/results/{args.cfg}_{args.exp}/models/'
    os.makedirs(weights_dir, exist_ok=True)
    pathnet_path = weights_dir + 'pathnet_{:03d}.pt'.format(start_epoch)
    posenet_path = weights_dir + 'posenet_{:03d}.pt'.format(start_epoch)
    if start_epoch > 0 and os.path.exists(pathnet_path) and os.path.exists(posenet_path):
        pose_predictor.load_model(pathnet_path, posenet_path, device)

    log_dir = os.getcwd() + f'/results/{args.cfg}_{args.exp}/tb'
    os.makedirs(log_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir)
    shutil.copyfile(cfg_name, '%s/%s.yml' % (f'results/{args.cfg}_{args.exp}', f'{args.cfg}_{args.exp}'))

    for i in range(start_epoch, start_epoch + epoch):

        loss_train = train_one_epoch(pose_predictor, dataloader_train, optimizer, loss_func, cfg)
        loss_test = test_model(pose_predictor, dataloader_test, loss_func, cfg)

        scheduler.step()

        if (i + 1) % save_freq == 0:
            pathnet_path_save = weights_dir + 'pathnet_{:03d}.pt'.format(i + 1)
            posenet_path_save = weights_dir + 'posenet_{:03d}.pt'.format(i + 1)
            pose_predictor.save_model(pathnet_path_save, posenet_path_save)

        if (i + 1) % 1 == 0:
            print("Epoch {}. Avg. train loss {}. test loss {}".format(i, loss_train, loss_test))
        #     writer.add_scalar('Loss/train', loss_train, i)
        #     writer.add_scalar('Loss/test', loss_test, i)

            # with torch.no_grad():
            #     inps, labels = next(iter(dataloader_test))  # bs, n_frames_inp, 21, 21, 18
            #     poses_outp = pose_predictor(inps.to(device))
            #     grid = torchvision.utils.make_grid(poses_outp[0, :, :, :, 0].unsqueeze(1), nrow=n_frames_inp)
            #     writer.add_image("images {}".format(i), grid)
            #     writer.add_graph(pose_predictor, inps.to(device))

    # writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='gta_path_GRU')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--train_pathnet', action='store_true', default=False)
    parser.add_argument('--train_posenet', action='store_true', default=False)
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--epoch', default=1200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0, type=float, help='initial learning rate')
    parser.add_argument('--exp', default='exp_00', type=str, help='experiment name')

    args = parser.parse_args()

    main(args)
