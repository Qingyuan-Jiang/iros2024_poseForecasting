"""
GTA-IM Dataset
"""

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import torch

LIMBS = [
    (0, 1),  # head_center -> neck
    (1, 2),  # neck -> right_clavicle
    (2, 3),  # right_clavicle -> right_shoulder
    (3, 4),  # right_shoulder -> right_elbow
    (4, 5),  # right_elbow -> right_wrist
    (1, 6),  # neck -> left_clavicle
    (6, 7),  # left_clavicle -> left_shoulder
    (7, 8),  # left_shoulder -> left_elbow
    (8, 9),  # left_elbow -> left_wrist
    (1, 10),  # neck -> spine0
    (10, 11),  # spine0 -> spine1
    (11, 12),  # spine1 -> spine2
    (12, 13),  # spine2 -> spine3
    (13, 14),  # spine3 -> pos_spine
    (14, 15),  # pos_spine -> right_hip
    (15, 16),  # right_hip -> right_knee
    (16, 17),  # right_knee -> right_ankle
    (14, 18),  # pos_spine -> left_hip
    (18, 19),  # left_hip -> left_knee
    (19, 20),  # left_knee -> left_ankle
]

LIMBS_prox = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21)
]


####################
# camera utils.
def get_focal_length(cam_near_clip, cam_field_of_view):
    near_clip_height = (
            2 * cam_near_clip * np.tan(cam_field_of_view / 2.0 * (np.pi / 180.0))
    )

    # camera focal length
    return 1080.0 / near_clip_height * cam_near_clip


def get_2d_from_3d(
        vertex,
        cam_coords,
        cam_rotation,
        cam_near_clip,
        cam_field_of_view,
        WIDTH=1920,
        HEIGHT=1080,
):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    WORLD_UP = np.array([0.0, 0.0, 1.0], 'double')
    WORLD_EAST = np.array([1.0, 0.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    clip_plane_center = cam_coords + cam_near_clip * cam_dir
    camera_center = -cam_near_clip * cam_dir
    near_clip_height = (
            2 * cam_near_clip * np.tan(cam_field_of_view / 2.0 * (np.pi / 180.0))
    )
    near_clip_width = near_clip_height * WIDTH / HEIGHT

    cam_up = rotate(WORLD_UP, theta)
    cam_east = rotate(WORLD_EAST, theta)
    near_clip_to_target = vertex - clip_plane_center

    camera_to_target = near_clip_to_target - camera_center

    camera_to_target_unit_vector = camera_to_target * (
            1.0 / np.linalg.norm(camera_to_target)
    )

    view_plane_dist = cam_near_clip / cam_dir.dot(camera_to_target_unit_vector)

    new_origin = (
            clip_plane_center
            + (near_clip_height / 2.0) * cam_up
            - (near_clip_width / 2.0) * cam_east
    )

    view_plane_point = (
                               view_plane_dist * camera_to_target_unit_vector
                       ) + camera_center
    view_plane_point = (view_plane_point + clip_plane_center) - new_origin
    viewPlaneX = view_plane_point.dot(cam_east)
    viewPlaneZ = view_plane_point.dot(cam_up)
    screenX = viewPlaneX / near_clip_width
    screenY = -viewPlaneZ / near_clip_height

    # screenX and screenY between (0, 1)
    ret = np.array([screenX, screenY], 'double')
    return ret


def screen_x_to_view_plane(x, cam_near_clip, cam_field_of_view):
    # x in (0, 1)
    near_clip_height = (
            2 * cam_near_clip * np.tan(cam_field_of_view / 2.0 * (np.pi / 180.0))
    )
    near_clip_width = near_clip_height * 1920.0 / 1080.0

    viewPlaneX = x * near_clip_width

    return viewPlaneX


def generate_id_map(map_path):
    id_map = cv2.imread(map_path, -1)
    h, w, _ = id_map.shape
    id_map = np.concatenate(
        (id_map, np.zeros((h, w, 1), dtype=np.uint8)), axis=2
    )
    id_map.dtype = np.uint32
    return id_map


def get_depth(
        vertex, cam_coords, cam_rotation, cam_near_clip, cam_field_of_view
):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    clip_plane_center = cam_coords + cam_near_clip * cam_dir
    camera_center = -cam_near_clip * cam_dir

    near_clip_to_target = vertex - clip_plane_center

    camera_to_target = near_clip_to_target - camera_center
    camera_to_target_unit_vector = camera_to_target * (
            1.0 / np.linalg.norm(camera_to_target)
    )

    depth = np.linalg.norm(camera_to_target) * cam_dir.dot(
        camera_to_target_unit_vector
    )
    depth = depth - cam_near_clip

    return depth


def get_kitti_format_camera_coords(
        vertex, cam_coords, cam_rotation, cam_near_clip
):
    cam_dir, cam_up, cam_east = get_cam_dir_vecs(cam_rotation)

    clip_plane_center = cam_coords + cam_near_clip * cam_dir

    camera_center = -cam_near_clip * cam_dir

    near_clip_to_target = vertex - clip_plane_center

    camera_to_target = near_clip_to_target - camera_center
    camera_to_target_unit_vector = camera_to_target * (
            1.0 / np.linalg.norm(camera_to_target)
    )

    z = np.linalg.norm(camera_to_target) * cam_dir.dot(
        camera_to_target_unit_vector
    )
    y = -np.linalg.norm(camera_to_target) * cam_up.dot(
        camera_to_target_unit_vector
    )
    x = np.linalg.norm(camera_to_target) * cam_east.dot(
        camera_to_target_unit_vector
    )

    return np.array([x, y, z])


def get_cam_dir_vecs(cam_rotation):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    WORLD_UP = np.array([0.0, 0.0, 1.0], 'double')
    WORLD_EAST = np.array([1.0, 0.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    cam_up = rotate(WORLD_UP, theta)
    cam_east = rotate(WORLD_EAST, theta)

    return cam_dir, cam_up, cam_east


def is_before_clip_plane(
        vertex,
        cam_coords,
        cam_rotation,
        cam_near_clip,
        cam_field_of_view,
        WIDTH=1920,
        HEIGHT=2080,
):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    clip_plane_center = cam_coords + cam_near_clip * cam_dir
    camera_center = -cam_near_clip * cam_dir

    near_clip_to_target = vertex - clip_plane_center

    camera_to_target = near_clip_to_target - camera_center

    camera_to_target_unit_vector = camera_to_target * (
            1.0 / np.linalg.norm(camera_to_target)
    )

    if cam_dir.dot(camera_to_target_unit_vector) > 0:
        return True
    else:
        return False


def get_clip_center_and_dir(cam_coords, cam_rotation, cam_near_clip):
    WORLD_NORTH = np.array([0.0, 1.0, 0.0], 'double')
    theta = (np.pi / 180.0) * cam_rotation
    cam_dir = rotate(WORLD_NORTH, theta)
    clip_plane_center = cam_coords + cam_near_clip * cam_dir
    return clip_plane_center, cam_dir


def rotate(a, t):
    d = np.zeros(3, 'double')
    d[0] = np.cos(t[2]) * (
            np.cos(t[1]) * a[0]
            + np.sin(t[1]) * (np.sin(t[0]) * a[1] + np.cos(t[0]) * a[2])
    ) - (np.sin(t[2]) * (np.cos(t[0]) * a[1] - np.sin(t[0]) * a[2]))
    d[1] = np.sin(t[2]) * (
            np.cos(t[1]) * a[0]
            + np.sin(t[1]) * (np.sin(t[0]) * a[1] + np.cos(t[0]) * a[2])
    ) + (np.cos(t[2]) * (np.cos(t[0]) * a[1] - np.sin(t[0]) * a[2]))
    d[2] = -np.sin(t[1]) * a[0] + np.cos(t[1]) * (
            np.sin(t[0]) * a[1] + np.cos(t[0]) * a[2]
    )
    return d


def get_intersect_point(center_pt, cam_dir, vertex1, vertex2):
    c1 = center_pt[0]
    c2 = center_pt[1]
    c3 = center_pt[2]
    a1 = cam_dir[0]
    a2 = cam_dir[1]
    a3 = cam_dir[2]
    x1 = vertex1[0]
    y1 = vertex1[1]
    z1 = vertex1[2]
    x2 = vertex2[0]
    y2 = vertex2[1]
    z2 = vertex2[2]

    k_up = a1 * (x1 - c1) + a2 * (y1 - c2) + a3 * (z1 - c3)
    k_down = a1 * (x1 - x2) + a2 * (y1 - y2) + a3 * (z1 - z2)
    k = k_up / k_down
    inter_point = (1 - k) * vertex1 + k * vertex2

    return inter_point


####################
# dataset utils.
def is_inside(x, y):
    return x >= 0 and x <= 1 and y >= 0 and y <= 1


def get_cut_edge(x1, y1, x2, y2):
    # (x1, y1) inside while (x2, y2) outside
    dx = x2 - x1
    dy = y2 - y1
    ratio_pool = []
    if x2 < 0:
        ratio = (x1 - 0) / (x1 - x2)
        ratio_pool.append(ratio)
    if x2 > 1:
        ratio = (1 - x1) / (x2 - x1)
        ratio_pool.append(ratio)
    if y2 < 0:
        ratio = (y1 - 0) / (y1 - y2)
        ratio_pool.append(ratio)
    if y2 > 1:
        ratio = (1 - y1) / (y2 - y1)
        ratio_pool.append(ratio)
    actual_ratio = min(ratio_pool)
    return x1 + actual_ratio * dx, y1 + actual_ratio * dy


def get_min_max_x_y_from_line(x1, y1, x2, y2):
    if is_inside(x1, y1) and is_inside(x2, y2):
        return min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
    if (not is_inside(x1, y1)) and (not is_inside(x2, y2)):
        return None, None, None, None
    if is_inside(x1, y1) and not is_inside(x2, y2):
        x2, y2 = get_cut_edge(x1, y1, x2, y2)
        return min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
    if is_inside(x2, y2) and not is_inside(x1, y1):
        x1, y1 = get_cut_edge(x2, y2, x1, y1)
        return min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)


def get_angle_in_2pi(unit_vec):
    theta = np.arccos(unit_vec[0])
    if unit_vec[1] > 0:
        return theta
    else:
        return 2 * np.pi - theta


####################
# math utils.
def vec_cos(a, b):
    prod = a.dot(b)
    prod = prod * 1.0 / np.linalg.norm(a) / np.linalg.norm(b)
    return prod


def compute_bbox_ratio(bbox2, bbox):
    # bbox2 is inside bbox
    s = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    s2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return s2 * 1.0 / s


def compute_iou(boxA, boxB):
    if (
            boxA[0] > boxB[2]
            or boxB[0] > boxA[2]
            or boxA[1] > boxB[3]
            or boxB[1] > boxA[3]
    ):
        return 0
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def project2dline(
        p1,
        p2,
        cam_coords,
        cam_rotation,
        cam_near_clip=0.15,
        cam_field_of_view=50.0,
        WIDTH=1920,
        HEIGHT=2080,
):
    before1 = is_before_clip_plane(
        p1, cam_coords, cam_rotation, cam_near_clip, cam_field_of_view
    )
    before2 = is_before_clip_plane(
        p2, cam_coords, cam_rotation, cam_near_clip, cam_field_of_view
    )
    if not (before1 or before2):
        return None
    if before1 and before2:
        cp1 = get_2d_from_3d(
            p1,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        cp2 = get_2d_from_3d(
            p2,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        x1 = int(cp1[0] * WIDTH)
        x2 = int(cp2[0] * WIDTH)
        y1 = int(cp1[1] * HEIGHT)
        y2 = int(cp2[1] * HEIGHT)
        return [[x1, y1], [x2, y2]]
    center_pt, cam_dir = get_clip_center_and_dir(
        cam_coords, cam_rotation, cam_near_clip
    )
    if before1 and not before2:
        inter2 = get_intersect_point(center_pt, cam_dir, p1, p2)
        cp1 = get_2d_from_3d(
            p1,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        cp2 = get_2d_from_3d(
            inter2,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        x1 = int(cp1[0] * WIDTH)
        x2 = int(cp2[0] * WIDTH)
        y1 = int(cp1[1] * HEIGHT)
        y2 = int(cp2[1] * HEIGHT)
        return [[x1, y1], [x2, y2]]
    if before2 and not before1:
        inter1 = get_intersect_point(center_pt, cam_dir, p1, p2)
        cp2 = get_2d_from_3d(
            p2,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        cp1 = get_2d_from_3d(
            inter1,
            cam_coords,
            cam_rotation,
            cam_near_clip,
            cam_field_of_view,
            WIDTH,
            HEIGHT,
        )
        x1 = int(cp1[0] * WIDTH)
        x2 = int(cp2[0] * WIDTH)
        y1 = int(cp1[1] * HEIGHT)
        y2 = int(cp2[1] * HEIGHT)
        return [[x1, y1], [x2, y2]]


####################
# io utils.
def read_depthmap(name, cam_near_clip, cam_far_clip):
    depth = cv2.imread(name)
    depth = np.concatenate(
        (depth, np.zeros_like(depth[:, :, 0:1], dtype=np.uint8)), axis=2
    )
    depth.dtype = np.uint32
    depth = 0.05 * 1000 / (depth.astype('float') + 1e-10)
    depth = (
            cam_near_clip
            * cam_far_clip
            / (cam_near_clip + depth * (cam_far_clip - cam_near_clip))
    )
    return depth


def create_skeleton_viz_data(nskeletons, njoints, col):
    lines = []
    colors = []
    for i in range(nskeletons):
        cur_lines = np.asarray(LIMBS)
        cur_lines += i * njoints
        lines.append(cur_lines)

        single_color = np.zeros([njoints, 3])
        single_color[:] = col
        colors.append(single_color[1:])

    lines = np.concatenate(lines, axis=0)
    colors = np.asarray(colors).reshape(-1, 3)
    return lines, colors


def add_skeleton(vis, joints, col, line_set=None, sphere_list=None, jlast=None):
    # add gt
    tl, jn, _ = joints.shape
    joints = joints.reshape(-1, 3)
    if jlast is not None:
        jlast = jlast.reshape(-1, 3)

    # plot history
    nskeletons = tl
    lines, colors = create_skeleton_viz_data(nskeletons, jn, col=col)
    if line_set is None:
        line_set = o3d.geometry.LineSet()
        if vis is not None:
            vis.add_geometry(line_set)

    line_set.points = o3d.utility.Vector3dVector(joints)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    if vis is not None:
        vis.update_geometry(line_set)

    count = 0
    sphere_list_tmp = []
    for j in range(joints.shape[0]):
        # spine joints
        if j % jn == 11 or j % jn == 12 or j % jn == 13:
            continue
        transformation = np.identity(4)
        if jlast is not None:
            transformation[:3, 3] = joints[j] - jlast[j]
        else:
            transformation[:3, 3] = joints[j]
        # head joint
        if j % jn == 0:
            r = 0.07
        else:
            r = 0.03
        if sphere_list is None:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
            if vis is not None:
                vis.add_geometry(sphere)
            sphere_list_tmp.append(sphere)
        else:
            sphere = sphere_list[count]
        sphere.paint_uniform_color(col)
        sphere.transform(transformation)
        if vis is not None:
            vis.update_geometry(sphere)
        count += 1
    if sphere_list is None:
        sphere_list = sphere_list_tmp
    return line_set, sphere_list


def compute_heading_direction(pos_curr, pos_last):
    vec_front = pos_curr - pos_last
    yaw = torch.atan2(vec_front[:, 1], vec_front[:, 0])
    rot_matrix = Rotation.from_euler('z', yaw).as_matrix()
    pose_a_mat = torch.eye(4)
    pose_a_mat[:3, :3] = torch.from_numpy(rot_matrix)
    pose_a_mat[:3, 3] = pos_curr.squeeze()
    return pose_a_mat


def crop_pcd_np(pcd_np, pos, dist):
    x, y, z = pos
    idx_x = np.logical_and((pcd_np[:, 0] >= x - dist), (pcd_np[:, 0] <= x + dist))
    idx_y = np.logical_and((pcd_np[:, 1] >= y - dist), (pcd_np[:, 1] <= y + dist))
    idx_z = np.logical_and((pcd_np[:, 2] >= z - dist), (pcd_np[:, 2] <= z + dist))
    idx_bool = idx_x * idx_y * idx_z
    idx = np.where(idx_bool)[0]
    return idx


def pcd2occmap(pcd, bound_min, bound_max, voxel_size, map_size):
    x_min, y_min, z_min = bound_min
    x_max, y_max, z_max = bound_max
    z_size = int((z_max - z_min) / voxel_size)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, map_size),
                             np.linspace(y_min, y_max, map_size),
                             np.linspace(z_min, z_max, z_size))
    pts_cub = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3) + voxel_size/2
    pts_occ = voxel_grid.check_if_included(o3d.utility.Vector3dVector(pts_cub))
    occ_map3d = np.asarray(pts_occ).reshape((map_size, map_size, z_size))
    occ_map2d = np.sum(occ_map3d, axis=2)
    # plt.imshow(occ_map2d > 0)
    # plt.show()
    return occ_map2d


def traj2map(traj, d, resolution, map_shape, device=None, binary=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    n_batch, n_frame = traj.shape[:2]
    idx = torch.clip((traj + d) / resolution, 0, map_shape[2] - 1)

    if binary:
        # Only the trajectory pixel is one, others are zero.
        # Comment this out since this is very sparse.
        traj_map = torch.zeros(map_shape).to(device)
        bs = torch.linspace(0, map_shape[0] - 1, map_shape[0]).int()
        fs = torch.linspace(0, map_shape[1] - 1, map_shape[1]).int()
        idx_prefix = torch.stack(torch.meshgrid(bs, fs, indexing='ij'), dim=-1).to(device)
        idx_full = torch.cat((idx_prefix, idx), dim=-1).long()
        traj_map[idx_full[:, :, 0], idx_full[:, :, 1], idx_full[:, :, 3], idx_full[:, :, 2]] = 1
    else:
        xs = torch.linspace(0, map_shape[3] - 1, map_shape[3]).int()
        ys = torch.linspace(0, map_shape[2] - 1, map_shape[2]).int()
        xy_mesh = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).to(device)
        xy_mesh_ext = xy_mesh.unsqueeze(0).unsqueeze(0).repeat(n_batch, n_frame, 1, 1, 1)
        idx_ext = idx.unsqueeze(2).unsqueeze(2).repeat(1, 1, map_shape[2], map_shape[3], 1)
        dist_xy = torch.norm(xy_mesh_ext - idx_ext, dim=-1)
        dist_yx = dist_xy.permute(0, 1, 3, 2)
        normal_distribution = torch.distributions.normal.Normal(0, 2)
        log_prob = normal_distribution.log_prob(dist_yx)
        prob = torch.exp(log_prob)
        traj_map = prob.unsqueeze(-1)
    return traj_map


def map2traj(map, d, resolution, device=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    bs, nf, h, w, chn = map.shape
    # map_ = map.squeeze().reshape(bs, nf, -1)
    # idx_1d = torch.argmax(map_, dim=-1)
    # y, x = idx_1d // h, idx_1d % w
    # idx = torch.stack((x, y), dim=-1)
    idx = softargmax2d(map[:, :, :, :, 0], beta=100).float()
    traj = (idx * resolution - d).float()
    return traj


def softargmax2d(input, beta=100):
    *_, h, w = input.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input = input.reshape(*_, h * w)
    input = torch.nn.functional.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w))).to(device)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w))).to(device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_c, result_r], dim=-1)

    return result


def get_relative_pose(pose, root_idx, n_joints):
    pos_root = pose[:, :, root_idx, :]  # bs * nf * 3
    pos_root_nj = pos_root.unsqueeze(2).repeat((1, 1, n_joints, 1))  # bs * nf * nj * 3
    pose_rel = pose - pos_root_nj  # Relative poses. bs * nf * nj * 3
    return pose_rel