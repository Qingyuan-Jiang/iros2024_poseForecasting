import copy
from math import pi

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def draw_frame(origin=[0, 0, 0], q=[0, 0, 0, 1], scale=1):
    # open3d quaternion format qw qx qy qz

    o3d_quat = np.array([q[3], q[0], q[1], q[2]])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=origin)
    frame_rot = copy.deepcopy(mesh_frame).rotate(mesh_frame.get_rotation_matrix_from_quaternion(o3d_quat))
    return frame_rot


def draw_T(T=np.eye(4), scale=1):
    r, t = T[:3, :3], T[:3, -1]
    q = Rotation.from_matrix(r).as_quat().tolist()
    return draw_frame(t, q, scale)


def draw_camera(origin=[0, 0, 0], scale=0.5):
    cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01 * scale, height=scale)
    cyld_0.paint_uniform_color([0, 0, 0])
    cyld_0.translate(np.array([0, 0, scale / 2]))
    cyld_0.rotate(
        cyld_0.get_rotation_matrix_from_xyz(np.array([0, pi / 2, 0])), center=[0, 0, 0])
    cyld_1 = copy.deepcopy(cyld_0).rotate(
        cyld_0.get_rotation_matrix_from_zyx(np.array([pi / 4, pi / 5, 0])), center=[0, 0, 0])
    cyld_2 = copy.deepcopy(cyld_0).rotate(
        cyld_0.get_rotation_matrix_from_zyx(np.array([-pi / 4, pi / 5, 0])), center=[0, 0, 0])
    cyld_3 = copy.deepcopy(cyld_0).rotate(
        cyld_0.get_rotation_matrix_from_zyx(np.array([pi / 4, -pi / 5, 0])), center=[0, 0, 0])
    cyld_4 = copy.deepcopy(cyld_0).rotate(
        cyld_0.get_rotation_matrix_from_zyx(np.array([-pi / 4, -pi / 5, 0])), center=[0, 0, 0])

    mesh = [cyld_1, cyld_2, cyld_3, cyld_4]

    cyld_0 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01 * scale, height=1.16 * scale)
    cyld_0.paint_uniform_color([0, 0, 0])
    cyld_5 = copy.deepcopy(cyld_0).translate(np.array([0.5773 * scale, -0.5773 * scale, 0]))
    cyld_6 = copy.deepcopy(cyld_0).translate(np.array([0.5773 * scale, 0.5773 * scale, 0]))

    cyld_0.rotate(
        cyld_0.get_rotation_matrix_from_xyz(np.array([pi / 2, 0, 0])), center=[0, 0, 0])
    cyld_7 = copy.deepcopy(cyld_0).translate(np.array([0.5773 * scale, 0, -0.5773 * scale]))
    cyld_8 = copy.deepcopy(cyld_0).translate(np.array([0.5773 * scale, 0, 0.5773 * scale]))

    mesh += [cyld_5, cyld_6, cyld_7, cyld_8]

    for m in mesh:
        m.translate(np.array(origin), relative=True)
    return mesh


def pcd2np(pcd):
    return np.asarray(pcd.points)


def np2pcd_xyzrgb(pcd_np_xyzrgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np_xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pcd_np_xyzrgb[:, 3:])
    return pcd


def np2pcd_xyz(pcd_np_xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np_xyz.reshape(-1, 3))
    return pcd


def pcd2np_xyzrgb(pcd):
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    return np.hstack((xyz, rgb))


def pcd_update_points(pcd, points):
    pcd_new = copy.deepcopy(pcd)
    pcd_new.points = o3d.utility.Vector3dVector(points)
    return pcd_new


def pcd_cat_points_colors(pcd, points, colors):
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)
    pcd_new = copy.deepcopy(pcd)
    pcd_new.points = o3d.utility.Vector3dVector(np.vstack((pcd_points, points)))
    pcd_new.colors = o3d.utility.Vector3dVector(np.vstack((pcd_colors, colors)))
    return pcd_new


def pcd_cat(pcd1, pcd2):
    pcd2_points = np.asarray(pcd2.points)
    pcd2_colors = np.asarray(pcd2.colors)
    return pcd_cat_points_colors(pcd1, pcd2_points, pcd2_colors)


def pcd_cat_list(pcd_list):
    assert len(pcd_list) >= 1, "List does not contain point cloud."
    pcd = pcd_list[0]
    for i in range(1, len(pcd_list)):
        pcd = pcd_cat(pcd, pcd_list[i])
    return pcd


def pcd_remove_nan_np(pcd_xyz):
    mask = np.logical_and(np.isfinite(pcd_xyz)[:, 0], np.isfinite(pcd_xyz)[:, 1], np.isfinite(pcd_xyz)[:, 2])
    return pcd_xyz[mask]


def pcd_from_rgbd_mask(K, rgb, depth, mask_img, h=576, w=1024):
    xx, yy = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    uv_h = np.reshape(np.concatenate((xx, yy, np.ones_like(xx)), axis=0), (3, h * w))
    uv_3d = np.linalg.inv(K) @ uv_h
    rays_d = uv_3d / uv_3d[-1, :]
    pcd_c = rays_d.T * np.tile(depth.reshape(-1, 1), (1, 3))

    mask = (mask_img > 0).reshape(-1, 3)
    pcd_actor_w = pcd_c[mask[:, 0], :3]
    color_actor = rgb.reshape(-1, 3)[mask[:, 0]] / 255
    return np.hstack((pcd_actor_w, color_actor))
