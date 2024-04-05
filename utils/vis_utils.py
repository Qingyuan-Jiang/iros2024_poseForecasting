import numpy as np
from utils.gta_utils import LIMBS
from utils.realIM_utils import LIMBS_coco
import open3d as o3d


def create_skeleton_viz_data(nskeletons, njoints, color='b', dataset='GTA'):
    lines = []
    colors = []
    for i in range(nskeletons):
        limbs = LIMBS if dataset == 'GTA' else LIMBS_coco
        cur_lines = np.asarray(limbs)
        cur_lines += i * njoints
        lines.append(cur_lines)
        n_viz = len(limbs)

        single_color = np.zeros([n_viz, 3])
        single_color[:] = [0.0, 0, 1.0] if color == 'b' else single_color
        single_color[:] = [0, 1.0, 0.0] if color == 'g' else single_color
        single_color[:] = [1.0, 0.0, 0] if color == 'r' else single_color
        colors.append(single_color)

    lines = np.concatenate(lines, axis=0)
    colors = np.asarray(colors).reshape(-1, 3)
    return lines, colors


def vis_skeleton_pcd(n_pred, n_joints, pose, color='b', dataset='GTA'):
    lines, colors = create_skeleton_viz_data(n_pred, n_joints, color, dataset)
    pose = pose.reshape(-1, 3)
    line_set = o3d.geometry.LineSet()
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.points = o3d.utility.Vector3dVector(pose)

    vis_list = [line_set]
    for j in range(pose.shape[0]):
        # spine joints
        if j % n_joints == 11 or j % n_joints == 12 or j % n_joints == 13:
            continue
        transformation = np.identity(4)
        transformation[:3, 3] = pose[j]
        # head joint
        r = 0.07 if j == 0 else 0.03

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=5)
        single_color = np.zeros([0, 0, 0])
        single_color = [0.0, 0, 1.0] if color == 'b' else single_color
        single_color = [0, 1.0, 0.0] if color == 'g' else single_color
        single_color = [1.0, 0.0, 0] if color == 'r' else single_color
        sphere.paint_uniform_color(single_color)
        vis_list.append(sphere.transform(transformation))
    return vis_list


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def o3d_save_visualization(vis_list, save_path, rotate=(0, 0), zoom=1):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    # vis.create_window()
    for vis_item in vis_list:
        vis.add_geometry(vis_item)
        vis.update_geometry(vis_item)
    vis.get_view_control().rotate(rotate[0], rotate[1])
    vis.get_view_control().set_zoom(zoom)
    # ctr = vis.get_view_control()
    # ctr.rotate(90.0, 90.0)
    # vis.add_geometry(pcd_crop)
    # vis.update_geometry(pcd_crop)
    # vis.add_geometry(pcd_i_can)
    # vis.update_geometry(pcd_i_can)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)
    vis.destroy_window()
