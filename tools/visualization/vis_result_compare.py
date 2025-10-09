import sys
sys.path.append('')
import os
import argparse
import os.path as osp
from PIL import Image
from tqdm import tqdm
from typing import List, Dict

import cv2
import mmcv
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from matplotlib.patches import Rectangle
from mmdet.datasets.pipelines import to_tensor
from matplotlib.collections import LineCollection
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import CustomNuscenesBox, CustomDetectionBox, color_map
from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import VectorizedLocalMap, LiDARInstanceLines

# Define colors for map lines and agents
map_colors = ['blue', 'green', 'red']

agent_color_dict={
    'car': 'cyan', 
    'truck': 'orange', 
    'construction_vehicle': 'purple', 
    'bus': 'yellow', 
    'trailer': 'brown', 
    'barrier': 'pink',
    'motorcycle': 'lime', 
    'bicycle': 'magenta', 
    'pedestrian': 'gray', 
    'traffic_cone': 'pink'
}

cams = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

data_root = 'absolute path to your nuscenes dataset'
info_path = 'absolute path to vad_nuscenes_infos_temporal_val.pkl'
baseline_result_path = '' # if you do not have a baseline to compare, just ignore
output_result_path = 'absolute path to results.pkl'

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sensor2lidar_rotation = R.T  # points @ R.T + T
    sensor2lidar_translation = T

    return sensor2lidar_rotation, sensor2lidar_translation

def get_gt_vec_maps(
    sample_token,
    data_root='data/nuscenes/',
    pc_range=[-15.0, -30.0, -4.0, 15.0, 30.0, 4.0],
    padding_value=-10000,
    map_classes=['divider', 'ped_crossing', 'boundary'],
    map_fixed_ptsnum_per_line=20
) -> None:
    """
    Get gt vec map for a given sample.
    """
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    lidar2ego_translation = cs_record['translation'],
    lidar2ego_rotation = cs_record['rotation'],
    ego2global_translation = pose_record['translation'],
    ego2global_rotation = pose_record['rotation'],
    map_location = nusc.get('log', nusc.get('scene', sample_rec['scene_token'])['log_token'])['location']

    lidar2ego = np.eye(4)
    lidar2ego[:3,:3] = Quaternion(cs_record['rotation']).rotation_matrix
    lidar2ego[:3, 3] = cs_record['translation']
    ego2global = np.eye(4)
    ego2global[:3,:3] = Quaternion(pose_record['rotation']).rotation_matrix
    ego2global[:3, 3] = pose_record['translation']
    lidar2global = ego2global @ lidar2ego
    lidar2global_translation = list(lidar2global[:3,3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_size = (patch_h, patch_w)

    vector_map = VectorizedLocalMap(data_root, patch_size=patch_size,
                                    map_classes=map_classes, 
                                    fixed_ptsnum_per_line=map_fixed_ptsnum_per_line,
                                    padding_value=padding_value)


    anns_results = vector_map.gen_vectorized_samples(
        map_location, lidar2global_translation, lidar2global_rotation
    )
    
    '''
    anns_results, type: dict
        'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
        'gt_vecs_pts_num': list[num_vecs], vec with num_points
        'gt_vecs_label': list[num_vecs], vec with cls index
    '''
    gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
    if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
        gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
    else:
        gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
        try:
            gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
        except:
            gt_vecs_pts_loc = gt_vecs_pts_loc
    
    return gt_vecs_pts_loc, gt_vecs_label


def visualize_sample_perception_free(
                                    sample_token: str,
                                    gt_vecs_pts_loc,
                                    gt_vecs_label,
                                    agent_gt_bboxes,
                                    agent_gt_labels,
                                    gt_agent_fut_trajs,
                                    gt_traj,
                                    fut_valid_flag,
                                    baseline_traj,
                                    output_traj,
                                    pc_range: list = [-30.0, -30.0, -4.0, 30.0, 30.0, 4.0],
                                    savepath: str = None,
                                ) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Init axes.
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.xlim(xmin=-30, xmax=30)
    plt.ylim(ymin=-30, ymax=30)

    map_gt_lines = [np.array(line.coords) for line in gt_vecs_pts_loc.instance_list]

    # === Plot map lines ===
    for line, label in zip(map_gt_lines, gt_vecs_label):
        x, y = line[:, 0], line[:, 1]
        ax.plot(x, y, color=map_colors[label], linewidth=0.5)

    # Plot agents
    for box, label, agent_traj in zip(agent_gt_bboxes, agent_gt_labels, gt_agent_fut_trajs):
        x_center, y_center, _, x_size, y_size, _, lidar_yaw = box[:7]
        agent_traj[:,0] += x_center
        agent_traj[:,1] += y_center
        # refer to `tools/data_converter/vad_nuscenes_converter.py` Line 345
        yaw_pitch_roll = -(lidar_yaw + np.pi / 2)
        # Create a rotation matrix
        rotation_matrix = np.array([
            [np.cos(yaw_pitch_roll-np.pi/2), -np.sin(yaw_pitch_roll-np.pi/2)],
            [np.sin(yaw_pitch_roll-np.pi/2), np.cos(yaw_pitch_roll-np.pi/2)]
        ])
        corners = np.array([
            [-x_size / 2, -y_size / 2],
            [x_size / 2, -y_size / 2],
            [x_size / 2, y_size / 2],
            [-x_size / 2, y_size / 2]
        ])
        rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x_center, y_center])
        edgecolor = agent_color_dict.get(label, "gray")  # fallback to gray if not found
        polygon = plt.Polygon(rotated_corners, closed=True, fill=None, edgecolor=edgecolor, linewidth=1)
        ax.add_patch(polygon)

        # Add heading direction arrow
        arrow_dx = 1.5 * np.cos(yaw_pitch_roll)  # Adjust the arrow size
        arrow_dy = 1.5 * np.sin(yaw_pitch_roll)
        ax.arrow(x_center, y_center, arrow_dx, arrow_dy, head_width=0.5, head_length=0.5,
                 fc=edgecolor, ec=edgecolor)
        
        # Plot agent trajectory
        agent_traj = np.vstack(([x_center, y_center], agent_traj))
        ego_x, ego_y = agent_traj[:, 0], agent_traj[:, 1]

        # Smooth the trajectory using interpolation for better visualization
        ax.plot(ego_x, ego_y, color=edgecolor, linestyle='-', linewidth=0.5)        
        

    # Plot ego car bounding box
    ego_width, ego_height = 1.8, 4  # Reasonable width and height for a car (in meters)
    ego_box = Rectangle((-ego_width / 2, -ego_height / 2), ego_width, ego_height,
                        linewidth=1, edgecolor='black', facecolor='none', label="Ego Car")
    ax.add_patch(ego_box)

    # === Baseline Traj ===
    if baseline_traj is not None:
        baseline_traj_pred = np.vstack(([0, 0], baseline_traj))  # Include (0, 0) as start point
        ego_x, ego_y = baseline_traj_pred[:, 0], baseline_traj_pred[:, 1]

        # Smooth the trajectory using interpolation for better visualization
        ax.plot(ego_x, ego_y, color='black', linestyle='--', linewidth=2, label="LAW Traj")
    # === LAW+CoDrive Traj ===
    output_traj_pred = np.vstack(([0, 0], output_traj))  # Include (0, 0) as start point
    ego_x, ego_y = output_traj_pred[:, 0], output_traj_pred[:, 1]

    # Smooth the trajectory using interpolation for better visualization
    ax.plot(ego_x, ego_y, color='blue', linestyle='-', linewidth=2, label="LAW+CoDrive Traj")

    # plot the ground truth trajectory and predicted trajectory of the ego car
    if fut_valid_flag:
        ego_trajectory = np.vstack(([0, 0], gt_traj))  # Include (0, 0) as start point
        ego_x, ego_y = ego_trajectory[:, 0], ego_trajectory[:, 1]

        # Smooth the trajectory using interpolation for better visualization
        ax.plot(ego_x, ego_y, color='red', linestyle='-', linewidth=2, label="GT Traj")

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.axis('off')
    fig.legend()
    fig.set_tight_layout(True)
    fig.canvas.draw()
    plt.savefig(savepath+'/bev+plan_traj.png', bbox_inches='tight', dpi=200)
    plt.close()

def proj_traj2front_camera(traj, lidar2img_rt, ax, linestyle, label):
    assert label in ['baseline', 'output', 'gt'], "only support label='baseline'|'output'|'gt'"
    traj = np.concatenate((
        traj[:, [0]],
        traj[:, [1]],
        -1.0*np.ones((traj.shape[0], 1)),
        np.ones((traj.shape[0], 1)),
    ), axis=1)
    # add the start point in lcf
    traj = np.concatenate((np.zeros((1, traj.shape[1])), traj), axis=0)
    # plan_traj[0, :2] = 2*plan_traj[1, :2] - plan_traj[2, :2]
    traj[0, 0] = 0.3
    traj[0, 2] = -1.0
    traj[0, 3] = 1.0

    traj = lidar2img_rt @ traj.T
    traj = traj[0:2, ...] / np.maximum(
        traj[2:3, ...], np.ones_like(traj[2:3, ...]) * 1e-5)
    traj = traj.T
    traj = np.stack((traj[:-1], traj[1:]), axis=1)

    plan_vecs = None
    for i in range(traj.shape[0]):
        plan_vec_i = traj[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

    # Create gradients for black (baseline) and blue (output)
    n_segments = len(plan_vecs)   # number of line segments

    if label == 'baseline':
        colors = plt.cm.Greys(np.linspace(0.9, 1.0, plan_vecs))  # dark gray -> white
    if label == 'output':
        colors = plt.cm.Blues(np.linspace(0.9, 1.0, n_segments))  # dark blue -> light blue
    if label == 'gt':
        colors = plt.cm.Reds(np.linspace(0.9, 1.0, n_segments))  # dark red -> light red
    
    # Create line collections
    line_segments = LineCollection(
        plan_vecs,
        colors=colors,
        linewidths=2,
        linestyles=linestyle,
        label=label
    )
    ax.add_collection(line_segments)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize VAD predictions')
    parser.add_argument('--save-path', help='the dir to save visualization results')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    out_path = args.save_path
    mmcv.mkdir_or_exist(out_path)

    with open(info_path, 'rb') as f:
        info_data = pickle.load(f)
        info_data = info_data['infos'] # list of dict [{}, ...] with keys(): dict_keys(['lidar_path', 'token', 'prev', 'next', 'can_bus', 'frame_idx', 'sweeps', 'cams', 'scene_token', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'fut_valid_flag', 'map_location', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag', 'gt_agent_fut_trajs', 'gt_agent_fut_masks', 'gt_agent_lcf_feat', 'gt_agent_fut_yaw', 'gt_agent_fut_goal', 'gt_ego_his_trajs', 'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'gt_ego_lcf_feat'])
    if results_path == "":
        baseline_result = None
    else:
        with open(baseline_result_path, 'rb') as f:
            baseline_result = pickle.load(f) # dict, with key=sample_token, value is a dict with keys: ['scene_token', eval_metrics, 'ego_fut_traj']
    with open(output_result_path, 'rb') as f:
        output_result = pickle.load(f) # dict, with key=sample_token, value is a dict with keys: ['scene_token', eval_metrics, 'ego_fut_traj']

    nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    scene_token_vised = ""
    frame_idx_vised = 0

    for idx in tqdm(range(len(info_data))):
        fut_valid_flag = info_data[idx]['fut_valid_flag']
        sample_token = info_data[idx]['token']
        scene_token = info_data[idx]['scene_token']
        ego_fut_cmd = info_data[idx]['gt_ego_fut_cmd'] # e.g. array([0., 0., 1.], dtype=float32)
        cam_metas = info_data[idx]['cams']

        # determine whether or not finished a scenario
        if scene_token_vised == "":
            scene_token_vised = scene_token
            # first, initialize scenario dir and video file
            scenario_out_path = os.path.join(out_path, scene_token)
            os.makedirs(scenario_out_path, exist_ok=True)

            video_path = osp.join(scenario_out_path, 'vis.mp4')
            video = cv2.VideoWriter(video_path, fourcc, 10, (3836, 1046), True)
        elif scene_token_vised != scene_token:
            # finish this scenario: save the video
            # and begine the next scenraio: set new save path
            scene_token_vised = scene_token
            frame_idx_vised = 0
            video.release()
            cv2.destroyAllWindows()

            scenario_out_path = os.path.join(out_path, scene_token)
            os.makedirs(scenario_out_path, exist_ok=True)

            video_path = osp.join(scenario_out_path, 'vis.mp4')
            video = cv2.VideoWriter(video_path, fourcc, 10, (3836, 1046), True)

        # This function is used to visualize
        # 1) GT map
        gt_vecs_pts_loc, gt_vecs_label = get_gt_vec_maps(sample_token, data_root=data_root)
        # 2) GT agent box and their moving traj
        agent_gt_bboxes = info_data[idx]['gt_boxes'] # [n_agent, 7]
        agent_gt_labels = info_data[idx]['gt_names'] # [n_agent,]
        gt_agent_fut_trajs = info_data[idx]['gt_agent_fut_trajs'] # [n_agent, 6, 2]
        n_agent = agent_gt_labels.shape[0]
        gt_agent_fut_trajs = gt_agent_fut_trajs.reshape(n_agent, 6, 2)
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        # 3) ego trajectory (GT, baseline, and ours)
        gt_traj = info_data[idx]['gt_ego_fut_trajs'] # [6, 2]
        gt_traj = np.cumsum(gt_traj, axis=0)
        if baseline_result is None:
            baseline_traj = None
        else:
            baseline_traj = baseline_result[sample_token]['ego_fut_traj'] # [6, 2]
        output_traj = output_result[sample_token]['ego_fut_traj'] # [6, 2]

        visualize_sample_perception_free(sample_token=sample_token,
                                        gt_vecs_pts_loc=gt_vecs_pts_loc,
                                        gt_vecs_label=gt_vecs_label,
                                        agent_gt_bboxes=agent_gt_bboxes,
                                        agent_gt_labels=agent_gt_labels,
                                        gt_agent_fut_trajs=gt_agent_fut_trajs,
                                        gt_traj=gt_traj,
                                        fut_valid_flag=fut_valid_flag,
                                        baseline_traj=baseline_traj,
                                        output_traj=output_traj,
                                        savepath=out_path)

        pred_path = osp.join(out_path, 'bev+plan_traj.png')
        pred_img = cv2.imread(pred_path)
        os.remove(pred_path)

        sample = nusc.get('sample', sample_token)
        cam_imgs = []
        for cam in cams:
            sample_data_token = sample['data'][cam]
            sd_record = nusc.get('sample_data', sample_data_token)
            sensor_modality = sd_record['sensor_modality']
            if sensor_modality in ['lidar', 'radar']:
                assert False
            elif sensor_modality == 'camera':
                img_path = cam_metas[cam]['data_path']
                camera_intrinsic = cam_metas[cam]['cam_intrinsic']
                data = Image.open(img_path)
 
                # Show image.
                _, ax = plt.subplots(1, 1, figsize=(6, 12))
                ax.imshow(data)

                if cam == 'CAM_FRONT':
                    lidar_sd_record =  nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    lidar_cs_record = nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
                    lidar_pose_record = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])

                    # get plan traj [x,y,z,w] quaternion, w=1
                    # we set z=-1 to get points near the ground in lidar coord system
                    plan_cmd = np.argmax(ego_fut_cmd)

                    l2e_r = lidar_cs_record['rotation']
                    l2e_t = lidar_cs_record['translation']
                    e2g_r = lidar_pose_record['rotation']
                    e2g_t = lidar_pose_record['translation']
                    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
                    s2l_r, s2l_t = obtain_sensor2top(nusc, sample_data_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(s2l_r)
                    lidar2cam_t = s2l_t @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    viewpad = np.eye(4)
                    viewpad[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                    if baseline_traj is not None:
                        proj_traj2front_camera(traj=baseline_traj, lidar2img_rt=lidar2img_rt, ax=ax, linestyle='-', label='baseline')

                    proj_traj2front_camera(traj=output_traj, lidar2img_rt=lidar2img_rt, ax=ax, linestyle='-', label='output')
                    
                    if fut_valid_flag:
                        proj_traj2front_camera(traj=gt_traj, lidar2img_rt=lidar2img_rt, ax=ax, linestyle='-', label='gt')

                ax.legend(labelcolor='white', facecolor='black')
                ax.set_xlim(0, data.size[0])
                ax.set_ylim(data.size[1], 0)
                ax.axis('off')
                if out_path is not None:
                    savepath = osp.join(out_path, f'{cam}_PRED.png')
                    plt.savefig(savepath, bbox_inches='tight', dpi=200, pad_inches=0.0)
                plt.close()

                # Load boxes and image.
                data_path = osp.join(out_path, f'{cam}_PRED.png')
                cam_img = cv2.imread(data_path)
                lw = 6
                tf = max(lw - 3, 1)
                w, h = cv2.getTextSize(cam, 0, fontScale=lw / 6, thickness=tf)[0]  # text width, height
                # color=(0, 0, 0)
                txt_color=(255, 255, 255)
                cv2.putText(cam_img,
                            cam, (10, h + 10),
                            0,
                            lw / 6,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
                cam_imgs.append(cam_img)
            else:
                raise ValueError("Error: Unknown sensor modality!")

        plan_cmd = np.argmax(ego_fut_cmd)
        cmd_list = ['Turn Right', 'Turn Left', 'Go Straight']
        plan_cmd_str = cmd_list[plan_cmd]
        pred_img = cv2.copyMakeBorder(pred_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1
        # Line thickness of 2 px
        thickness = 3
        # org
        org = (20, 40)      
        # Blue color in BGR
        color = (0, 0, 0)
        # Using cv2.putText() method
        pred_img = cv2.putText(pred_img, 'BEV', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        pred_img = cv2.putText(pred_img, plan_cmd_str, (20, 770), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        sample_img = pred_img
        cam_img_top = cv2.hconcat([cam_imgs[0], cam_imgs[1], cam_imgs[2]])
        cam_img_down = cv2.hconcat([cam_imgs[3], cam_imgs[4], cam_imgs[5]])
        cam_img = cv2.vconcat([cam_img_top, cam_img_down])
        size = (1046, 1046)
        sample_img = cv2.resize(sample_img, size)
        vis_img = cv2.hconcat([cam_img, sample_img])

        fig_save_path = os.path.join(scenario_out_path, f'frame_{frame_idx_vised}.png')
        frame_idx_vised += 1
        cv2.imwrite(fig_save_path, vis_img)  # concat_img is the result of hconcat

        video.write(vis_img)
    
    video.release()
    cv2.destroyAllWindows()