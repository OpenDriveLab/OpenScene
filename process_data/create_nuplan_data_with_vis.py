"""
convert nuscenes data and generate occupancy gt
accumulate the lidar data: sample+sweep
(1) accumulate the background and foreground objects separately
(2) introduce the lidarseg in sample data
(3) generate the occupancy gt
"""
import enum
from itertools import accumulate
import shutil
import mmcv
import numpy as np
import os
from collections import OrderedDict
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

# from mmdet3d.core.bbox.box_np_ops import points_cam2img
# from mmdet3d.datasets import NuScenesDataset

from utils import *
import cv2
import json
import os.path as osp
import yaml

from single_frame_occ_generator import SingleFrameOCCGTGenerator
from tqdm import tqdm
from KITTI_VIZ_BEV.plot import point_bev

# import mmengine
import os

import threading
import multiprocessing
from multiprocessing import Process
import pickle

thread_num = 64

# import sys
# sys.path.append('/home/zhouyunsong/zhouys/Git_repos/nuplan-devkit/nuplan')

from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.image import Image
from nuplan.database.nuplan_db_orm.lidar import Lidar
from nuplan.database.nuplan_db_orm.category import Category
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.log import Log
from nuplan.database.nuplan_db_orm.scene import Scene
from nuplan.database.nuplan_db_orm.scenario_tag import ScenarioTag
from nuplan.database.nuplan_db_orm.traffic_light_status import TrafficLightStatus
from nuplan.database.nuplan_db_orm.track import Track
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_ego_state_for_lidarpc_token_from_db,
    get_end_sensor_time_from_db,
    get_images_from_lidar_tokens,
    get_cameras,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data


from os import listdir
from os.path import isfile, join
from mmdet3d.core.bbox import LiDARInstance3DBoxes

filtered_classes = ['traffic_cone', 'barrier', 'czone_sign', 'generic_object']

def split_scenes(scenes):
    scenes = sorted(scenes)
    # num_tasks = int(os.environ['SLURM_NTASKS'])
    # cur_id = int(os.environ['SLURM_PROCID'])
    num_tasks = thread_num
    cur_id = int(multiprocessing.current_process().name)
    # cur_id = int(threading.current_thread().name)
    # print(multiprocessing.current_process().name)
    # 
    # num_tasks = 1
    # cur_id = 0
    # train_scenes
    num_scene = len(scenes)
    a = num_scene//num_tasks
    b = num_scene % num_tasks

    if cur_id == 0:
        print('num_scene:', num_scene)

    process_num = []
    count = 0
    for id in range(num_tasks):
        if id >= b:
            process_num.append(a)
        else:
            process_num.append(a+1)
    addsum = np.cumsum(process_num)

    if cur_id == 0:
        start = 0
        end = addsum[0]
    else:
        start = addsum[cur_id-1]
        end = addsum[cur_id]

    return scenes[start:end], start



def create_nuplan_info(nuplandb_wrapper, test, max_sweeps=1, sweep_interval=1, sample_interval=10, out_dir='./', occ_resolution='normal', save_flow_info=True, filter_instance=True, use_cuda=False, is_nuplan=True, scene_process_type='reprocess', set_background_label=False):

    file_names_with_extension = [f for f in listdir(NUPLAN_DB_FILES) if isfile(join(NUPLAN_DB_FILES, f))]

    file_names = [name[:-3] for name in file_names_with_extension]

    file_names.sort()
    # file_names = file_names[:2]
    file_names_splited, start = split_scenes(file_names)
    # file_names_splited = file_names 
    # start = 0
    scene_dict = {}
    save_bev_images = False
    save_surround_images = False
    # save_bev_images = True
    # save_surround_images = True
    save_scene_background_point = False
    save_instance_point = False 
    save_single_frame_data = False  # save accumulated lidar frame-by-frame
    train_flag = True

    log_sensors = os.listdir(NUPLAN_SENSOR_ROOT)


    log_idx = start
    # for log_file_name in tqdm(file_names, desc='outer loop'):
    for log_file_name in file_names_splited:
        log_db_name = log_file_name
        log_db = nuplandb_wrapper.get_log_db(log_db_name)

        log_file = os.path.join(NUPLAN_DB_FILES, log_db_name + '.db')

        if log_db_name not in log_sensors:
            continue

        scenes = []
        frame_idx = 0
        log_idx += 1

        lidar_pc_list = log_db.lidar_pc
        lidar_pcs = lidar_pc_list
        # print(lidar_pcs)
        # lidar_pcs.sort(key=lambda x: x.timestamp)

        # get log cam infos
        log_cam_infos = {}
        for cam in get_cameras(log_file, [str(channel.value) for channel in CameraChannel]):
            intrinsics = np.array(pickle.loads(cam.intrinsic), dtype=np.float32)
            translation = np.array(pickle.loads(cam.translation), dtype=np.float32)
            rotation = np.array(pickle.loads(cam.rotation), dtype=np.float32)
            rotation = Quaternion(rotation).rotation_matrix
            distortion = np.array(pickle.loads(cam.distortion), dtype=np.float32)
            c = dict(
                intrinsic=intrinsics,
                distortion=distortion,
                translation=translation,
                rotation=rotation,
            )
            log_cam_infos[cam.token] = c
        # select start_idx, which is the closest to the first image
        for start_idx in range(0, len(lidar_pcs)):
            # print([lidar_pcs[start_idx].token])
            retrieved_images = get_images_from_lidar_tokens(
                log_file, [lidar_pcs[start_idx].token], [str(channel.value) for channel in CameraChannel]
            )
            # print(len(list(retrieved_images)))
            if len(list(retrieved_images)) == 8:
                break
        # print(retrieved_images)
        # print(len(list(retrieved_images)))
        retrieved_images = get_images_from_lidar_tokens(
            log_file, [lidar_pcs[start_idx].token], ['CAM_F0']
        )

        
        # diff_0 = abs(next(retrieved_images).timestamp - lidar_pcs[start_idx].timestamp)
        diff_0 = abs(list(retrieved_images)[0].timestamp - lidar_pcs[start_idx].timestamp)

        retrieved_images = get_images_from_lidar_tokens(
            log_file, [lidar_pcs[start_idx + 1].token], ['CAM_F0']
        )
        # diff_1 = abs(next(retrieved_images).timestamp - lidar_pcs[start_idx + 1].timestamp)
        diff_1 = abs(list(retrieved_images)[0].timestamp - lidar_pcs[start_idx + 1].timestamp)

        start_idx = start_idx if diff_0 < diff_1 else start_idx + 1


        # for lidar_pc in tqdm(log_db.lidar_pc, desc='inner loop', leave=False):
        
        lidar_pc_list = lidar_pc_list[start_idx::sample_interval]
        index = -1
        for lidar_pc in tqdm(lidar_pc_list, dynamic_ncols=True):

            index += 1
            lidar_pc_token = lidar_pc.token
            ego_pose_token = lidar_pc.ego_pose_token
            scene_token = lidar_pc.scene_token
            pc_file_name = lidar_pc.filename
            next_token = lidar_pc.next_token
            prev_token = lidar_pc.prev_token
            lidar_token = lidar_pc.lidar_token
            time_stamp = lidar_pc.timestamp

            next = lidar_pc.next
            prev = lidar_pc.prev
            ego_pose = lidar_pc.ego_pose
            scene = lidar_pc.scene
            log_str = '%04d' % log_idx
            # scene_name = 'log-'+log_str+'-'+lidar_pc.scene.name
            scene_name = 'log-'+log_str+'-'+lidar_pc.scene.name
            lidar_boxes = lidar_pc.lidar_boxes


            cur_frame_idx = frame_idx

            if scene_token not in scene_dict.keys():
                scene_dict[scene_token] = {}
                frame_idx = 0
            if frame_idx == 0:
                scene_dict[scene_token] = {}

            
        
            # scene_save_dir = os.path.join(out_dir, 'trainval', scene_name)
            # os.makedirs(scene_save_dir, exist_ok=True)

            # if log_idx % 5 == 0:
            #     train_flag = False
            # else:
            #     train_flag = True

            if train_flag:
                scene_save_dir = os.path.join(out_dir, 'train', scene_name)
            else:
                scene_save_dir = os.path.join(out_dir, 'val', scene_name)
            os.makedirs(scene_save_dir, exist_ok=True)


            scene_pkl_file = os.path.join(scene_save_dir, 'scene_info.pkl')
            if os.path.exists(scene_pkl_file):  # this scene has been processed
                if scene_process_type == 'skip':
                    continue


            # can_bus = np.zeros(18) # qx
            can_bus = [lidar_pc.ego_pose.x, lidar_pc.ego_pose.y, lidar_pc.ego_pose.z, lidar_pc.ego_pose.qw, lidar_pc.ego_pose.qx, lidar_pc.ego_pose.qy, lidar_pc.ego_pose.qz, lidar_pc.ego_pose.acceleration_x, lidar_pc.ego_pose.acceleration_y, lidar_pc.ego_pose.acceleration_z, lidar_pc.ego_pose.vx, lidar_pc.ego_pose.vy, lidar_pc.ego_pose.vz, lidar_pc.ego_pose.angular_rate_x, lidar_pc.ego_pose.angular_rate_y, lidar_pc.ego_pose.angular_rate_z]
            can_bus.extend([0., 0.])
            can_bus = np.array(can_bus)

            lidar = log_db.session.query(Lidar) \
            .filter(Lidar.token == lidar_token) \
            .all()

            pc_file_name = os.path.join(NUPLAN_SENSOR_ROOT, pc_file_name)

            if not os.path.exists(pc_file_name):  # this scene has been processed
                # print(pc_file_name)
                with open('./nofile.log','a') as f:
                    f.write(pc_file_name)
                    f.write('\n')
                continue

            info = {
                'lidar_path': pc_file_name,
                'lidar_seg_path': None,
                'token': lidar_pc_token,
                'prev': prev_token,
                'next': next_token,
                'can_bus': can_bus,
                'frame_idx': frame_idx,  # temporal related info
                'sweeps': [],
                'cams': dict(),
                'scene_token': scene_token,  # temporal related info
                'scene_name': scene_name,  # additional info
                'lidar2ego_translation': lidar[0].translation_np,
                'lidar2ego_rotation': [lidar[0].rotation.w, lidar[0].rotation.x, lidar[0].rotation.y, lidar[0].rotation.z],
                'ego2global_translation': can_bus[:3],
                'ego2global_rotation': can_bus[3:7],
                'timestamp': time_stamp,
            }
            info['sample_prev'] = None
            info['sample_next'] = None
            if index > 0:
                info['sample_prev'] = lidar_pc_list[index-1].token
            if index < len(lidar_pc_list)-1:
                next_key_token = lidar_pc_list[index+1].token
                next_key_scene = lidar_pc_list[index+1].scene_token
                info['sample_next'] = lidar_pc_list[index+1].token
            else:
                next_key_token, next_key_scene = None, None

            if next_key_token == None or next_key_token == '':
                frame_idx = 0
            else:
                # next_pc= log_db.session.query(LidarPc) \
                # .filter(LidarPc.token == next_token) \
                # .all()
                # next_pc = next_pc[0]
                # if next_pc.scene_token != scene_token:
                #     frame_idx = 0
                # else:
                #     frame_idx += 1
                if next_key_scene != scene_token:
                    frame_idx = 0
                else:
                    frame_idx += 1

          

            l2e_r = info['lidar2ego_rotation']
            l2e_t = info['lidar2ego_translation']
            e2g_r = info['ego2global_rotation']
            e2g_t = info['ego2global_translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # add lidar2global: map point coord in lidar to point coord in the global
            l2e = np.eye(4)
            l2e[:3, :3] = l2e_r_mat
            l2e[:3, -1] = l2e_t
            e2g = np.eye(4)
            e2g[:3, :3] = e2g_r_mat
            e2g[:3, -1] = e2g_t
            lidar2global  = np.dot(e2g, l2e)
            info['ego2global'] = e2g
            info['lidar2ego'] = l2e
            info['lidar2global'] = lidar2global  # additional info

            # obtain 8 image's information per frame
            retrieved_images = get_images_from_lidar_tokens(
                log_file, [lidar_pc.token], [str(channel.value) for channel in CameraChannel]
            )

            log_name = log_db_name
            cams = {}
            for img in retrieved_images:
                channel = img.channel
                filename = img.filename_jpg
                timestamp = img.timestamp

                filepath = os.path.join(NUPLAN_SENSOR_ROOT, filename)
                if not os.path.exists(filepath):
                    frame_str = f'{log_name}, {lidar_pc_token}'
                    tqdm.tqdm.write(f'camera file missing: {frame_str}')
                    continue
                cam_info = log_cam_infos[img.camera_token]
                cams[channel] = dict(
                    data_path = filepath,
                    sensor2lidar_rotation = cam_info['rotation'],
                    sensor2lidar_translation = cam_info['translation'],
                    cam_intrinsic = cam_info['intrinsic'],
                    distortion = cam_info['distortion'],
                )
            if len(cams) != 8:
                frame_str = f'{log_name},  {lidar_pc_token}'
                tqdm.tqdm.write(f'not all cameras are available: {frame_str}')
                continue

            info['cams'] = cams
            

            sweeps = []
            tmp_info = info
            count = 0
            while len(sweeps) < max_sweeps:
                

                if not tmp_info['prev'] == None:

                    sweep = obtain_sensor2top(tmp_info['prev'], log_db, l2e_t,
                                            l2e_r_mat, e2g_t, e2g_r_mat, 'lidar_pc_token')
                    # temp = nusc.get('sample_data', sweep['sample_data_token'])
                    # if temp['is_key_frame']:   # TODO we only add the real sweep data (not key frame)
                    #     break
                    # sweeps.append(sweep)
                    # sd_rec = nusc.get('sample_data', sd_rec['prev'])
                    tmp_info = sweep

                    if count == sweep_interval:
                        if os.path.exists(sweep['data_path']):
                            sweeps.append(sweep)
                        count = 0
                    else:
                        count += 1

                else:
                    break
                
            info['sweeps'] = sweeps
            # obtain annotation
            if not test:
                # annotations = [
                #     nusc.get('sample_annotation', token)
                #     for token in sample['anns']
                # ]
                if filter_instance:
                    fg_lidar_boxes = [box for box in lidar_boxes if box.category.name not in filtered_classes]
                annotations = fg_lidar_boxes
                boxes = fg_lidar_boxes
                # get the box id for tracking the box in the scene
                instance_tokens = [item.token for item in annotations]
                track_tokens = [item.track_token for item in annotations]

                inv_ego_r = lidar_pc.ego_pose.trans_matrix_inv
                ego_r = lidar_pc.ego_pose.trans_matrix
                ego_yaw = lidar_pc.ego_pose.quaternion.yaw_pitch_roll[0]

                locs = np.array([np.dot(inv_ego_r[:3,:3], (b.translation_np-lidar_pc.ego_pose.translation_np).T).T for b in boxes]).reshape(-1,3)

                # locs = np.array([(np.dot(ego_r[:3,:3], b.translation_np.T).T-lidar_pc.ego_pose.translation_np) for b in boxes]).reshape(-1,3)
                

                # locs = np.array([[b.x-can_bus[0], b.y-can_bus[1], b.z-can_bus[2]] for b in boxes]).reshape(-1, 3)
                dims = np.array([[b.width, b.length, b.height] for b in boxes]).reshape(-1, 3)
                dims_lwh = np.concatenate([dims[:, 1:2], dims[:, 0:1], dims[:, 2:]], axis=-1)

                # rots = np.array([np.deg2rad(b.yaw) for b in boxes]).reshape(-1, 1)
                # TODO: check the rotation
                rots = np.array([b.yaw for b in fg_lidar_boxes]).reshape(-1, 1)
                rots = rots - ego_yaw

                velocity = np.array([[b.vx, b.vy] for b in boxes]).reshape(-1, 2)
                velocity_3d = np.array([[b.vx, b.vy, b.vz] for b in boxes]).reshape(-1, 3)
                # velocity = np.array(
                #     [nusc.box_velocity(token)[:2] for token in sample['anns']])
                # velocity_3d = np.array([nusc.box_velocity(token) for token in sample['anns']])

                valid_flag = np.array([True for anno in annotations], dtype=bool).reshape(-1)
                # convert velo from global to lidar: only need the rotation matrix
                for i in range(len(boxes)):
                    velo = np.array([*velocity[i], 0.0])
                    velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                        l2e_r_mat).T
                    velocity[i] = velo[:2]

                for i in range(len(boxes)):
                    velo = velocity_3d[i]
                    velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                        l2e_r_mat).T
                    velocity_3d[i] = velo

                names = [box.category.name for box in fg_lidar_boxes]
                names = np.array(names)

                # we need to convert rot to SECOND format.
                gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                gt_boxes_st = np.concatenate([locs, dims_lwh, rots], axis=1)
                #############################################################################
                # vis_point = lidar_pc.load(log_db).points.T
                # gt_boxes_vis = np.concatenate([locs, dims_lwh, -rots - np.pi / 2], axis=1)
                # point_bev(vis_point,gt_boxes_st)
                #############################################################################
                assert len(gt_boxes) == len(
                    annotations), f'{len(gt_boxes)}, {len(annotations)}'
                info['gt_boxes'] = gt_boxes
                info['gt_names'] = names
                info['gt_velocity'] = velocity.reshape(-1, 2)
                info['gt_velocity_3d'] = velocity_3d.reshape(-1, 3)
                info['num_lidar_pts'] = np.ones(names.shape) # suppose have points
                info['num_radar_pts'] = None
                info['valid_flag'] = valid_flag

                # additional info
                info['instance_tokens'] = instance_tokens
                info['track_tokens'] = track_tokens
                info['dims_lwh'] = dims_lwh
                info['gt_boxes_st'] = gt_boxes_st # standard definition of gt_boxes_st



                bg_lidar_boxes = [box for box in lidar_boxes if box.category.name in filtered_classes]
                locs = np.array([np.dot(inv_ego_r[:3,:3], (b.translation_np-lidar_pc.ego_pose.translation_np).T).T for b in bg_lidar_boxes]).reshape(-1,3)


                dims = np.array([[b.width, b.length, b.height] for b in bg_lidar_boxes]).reshape(-1, 3)
                dims_lwh = np.concatenate([dims[:, 1:2], dims[:, 0:1], dims[:, 2:]], axis=-1)
                rots = np.array([b.yaw for b in bg_lidar_boxes]).reshape(-1, 1)
                rots = rots - ego_yaw

                # velocity = np.array([[b.vx, b.vy] for b in bg_lidar_boxes]).reshape(-1, 2)
                # velocity_3d = np.array([[b.vx, b.vy, b.vz] for b in bg_lidar_boxes]).reshape(-1, 3)

                bg_names = [box.category.name for box in bg_lidar_boxes]
                bg_names = np.array(bg_names)

                # we need to convert rot to SECOND format.
                bg_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                bg_boxes_st = np.concatenate([locs, dims_lwh, rots], axis=1)


                info['bg_names'] = bg_names
                info['bg_boxes_st'] = bg_boxes_st


                back_instance_info = extract_frame_background_instance_lidar(lidar_pc, log_db, info, use_cuda)
                back_instance_info_sweeps, sweep_infos = extract_and_split_sweep_lidar(log_db, info)
                background_lidar_seg_info = extract_background_lidar_seg(back_instance_info['background_points'], log_db, None, info, set_background_label)


                save_sample_lidar = False
                if save_sample_lidar:
                    sample_lidar_dir = os.path.join(scene_save_dir, 'samples')
                    os.makedirs(sample_lidar_dir, exist_ok=True)
                    sample_save_path = os.path.join(sample_lidar_dir, '{:03}.bin'.format(cur_frame_idx))
                    shutil.copy(info['lidar_path'], sample_save_path)

                check_sweep_data = False
                if check_sweep_data:  # check lidar and 3D box
                    sweep_save_dir = os.path.join(scene_save_dir, 'sweeps')
                    os.makedirs(sweep_save_dir, exist_ok=True)
                    for sweep_index, sweep_info in enumerate(sweep_infos):
                        sweep_lidar_path = sweep_info['lidar_path']
                        sweep_save_path = os.path.join(sweep_save_dir, '{:03}_{}_.bin'.format(cur_frame_idx, sweep_index))
                        shutil.copy(sweep_lidar_path, sweep_save_path)
                        result_box = {'gt_bboxes_3d': sweep_info['gt_boxes_st'],
                                    'gt_labels_3d': sweep_info['gt_names']}
                        box_save_path = os.path.join(sweep_save_dir, '{:03}_{}_box.pkl'.format(cur_frame_idx, sweep_index))
                        mmcv.dump(result_box, box_save_path)

                scene_dict[scene_token][cur_frame_idx] = {}
                scene_dict[scene_token][cur_frame_idx]['basic_info'] = info
                scene_dict[scene_token][cur_frame_idx]['back_instance_info'] = back_instance_info
                scene_dict[scene_token][cur_frame_idx]['back_instance_info_sweeps'] = back_instance_info_sweeps
                scene_dict[scene_token][cur_frame_idx]['background_lidar_seg_info'] = background_lidar_seg_info


                

                accum_sweep = True
                if frame_idx == 0:  # end of the current scene

                    if scene_process_type == 'resume':
                        save_frame_idxs = sorted(scene_dict[scene_token].keys())
                        resume_path = scene_save_dir
                    
                        cur_scene_infos = []
                        for cur_frame_idx in save_frame_idxs:
                            cur_info = scene_dict[scene_token][cur_frame_idx]['basic_info']

                            occ_file_str = '%03d' % frame_idx
                            occ_gt_final_path = os.path.join(resume_path, 'occ_gt', (occ_file_str+'_occ_final.npy'))
                            flow_gt_final_path = os.path.join(resume_path, 'occ_gt', (occ_file_str+'_flow_final.npy'))
                            occ_invalid_path = os.path.join(resume_path, 'occ_gt', (occ_file_str+'_occ_invalid.npy'))
                        

                            cur_info['occ_gt_final_path'] = occ_gt_final_path if os.path.exists(occ_gt_final_path) else None
                            cur_info['flow_gt_final_path'] = flow_gt_final_path if os.path.exists(flow_gt_final_path) else None
                            cur_info['occ_invalid_path'] = occ_invalid_path if os.path.exists(occ_invalid_path) else None

                            cur_scene_infos.append(cur_info)

                        mmcv.dump(cur_scene_infos, scene_pkl_file)

                    else:

                        # print('end of the current scene')
                        background_track, instance_track = accumulate_background_box_point(scene_dict[scene_token], accum_sweep)

                        lidarseg_background_points, lidarseg_background_labels = accumulate_lidarseg_background(scene_dict[scene_token])


                        # 1. save wholce scene accumulated background and box points separately
                        # save whole scene accumulated background points in the global system
                        back_points = background_track['accu_global']
                        if save_scene_background_point:
                            scene_background_path = os.path.join(scene_save_dir,'scene_background_point.bin')
                            back_points_save = back_points.astype(np.float32)
                            back_points_save.tofile(scene_background_path)  

                        # save the accumulated box points in the local box system
                        if save_instance_point:
                            instance_points_dir = os.path.join(scene_save_dir, 'instance_points')
                            os.makedirs(instance_points_dir, exist_ok=True)
                            for instance_id in instance_track:
                                instance_object_track = instance_track[instance_id]
                                box_points = instance_object_track.accu_points
                                box_points = box_points.astype(np.float32)
                                class_name = instance_object_track.class_name
                                class_dir = os.path.join(instance_points_dir, class_name)
                                os.makedirs(class_dir, exist_ok=True)

                                if instance_object_track.is_stationary:
                                    box_points.tofile(os.path.join(class_dir,instance_id+'_sta.bin'))
                                else:
                                    box_points.tofile(os.path.join(class_dir,instance_id+'_mov.bin'))
                        
                        
                        # 2. save the accu points  and gt box in the lidar system frame by frame
                        save_frame_idxs = sorted(scene_dict[scene_token].keys())
                        
                        if save_single_frame_data:
                            single_frame_dir = os.path.join(scene_save_dir, 'sing_frame_data')
                            os.makedirs(single_frame_dir, exist_ok=True)
                        cur_scene_infos = []
                        for cur_frame_idx in save_frame_idxs:
                            # save accumulate sample lidar data
                            cur_info = scene_dict[scene_token][cur_frame_idx]['basic_info']
                            lidar2global =cur_info['lidar2global']

                            back_points_in_lidar = transform_points_global2lidar(back_points, lidar2global, filter=True)
                            flag, box_points_in_lidar, points_velocitys, points_labels = transform_points_box2lidar(instance_track, cur_frame_idx)
                            if flag:
                                points_in_lidar = np.concatenate([box_points_in_lidar, back_points_in_lidar], axis=0)
                            else:
                                points_in_lidar = back_points_in_lidar
                            points_in_lidar = points_in_lidar.astype(np.float32)

                            if save_single_frame_data:
                                cur_frame_save_path = os.path.join(single_frame_dir, '{:03d}.bin'.format(cur_frame_idx))
                                points_in_lidar.tofile(cur_frame_save_path)
                                cur_info['accumulate_lidar_path'] = cur_frame_save_path


                            # save sample data lidarseg
                            lidarseg_background_points_frame, lidarseg_background_labels_frame = transform_lidarseg_back_global2lidar(
                                lidarseg_background_points,
                                lidarseg_background_labels,
                                lidar2global,
                                filter=True)
                            lidarseg_background_points_frame = lidarseg_background_points_frame.astype(np.float32)
                            assert len(lidarseg_background_points_frame) == len(lidarseg_background_labels_frame)
                            

                            ###########################################################################
                            if save_bev_images:
                                bev_save_dir = os.path.join(scene_save_dir, 'bev_images')
                                os.makedirs(bev_save_dir, exist_ok=True)


                                lidar_pc = cur_info['token']
                                lidar_pc= log_db.session.query(LidarPc) \
                                .filter(LidarPc.token == lidar_pc) \
                                .all()
                                lidar_pc = lidar_pc[0]
                                vis_point = lidar_pc.load(log_db).points.T
                                gt_boxes_st = cur_info['gt_boxes_st']
                                gt_names = cur_info['gt_names']
                                bg_boxes_st = cur_info['bg_boxes_st']
                                bg_names = cur_info['bg_names']

                                all_boxes_st = np.concatenate((gt_boxes_st,bg_boxes_st),0)
                                all_names = np.concatenate((gt_names,bg_names),0)
                                # fig_save_dir = os.path.join('./train', scene_name)
                                # os.makedirs(fig_save_dir, exist_ok=True)
                                bev_image_path = os.path.join(bev_save_dir, '{:03d}.jpg'.format(cur_frame_idx))
                                # fig_path = os.path.join(fig_save_dir, (str(cur_frame_idx)+'.png'))
                                # print(fig_path)
                                # point_bev(lidarseg_background_points_frame,gt_boxes_st, fig_path)
                                point_bev(points_in_lidar,all_boxes_st,all_names, bev_image_path)
                            # continue
                            ###########################################################################
                            
                            if save_single_frame_data:
                                lidarseg_point_path = os.path.join(single_frame_dir, '{:03d}_lidarseg_back_points.bin'.format(cur_frame_idx))
                                lidarseg_background_points_frame.tofile(lidarseg_point_path)
                                cur_info['lidarseg_back_points_path'] = None
                                lidarseg_label_path = os.path.join(single_frame_dir, '{:03d}_lidarseg_back_labels.npy'.format(cur_frame_idx))
                                np.save(lidarseg_label_path, lidarseg_background_labels_frame)
                                cur_info['lidarseg_label_path'] = None

                            # save the gt box
                            if save_single_frame_data:
                                result_box = {'gt_bboxes_3d': cur_info['gt_boxes_st'],
                                            'gt_labels_3d': cur_info['gt_names'],
                                            'gt_velocity': cur_info['gt_velocity']}
                                box_save_path = os.path.join(single_frame_dir, '{:03}_box.pkl'.format(cur_frame_idx))
                                mmcv.dump(result_box, box_save_path)

                            # save eight images
                            if save_surround_images:
                                img_save_dir = os.path.join(scene_save_dir, 'surround_images')
                                os.makedirs(img_save_dir, exist_ok=True)

                                cams = cur_info['cams']
                                
                                cam_types = ['CAM_L0', 'CAM_F0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_B0', 'CAM_R2']
                                height, width = 1080, 1920
                                images = []
                                gt_boxes = cur_info['gt_boxes_st']
                                gt_names = cur_info['gt_names']
                                velocity_3d = cur_info['gt_velocity_3d']
                                CLASSES = ['vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier', 'czone_sign', 'generic_object']
                                for cam_type in cam_types:
                                    data_path = cams[cam_type]['data_path']
                                    img = cv2.imread(data_path)
                                    cam2lidar = np.eye(4)
                                    cam2lidar[:3, :3] = cams[cam_type]['sensor2lidar_rotation']
                                    cam2lidar[:3, 3] = cams[cam_type]['sensor2lidar_translation']
                                    lidar2cam_rt = np.linalg.inv(cam2lidar)

                                    intrinsic = cams[cam_type]['cam_intrinsic']
                                    viewpad = np.eye(4)
                                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                                    img = img_box_vis(img, gt_boxes, gt_names, lidar2cam_rt, viewpad, velocity_3d)
                                    images.append(img)
                                six_image = np.zeros((height*3, width*3, 3))
                                six_image[:height, :width] = images[0]
                                six_image[:height, width:width*2] = images[1]
                                six_image[:height, width*2:width*3] = images[2]
                                six_image[height:2*height, :width] = images[3]
                                six_image[height:2*height, width*2:width*3] = images[4]
                                six_image[2*height:3*height, :width] = images[5]
                                six_image[2*height:3*height, width:width*2] = images[6]
                                six_image[2*height:3*height, width*2:width*3] = images[7]
                                surround_image_path = os.path.join(img_save_dir, '{:03d}.jpg'.format(cur_frame_idx))
                                cv2.imwrite(surround_image_path, six_image)
                                cur_info['surround_image_path'] = surround_image_path
                            

                            generator = SingleFrameOCCGTGenerator(cur_info, scene_save_dir, train_flag=train_flag,
                                                                occ_resolution=occ_resolution, voxel_point_threshold=0,
                                                                save_flow_info=save_flow_info)

                            occ_paths = generator.save_occ_gt(points_in_lidar, points_velocitys, points_labels,
                                                lidarseg_background_points_frame, 
                                                lidarseg_background_labels_frame,is_nuplan)
                            
                            occ_gt_final_path, flow_gt_final_path, occ_invalid_path = occ_paths
                            cur_info['occ_gt_final_path'] = occ_gt_final_path
                            cur_info['flow_gt_final_path'] = flow_gt_final_path
                            cur_info['occ_invalid_path'] = occ_invalid_path

                            # save scene info
                            cur_scene_infos.append(cur_info)

                        mmcv.dump(cur_scene_infos, scene_pkl_file)
           
def img_box_vis(raw_image, gt_boxes, gt_names, lidar2cam_rt, viewpad, velocity_3d):
    lidar2img = (viewpad @ lidar2cam_rt)

    corners = LiDARInstance3DBoxes(gt_boxes, origin=(0.5, 0.5, 0.5)).corners
    image = raw_image.copy()
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    color_map = {'vehicle':(0, 255, 0), 'bicycle': (255, 0, 0), 'pedestrian': (0, 0, 255)}
    i = 0
    for box in corners:
        color = color_map[gt_names[i]]
        pts_4d = torch.cat([box, torch.ones_like(box[:, :1])], dim=-1)
        pts_2d = pts_4d @ lidar2img.T
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
        pts_2d[:, :2] /= pts_2d[:, 2:3]
        pts_2d = pts_2d[:, :2].numpy().astype(int)

        fov_inds = ((pts_2d[:, 0] < image.shape[1])
                    & (pts_2d[:, 0] >= 0)
                    & (pts_2d[:, 1] < image.shape[0])
                    & (pts_2d[:, 1] >= 0))
        if fov_inds.sum() < 1:
            i += 1
            continue
        for start, end in line_indices:
            cv2.line(image, (pts_2d[start, 0], pts_2d[start, 1]),
                        (pts_2d[end, 0], pts_2d[end, 1]), color, 1, cv2.LINE_AA)
        i += 1
    center = gt_boxes[:, :3]
    dst = center + velocity_3d * 0.5
    for idx in range(len(gt_boxes)):
        # color = color_map[gt_names[idx]]
        pts_3d = np.stack([center[idx], dst[idx]])
        pts_4d = np.concatenate([pts_3d, np.ones_like(pts_3d[:, :1])], axis=-1)
        pts_2d = pts_4d @ lidar2img.T
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
        pts_2d[:, :2] /= pts_2d[:, 2:3]
        pts_2d = pts_2d[:, :2].astype(int)
        fov_inds = ((pts_2d[:, 0] < image.shape[1])
                    & (pts_2d[:, 0] >= 0)
                    & (pts_2d[:, 1] < image.shape[0])
                    & (pts_2d[:, 1] >= 0))
        if fov_inds.sum() < 1:
            continue
        
        cv2.line(image, (pts_2d[0, 0], pts_2d[0, 1]), (pts_2d[1, 0], pts_2d[1, 1]), (255, 0, 255), 1, cv2.LINE_AA)

    return image
         


def obtain_sensor2top(sensor,
                      log_db,
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
    if sensor_type == 'lidar_pc_token':
        lidar_pc= log_db.session.query(LidarPc) \
        .filter(LidarPc.token == sensor) \
        .all()
        lidar_pc = lidar_pc[0]

        lidar_sensor = log_db.session.query(Lidar) \
        .filter(Lidar.token == lidar_pc.lidar_token) \
        .all()
        lidar_sensor = lidar_sensor[0]

        sweep = {
        'prev': lidar_pc.prev_token,
        'data_path': os.path.join(NUPLAN_SENSOR_ROOT, lidar_pc.filename),
        'type': lidar_sensor.channel,
        'sample_data_token': lidar_pc.token,
        'sensor2ego_translation': lidar_sensor.translation_np,
        'sensor2ego_rotation': lidar_sensor.quaternion,
        # 'ego2global_translation': [lidar_pc.ego_pose.x, lidar_pc.ego_pose.y, lidar_pc.ego_pose.z],
        # 'ego2global_rotation': [lidar_pc.ego_pose.qw, lidar_pc.ego_pose.qx, lidar_pc.ego_pose.qy, lidar_pc.ego_pose.qz],
        'ego2global_translation': lidar_pc.ego_pose.translation_np,
        'ego2global_rotation': [lidar_pc.ego_pose.qw, lidar_pc.ego_pose.qx, lidar_pc.ego_pose.qy, lidar_pc.ego_pose.qz],
        'timestamp': lidar_pc.timestamp
        }
    elif sensor_type == 'lidar':
        lidar_sensor = log_db.session.query(Lidar) \
        .filter(Lidar.token == sensor.lidar_token) \
        .all()
        lidar_sensor = lidar_sensor[0]

        sweep = {
        'prev': sensor.prev_token,
        'data_path': os.path.join(NUPLAN_SENSOR_ROOT, sensor.filename),
        'type': lidar_sensor.channel,
        'sample_data_token': sensor.token,
        'sensor2ego_translation': lidar_sensor.translation_np,
        'sensor2ego_rotation': lidar_sensor.quaternion,
        # 'ego2global_translation': [sensor.ego_pose.x, sensor.ego_pose.y, sensor.ego_pose.z],
        'ego2global_translation': sensor.ego_pose.translation_np,
        'ego2global_rotation': [sensor.ego_pose.qw, sensor.ego_pose.qx, sensor.ego_pose.qy, sensor.ego_pose.qz],
        'timestamp': sensor.timestamp
        }
    else:
        sweep = {
            'data_path': os.path.join(NUPLAN_SENSOR_ROOT, sensor.filename_jpg),
            'type': sensor.camera.channel,
            'sample_data_token': sensor.token,
            'sensor2ego_translation': sensor.camera.translation,
            'sensor2ego_rotation': sensor.camera.rotation,
            # 'ego2global_translation': [sensor.ego_pose.x, sensor.ego_pose.y, sensor.ego_pose.z],
            'ego2global_translation': sensor.ego_pose.translation_np,
            'ego2global_rotation': [sensor.ego_pose.qw, sensor.ego_pose.qx, sensor.ego_pose.qy, sensor.ego_pose.qz],
            'timestamp': sensor.timestamp
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
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def merge_scene(root_dir):
    # for data_type in ['train', 'val']:
    for data_type in ['trainval']:
        print('process scenes:', data_type)
        scenes_dir = os.path.join(root_dir, data_type)
        if not os.path.exists(scenes_dir):
            continue
        datas = []
        for scene in sorted(os.listdir(scenes_dir)):
            scene_path = os.path.join(scenes_dir, scene, 'scene_info.pkl')
            if os.path.exists(scene_path):
                data = mmcv.load(scene_path)
                datas.extend(data)
        # if data_type == 'train':
        #     assert len(datas) == 28130
        # else:
        #     assert len(datas) == 6019
        save_path = os.path.join(root_dir, 'nuscenes_infos_temporal_{}.pkl'.format(data_type))
        metadata = dict(version='v1.0-trainval')
        save_data = dict(infos=datas, metadata=metadata)
        mmcv.dump(save_data, save_path)

def split_scene(root_dir):
    for data_type in ['train']:
        print('process scenes:', data_type)
        scenes_dir = os.path.join(root_dir, data_type)
        if not os.path.exists(scenes_dir):
            continue
        datas = []
        scene_len = int(len(os.listdir(scenes_dir))*0.8)
        print("80")
        for scene in sorted(os.listdir(scenes_dir)[:scene_len]):
            print(scene)
            with open ('train_scenes.txt','a') as f:
                f.write(str(scene))
                f.write('\n')
            data = mmcv.load(os.path.join(scenes_dir, scene, 'scene_info.pkl'))
            datas.extend(data)
            
        # if data_type == 'train':
        #     assert len(datas) == 28130
        # else:
        #     assert len(datas) == 6019
        # save_path = os.path.join(root_dir, 'nuscenes_infos_temporal_80_{}.pkl'.format(data_type))
        # metadata = dict(version='v1.0-trainval')
        # save_data = dict(infos=datas, metadata=metadata)
        # mmengine.dump(save_data, save_path)
        #################################################################################################
        print("20")
        datas = []
        for scene in sorted(os.listdir(scenes_dir)[scene_len:]):
            print(scene)
            with open ('val_scenes.txt','a') as f:
                f.write(str(scene))
                f.write('\n')
            data = mmcv.load(os.path.join(scenes_dir, scene, 'scene_info.pkl'))
            datas.extend(data)
            
        # if data_type == 'train':
        #     assert len(datas) == 28130
        # else:
        #     assert len(datas) == 6019
        # save_path = os.path.join(root_dir, 'nuscenes_infos_temporal_20_{}.pkl'.format(data_type))
        # metadata = dict(version='v1.0-trainval')
        # save_data = dict(infos=datas, metadata=metadata)
        # mmengine.dump(save_data, save_path)

if __name__ == '__main__':
    

    occ_resolution = 'normal' 
    out_dir = '/cpfs01/user/zhouyunsong/zhouys/Datasets/nuPlan/occupancy'
    out_dir = '/cpfs01/user/zhouyunsong/zhouys/Datasets/nuPlan_PlayGround/occupancy_back_seg'
    # out_dir = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/zhouys/occupancy_all'
    
    # occ_resolution = 'coarse' 
    # out_dir = './data/nuscenes_occupancy_coarse'

    save_flow_info=True
    save_flow_info=False
    

    NUPLAN_DATA_ROOT = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/nuplan-v1.1/"
    NUPLAN_DB_FILES = f"{NUPLAN_DATA_ROOT}/splits/mini"
    NUPLAN_SENSOR_ROOT = f"{NUPLAN_DATA_ROOT}/sensor_blobs"
    # NUPLAN_SENSOR_ROOT = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/nuplan-v1.1/splits/trainval_sensor'
    NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
    NUPLAN_MAPS_ROOT = f"/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/maps"


    nuplandb_wrapper = NuPlanDBWrapper(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        db_files=NUPLAN_DB_FILES,
        map_version=NUPLAN_MAP_VERSION,
    )

    test = False
    # max_frame = 40
    # max_sweeps = 2
    # sweep_interval = 4

    # max_frame = 15
    max_sweeps = 1
    sweep_interval =5
    sample_interval = 10
    ilter_instance=True
    use_cuda=False
    is_nuplan = False
    scene_process_type='skip' # reprocess resume skip
    save_flow_info = False
    set_background_label=True

    # if multiprocessing.get_start_method() != 'spawn':
    #     multiprocessing.set_start_method('spawn')


    threads=[]
    # thread_num = 50
    for x in range(thread_num):
        t=Process(target=create_nuplan_info, name = str(x), args=(nuplandb_wrapper, test, max_sweeps, sweep_interval,sample_interval, out_dir, occ_resolution, save_flow_info, ilter_instance, use_cuda, is_nuplan, scene_process_type, set_background_label))
        threads.append(t)
    for thr in threads:
        thr.start()
    for thr in threads:
        if thr.is_alive():
            thr.join()

    # create_nuplan_info(nuplandb_wrapper, test=False, max_frame=10, max_sweeps=2, sweep_interval=0, out_dir=out_dir, occ_resolution=occ_resolution, save_flow_info=save_flow_info)

    # merge_scene(out_dir)
