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

# from utils import *
import cv2
import json
import os.path as osp
import yaml
# from single_frame_occ_generator import SingleFrameOCCGTGenerator
from tqdm import tqdm

# import mmengine
import os



# import sys
# sys.path.append('/home/liyang/zhouys/Git_repos/nuplan-devkit/nuplan')

# from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
# from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
# from nuplan.database.nuplan_db_orm.image import Image
# from nuplan.database.nuplan_db_orm.lidar import Lidar
# from nuplan.database.nuplan_db_orm.category import Category
# from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
# from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
# from nuplan.database.nuplan_db_orm.log import Log
# from nuplan.database.nuplan_db_orm.scene import Scene
# from nuplan.database.nuplan_db_orm.scenario_tag import ScenarioTag
# from nuplan.database.nuplan_db_orm.traffic_light_status import TrafficLightStatus
# from nuplan.database.nuplan_db_orm.track import Track
# from nuplan.database.nuplan_db.nuplan_scenario_queries import (
#     get_ego_state_for_lidarpc_token_from_db,
#     get_end_sensor_time_from_db,
#     get_images_from_lidar_tokens,
# )



from os import listdir
from os.path import isfile, join
import pickle

root_dir = '/cpfs01/user/liyang/zhouys/Datasets/nuPlan/occupancy'
root_dir = '/cpfs01/user/zhouyunsong/zhouys/Datasets/nuPlan_PlayGround/occupancy'
# root_dir = '/nvme/liyang/Datasets/nuScenes/occupancy'

origin_path = ''
new_path = ''
process_mode = 'none' # chaneg or add


def convert_data(data):
    cur_scene_infos = []
    for cur_info in data:
        occ_gt_final_path = cur_info['occ_gt_final_path']
        flow_gt_final_path = cur_info['flow_gt_final_path']
        occ_invalid_path = cur_info['occ_invalid_path']
        if process_mode == 'change':
            cur_info['occ_gt_final_path'] = occ_gt_final_path.replace(origin_path, new_path)
            cur_info['flow_gt_final_path'] = flow_gt_final_path.replace(origin_path, new_path)
            cur_info['occ_invalid_path'] = occ_invalid_path.replace(origin_path, new_path)
        elif process_mode == 'add':
            cur_info['occ_gt_final_path'] = os.path.join(new_path, occ_gt_final_path)
            cur_info['flow_gt_final_path'] = os.path.join(new_path, flow_gt_final_path)
            cur_info['occ_invalid_path'] = os.path.join(new_path, occ_invalid_path)

        cams = cur_info['cams']

        cam_types = ['CAM_L0', 'CAM_F0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_B0', 'CAM_R2']
        for cam_type in cam_types:
            data_path = cams[cam_type]['data_path']
            if process_mode == 'change':
                cams[cam_type]['data_path'] = data_path.replace(origin_path, new_path)
            elif process_mode == 'add':
                cams[cam_type]['data_path'] = os.path.join(new_path, data_path)

        cur_scene_infos.append(cur_info)
    return cur_scene_infos


for data_type in ['train_fixed']:
    print('process scenes:', data_type)
    scenes_dir = os.path.join(root_dir, data_type)
    if not os.path.exists(scenes_dir):
        continue
    datas = []
    for scene in sorted(os.listdir(scenes_dir)):
        scene_path = os.path.join(scenes_dir, scene, 'scene_info.pkl')
        if os.path.exists(scene_path):
            # data = mmengine.load(scene_path)
            data = mmcv.load(scene_path)
            data = convert_data(data)
            datas.extend(data)
    if data_type == 'val':
        datas = datas[:int(0.6*len(datas))]
    # if data_type == 'train':
    #     assert len(datas) == 28130
    # else:
    #     assert len(datas) == 6019
    save_path = os.path.join(root_dir, 'nuscenes_infos_temporal_{}.pkl'.format(data_type))
    # save_path = './nuscenes_val.pkl'
    metadata = dict(version='v1.0-trainval')
    save_data = dict(infos=datas, metadata=metadata)
    # mmengine.dump(save_data, save_path)
    mmcv.dump(save_data, save_path)
    # with open(save_path, 'wb') as f:
    #     pickle.dump(save_data, f)
