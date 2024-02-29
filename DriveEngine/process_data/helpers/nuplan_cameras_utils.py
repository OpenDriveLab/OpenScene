import os
import copy
import shapely
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
import numpy as np
import mmcv
from pyquaternion import Quaternion
import pickle
import cv2
import matplotlib.pyplot as plt
import torch

from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_images_from_lidar_tokens,
    get_sensor_data_from_sensor_data_tokens_from_db,
    get_cameras,
    get_scenarios_from_db,
    get_lidarpc_tokens_with_scenario_tag_from_db
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario, CameraChannel, LidarChannel

NUPLAN_DB_PATH = os.environ["NUPLAN_DB_PATH"]
NUPLAN_SENSOR_PATH = os.environ["NUPLAN_SENSOR_PATH"]

def get_log_cam_info(log):
    log_name = log.logfile
    log_file = os.path.join(NUPLAN_DB_PATH, log_name + '.db')

    log_cam_infos = {}
    for cam in get_cameras(log_file, [str(channel.value) for channel in CameraChannel]):
        intrinsics = np.array(pickle.loads(cam.intrinsic))
        translation = np.array(pickle.loads(cam.translation))
        rotation = np.array(pickle.loads(cam.rotation))
        rotation = Quaternion(rotation).rotation_matrix
        distortion = np.array(pickle.loads(cam.distortion))
        c = dict(
            intrinsic=intrinsics,
            distortion=distortion,
            translation=translation,
            rotation=rotation,
        )
        log_cam_infos[cam.token] = c

    return log_cam_infos

def get_closest_start_idx(log, lidar_pcs):
    log_name = log.logfile
    log_file = os.path.join(NUPLAN_DB_PATH, log_name + '.db')

    # Find the first valid point clouds, with all 8 cameras available.
    for start_idx in range(0, len(lidar_pcs)):
        retrieved_images = get_images_from_lidar_tokens(
            log_file, [lidar_pcs[start_idx].token], [str(channel.value) for channel in CameraChannel]
        )
        if len(list(retrieved_images)) == 8:
            break

    # Find the true LiDAR start_idx with the minimum timestamp difference with CAM_F0.
    retrieved_images = get_images_from_lidar_tokens(
        log_file, [lidar_pcs[start_idx].token], ['CAM_F0']
    )
    diff_0 = abs(next(retrieved_images).timestamp - lidar_pcs[start_idx].timestamp)

    retrieved_images = get_images_from_lidar_tokens(
        log_file, [lidar_pcs[start_idx + 1].token], ['CAM_F0']
    )
    diff_1 = abs(next(retrieved_images).timestamp - lidar_pcs[start_idx + 1].timestamp)

    start_idx = start_idx if diff_0 < diff_1 else start_idx + 1
    return start_idx

def get_cam_info_from_lidar_pc(log, lidar_pc, log_cam_infos):
    log_name = log.logfile
    log_file = os.path.join(NUPLAN_DB_PATH, log_name + '.db')
    retrieved_images = get_images_from_lidar_tokens(
        log_file, [lidar_pc.token], [str(channel.value) for channel in CameraChannel]
    )
    cams = {}
    for img in retrieved_images:
        channel = img.channel
        filename = img.filename_jpg

        filepath = os.path.join(NUPLAN_SENSOR_PATH, filename)
        if not os.path.exists(filepath):
            return None
        cam_info = log_cam_infos[img.camera_token]
        cams[channel] = dict(
            data_path = filename,
            sensor2lidar_rotation = cam_info['rotation'],
            sensor2lidar_translation = cam_info['translation'],
            cam_intrinsic = cam_info['intrinsic'],
            distortion = cam_info['distortion'],
        )
    if len(cams) != 8:
        return None
    return cams
