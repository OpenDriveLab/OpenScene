import argparse
import shutil
from typing import Dict, List

# import mmcv
import numpy as np
from os import listdir
from os.path import isfile, join

from pyquaternion import Quaternion
import time
import cv2

from tqdm import tqdm

import os

import multiprocessing
import pickle
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.lidar import Lidar
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_traffic_light_status_for_lidarpc_token_from_db
)

from helpers.multiprocess_helper import get_scenes_per_thread
from helpers.canbus import CanBus
from helpers.driving_command import get_driving_command
from helpers.nuplan_cameras_utils import (
    get_log_cam_info, get_closest_start_idx, get_cam_info_from_lidar_pc
)

NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]
filtered_classes = ["traffic_cone", "barrier", "czone_sign", "generic_object"]

def create_nuplan_info(args):
    nuplan_sensor_root = args.nuplan_sensor_path
    # get all db files & assign db files for current thread.
    log_sensors = os.listdir(nuplan_sensor_root)
    nuplan_db_path = args.nuplan_db_path
    db_names_with_extension = [
        f for f in listdir(nuplan_db_path) if isfile(join(nuplan_db_path, f))]
    db_names = [name[:-3] for name in db_names_with_extension]
    db_names.sort()
    db_names_splited, start = get_scenes_per_thread(db_names, args.thread_num)
    log_idx = start

    # For each sequence...
    for log_db_name in db_names_splited:

        frame_infos = []
        scene_list = []
        broken_frame_tokens = []

        log_db = NuPlanDB(args.nuplan_root_path, join(nuplan_db_path, log_db_name + ".db"), None)
        log_name = log_db.log_name
        log_token = log_db.log.token
        map_location = log_db.log.map_version
        vehicle_name = log_db.log.vehicle_name

        map_api = get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_location)  # NOTE: lru cached

        log_file = os.path.join(nuplan_db_path, log_db_name + ".db")
        if log_db_name not in log_sensors:
            continue

        frame_idx = 0
        log_idx += 1

        # list (sequence) of point clouds (each frame).
        lidar_pc_list = log_db.lidar_pc
        lidar_pcs = lidar_pc_list

        log_cam_infos = get_log_cam_info(log_db.log)
        start_idx = get_closest_start_idx(log_db.log, lidar_pcs)

        # Find key_frames (controled by args.sample_interval)
        lidar_pc_list = lidar_pc_list[start_idx :: args.sample_interval]
        index = -1
        for lidar_pc in tqdm(lidar_pc_list, dynamic_ncols=True):
            index += 1
            # LiDAR attributes.
            lidar_pc_token = lidar_pc.token
            scene_token = lidar_pc.scene_token
            pc_file_name = lidar_pc.filename
            next_token = lidar_pc.next_token
            prev_token = lidar_pc.prev_token
            lidar_token = lidar_pc.lidar_token
            time_stamp = lidar_pc.timestamp
            scene_name = f"log-{log_idx:04d}-{lidar_pc.scene.name}"
            lidar_boxes = lidar_pc.lidar_boxes
            roadblock_ids = [
                str(roadblock_id)
                for roadblock_id in str(lidar_pc.scene.roadblock_ids).split(" ")
                if len(roadblock_id) > 0
            ]

            if scene_token not in scene_list:
                scene_list.append(scene_token)
                frame_idx = 0

            can_bus = CanBus(lidar_pc).tensor
            lidar = log_db.session.query(Lidar).filter(Lidar.token == lidar_token).all()
            pc_file_path = os.path.join(args.nuplan_sensor_path, pc_file_name)
            if not os.path.exists(pc_file_path):  # some lidar files are missing.
                broken_frame_tokens.append(lidar_pc_token)
                frame_str = f"{log_db_name}, {lidar_pc_token}"
                tqdm.write(f"missing lidar files: {frame_str}")
                continue

            traffic_lights = []
            for traffic_light_status in get_traffic_light_status_for_lidarpc_token_from_db(
                log_file, lidar_pc_token
            ):
                lane_connector_id: int = traffic_light_status.lane_connector_id
                is_red: bool = traffic_light_status.status.value == 2
                traffic_lights.append((lane_connector_id, is_red))

            ego_pose = StateSE2(
                lidar_pc.ego_pose.x,
                lidar_pc.ego_pose.y,
                lidar_pc.ego_pose.quaternion.yaw_pitch_roll[0],
            )
            driving_command = get_driving_command(ego_pose, map_api, roadblock_ids)

            info = {
                "token": lidar_pc_token,
                "frame_idx": frame_idx,
                "timestamp": time_stamp,
                "log_name": log_name,
                "log_token": log_token,
                "scene_name": scene_name,
                "scene_token": scene_token,
                "map_location": map_location,
                "roadblock_ids": roadblock_ids,
                "vehicle_name": vehicle_name,
                "can_bus": can_bus,
                "lidar_path": pc_file_name,  # use the relative path.
                "lidar2ego_translation": lidar[0].translation_np,
                "lidar2ego_rotation": [
                    lidar[0].rotation.w,
                    lidar[0].rotation.x,
                    lidar[0].rotation.y,
                    lidar[0].rotation.z,
                ],
                "ego2global_translation": can_bus[:3],
                "ego2global_rotation": can_bus[3:7],
                "ego_dynamic_state": [
                    lidar_pc.ego_pose.vx,
                    lidar_pc.ego_pose.vy,
                    lidar_pc.ego_pose.acceleration_x,
                    lidar_pc.ego_pose.acceleration_y,
                ],
                "traffic_lights": traffic_lights,
                "driving_command": driving_command, 
                "cams": dict(),
            }
            info["sample_prev"] = None
            info["sample_next"] = None

            if index > 0:  # find prev.
                info["sample_prev"] = lidar_pc_list[index - 1].token
            if index < len(lidar_pc_list) - 1:  # find next.
                next_key_token = lidar_pc_list[index + 1].token
                next_key_scene = lidar_pc_list[index + 1].scene_token
                info["sample_next"] = next_key_token
            else:
                next_key_token, next_key_scene = None, None

            if next_key_token == None or next_key_token == "":
                frame_idx = 0
            else:
                if next_key_scene != scene_token:
                    frame_idx = 0
                else:
                    frame_idx += 1

            # Parse lidar2ego translation.
            l2e_r = info["lidar2ego_rotation"]
            l2e_t = info["lidar2ego_translation"]
            e2g_r = info["ego2global_rotation"]
            e2g_t = info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # add lidar2global: map point coord in lidar to point coord in the global
            l2e = np.eye(4)
            l2e[:3, :3] = l2e_r_mat
            l2e[:3, -1] = l2e_t
            e2g = np.eye(4)
            e2g[:3, :3] = e2g_r_mat
            e2g[:3, -1] = e2g_t
            lidar2global = np.dot(e2g, l2e)
            info["ego2global"] = e2g
            info["lidar2ego"] = l2e
            info["lidar2global"] = lidar2global

            # obtain 8 image's information per frame
            cams = get_cam_info_from_lidar_pc(log_db.log, lidar_pc, log_cam_infos)
            if cams == None:
                broken_frame_tokens.append(lidar_pc_token)
                frame_str = f"{log_db_name}, {lidar_pc_token}"
                tqdm.write(f"not all cameras are available: {frame_str}")
                continue
            info["cams"] = cams

            # Parse 3D object labels.
            if not args.is_test:
                if args.filter_instance:
                    fg_lidar_boxes = [
                        box for box in lidar_boxes if box.category.name not in filtered_classes
                    ]
                else:
                    fg_lidar_boxes = lidar_boxes

                instance_tokens = [item.token for item in fg_lidar_boxes]
                track_tokens = [item.track_token for item in fg_lidar_boxes]

                inv_ego_r = lidar_pc.ego_pose.trans_matrix_inv
                ego_yaw = lidar_pc.ego_pose.quaternion.yaw_pitch_roll[0]

                locs = np.array(
                    [
                        np.dot(
                            inv_ego_r[:3, :3],
                            (b.translation_np - lidar_pc.ego_pose.translation_np).T,
                        ).T
                        for b in fg_lidar_boxes
                    ]
                ).reshape(-1, 3)
                dims = np.array([[b.length, b.width, b.height] for b in fg_lidar_boxes]).reshape(
                    -1, 3
                )
                rots = np.array([b.yaw for b in fg_lidar_boxes]).reshape(-1, 1)
                rots = rots - ego_yaw

                velocity_3d = np.array([[b.vx, b.vy, b.vz] for b in fg_lidar_boxes]).reshape(-1, 3)
                for i in range(len(fg_lidar_boxes)):
                    velo = velocity_3d[i]
                    velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    velocity_3d[i] = velo

                names = [box.category.name for box in fg_lidar_boxes]
                names = np.array(names)
                gt_boxes_nuplan = np.concatenate([locs, dims, rots], axis=1)
                info["anns"] = dict(
                    gt_boxes=gt_boxes_nuplan,
                    gt_names=names,
                    gt_velocity_3d=velocity_3d.reshape(-1, 3),
                    instance_tokens=instance_tokens,
                    track_tokens=track_tokens,
                )
            frame_infos.append(info)

        del map_api

        # after check.
        for info in frame_infos:
            if info["sample_prev"] in broken_frame_tokens:
                info["sample_prev"] = None
            if info["sample_next"] in broken_frame_tokens:
                info["sample_next"] = None

        pkl_file_path = f"{args.out_dir}/{log_name}.pkl"
        os.makedirs(args.out_dir, exist_ok=True)

        with open(pkl_file_path, "wb") as f:
            pickle.dump(frame_infos, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument(
        "--thread-num", type=int, default=16, help="number of threads for multi-processing."
    )

    # directory configurations.
    parser.add_argument("--nuplan-root-path", help="the path to nuplan root path.")
    parser.add_argument("--nuplan-db-path", help="the dir saving nuplan db.")
    parser.add_argument("--nuplan-sensor-path", help="the dir to nuplan sensor data.")
    parser.add_argument("--nuplan-map-version", help="nuplan mapping dataset version.")
    parser.add_argument("--nuplan-map-root", help="path to nuplan map data.")
    parser.add_argument("--out-dir", help="output path.")

    parser.add_argument(
        "--sample-interval", type=int, default=10, help="interval of key frame samples."
    )

    # split.
    parser.add_argument("--is-test", action="store_true", help="Dealing with Test set data.")
    parser.add_argument(
        "--filter-instance", action="store_true", help="Ignore instances in filtered_classes."
    )
    parser.add_argument("--split", type=str, default="train", help="Train/Val/Test set.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    nuplan_root_path = args.nuplan_root_path
    nuplan_db_path = args.nuplan_db_path
    nuplan_sensor_path = args.nuplan_sensor_path
    nuplan_map_version = args.nuplan_map_version
    nuplan_map_root = args.nuplan_map_root
    out_dir = args.out_dir

    manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    threads = []
    for x in range(args.thread_num):
        t = multiprocessing.Process(
            target=create_nuplan_info,
            name=str(x),
            args=(args,),
        )
        threads.append(t)
    for thr in threads:
        thr.start()
    for thr in threads:
        if thr.is_alive():
            thr.join()
