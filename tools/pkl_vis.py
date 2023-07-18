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

import cv2
import json
import os.path as osp
import yaml
from tqdm import tqdm

import mmengine
import os
import json



# import sys
# sys.path.append('/home/liyang/zhouys/Git_repos/nuplan-devkit/nuplan')

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
)



from os import listdir
from os.path import isfile, join


scene_path = '/nvme/liyang/Datasets/nuPlan/occupancy/nuscenes_infos_temporal_trainval.pkl'
# scene_path = '/nvme/liyang/Datasets/nuScenes/occupancy/nuscenes_infos_temporal_val_occ_gt.pkl'
data = mmcv.load(scene_path)
print(data)
# with open('val_data.json', 'a') as f:
#     json.dump(data, f)











