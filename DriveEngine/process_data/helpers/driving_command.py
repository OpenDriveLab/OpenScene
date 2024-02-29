from re import A
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import numpy.typing as npt

from shapely.geometry import Point

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)

from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import (
    get_current_roadblock_candidates,
)
from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import (
    Dijkstra,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    SE2Index,
)

from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.bfs_roadblock import (
    BreadthFirstSearchRoadBlock,
)

# shapely runtime warning
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_driving_command(
    ego_pose: StateSE2,
    map_api: AbstractMap,
    route_roadblock_ids: List[str],
    distance: float = 20,
    lateral_offset: float = 2,
) -> npt.NDArray[np.int]:
    """
    Creates the one-hot (left, forward, right, unknown) driving command for the ego vehicle
    :param ego_pose: (x, y, heading) object for global ego pose
    :param map_api: nuPlan's map interface
    :param route_roadblock_ids: roadblock ids of route
    :param distance: longitudinal distance to interpolate on th centerline, defaults to 10
    :param lateral_offset: lateral offset for left/right threshold, to defaults to 2
    :return: numpy one-hot array of driving_command
    """

    driving_command = np.zeros((4,), dtype=int)  # one-hot: (left, forward, right, unknown)

    # Apply route correction on route_roadblock_ids
    route_roadblock_dict, _ = get_route_dicts(map_api, route_roadblock_ids)
    corrected_route_roadblock_ids = route_roadblock_correction(
        ego_pose, map_api, route_roadblock_dict
    )

    # If no route is available or route is too damaged, return unknown command
    if corrected_route_roadblock_ids is None:
        driving_command[-1] = 1
        return driving_command

    route_roadblock_dict, route_lane_dict = get_route_dicts(map_api, corrected_route_roadblock_ids)

    # Find the nearest lane, graph search, and centerline extraction
    current_lane = get_current_lane(ego_pose, route_lane_dict)
    discrete_centerline = get_discrete_centerline(
        current_lane, route_roadblock_dict, route_lane_dict
    )
    centerline = PDMPath(discrete_centerline)

    # Interpolate target distance on centerline
    current_progress = centerline.project(Point(*ego_pose.array))
    target_progress = current_progress + distance

    current_pose_array, target_pose_array = centerline.interpolate(
        [current_progress, target_progress], as_array=True
    )
    target_pose_array = convert_absolute_to_relative_se2_array(
        StateSE2(*current_pose_array), target_pose_array[None,...]
    )[0]

    # Threshold for driving command
    if target_pose_array[SE2Index.Y] >= lateral_offset:
        driving_command[0] = 1
    elif target_pose_array[SE2Index.Y] <= -lateral_offset:
        driving_command[2] = 1
    else:
        driving_command[1] = 1

    # delete some variables for memory management
    del route_roadblock_dict, route_lane_dict, _, centerline
    return driving_command


def get_route_dicts(
    map_api: AbstractMap, route_roadblock_ids: List[str]
) -> Tuple[Dict[str, RoadBlockGraphEdgeMapObject], Dict[str, LaneGraphEdgeMapObject]]:
    """
    Loads the roadblock and lane dicts
    :param map_api: nuPlan's map interface
    :param route_roadblock_ids: roadblock ids of route
    :return: tuple of roadblock and lane dicts
    """

    # remove repeated ids while remaining order in list
    route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject] = {}
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject] = {}

    for id_ in route_roadblock_ids:
        block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
        block = block or map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        route_roadblock_dict[block.id] = block
        for lane in block.interior_edges:
            route_lane_dict[lane.id] = lane

    return route_roadblock_dict, route_lane_dict


def get_current_lane(
    ego_pose: StateSE2, route_lane_dict: Dict[str, LaneGraphEdgeMapObject]
) -> LaneGraphEdgeMapObject:
    """
    Find current lane, either if intersection with ego pose, or by distance.
    :param ego_pose: (x, y, heading) object for global ego pose
    :param route_lane_dict: Dictionary of roadblock ids and objects
    :return: Lane object
    """

    closest_distance = np.inf
    starting_lane = None
    for edge in route_lane_dict.values():
        if edge.contains_point(ego_pose):
            starting_lane = edge
            break

        distance = edge.polygon.distance(Point(*ego_pose.array))
        if distance < closest_distance:
            starting_lane = edge
            closest_distance = distance

    return starting_lane


def get_discrete_centerline(
    current_lane: LaneGraphEdgeMapObject,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    search_depth: int = 30,
) -> List[StateSE2]:
    """
    Given the current lane, apply graph search, and extract centerline.
    :param current_lane: Lane object closest to ego
    :param route_roadblock_dict: Dictionary of roadblock ids and objects
    :param route_lane_dict: Dictionary of lane ids and objects
    :param search_depth: max search depth of Dijkstra, defaults to 30
    :return: List of (x, y, heading) objects
    """

    roadblocks = list(route_roadblock_dict.values())
    roadblock_ids = list(route_roadblock_dict.keys())

    # find current roadblock index
    start_idx = np.argmax(np.array(roadblock_ids) == current_lane.get_roadblock_id())
    roadblock_window = roadblocks[start_idx : start_idx + search_depth]

    graph_search = Dijkstra(current_lane, list(route_lane_dict.keys()))
    route_plan, _ = graph_search.search(roadblock_window[-1])

    centerline_discrete_path: List[StateSE2] = []
    for lane in route_plan:
        centerline_discrete_path.extend(lane.baseline_path.discrete_path)

    return centerline_discrete_path

def route_roadblock_correction(
    ego_pose: StateSE2,
    map_api: AbstractMap,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
) -> Optional[List[str]]:
    """
    Applies several methods to correct route roadblocks.
    :param ego_pose: class containing ego position
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param search_depth_backward: depth of forward BFS search, defaults to 15
    :param search_depth_forward:  depth of backward BFS search, defaults to 30
    :return: list of roadblock id's of corrected route
    """
    
    if len(route_roadblock_dict) == 0:
        return None
    
    # TODO: Refactor code for readability
    starting_block, starting_block_candidates = get_current_roadblock_candidates(
        ego_pose, map_api, route_roadblock_dict
    )
    starting_block_ids = [roadblock.id for roadblock in starting_block_candidates]

    route_roadblocks = list(route_roadblock_dict.values())
    route_roadblock_ids = list(route_roadblock_dict.keys())

    # Fix 1: when agent starts off-route
    if starting_block.id not in route_roadblock_ids:
        # Backward search if current roadblock not in route
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[0], map_api, forward_search=False
        )
        (path, path_id), path_found = graph_search.search(
            starting_block_ids, max_depth=search_depth_backward
        )

        if path_found:
            route_roadblocks[:0] = path[:-1]
            route_roadblock_ids[:0] = path_id[:-1]
            
        else:
            return None

    # Fix 2: check if roadblocks are linked, search for links if not
    roadblocks_to_append = {}
    for i in range(len(route_roadblocks) - 1):
        next_incoming_block_ids = [
            _roadblock.id for _roadblock in route_roadblocks[i + 1].incoming_edges
        ]
        is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue

        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[i], map_api, forward_search=True
        )
        (path, path_id), path_found = graph_search.search(
            route_roadblock_ids[i + 1], max_depth=search_depth_forward
        )

        if path_found and path and len(path) >= 3:
            path, path_id = path[1:-1], path_id[1:-1]
            roadblocks_to_append[i] = (path, path_id)

    # append missing intermediate roadblocks
    offset = 1
    for i, (path, path_id) in roadblocks_to_append.items():
        route_roadblocks[i + offset : i + offset] = path
        route_roadblock_ids[i + offset : i + offset] = path_id
        offset += len(path)
        
    return route_roadblock_ids
