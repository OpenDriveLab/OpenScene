import numpy as np


class CanBus:
    """Wrapper class to convert lidar_can_bus to numpy array"""

    def __init__(self, lidar_pc):
        self.x = lidar_pc.ego_pose.x
        self.y = lidar_pc.ego_pose.y
        self.z = lidar_pc.ego_pose.z

        self.qw = lidar_pc.ego_pose.qw
        self.qx = lidar_pc.ego_pose.qx
        self.qy = lidar_pc.ego_pose.qy
        self.qz = lidar_pc.ego_pose.qz

        self.acceleration_x = lidar_pc.ego_pose.acceleration_x
        self.acceleration_y = lidar_pc.ego_pose.acceleration_y
        self.acceleration_z = lidar_pc.ego_pose.acceleration_z

        self.vx = lidar_pc.ego_pose.vx
        self.vy = lidar_pc.ego_pose.vy
        self.vz = lidar_pc.ego_pose.vz

        self.angular_rate_x = lidar_pc.ego_pose.angular_rate_x
        self.angular_rate_y = lidar_pc.ego_pose.angular_rate_y
        self.angular_rate_z = lidar_pc.ego_pose.angular_rate_z

        self.tensor = np.array(
            [
                self.x,
                self.y,
                self.z,
                self.qw,
                self.qx,
                self.qy,
                self.qz,
                self.acceleration_x,
                self.acceleration_y,
                self.acceleration_z,
                self.vx,
                self.vy,
                self.vz,
                self.angular_rate_x,
                self.angular_rate_y,
                self.angular_rate_z,
                0.0,
                0.0,
            ]
        )
