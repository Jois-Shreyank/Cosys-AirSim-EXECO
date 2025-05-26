#!/usr/bin/env python3

import cosysairsim as airsim
import rospy
import struct
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

class AirSimLidarPosePublisher:
    def __init__(self,
                 vehicle_name="Car1",
                 lidar_name="GPULidar",
                 lidar_topic="lidar_points",
                 pose_topic="vehicle_pose",
                 frame_id="base_link",
                 rate_hz=5):
        # AirSim client
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name
        self.frame_id = frame_id

        # ROS init
        rospy.init_node('airsim_lidar_pose_publisher', anonymous=True)
        self.lidar_pub = rospy.Publisher(lidar_topic, PointCloud2, queue_size=1)
        self.pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
        self.rate = rospy.Rate(rate_hz)

    def get_lidar_points(self):
        data = self.client.getLidarData(self.lidar_name, self.vehicle_name)
        if not data or len(data.point_cloud) < 3:
            rospy.logwarn("No Lidar data")
            return None
        points = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
        # Convert AirSim axes to ROS (if needed)
        points[:,1] *= -1
        points[:,2] *= -1
        return points

    def make_pointcloud2(self, points, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        pc = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * points.shape[0],
            is_dense=True,
            data=struct.pack('fff'*points.shape[0], *points.flatten())
        )
        return pc

    def get_pose(self):
        pose = self.client.simGetVehiclePose(self.vehicle_name)
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = self.frame_id
        # position
        p = pose.position
        ps.pose.position.x = p.x_val
        ps.pose.position.y = -p.y_val
        ps.pose.position.z = -p.z_val
        # orientation (quaternion)
        q = pose.orientation.inverse()
        ps.pose.orientation.x = q.x_val
        ps.pose.orientation.y = q.y_val
        ps.pose.orientation.z = q.z_val
        ps.pose.orientation.w = q.w_val
        return ps

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            # publish lidar
            pts = self.get_lidar_points()
            if pts is not None:
                pc2 = self.make_pointcloud2(pts, now)
                self.lidar_pub.publish(pc2)
            # publish pose
            pose_msg = self.get_pose()
            self.pose_pub.publish(pose_msg)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        pub = AirSimLidarPosePublisher(
            vehicle_name="Car1",
            lidar_name="GPULidar",
            lidar_topic="/airsim/lidar",
            pose_topic="/airsim/pose",
            frame_id="base_link",
            rate_hz=5
        )
        pub.run()
    except rospy.ROSInterruptException:
        pass
