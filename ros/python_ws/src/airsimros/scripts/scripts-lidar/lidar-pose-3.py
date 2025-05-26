#!/usr/bin/env python3

import cosysairsim as airsim
import rospy
import struct
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
import tf2_ros

class AirSimLidarPosePublisher:
    def __init__(self,
                 vehicle_name="Car1",
                 lidar_name="GPULidar",
                 lidar_topic="lidar_points",
                 pose_topic="vehicle_pose",
                 path_topic="vehicle_path",
                 frame_id="base_link",
                 parent_frame_id="map",
                 rate_hz=5):
        # AirSim client
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name
        self.frame_id = frame_id
        self.parent_frame_id = parent_frame_id

        # ROS init
        rospy.init_node('airsim_lidar_pose_publisher', anonymous=True)
        self.lidar_pub = rospy.Publisher(lidar_topic, PointCloud2, queue_size=1)
        self.pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
        self.path_pub = rospy.Publisher(path_topic, Path, queue_size=1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(rate_hz)

        # Path accumulator
        self.path = Path()
        self.path.header.frame_id = self.parent_frame_id

    def get_lidar_points(self):
        data = self.client.getLidarData(self.lidar_name, self.vehicle_name)
        if not data or len(data.point_cloud) < 3:
            rospy.logwarn("No LiDAR data")
            return None
        points = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
        # Convert NED to ENU for ROS
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
        pc2 = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * points.shape[0],
            is_dense=True,
            data=struct.pack('f'*points.size, *points.flatten())
        )
        return pc2

    def get_pose(self):
        # simGetVehiclePose returns NED pose
        sim_pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = self.parent_frame_id
        # position convert NED->ENU
        p = sim_pose.position
        ps.pose.position.x = p.x_val
        ps.pose.position.y = -p.y_val
        ps.pose.position.z = -p.z_val
        # orientation quaternion inverse for ROS
        q = sim_pose.orientation.inverse()
        ps.pose.orientation.x = q.x_val
        ps.pose.orientation.y = q.y_val
        ps.pose.orientation.z = q.z_val
        ps.pose.orientation.w = q.w_val
        return ps

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()

            # Publish LiDAR point cloud
            points = self.get_lidar_points()
            if points is not None:
                pc2_msg = self.make_pointcloud2(points, now)
                self.lidar_pub.publish(pc2_msg)

            # Publish PoseStamped
            pose_msg = self.get_pose()
            self.pose_pub.publish(pose_msg)

            # Update and publish path
            self.path.header.stamp = pose_msg.header.stamp
            self.path.poses.append(pose_msg)
            self.path_pub.publish(self.path)

            # Broadcast transform map -> base_link
            tf_msg = TransformStamped()
            tf_msg.header.stamp = pose_msg.header.stamp
            tf_msg.header.frame_id = self.parent_frame_id
            tf_msg.child_frame_id = self.frame_id
            tf_msg.transform.translation.x = pose_msg.pose.position.x
            tf_msg.transform.translation.y = pose_msg.pose.position.y
            tf_msg.transform.translation.z = pose_msg.pose.position.z
            tf_msg.transform.rotation = pose_msg.pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        publisher = AirSimLidarPosePublisher(
            vehicle_name="Car1",
            lidar_name="GPULidar",
            lidar_topic="/airsim/lidar",
            pose_topic="/airsim/pose",
            path_topic="/airsim/path",
            frame_id="base_link",
            parent_frame_id="map",
            rate_hz=5
        )
        publisher.run()
    except rospy.ROSInterruptException:
        pass
