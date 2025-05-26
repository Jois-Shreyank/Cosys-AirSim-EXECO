#!/usr/bin/env python3

import cosysairsim as airsim
import rospy
import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import time

class AirSimLidarPublisher:
    def __init__(self, vehicle_name="Car1", topic_name="lidar_points", lidar_name="GPULidar"):
        # Initialize the AirSim client
        self.client = airsim.CarClient()
        self.client.confirmConnection()

        # Set the vehicle name and lidar sensor name
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name

        # Initialize the ROS node
        rospy.init_node('airsim_lidar_to_ros', anonymous=True)

        # Create a publisher for the lidar point cloud
        self.lidar_pub = rospy.Publisher(topic_name, PointCloud2, queue_size=10)

        # Set the loop rate (10 Hz)
        self.rate = rospy.Rate(5)

    def get_lidar_data(self):
        # Get LiDAR data from the AirSim client
        lidar_data = self.client.getLidarData(self.lidar_name)

        if lidar_data:
            # The point cloud data from AirSim is in lidar_data.point_cloud (list of floats)
            lidar_points = np.array(lidar_data.point_cloud).reshape(-1, 3)
            lidar_points[:, 2] = -lidar_points[:, 2]
            lidar_points[:, 1] = -lidar_points[:, 1]
            return lidar_points
        else:
            rospy.logwarn("No LiDAR data received!")
            return None

    def create_pointcloud2(self, points, header):
        """Convert points to a PointCloud2 message"""
        # Create a flat array with all the points packed together in a single list
        pc_data = np.array(points, dtype=np.float32).flatten()

        # Define the fields of the PointCloud2 message (x, y, z)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        # Create and populate the PointCloud2 message
        pc2 = PointCloud2()
        pc2.header = header
        pc2.height = 1  # since we have a single row of points
        pc2.width = len(points)  # number of points
        pc2.fields = fields
        pc2.is_bigendian = False
        pc2.point_step = 12  # x, y, z, each float32 (4 bytes)
        pc2.row_step = 12 * len(points)
        pc2.is_dense = True
        pc2.data = struct.pack('f' * len(pc_data), *pc_data)  # packing data

        return pc2

    def publish_lidar_points(self):
        while not rospy.is_shutdown():
            # Get LiDAR points from AirSim
            lidar_points = self.get_lidar_data()

            if lidar_points is not None:
                # Convert the lidar points to ROS PointCloud2 message
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "base_link"  # Or any other frame you're using for the vehicle

                # Create the PointCloud2 message
                pc_data = self.create_pointcloud2(lidar_points, header)

                # Publish the PointCloud2 message to ROS topic
                self.lidar_pub.publish(pc_data)

            # Sleep to maintain the loop rate
            self.rate.sleep()

if __name__ == '__main__':
    try:
        # Initialize and start the Lidar publisher
        lidar_publisher = AirSimLidarPublisher(vehicle_name="Car1", lidar_name="GPULidar")
        lidar_publisher.publish_lidar_points()
    except rospy.ROSInterruptException:
        pass
