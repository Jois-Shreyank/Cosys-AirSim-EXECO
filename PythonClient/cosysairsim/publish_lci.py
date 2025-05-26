#!/usr/bin/env python3
import struct
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import PointCloud2, Image, Imu
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from cv_bridge import CvBridge

import cosysairsim as airsim

class AirSimLidarPoseVisionImu(Node):
    def __init__(self,
                 vehicle_name="Car1",
                 lidar_name="GPULidar",
                 left_cam="camera_left",
                 right_cam="camera_right",
                 imu_name="IMU",
                 lidar_topic="/airsim/lidar",
                 pose_topic="/airsim/pose",
                 path_topic="/airsim/path",
                 left_image_topic="/airsim/left/image_raw",
                 right_image_topic="/airsim/right/image_raw",
                 imu_topic="/airsim/imu",
                 frame_id="base_link",
                 parent_frame_id="map",
                 rate_hz=5):

        super().__init__('airsim_lidar_pose_vision_imu')

        # AirSim client (multirotor)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name
        self.left_cam = left_cam
        self.right_cam = right_cam
        self.imu_name = imu_name
        self.frame_id = frame_id
        self.parent_frame_id = parent_frame_id

        # CV bridge
        self.bridge = CvBridge()

        # QoS
        qos = QoSProfile(depth=1)

        # Publishers
        self.lidar_pub  = self.create_publisher(PointCloud2, lidar_topic, qos)
        self.pose_pub   = self.create_publisher(PoseStamped,  pose_topic, qos)
        self.path_pub   = self.create_publisher(Path,         path_topic, qos)
        self.left_pub   = self.create_publisher(Image,        left_image_topic,  qos)
        self.right_pub  = self.create_publisher(Image,        right_image_topic, qos)
        self.imu_pub    = self.create_publisher(Imu,          imu_topic,         qos)

        # Dynamic TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        # Static TF broadcaster
        self.static_broadcaster = StaticTransformBroadcaster(self)
        # Publish fixed sensor → body transforms
        self.publish_static_transforms()

        # Path accumulator
        self.path = Path()
        self.path.header.frame_id = parent_frame_id

        # Timer
        self.timer = self.create_timer(1.0 / rate_hz, self.timer_callback)

    def publish_static_transforms(self):
        now = self.get_clock().now().to_msg()
        static_transforms = []

        # IMU relative to body
        imu_tf = TransformStamped()
        imu_tf.header.stamp = now
        imu_tf.header.frame_id = self.frame_id
        imu_tf.child_frame_id = 'imu_link'
        imu_tf.transform.translation.x = 0.0
        imu_tf.transform.translation.y = 0.0
        imu_tf.transform.translation.z = 0.0
        imu_tf.transform.rotation.x = 0.0
        imu_tf.transform.rotation.y = 0.0
        imu_tf.transform.rotation.z = 0.0
        imu_tf.transform.rotation.w = 1.0
        static_transforms.append(imu_tf)

        # Left camera relative to body
        camL_tf = TransformStamped()
        camL_tf.header.stamp = now
        camL_tf.header.frame_id = self.frame_id
        camL_tf.child_frame_id = self.left_cam + '_link'
        camL_tf.transform.translation.x = 2.00
        camL_tf.transform.translation.y = -0.10
        camL_tf.transform.translation.z = -1.10
        camL_tf.transform.rotation.x = 0.0
        camL_tf.transform.rotation.y = 0.0
        camL_tf.transform.rotation.z = 0.0
        camL_tf.transform.rotation.w = 1.0
        static_transforms.append(camL_tf)

        # Right camera relative to body
        camR_tf = TransformStamped()
        camR_tf.header.stamp = now
        camR_tf.header.frame_id = self.frame_id
        camR_tf.child_frame_id = self.right_cam + '_link'
        camR_tf.transform.translation.x = 2.00
        camR_tf.transform.translation.y = 0.10
        camR_tf.transform.translation.z = -1.10
        camR_tf.transform.rotation.x = 0.0
        camR_tf.transform.rotation.y = 0.0
        camR_tf.transform.rotation.z = 0.0
        camR_tf.transform.rotation.w = 1.0
        static_transforms.append(camR_tf)

        # Send them out
        self.static_broadcaster.sendTransform(static_transforms)

    def get_lidar_points(self):
        data = self.client.getLidarData(self.lidar_name, self.vehicle_name)
        if not data or len(data.point_cloud) < 3:
            self.get_logger().warn("No LiDAR data")
            return None
        pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
        # NED→ENU
        pts[:,1] *= -1
        pts[:,2] *= -1
        return pts

    def make_pointcloud2(self, points, header):
        return point_cloud2.create_cloud_xyz32(header, points.tolist())

    def get_pose(self, header):
        sim_pose = self.client.simGetVehiclePose(self.vehicle_name)
        ps = PoseStamped()
        ps.header = header
        ps.header.frame_id = self.parent_frame_id

        p = sim_pose.position
        ps.pose.position.x = float(p.x_val)
        ps.pose.position.y = float(-p.y_val)
        ps.pose.position.z = float(-p.z_val)

        q = sim_pose.orientation.inverse()
        ps.pose.orientation.x = float(q.x_val)
        ps.pose.orientation.y = float(q.y_val)
        ps.pose.orientation.z = float(q.z_val)
        ps.pose.orientation.w = float(q.w_val)
        return ps

    def get_images(self, header):
        reqs = [
            airsim.ImageRequest(self.left_cam,  airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(self.right_cam, airsim.ImageType.Scene, False, False)
        ]
        responses = self.client.simGetImages(reqs, vehicle_name=self.vehicle_name)
        imgs = {}
        for req, resp in zip(reqs, responses):
            if resp.width == 0 or resp.height == 0:
                self.get_logger().warn(f"No image from {req.camera_name}")
                continue
            arr = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            img = arr.reshape(resp.height, resp.width, 3)
            ros_img = self.bridge.cv2_to_imgmsg(img, encoding='rgb8')
            ros_img.header = header
            ros_img.header.frame_id = self.frame_id
            imgs[req.camera_name] = ros_img
        return imgs

    def get_imu(self, header):
        # positional args: imu_name, vehicle_name
        imu_data = self.client.getImuData(self.imu_name, self.vehicle_name)
        m = Imu()
        m.header = header
        m.linear_acceleration.x = float(imu_data.linear_acceleration.x_val)
        m.linear_acceleration.y = float(-imu_data.linear_acceleration.y_val)
        m.linear_acceleration.z = float(-imu_data.linear_acceleration.z_val)
        m.angular_velocity.x = float(imu_data.angular_velocity.x_val)
        m.angular_velocity.y = float(-imu_data.angular_velocity.y_val)
        m.angular_velocity.z = float(-imu_data.angular_velocity.z_val)
        return m

    def timer_callback(self):
        now = self.get_clock().now().to_msg()
        header = type(PoseStamped().header)()
        header.stamp = now

        pts = self.get_lidar_points()
        if pts is not None:
            pc2 = self.make_pointcloud2(pts, header)
            pc2.header.frame_id = self.frame_id
            self.lidar_pub.publish(pc2)

        pose_msg = self.get_pose(header)
        self.pose_pub.publish(pose_msg)

        self.path.header.stamp = now
        self.path.poses.append(pose_msg)
        self.path_pub.publish(self.path)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = self.parent_frame_id
        tf.child_frame_id = self.frame_id
        tf.transform.translation.x = float(pose_msg.pose.position.x)
        tf.transform.translation.y = float(pose_msg.pose.position.y)
        tf.transform.translation.z = float(pose_msg.pose.position.z)
        tf.transform.rotation.x = float(pose_msg.pose.orientation.x)
        tf.transform.rotation.y = float(pose_msg.pose.orientation.y)
        tf.transform.rotation.z = float(pose_msg.pose.orientation.z)
        tf.transform.rotation.w = float(pose_msg.pose.orientation.w)
        self.tf_broadcaster.sendTransform(tf)

        imgs = self.get_images(header)
        if self.left_cam in imgs:
            self.left_pub.publish(imgs[self.left_cam])
        if self.right_cam in imgs:
            self.right_pub.publish(imgs[self.right_cam])

        imu_msg = self.get_imu(header)
        self.imu_pub.publish(imu_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AirSimLidarPoseVisionImu(rate_hz=5)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()