#!/usr/bin/env python3
import struct
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor

# Attempt cv_bridge import
try:
    from cv_bridge import CvBridge
    _HAS_CVBRIDGE = True
except ImportError:
    _HAS_CVBRIDGE = False

from sensor_msgs.msg import PointCloud2, Image, Imu
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import cosysairsim as airsim

class AirSimLidarPoseVisionImu(Node):
    def __init__(self,
                 vehicle_name="Car1",
                 lidar_name="GPULidar",
                 left_cam="camera_left",
                 right_cam="camera_right",
                 imu_name="IMU",
                 frame_id="base_link",
                 parent_frame_id="map",
                 imu_rate_hz=100.0,
                 slow_rate_hz=5.0):
        super().__init__('airsim_lidar_pose_vision_imu')

        # CV Bridge
        if _HAS_CVBRIDGE:
            self.bridge = CvBridge()
        else:
            self.get_logger().warning(
                'cv_bridge unavailable: image topics disabled. '
                'Install numpy<2 and ros-<distro>-cv-bridge.'
            )

        # AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.vehicle_name = vehicle_name
        self.lidar_name = lidar_name
        self.left_cam = left_cam
        self.right_cam = right_cam
        self.imu_name = imu_name
        self.frame_id = frame_id
        self.parent_frame_id = parent_frame_id

        qos = QoSProfile(depth=1)
        # Publishers
        self.lidar_pub = self.create_publisher(PointCloud2, '/airsim/lidar', qos)
        self.pose_pub  = self.create_publisher(PoseStamped, '/airsim/pose', qos)
        self.path_pub  = self.create_publisher(Path,        '/airsim/path', qos)
        if _HAS_CVBRIDGE:
            self.left_pub  = self.create_publisher(Image, '/airsim/left/image_raw', qos)
            self.right_pub = self.create_publisher(Image, '/airsim/right/image_raw', qos)
        self.imu_pub   = self.create_publisher(Imu, '/airsim/imu', qos)

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        # Path
        self.path = Path()
        self.path.header.frame_id = parent_frame_id

        # Timers
        self.create_timer(1.0/imu_rate_hz, self._imu_callback)
        self.create_timer(1.0/slow_rate_hz, self._slow_callback)

    def _publish_static_transforms(self):
        now = self.get_clock().now().to_msg()
        tfs = []
        # IMU
        imu_tf = TransformStamped()
        imu_tf.header.stamp = now
        imu_tf.header.frame_id = self.frame_id
        imu_tf.child_frame_id = 'imu_link'
        imu_tf.transform.rotation.w = 1.0
        tfs.append(imu_tf)
        # Left cam
        camL = TransformStamped(); camL.header.stamp = now
        camL.header.frame_id = self.frame_id
        camL.child_frame_id = f'{self.left_cam}_link'
        camL.transform.translation.x = 2.0
        camL.transform.translation.y = -0.1
        camL.transform.translation.z = -1.1
        camL.transform.rotation.w = 1.0
        tfs.append(camL)
        # Right cam
        camR = TransformStamped(); camR.header.stamp = now
        camR.header.frame_id = self.frame_id
        camR.child_frame_id = f'{self.right_cam}_link'
        camR.transform.translation.x = 2.0
        camR.transform.translation.y = 0.1
        camR.transform.translation.z = -1.1
        camR.transform.rotation.w = 1.0
        tfs.append(camR)
        self.static_broadcaster.sendTransform(tfs)

    def _imu_callback(self):
        data = self.client.getImuData(self.imu_name, self.vehicle_name)
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        msg.linear_acceleration.x = float(data.linear_acceleration.x_val)
        msg.linear_acceleration.y = float(-data.linear_acceleration.y_val)
        msg.linear_acceleration.z = float(-data.linear_acceleration.z_val)
        msg.angular_velocity.x = float(data.angular_velocity.x_val)
        msg.angular_velocity.y = float(-data.angular_velocity.y_val)
        msg.angular_velocity.z = float(-data.angular_velocity.z_val)
        self.imu_pub.publish(msg)

    def _slow_callback(self):
        now = self.get_clock().now().to_msg()
        header = PoseStamped().header; header.stamp = now

        # LiDAR
        d = self.client.getLidarData(self.lidar_name, self.vehicle_name)
        if d and len(d.point_cloud)>=3:
            pts = np.array(d.point_cloud, dtype=np.float32).reshape(-1,3)
            pts[:,1]=-1; pts[:,2]=-1
            cloud = point_cloud2.create_cloud_xyz32(header, pts.tolist())
            cloud.header.frame_id = self.frame_id
            self.lidar_pub.publish(cloud)

        # Pose
        sp = self.client.simGetVehiclePose(self.vehicle_name)
        pose = PoseStamped(); pose.header = header
        pose.header.frame_id = self.parent_frame_id
        p=sp.position
        pose.pose.position.x = float(p.x_val)
        pose.pose.position.y = float(-p.y_val)
        pose.pose.position.z = float(-p.z_val)
        q=sp.orientation.inverse()
        pose.pose.orientation.x = float(q.x_val)
        pose.pose.orientation.y = float(q.y_val)
        pose.pose.orientation.z = float(q.z_val)
        pose.pose.orientation.w = float(q.w_val)
        self.pose_pub.publish(pose)

        # Path
        self.path.header.stamp = now
        self.path.poses.append(pose)
        self.path_pub.publish(self.path)

        # TF dynamic
        tf = TransformStamped(); tf.header.stamp=now
        tf.header.frame_id=self.parent_frame_id; tf.child_frame_id=self.frame_id
        tf.transform.translation = pose.pose.position
        tf.transform.rotation    = pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

        # Images
        if _HAS_CVBRIDGE:
            reqs = [
                airsim.ImageRequest(self.left_cam, airsim.ImageType.Scene, False, False),
                airsim.ImageRequest(self.right_cam, airsim.ImageType.Scene, False, False)
            ]
            resps = self.client.simGetImages(reqs, vehicle_name=self.vehicle_name)
            for req,resp in zip(reqs,resps):
                if resp.width and resp.height:
                    img=np.frombuffer(resp.image_data_uint8,dtype=np.uint8)
                    img=img.reshape(resp.height,resp.width,3)
                    imsg=self.bridge.cv2_to_imgmsg(img,'rgb8')
                    imsg.header.stamp=now;imsg.header.frame_id=self.frame_id
                    if req.camera_name==self.left_cam: self.left_pub.publish(imsg)
                    else: self.right_pub.publish(imsg)


def main(args=None):
    rclpy.init(args=args)
    node=AirSimLidarPoseVisionImu()
    exec=MultiThreadedExecutor()
    exec.add_node(node)
    try: exec.spin()
    except KeyboardInterrupt: pass
    node.destroy_node(); rclpy.shutdown()

if __name__=='__main__': main()