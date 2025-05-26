#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from cv_bridge import CvBridge
import numpy as np
import cosysairsim as airsim

class SensorsPublisher(Node):
    def __init__(self):
        super().__init__('airsim_sensors_publisher')
        self.client = airsim.MultirotorClient(); self.client.confirmConnection()
        qos = QoSProfile(depth=1)
        self.lidar_pub = self.create_publisher(PointCloud2, '/airsim/lidar', qos)
        self.left_pub  = self.create_publisher(Image, '/airsim/left/image_raw', qos)
        self.right_pub = self.create_publisher(Image, '/airsim/right/image_raw', qos)
        self.pose_pub  = self.create_publisher(PoseStamped, '/airsim/pose', qos)
        self.path_pub  = self.create_publisher(Path, '/airsim/path', qos)
        self.tf_broad  = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)
        self.bridge    = CvBridge()
        self.path = Path(); self.path.header.frame_id = 'map'
        self._publish_static()
        self.timer = self.create_timer(0.2, self.cb)  # 5 Hz

    def _publish_static(self):
        now = self.get_clock().now().to_msg()
        tfs = []
        for name, xyz in [('imu_link',(0,0,0)),('camera_left_link',(2,-0.1,-1.1)),('camera_right_link',(2,0.1,-1.1))]:
            tf = TransformStamped()
            tf.header.stamp = now; tf.header.frame_id='base_link'; tf.child_frame_id=name
            tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z = xyz
            tf.transform.rotation.w = 1.0
            tfs.append(tf)
        self.tf_static.sendTransform(tfs)

    def cb(self):
        now = self.get_clock().now().to_msg()
        header = PoseStamped().header; header.stamp = now
        # LiDAR
        d = self.client.getLidarData('GPULidar','Car1')
        if d.point_cloud:
            pts = np.array(d.point_cloud,dtype=np.float32).reshape(-1,3)
            pts[:,1]=-1; pts[:,2]=-1
            pc2 = point_cloud2.create_cloud_xyz32(header, pts.tolist())
            pc2.header.frame_id='base_link'
            self.lidar_pub.publish(pc2)
        # Pose & Path & TF
        sp = self.client.simGetVehiclePose('Car1')
        pmsg = PoseStamped(); pmsg.header=header; pmsg.header.frame_id='map'
        p=p=sp.position; q=sp.orientation.inverse()
        pmsg.pose.position.x = float(p.x_val)
        pmsg.pose.position.y = float(-p.y_val)
        pmsg.pose.position.z = float(-p.z_val)
        pmsg.pose.orientation.x = float(q.x_val)
        pmsg.pose.orientation.y = float(q.y_val)
        pmsg.pose.orientation.z = float(q.z_val)
        pmsg.pose.orientation.w = float(q.w_val)
        self.pose_pub.publish(pmsg)
        self.path.header.stamp = now; self.path.poses.append(pmsg)
        self.path_pub.publish(self.path)
        tf = TransformStamped(); tf.header.stamp=now
        tf.header.frame_id='map'; tf.child_frame_id='base_link'
        tf.transform.translation = pmsg.pose.position
        tf.transform.rotation    = pmsg.pose.orientation
        self.tf_broad.sendTransform(tf)
        # Images
        for cam, pub in [('camera_left',self.left_pub),('camera_right',self.right_pub)]:
            resp = self.client.simGetImage(cam, airsim.ImageType.Scene)
            if resp:
                arr = np.frombuffer(resp,dtype=np.uint8).reshape(480,640,3)
                img = self.bridge.cv2_to_imgmsg(arr,'rgb8')
                img.header.stamp=now; img.header.frame_id='base_link'
                pub.publish(img)


def main():
    rclpy.init()
    node=SensorsPublisher()
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__=='__main__': main()