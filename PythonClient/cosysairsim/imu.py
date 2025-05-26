#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Imu
import cosysairsim as airsim

class ImuPublisher(Node):
    def __init__(self):
        super().__init__('airsim_imu_publisher')
        self.client = airsim.MultirotorClient(); self.client.confirmConnection()
        qos = QoSProfile(depth=1)
        self.pub = self.create_publisher(Imu, '/airsim/imu', qos)
        self.timer = self.create_timer(0.01, self.timer_cb)  # 100 Hz

    def timer_cb(self):
        data = self.client.getImuData('IMU', 'Car1')
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        accel = data.linear_acceleration
        gyro  = data.angular_velocity
        msg.linear_acceleration.x = float(accel.x_val)
        msg.linear_acceleration.y = float(-accel.y_val)
        msg.linear_acceleration.z = float(-accel.z_val)
        msg.angular_velocity.x = float(gyro.x_val)
        msg.angular_velocity.y = float(-gyro.y_val)
        msg.angular_velocity.z = float(-gyro.z_val)
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = ImuPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__': main()