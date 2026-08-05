#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class PoseToOdom(Node):
    def __init__(self):
        super().__init__("pose_to_odom")

        self.declare_parameter("pose_topic", "/vrpn_client_node/UWBTest/pose")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("odom_frame_id", "world")
        self.declare_parameter("child_frame_id", "cam_1_depth_optical_frame")
        self.declare_parameter("linear_covariance", 1.0e-4)
        self.declare_parameter("angular_covariance", 1.0e-4)
        self.declare_parameter("twist_covariance", 1.0e-3)
        self.declare_parameter("queue_size", 10)

        pose_topic = self.get_parameter("pose_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        self.odom_frame_id = self.get_parameter("odom_frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.linear_covariance = float(self.get_parameter("linear_covariance").value)
        self.angular_covariance = float(self.get_parameter("angular_covariance").value)
        self.twist_covariance = float(self.get_parameter("twist_covariance").value)
        queue_size = int(self.get_parameter("queue_size").value)

        self._validate_covariance("linear_covariance", self.linear_covariance)
        self._validate_covariance("angular_covariance", self.angular_covariance)
        self._validate_covariance("twist_covariance", self.twist_covariance)

        sub_qos = QoSProfile(depth=queue_size, reliability=ReliabilityPolicy.BEST_EFFORT)
        pub_qos = QoSProfile(depth=queue_size, reliability=ReliabilityPolicy.RELIABLE)

        self.publisher = self.create_publisher(Odometry, odom_topic, pub_qos)
        self.subscription = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            sub_qos,
        )

        self.get_logger().info(
            "Converting PoseStamped %s to Odometry %s (%s -> %s)"
            % (pose_topic, odom_topic, self.odom_frame_id, self.child_frame_id)
        )

    @staticmethod
    def _validate_covariance(name, value):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative value")

    def pose_callback(self, msg):
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = msg.header.frame_id or self.odom_frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = msg.pose

        odom.pose.covariance[0] = self.linear_covariance
        odom.pose.covariance[7] = self.linear_covariance
        odom.pose.covariance[14] = self.linear_covariance
        odom.pose.covariance[21] = self.angular_covariance
        odom.pose.covariance[28] = self.angular_covariance
        odom.pose.covariance[35] = self.angular_covariance

        odom.twist.covariance[0] = self.twist_covariance
        odom.twist.covariance[7] = self.twist_covariance
        odom.twist.covariance[14] = self.twist_covariance
        odom.twist.covariance[21] = self.twist_covariance
        odom.twist.covariance[28] = self.twist_covariance
        odom.twist.covariance[35] = self.twist_covariance

        self.publisher.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = PoseToOdom()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
