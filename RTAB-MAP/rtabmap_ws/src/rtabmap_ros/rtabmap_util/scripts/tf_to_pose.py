#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener


class TfToPose(Node):
    def __init__(self):
        super().__init__("tf_to_pose")

        self.declare_parameter("target_frame", "map")
        self.declare_parameter("source_frame", "base_link")
        self.declare_parameter("rate", 10.0)
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.source_frame = self.get_parameter("source_frame").get_parameter_value().string_value
        rate = self.get_parameter("rate").get_parameter_value().double_value
        if rate <= 0.0:
            raise ValueError("rate must be greater than 0")

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(PoseStamped, "pose", 1)
        self.timer = self.create_timer(1.0 / rate, self.publish_pose)

    def publish_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            self.get_logger().debug(
                f"Waiting for transform {self.target_frame} -> {self.source_frame}: {exc}"
            )
            return

        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        self.publisher.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = TfToPose()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
