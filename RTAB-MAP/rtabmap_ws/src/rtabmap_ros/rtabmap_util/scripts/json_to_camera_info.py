#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image


def _float_list(values, expected_len, name):
    if not isinstance(values, list) or len(values) != expected_len:
        raise ValueError(f"{name} must be a list with {expected_len} values")
    return [float(value) for value in values]


def _intrinsics_to_camera_info(path):
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    msg = CameraInfo()
    msg.width = int(data.get("width", data.get("image_width", 0)) or 0)
    msg.height = int(data.get("height", data.get("image_height", 0)) or 0)

    if "cam_K" in data:
        msg.k = _float_list(data["cam_K"], 9, "cam_K")
    elif "camera_matrix" in data and isinstance(data["camera_matrix"], dict):
        msg.k = _float_list(data["camera_matrix"].get("data"), 9, "camera_matrix.data")
    else:
        try:
            fx = float(data["fx"])
            fy = float(data["fy"])
            cx = float(data["cx"])
            cy = float(data["cy"])
        except KeyError as exc:
            raise ValueError("JSON must contain cam_K, camera_matrix.data, or fx/fy/cx/cy") from exc
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    if "projection_matrix" in data and isinstance(data["projection_matrix"], dict):
        msg.p = _float_list(data["projection_matrix"].get("data"), 12, "projection_matrix.data")
    elif "cam_P" in data:
        msg.p = _float_list(data["cam_P"], 12, "cam_P")
    else:
        msg.p = [
            msg.k[0],
            0.0,
            msg.k[2],
            0.0,
            0.0,
            msg.k[4],
            msg.k[5],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

    if "rectification_matrix" in data and isinstance(data["rectification_matrix"], dict):
        msg.r = _float_list(data["rectification_matrix"].get("data"), 9, "rectification_matrix.data")
    elif "cam_R" in data:
        msg.r = _float_list(data["cam_R"], 9, "cam_R")
    else:
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    if "distortion_coefficients" in data and isinstance(data["distortion_coefficients"], dict):
        msg.d = [float(value) for value in data["distortion_coefficients"].get("data", [])]
        msg.distortion_model = data.get(
            "distortion_model",
            data["distortion_coefficients"].get("model", "plumb_bob"),
        )
    else:
        msg.d = [float(value) for value in data.get("cam_D", data.get("d", []))]
        msg.distortion_model = data.get("distortion_model", "plumb_bob")

    return msg


class JsonToCameraInfo(Node):
    def __init__(self):
        super().__init__("json_to_camera_info")

        self.declare_parameter("json_path", "")
        self.declare_parameter("frame_id", "")
        json_path = self.get_parameter("json_path").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        if not json_path:
            self.get_logger().error("json_path parameter should be set")
            sys.exit(1)

        try:
            self.camera_info_msg = _intrinsics_to_camera_info(json_path)
        except Exception as exc:
            self.get_logger().error(f"Failed to load camera intrinsics JSON: {exc}")
            sys.exit(1)

        self.publisher = self.create_publisher(CameraInfo, "camera_info", 1)
        self.subscription = self.create_subscription(
            Image,
            "image",
            self.callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )

    def callback(self, image):
        self.camera_info_msg.header = image.header
        if self.frame_id:
            self.camera_info_msg.header.frame_id = self.frame_id
        if not self.camera_info_msg.width:
            self.camera_info_msg.width = image.width
        if not self.camera_info_msg.height:
            self.camera_info_msg.height = image.height
        self.publisher.publish(self.camera_info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JsonToCameraInfo()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
