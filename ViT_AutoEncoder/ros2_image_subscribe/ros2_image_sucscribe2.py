#!/usr/bin/env python

import os

# ==========================
# 1. ROS2 환경 설정
# ==========================
os.environ["ROS_DOMAIN_ID"] = "20"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

print("[ENV] ROS_DOMAIN_ID      =", os.environ.get("ROS_DOMAIN_ID"))
print("[ENV] RMW_IMPLEMENTATION =", os.environ.get("RMW_IMPLEMENTATION"))

# ==========================
# 2. ROS2 및 OpenCV import
# ==========================
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage


class MultiCamCompressedSaver(Node):
    def __init__(
        self,
        cams=None,
        max_messages_per_cam=100,
        save_root="frames_compressed",
    ):
        super().__init__("multi_cam_compressed_saver")

        # 사용할 카메라 이름들
        if cams is None:
            cams = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"]
        self.cams = cams

        self.max_messages_per_cam = max_messages_per_cam
        self.save_root = save_root

        # 카메라별 저장 개수 카운터
        self.counts = {cam: 0 for cam in self.cams}

        # 저장 루트 폴더 및 카메라별 폴더 생성
        os.makedirs(self.save_root, exist_ok=True)
        for cam in self.cams:
            cam_dir = os.path.join(self.save_root, cam)
            os.makedirs(cam_dir, exist_ok=True)

        # QoS 설정
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        # 구독자 생성
        for cam in self.cams:
            topic_name = f"/{cam}/image_raw/compressed"
            self.create_subscription(
                CompressedImage,
                topic_name,
                lambda msg, cam_name=cam: self.callback(msg, cam_name),
                qos_profile,
            )
            self.get_logger().info(f"Subscribed to: {topic_name}")

        self.get_logger().info(
            f"Saving {self.max_messages_per_cam} JPEGs per camera "
            f"into '{self.save_root}/camX/'"
        )

    # ---------------------------------------------------------
    # 콜백: camX_f0001 형식으로 저장
    # ---------------------------------------------------------
    def callback(self, msg: CompressedImage, cam_name: str):

        # 이미 이 카메라는 100장을 다 저장했으면 무시
        if self.counts[cam_name] >= self.max_messages_per_cam:
            return

        # 카운터 증가
        self.counts[cam_name] += 1
        idx = self.counts[cam_name]

        # 저장 파일명 cam0_f0001.jpg 형태
        filename = f"{cam_name}_f{idx:04d}.jpg"

        cam_dir = os.path.join(self.save_root, cam_name)
        filepath = os.path.join(cam_dir, filename)

        # JPEG 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().error(
                f"[{cam_name}] Failed to decode JPEG frame "
                f"({idx}/{self.max_messages_per_cam})"
            )
            return

        # JPEG 파일로 저장
        cv2.imwrite(filepath, frame)

        self.get_logger().info(
            f"[{cam_name}] Saved {filepath} "
            f"({idx}/{self.max_messages_per_cam})"
        )

        # 모든 카메라가 목표 개수에 도달했으면 종료
        if all(cnt >= self.max_messages_per_cam for cnt in self.counts.values()):
            self.get_logger().info("All cameras reached target count. Shutting down...")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = MultiCamCompressedSaver(
        cams=["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"],
        max_messages_per_cam=100,
        save_root="frames_compressed",
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
