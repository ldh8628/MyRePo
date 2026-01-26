#!/usr/bin/env python

import os
import math
from typing import Dict, Optional, List

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage

from ultralytics import YOLO


# =========================
# 0. ROS2 환경변수 (필요 시)
# =========================
os.environ.setdefault("ROS_DOMAIN_ID", "20")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")


# =========================
# 1. 기본 설정 (B.py 기반, 3카메라용)
# =========================

# cam1, cam2, cam3만 사용
CAM_NAMES = ["cam1", "cam2", "cam3"]
NUM_CAMS = len(CAM_NAMES)

WIDTH = 640
HEIGHT = 360

fx = 401.5
fy = 401.5
cx = 320.0
cy = 180.0

K1 = -0.343698
K2 = 0.105503
K3 = 0.0
P1 = -0.001558
P2 = -0.002416

CYL_H_FOV_DEG = 77.0
CYL_V_FOV_DEG = 51.3

RESIZE_FACTOR = 1.0

PANORAMA_CROP_HEIGHT_SCALE = 0.90
PANORAMA_CROP_WIDTH_SCALE  = 0.98

TARGET_W = 1280
TARGET_H = 324

OVERLAP_PX = 0
FEATHER_STRENGTH = 1.0


# =========================
# 2. Cylindrical mapping
# =========================

def generate_cylindrical_map(
    resized_w: int,
    resized_h: int,
    fov_h_deg: float,
    fov_v_deg: float,
    fx_src: float,
    fy_src: float,
    cx_src: float,
    cy_src: float,
    resize_factor: float = 1.0,
):
    fx_r = fx_src * resize_factor
    fy_r = fy_src * resize_factor
    cx_r = cx_src * resize_factor
    cy_r = cy_src * resize_factor

    fov_h_rad = math.radians(fov_h_deg)
    fov_v_rad = math.radians(fov_v_deg)

    fx_cyl = resized_w / (2.0 * math.tan(fov_h_rad / 2.0))
    fy_cyl = resized_h / (2.0 * math.tan(fov_v_rad / 2.0))
    cx_cyl = resized_w / 2.0
    cy_cyl = resized_h / 2.0

    ys, xs = np.indices((resized_h, resized_w), dtype=np.float32)

    theta = (xs - cx_cyl) / fx_cyl
    hh    = (ys - cy_cyl) / fy_cyl

    xs_3d = np.sin(theta)
    ys_3d = hh
    zs_3d = np.cos(theta)

    x_n = xs_3d / zs_3d
    y_n = ys_3d / zs_3d

    r2 = x_n * x_n + y_n * y_n
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + K1 * r2 + K2 * r4 + K3 * r6
    dx = 2.0 * P1 * x_n * y_n + P2 * (r2 + 2.0 * x_n * x_n)
    dy = P1 * (r2 + 2.0 * y_n * y_n) + 2.0 * P2 * x_n * y_n

    x_d = x_n * radial + dx
    y_d = y_n * radial + dy

    map_x = x_d * fx_r + cx_r
    map_y = y_d * fy_r + cy_r

    return map_x.astype(np.float32), map_y.astype(np.float32)


# =========================
# 3. Feather Blend
# =========================

def feather_blend_with_pad(
    strips: List[np.ndarray],
    overlap: int,
    final_w: int,
    final_h: int,
    feather_strength: float = 1.0
) -> np.ndarray:
    """
    B.py와 동일한 feather blending.
    strips: [H_i x W_i x 3] 배열 리스트
    """
    h, w0, _ = strips[0].shape
    pano = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    ypad = (final_h - h) // 2

    overlap = max(1, overlap)
    alpha = np.zeros(overlap, dtype=np.float32)
    inv_overlap = 1.0 / overlap
    gamma = 1.0 / max(0.1, feather_strength)

    for x in range(overlap):
        t = (x + 0.5) * inv_overlap
        alpha[x] = t ** gamma

    for y in range(final_h):
        sy = y - ypad
        if sy < 0 or sy >= h:
            continue

        dst = pano[y]
        # 첫 strip
        strip0 = strips[0][sy]
        curx = strip0.shape[0]
        dst[:curx] = strip0[:min(curx, final_w)]

        # 나머지 strips
        for i in range(1, len(strips)):
            strip = strips[i][sy]
            w = strip.shape[0]

            ov = min(overlap, w, curx)

            # feather 영역
            for x in range(ov):
                dst_x = curx - ov + x
                if dst_x >= final_w:
                    continue
                a = alpha[x]
                prev = dst[dst_x].astype(np.float32)
                curr = strip[x].astype(np.float32)
                dst[dst_x] = (prev * (1 - a) + curr * a).astype(np.uint8)

            # non-overlap
            tail = w - ov
            for x in range(tail):
                dst_x = curx + x
                if dst_x < final_w:
                    dst[dst_x] = strip[ov + x]

            curx += tail

    return pano


# =========================
# 4. ROS2 노드: 3캠 파노라마 + YOLOv8n + publish
# =========================

class PanoYoloNode(Node):
    def __init__(self):
        super().__init__("pano_yolov8_node")

        # 최신 프레임 버퍼
        self.buffers: Dict[str, Optional[np.ndarray]] = {cam: None for cam in CAM_NAMES}
        self.frame_index: int = 0

        # -------- 파노라마 파라미터 계산 --------
        self.resized_w = int(round(WIDTH * RESIZE_FACTOR))
        self.resized_h = int(round(HEIGHT * RESIZE_FACTOR))

        self.crop_w = int(round(self.resized_w * PANORAMA_CROP_WIDTH_SCALE))
        self.crop_h = int(round(self.resized_h * PANORAMA_CROP_HEIGHT_SCALE))
        self.crop_x0 = (self.resized_w - self.crop_w) // 2
        self.crop_y0 = (self.resized_h - self.crop_h) // 2

        # cylindrical map (한 번만 계산)
        self.get_logger().info("Generating cylindrical map...")
        self.map_x, self.map_y = generate_cylindrical_map(
            self.resized_w,
            self.resized_h,
            CYL_H_FOV_DEG,
            CYL_V_FOV_DEG,
            fx,
            fy,
            cx,
            cy,
            RESIZE_FACTOR,
        )
        self.get_logger().info("Cylindrical map generated.")

        # per-camera width 계산 (NUM_CAMS=3 기준)
        overlap_resized = int(round(OVERLAP_PX * RESIZE_FACTOR))
        pano_w_crop = NUM_CAMS * self.crop_w - (NUM_CAMS - 1) * overlap_resized
        scale_x = TARGET_W / pano_w_crop

        self.overlap_target = int(round(overlap_resized * scale_x))
        per_w_base = int(round(self.crop_w * scale_x))

        self.per_cam_w = [per_w_base] * NUM_CAMS
        total_now = NUM_CAMS * per_w_base - (NUM_CAMS - 1) * self.overlap_target
        delta = TARGET_W - total_now

        i = 0
        while delta > 0 and i < NUM_CAMS:
            self.per_cam_w[i] += 1
            delta -= 1
            i += 1
        i = 0
        while delta < 0 and i < NUM_CAMS:
            if self.per_cam_w[i] > 1:
                self.per_cam_w[i] -= 1
                delta += 1
            i += 1

        self.per_cam_h = self.crop_h

        self.get_logger().info(
            f"per_cam_w = {self.per_cam_w}, overlap_target = {self.overlap_target}"
        )

        # -------- YOLOv8n 초기화 --------
        # yolov8n.pt 파일 경로는 환경에 맞게 수정
        model_path = "yolov8n.pt"
        self.get_logger().info(f"Loading YOLOv8n model from {model_path} ...")
        self.yolo_model = YOLO(model_path)
        # 필요 시 CUDA 강제:
        # self.yolo_model.to("cuda")
        self.get_logger().info("YOLOv8n model loaded.")

        # -------- Publisher / Subscriber 설정 --------
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        # 파노라마 + YOLO 결과 이미지 publish (CompressedImage)
        self.pub_pano = self.create_publisher(
            CompressedImage,
            "/panorama/image_raw/compressed",
            10,
        )

        # cam1, cam2, cam3만 구독
        for cam in CAM_NAMES:
            topic = f"/{cam}/image_raw/compressed"
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, cam_name=cam: self.cb_image(msg, cam_name),
                qos,
            )
            self.get_logger().info(f"Subscribed to {topic}")

    # ---------- 이미지 콜백 ----------
    def cb_image(self, msg: CompressedImage, cam_name: str):
        # JPEG 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().error(f"[{cam_name}] decode fail")
            return

        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        self.buffers[cam_name] = frame

        # cam1, cam2, cam3 모두 최신 프레임이 들어온 순간 파노라마 1장 생성
        if all(self.buffers[c] is not None for c in CAM_NAMES):
            self.frame_index += 1
            try:
                pano = self.create_panorama()
                pano_yolo = self.run_yolo_and_draw(pano)
                self.publish_pano(pano_yolo)
            except Exception as e:
                self.get_logger().error(f"Error in panorama/YOLO pipeline: {e}")
            finally:
                # 다음 프레임을 위해 버퍼 초기화
                self.buffers = {cam: None for cam in CAM_NAMES}

    # ---------- 파노라마 생성 ----------
    def create_panorama(self) -> np.ndarray:
        """
        self.buffers 에 있는 cam1, cam2, cam3 이미지를 이용해
        B.py와 동일한 방식으로 파노라마 생성.
        """
        # CAM_NAMES 순서를 기준으로 strip 생성
        strips: List[np.ndarray] = []

        for idx, cam_name in enumerate(CAM_NAMES):
            src = self.buffers[cam_name]
            resized = cv2.resize(src, (self.resized_w, self.resized_h))

            # cylindrical remap
            warped = cv2.remap(resized, self.map_x, self.map_y, cv2.INTER_LINEAR)

            # crop
            cropped = warped[
                self.crop_y0 : self.crop_y0 + self.crop_h,
                self.crop_x0 : self.crop_x0 + self.crop_w,
            ]

            target_w = self.per_cam_w[idx]
            strip = cv2.resize(cropped, (target_w, self.per_cam_h))
            strips.append(strip)

        pano = feather_blend_with_pad(
            strips,
            self.overlap_target,
            TARGET_W,
            TARGET_H,
            FEATHER_STRENGTH,
        )

        return pano

    # ---------- YOLOv8n inference + 박스 그리기 ----------
    def run_yolo_and_draw(self, pano_bgr: np.ndarray) -> np.ndarray:
        """
        YOLOv8n으로 파노라마에서 객체를 탐지하고,
        바운딩 박스를 그린 이미지를 반환.
        """
        # Ultralytics는 RGB 입력을 사용하는 것이 기본이지만,
        # OpenCV는 BGR이므로 변환 후 넣어도 되고,
        # BGR 그대로 넣어도 내부에서 크게 문제는 없지만
        # 명시적으로 RGB로 변환하는 편이 안전함.
        #pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

        # 결과는 리스트 형태로 반환됨 (frame 단위)
        results = self.yolo_model(
            pano_bgr,
            imgsz=640,      # 필요 시 조정
            conf=0.25,      # confidence threshold
            verbose=False,
        )

        # results[0].plot() 는 주석된 BGR 이미지를 numpy array로 반환
        annotated = results[0].plot()  # BGR 형식
        return annotated

    # ---------- 파노라마 publish ----------
    def publish_pano(self, pano_bgr: np.ndarray):
        """
        BGR 이미지를 JPEG로 인코딩하여 CompressedImage로 publish.
        """
        ret, jpeg_data = cv2.imencode(".jpg", pano_bgr)
        if not ret:
            self.get_logger().error("Failed to encode panorama to JPEG.")
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = jpeg_data.tobytes()

        self.pub_pano.publish(msg)
        self.get_logger().info(f"[FRAME {self.frame_index}] panorama published.")


# =========================
# 5. main
# =========================

def main(args=None):
    rclpy.init(args=args)
    node = PanoYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
