#!/usr/bin/env python

import os
import math
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy  # 사용은 안 하지만 그대로 둠

import onnxruntime as ort
import requests


# =========================
# 0. ROS2 환경변수
# =========================
os.environ["ROS_DOMAIN_ID"] = "20"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"


# =========================
# 1. 기본 파라미터 (B.py와 동일 + YOLO 관련)
# =========================

NUM_CAMS = 6
CAMS = [f"cam{i}" for i in range(NUM_CAMS)]

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

PANORAMA_CROP_HEIGHT_SCALE = 0.9
PANORAMA_CROP_WIDTH_SCALE  = 0.98

TARGET_W = 1280
TARGET_H = 324

OVERLAP_PX = 190
FEATHER_STRENGTH = 1.0

PANO_DIR = "panorama_output"

# --- YOLO-NAS + ONNXRuntime 관련 설정 ---
YOLO_ONNX_PATH = "yolo_nas_s.onnx"

# 실제로는 본인이 export한 yolo_nas_s.onnx를
# GitHub/HF/S3 등에 올린 뒤 그 URL로 교체해서 사용
YOLO_ONNX_URL = (
    "https://huggingface.co/hr16/yolo-nas-fp16/resolve/main/yolo_nas_s_fp16.onnx"
)

YOLO_INPUT_SIZE = 640                # onnx 입력 크기 (입력 shape: [1,3,640,640])
YOLO_CONF_THRESH = 0.25
YOLO_RUN_EVERY_N_FRAMES = 3          # 3프레임당 1번 YOLO 수행
YOLO_NUM_PATCHES = 3                 # 파노라마를 가로로 몇 개 패치로 나눌지

# === person 클래스만 남기기 위한 ID (COCO 기준 일반적으로 0이 person이지만, 실제 모델에 맞게 수정 필요) ===
PERSON_CLASS_ID = 0


# =========================
# 1-1. YOLO ONNX 자동 다운로드 함수
# =========================

def download_yolo_onnx_if_needed(path: str, url: str, logger) -> bool:
    """
    path에 파일이 없으면 url에서 다운로드 시도.
    성공하면 True, 실패하면 False.
    """
    if os.path.exists(path):
        return True

    logger.info(f"YOLO ONNX file not found at '{path}', downloading from:\n  {url}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, path)
        logger.info(f"Downloaded YOLO ONNX model to '{path}'")
        return True
    except Exception as e:
        logger.error(f"Failed to download YOLO ONNX model: {e}")
        return False


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
    dy = P1 * (2.0 * y_n * y_n + r2) + 2.0 * P2 * x_n * y_n

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
        strip0 = strips[0][sy]
        curx = strip0.shape[0]
        dst[:curx] = strip0[:min(curx, final_w)]

        for i in range(1, len(strips)):
            strip = strips[i][sy]
            w = strip.shape[0]

            ov = min(overlap, w, curx)

            for x in range(ov):
                dst_x = curx - ov + x
                if dst_x >= final_w:
                    continue
                a = alpha[x]
                prev = dst[dst_x].astype(np.float32)
                curr = strip[x].astype(np.float32)
                dst[dst_x] = (prev * (1.0 - a) + curr * a).astype(np.uint8)

            tail = w - ov
            for x in range(tail):
                dst_x = curx + x
                if dst_x < final_w:
                    dst[dst_x] = strip[ov + x]

            curx += tail

    return pano


# =========================
# 4. 아주 단순화된 SORT 스타일 트래커
# =========================

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: [x1, y1, x2, y2]"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class Track:
    def __init__(self, track_id: int, bbox: np.ndarray, score: float, cls_id: int):
        self.id = track_id
        self.bbox = bbox.astype(np.float32)
        self.score = float(score)
        self.cls = int(cls_id)
        self.age = 0
        self.time_since_update = 0

    def update(self, bbox: np.ndarray, score: float, cls_id: int):
        self.bbox = bbox.astype(np.float32)
        self.score = float(score)
        self.cls = int(cls_id)
        self.time_since_update = 0
        self.age += 1

    def predict_only(self):
        # motion model 없음: bbox 유지, 나이만 증가
        self.time_since_update += 1
        self.age += 1


class SortTracker:
    """
    Kalman filter 없이 IoU 기반 간단 멀티 객체 추적.
    - YOLO 수행 프레임에서만 detection으로 update
    - 그 사이 프레임에서는 bbox 고정, time_since_update만 증가
    """
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 5):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks: List[Track] = []
        self._next_id = 1

    def _associate(self, dets: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(self.tracks) == 0 or dets.shape[0] == 0:
            return [], list(range(len(self.tracks))), list(range(dets.shape[0]))

        iou_matrix = np.zeros((len(self.tracks), dets.shape[0]), dtype=np.float32)
        for t_idx, t in enumerate(self.tracks):
            for d_idx in range(dets.shape[0]):
                iou_matrix[t_idx, d_idx] = iou_xyxy(t.bbox, dets[d_idx])

        matches: List[Tuple[int, int]] = []
        used_tracks = set()
        used_dets = set()

        while True:
            max_iou = 0.0
            max_t = -1
            max_d = -1
            for t_idx in range(len(self.tracks)):
                if t_idx in used_tracks:
                    continue
                for d_idx in range(dets.shape[0]):
                    if d_idx in used_dets:
                        continue
                    if iou_matrix[t_idx, d_idx] > max_iou:
                        max_iou = iou_matrix[t_idx, d_idx]
                        max_t = t_idx
                        max_d = d_idx

            if max_iou < self.iou_thresh or max_t == -1:
                break

            matches.append((max_t, max_d))
            used_tracks.add(max_t)
            used_dets.add(max_d)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        unmatched_dets = [i for i in range(dets.shape[0]) if i not in used_dets]

        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: Optional[np.ndarray]):
        """
        detections: None 또는 [N, 6] (x1, y1, x2, y2, score, cls)
        """
        if detections is None:
            for t in self.tracks:
                t.predict_only()
        else:
            for t in self.tracks:
                t.predict_only()

            if detections.shape[0] > 0:
                det_bboxes = detections[:, 0:4]
                det_scores = detections[:, 4]
                det_clses = detections[:, 5].astype(int)

                matches, unmatched_tracks, unmatched_dets = self._associate(det_bboxes)

                for t_idx, d_idx in matches:
                    self.tracks[t_idx].update(det_bboxes[d_idx], det_scores[d_idx], det_clses[d_idx])

                for d_idx in unmatched_dets:
                    new_track = Track(
                        self._next_id,
                        det_bboxes[d_idx],
                        det_scores[d_idx],
                        det_clses[d_idx],
                    )
                    self._next_id += 1
                    self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def get_active_tracks(self) -> List[Track]:
        return self.tracks


# =========================
# 5. 오프라인 파노라마 + YOLO-NAS(ONNX) + 간단 SORT + FPS 로그
# =========================

class RealTimePanoramaNode(Node):
    def __init__(self):
        super().__init__("realtime_panorama_yolonas_node")

        self.buffers: Dict[str, Optional[np.ndarray]] = {cam: None for cam in CAMS}
        self.frame_index: int = 0

        # FPS 계산용 (전체 파이프라인)
        self.last_frame_time: Optional[float] = None
        self.avg_fps: float = 0.0
        self.fps_alpha: float = 0.9  # 지수이동평균 계수

        # panorama 연산 시간/Hz 측정용 변수
        self.pano_time_avg: float = 0.0     # seconds
        self.pano_hz_avg: float = 0.0
        self.pano_alpha: float = 0.9        # EMA 계수

        # ----- 파노라마 파라미터 계산 (B.py와 동일) -----
        self.resized_w = int(round(WIDTH * RESIZE_FACTOR))
        self.resized_h = int(round(HEIGHT * RESIZE_FACTOR))

        self.crop_w = int(round(self.resized_w * PANORAMA_CROP_WIDTH_SCALE))
        self.crop_h = int(round(self.resized_h * PANORAMA_CROP_HEIGHT_SCALE))
        self.crop_x0 = (self.resized_w - self.crop_w) // 2
        self.crop_y0 = (self.resized_h - self.crop_h) // 2

        self.map_x, self.map_y = generate_cylindrical_map(
            self.resized_w, self.resized_h,
            CYL_H_FOV_DEG, CYL_V_FOV_DEG,
            fx, fy, cx, cy,
            RESIZE_FACTOR
        )

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

        os.makedirs(PANO_DIR, exist_ok=True)

        # ----- YOLO-NAS ONNX Runtime 세션 초기화 (자동 다운로드 포함) -----
        if not download_yolo_onnx_if_needed(YOLO_ONNX_PATH, YOLO_ONNX_URL, self.get_logger()):
            self.get_logger().warn(
                f"Could not prepare YOLO ONNX file '{YOLO_ONNX_PATH}'. "
                f"Detection will be disabled."
            )
            self.ort_session = None
            self.onnx_input_name = None
            self.onnx_output_names = None
        else:
            self.get_logger().info(f"Loading YOLO-NAS ONNX model from {YOLO_ONNX_PATH}")
            self.ort_session = ort.InferenceSession(
                YOLO_ONNX_PATH,
                providers=["CPUExecutionProvider"]
            )
            inputs = self.ort_session.get_inputs()
            outputs = self.ort_session.get_outputs()
            self.onnx_input_name = inputs[0].name
            self.onnx_output_names = [o.name for o in outputs]
            self.get_logger().info(
                f"ONNX input='{self.onnx_input_name}', outputs={self.onnx_output_names}"
            )

        # ----- SORT 스타일 트래커 -----
        self.tracker = SortTracker(iou_thresh=0.3, max_age=5)

        # ROS2 Subscriber는 사용하지 않음 (오프라인 이미지 사용)

    # ---------- 파노라마 생성 ----------
    def create_panorama(self) -> np.ndarray:
        # panorama 연산 시간 측정 시작
        t0 = time.time()

        imgs = [self.buffers[c] for c in CAMS]
        strips: List[np.ndarray] = []

        for cam_idx in range(NUM_CAMS):
            src = imgs[cam_idx]
            resized = cv2.resize(src, (self.resized_w, self.resized_h))
            warped  = cv2.remap(resized, self.map_x, self.map_y, cv2.INTER_LINEAR)
            cropped = warped[
                self.crop_y0:self.crop_y0 + self.crop_h,
                self.crop_x0:self.crop_x0 + self.crop_w
            ]
            target_w = self.per_cam_w[cam_idx]
            strip = cv2.resize(cropped, (target_w, self.per_cam_h))
            strips.append(strip)

        pano = feather_blend_with_pad(
            strips,
            self.overlap_target,
            TARGET_W,
            TARGET_H,
            FEATHER_STRENGTH
        )

        # panorama 연산 시간/Hz 계산 및 로그
        t1 = time.time()
        dt = t1 - t0  # seconds

        if dt > 0:
            inst_hz = 1.0 / dt
        else:
            inst_hz = 0.0

        # 지수 이동 평균
        if self.pano_time_avg == 0.0:
            self.pano_time_avg = dt
            self.pano_hz_avg = inst_hz
        else:
            self.pano_time_avg = (
                self.pano_alpha * self.pano_time_avg + (1.0 - self.pano_alpha) * dt
            )
            self.pano_hz_avg = (
                self.pano_alpha * self.pano_hz_avg + (1.0 - self.pano_alpha) * inst_hz
            )

        self.get_logger().info(
            f"[PANO] time = {dt*1000.0:.2f} ms "
            f"(avg {self.pano_time_avg*1000.0:.2f} ms), "
            f"Hz = {inst_hz:.2f} (avg {self.pano_hz_avg:.2f})"
        )

        return pano

    # ---------- YOLO-NAS + 간단 SORT ----------
    def run_detection_and_tracking(self, pano_bgr: np.ndarray) -> np.ndarray:
        detections: Optional[np.ndarray]

        if self.ort_session is None:
            detections = None
        else:
            if self.frame_index % YOLO_RUN_EVERY_N_FRAMES == 0:
                detections = self.run_yolo_on_patches(pano_bgr)
            else:
                detections = None

        self.tracker.update(detections)
        tracks = self.tracker.get_active_tracks()
        pano_annot = self.draw_tracks(pano_bgr, tracks)
        return pano_annot

    def run_yolo_on_patches(self, pano_bgr: np.ndarray) -> np.ndarray:
        """
        파노라마를 가로 방향으로 YOLO_NUM_PATCHES개로 나누고,
        각 패치를 yolo_nas_s.onnx에 넣어서 detection 수행.

        현재 onnx 출력 구조 (사용자 실행 결과 기준):
          - preds[0]: (1, 1)          int64   -> num_detections 등 메타 정보 (사용 안함)
          - preds[1]: (1, 1000, 4)    float32 -> boxes (x_min, y_min, x_max, y_max)
          - preds[2]: (1, 1000)       float32 -> scores
          - preds[3]: (1, 1000)       int64   -> class ids
        """
        if self.ort_session is None:
            return np.zeros((0, 6), dtype=np.float32)

        h, w, _ = pano_bgr.shape
        patch_w = w // YOLO_NUM_PATCHES

        all_dets: List[List[float]] = []

        for i in range(YOLO_NUM_PATCHES):
            x0 = i * patch_w
            x1 = (i + 1) * patch_w if i < YOLO_NUM_PATCHES - 1 else w
            patch = pano_bgr[:, x0:x1, :]

            patch_resized = cv2.resize(patch, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))

            # BGR -> RGB, uint8 NCHW
            img = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))[None, :, :, :]  # (1,3,H,W)
            img = img.astype(np.uint8)

            preds = self.ort_session.run(
                self.onnx_output_names,
                {self.onnx_input_name: img}
            )

            # preds[0]: (1,1) int64 -> 사용 안 함
            boxes  = preds[1]  # (1, 1000, 4)
            scores = preds[2]  # (1, 1000)
            labels = preds[3]  # (1, 1000)

            if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4:
                self.get_logger().error(f"Unexpected boxes shape: {boxes.shape}")
                continue
            if scores.ndim != 2 or scores.shape[0] != 1:
                self.get_logger().error(f"Unexpected scores shape: {scores.shape}")
                continue
            if labels.ndim != 2 or labels.shape[0] != 1:
                self.get_logger().error(f"Unexpected labels shape: {labels.shape}")
                continue

            boxes  = boxes[0]          # (1000, 4)
            scores = scores[0]         # (1000,)
            labels = labels[0].astype(np.int32)  # (1000,)

            num_det = boxes.shape[0]
            if scores.shape[0] != num_det or labels.shape[0] != num_det:
                self.get_logger().error(
                    f"Inconsistent output sizes: "
                    f"boxes={boxes.shape}, scores={scores.shape}, labels={labels.shape}"
                )
                continue

            for j in range(num_det):
                conf = float(scores[j])
                if conf < YOLO_CONF_THRESH:
                    continue

                x_min, y_min, x_max, y_max = boxes[j]
                cls_id = int(labels[j])

                # === person 클래스만 남기기 ===
                if cls_id != PERSON_CLASS_ID:
                    continue

                # 좌표가 onnx 입력 해상도 기준 (0~YOLO_INPUT_SIZE) 라고 가정
                gx1 = x0 + x_min * (x1 - x0) / YOLO_INPUT_SIZE
                gy1 = y_min * h / YOLO_INPUT_SIZE
                gx2 = x0 + x_max * (x1 - x0) / YOLO_INPUT_SIZE
                gy2 = y_max * h / YOLO_INPUT_SIZE

                gx1 = max(0.0, min(float(w - 1), gx1))
                gy1 = max(0.0, min(float(h - 1), gy1))
                gx2 = max(0.0, min(float(w - 1), gx2))
                gy2 = max(0.0, min(float(h - 1), gy2))

                if gx2 <= gx1 or gy2 <= gy1:
                    continue

                all_dets.append([gx1, gy1, gx2, gy2, conf, cls_id])

        if len(all_dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        return np.array(all_dets, dtype=np.float32)

    def draw_tracks(self, pano_bgr: np.ndarray, tracks: List[Track]) -> np.ndarray:
        out = pano_bgr.copy()
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            x1_i = int(round(x1))
            y1_i = int(round(y1))
            x2_i = int(round(x2))
            y2_i = int(round(y2))

            cv2.rectangle(out, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            label = f"ID{t.id} C{t.cls} {t.score:.2f}"
            cv2.putText(
                out,
                label,
                (x1_i, max(0, y1_i - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return out


# =========================
# 6. main (오프라인 JPEG 100프레임 처리)
# =========================

def main(args=None):
    rclpy.init(args=args)
    node = RealTimePanoramaNode()

    try:
        # f0001 ~ f0100까지 반복
        for frame_idx in range(1, 101):
            node.frame_index = frame_idx

            # 6개 cam 이미지 로드
            all_ok = True
            for cam in CAMS:
                img_path = os.path.join(
                    "frames_compressed",
                    cam,
                    f"{cam}_f{frame_idx:04d}.jpg"
                )
                img = cv2.imread(img_path)
                if img is None:
                    node.get_logger().error(f"Failed to read image: {img_path}")
                    all_ok = False
                    break

                if img.shape[1] != WIDTH or img.shape[0] != HEIGHT:
                    img = cv2.resize(img, (WIDTH, HEIGHT))

                node.buffers[cam] = img

            if not all_ok:
                node.get_logger().warn(
                    f"Stopping at frame_idx={frame_idx} due to missing image."
                )
                break

            # 파노라마 + YOLO 파이프라인 실행
            pano = node.create_panorama()
            pano_annot = node.run_detection_and_tracking(pano)

            out_path = os.path.join(PANO_DIR, f"pano_f{frame_idx:04d}.jpg")
            cv2.imwrite(out_path, pano_annot)

            # 전체 파이프라인 FPS 측정
            now = time.time()
            if node.last_frame_time is not None:
                dt = now - node.last_frame_time
                if dt > 0:
                    inst_fps = 1.0 / dt
                    if node.avg_fps == 0.0:
                        node.avg_fps = inst_fps
                    else:
                        node.avg_fps = (
                            node.fps_alpha * node.avg_fps
                            + (1.0 - node.fps_alpha) * inst_fps
                        )
                    node.get_logger().info(
                        f"[FRAME {frame_idx}] "
                        f"inst FPS = {inst_fps:.2f}, avg FPS = {node.avg_fps:.2f}, "
                        f"saved: {out_path}"
                    )
            else:
                node.get_logger().info(
                    f"[FRAME {frame_idx}] first frame saved: {out_path}"
                )
            node.last_frame_time = now

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
