"""
B.py
- Load 6 JPEG images saved by A.py from:
    frames_compressed/cam0/cam0_fXXXX.jpg
    frames_compressed/cam1/cam1_fXXXX.jpg
    ...
    frames_compressed/cam5/cam5_fXXXX.jpg

- Apply undistortion + cylindrical projection
- Crop + scale + feather blend
- Save panorama to:
    frames_compressed/panorama/pano_fXXXX.jpg
"""

import os
import re
import math
import glob
from pathlib import Path
import cv2
import numpy as np

# =========================
# 1. 기본 설정
# =========================

BASE_DIR = Path("frames_compressed")

CAM_DIRS = [BASE_DIR / f"cam{i}" for i in range(6)]
PANORAMA_DIR = BASE_DIR / "panorama"
CACHE_DIR = BASE_DIR / "cache"

NUM_CAMS = 6

WIDTH = 640
HEIGHT = 360

fx = 401.5 #1204.505857
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

PANORAMA_CROP_HEIGHT_SCALE = 0.9#0.70
PANORAMA_CROP_WIDTH_SCALE = 0.98

TARGET_W = 1280
TARGET_H = 324

OVERLAP_PX = 0
FEATHER_STRENGTH = 1.0#1.0


# =========================
# Utility
# =========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_frames_in_camera_folder(cam_dir: Path, cam_index: int):
    """
    cam0_fXXXX.jpg → XXXX 추출하여 프레임 번호 목록 반환
    """
    pattern = str(cam_dir / f"cam{cam_index}_f*.jpg")
    files = glob.glob(pattern)

    regex = re.compile(rf"cam{cam_index}_f(\d+)\.jpg")

    frames = {}
    for f in files:
        name = os.path.basename(f)
        m = regex.match(name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        frames[frame_idx] = f

    return frames  # dict {frame_idx: fullpath}


def find_full_frame_groups(cam_dirs):
    """
    cam0, cam1, ... cam5 폴더 각각에서 공통 frame index를 찾아
    6장의 이미지가 모두 있는 프레임만 반환.
    """
    cam_frames = []
    for cam_idx, cam_dir in enumerate(cam_dirs):
        cam_frames.append(list_frames_in_camera_folder(cam_dir, cam_idx))

    # 가능한 모든 frame index = cam0 프레임 기준
    all_indices = set(cam_frames[0].keys())

    for i in range(1, NUM_CAMS):
        all_indices &= set(cam_frames[i].keys())

    result = []
    for idx in sorted(all_indices):
        paths = [cam_frames[c][idx] for c in range(NUM_CAMS)]
        result.append((idx, paths))

    return result


# =========================
# Cylindrical mapping
# =========================

def generate_cylindrical_map(resized_w, resized_h,
                             fov_h_deg, fov_v_deg,
                             fx_src, fy_src, cx_src, cy_src,
                             resize_factor=1.0):

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
    hh = (ys - cy_cyl) / fy_cyl

    xs_3d = np.sin(theta)
    ys_3d = hh
    zs_3d = np.cos(theta)

    x_n = xs_3d / zs_3d
    y_n = ys_3d / zs_3d

    r2 = x_n * x_n + y_n * y_n
    r4 = r2 * r2
    r6 = r4 * r2

    radial = 1.0 + K1*r2 + K2*r4 + K3*r6
    dx = 2.0 * P1 * x_n * y_n + P2 * (r2 + 2.0 * x_n * x_n)
    dy = P1 * (r2 + 2.0 * y_n * y_n) + 2.0 * P2 * x_n * y_n

    x_d = x_n * radial + dx
    y_d = y_n * radial + dy

    map_x = x_d * fx_r + cx_r
    map_y = y_d * fy_r + cy_r

    return map_x.astype(np.float32), map_y.astype(np.float32)


# =========================
# Feather Blend
# =========================

def feather_blend_with_pad(strips, overlap, final_w, final_h, feather_strength=1.0):
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
                dst[dst_x] = (prev * (1-a) + curr * a).astype(np.uint8)

            # non-overlap
            tail = w - ov
            for x in range(tail):
                dst_x = curx + x
                if dst_x < final_w:
                    dst[dst_x] = strip[ov+x]

            curx += tail

    return pano


def run_yolo_and_draw(self, pano_bgr: np.ndarray) -> np.ndarray:
        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

        results = self.yolo_model(
            pano_rgb,
            imgsz=640,
            conf=0.25,
            verbose=False,
        )

        annotated = results[0].plot()  # BGR
        return annotated

# =========================
# Main
# =========================

def main():
    ensure_dir(PANORAMA_DIR)
    ensure_dir(CACHE_DIR)

    # 1. cam0~cam5 폴더에서 공통 frame index 찾기
    frame_groups = find_full_frame_groups(CAM_DIRS)
    if not frame_groups:
        print("[WARN] No frame groups found.")
        return

    print(f"[INFO] Found {len(frame_groups)} frame groups.")

    resized_w = int(round(WIDTH * RESIZE_FACTOR))
    resized_h = int(round(HEIGHT * RESIZE_FACTOR))

    crop_w = int(round(resized_w * PANORAMA_CROP_WIDTH_SCALE))
    crop_h = int(round(resized_h * PANORAMA_CROP_HEIGHT_SCALE))
    crop_x0 = (resized_w - crop_w) // 2
    crop_y0 = (resized_h - crop_h) // 2

    # Cyl map cache
    cyl_cache = CACHE_DIR / f"cyl_map_{resized_w}x{resized_h}_rf{RESIZE_FACTOR:.3f}.npz"
    if cyl_cache.exists():
        print("[INFO] Loading cached mapping:", cyl_cache)
        data = np.load(cyl_cache)
        map_x, map_y = data["map_x"], data["map_y"]
    else:
        print("[INFO] Generating cylindrical map...")
        map_x, map_y = generate_cylindrical_map(
            resized_w, resized_h,
            CYL_H_FOV_DEG, CYL_V_FOV_DEG,
            fx, fy, cx, cy,
            RESIZE_FACTOR
        )
        np.savez_compressed(cyl_cache, map_x=map_x, map_y=map_y)
        print("[INFO] Saved cylindrical map.")

    # Per-camera width 계산
    overlap_resized = int(round(OVERLAP_PX * RESIZE_FACTOR))
    pano_w_crop = NUM_CAMS * crop_w - (NUM_CAMS - 1) * overlap_resized
    scale_x = TARGET_W / pano_w_crop

    overlap_target = int(round(overlap_resized * scale_x))
    per_w_base = int(round(crop_w * scale_x))

    per_cam_w = [per_w_base] * NUM_CAMS
    total_now = NUM_CAMS * per_w_base - (NUM_CAMS - 1) * overlap_target
    delta = TARGET_W - total_now

    i = 0
    while delta > 0 and i < NUM_CAMS:
        per_cam_w[i] += 1
        delta -= 1
        i += 1
    i = 0
    while delta < 0 and i < NUM_CAMS:
        if per_cam_w[i] > 1:
            per_cam_w[i] -= 1
            delta += 1
        i += 1

    per_cam_h = crop_h

    print(f"[INFO] per_cam_w = {per_cam_w}, overlap_target={overlap_target}")

    # =========================
    # 각 Frame 처리
    # =========================

    for frame_idx, paths in frame_groups:
        print(f"\n[FRAME {frame_idx}]")

        imgs = []
        for cam_idx, p in enumerate(paths):
            img = cv2.imread(p)
            if img is None:
                print(f"[ERROR] Failed to read {p}")
                imgs = []
                break
            if (img.shape[1] != WIDTH) or (img.shape[0] != HEIGHT):
                img = cv2.resize(img, (WIDTH, HEIGHT))
            imgs.append(img)

        if len(imgs) != NUM_CAMS:
            continue

        strips = []
        for cam_idx in range(NUM_CAMS):
            src = imgs[cam_idx]
            resized = cv2.resize(src, (resized_w, resized_h))

            warped = cv2.remap(resized, map_x, map_y, cv2.INTER_LINEAR)

            cropped = warped[crop_y0:crop_y0+crop_h, crop_x0:crop_x0+crop_w]

            target_w = per_cam_w[cam_idx]
            strip = cv2.resize(cropped, (target_w, per_cam_h))
            strips.append(strip)

        pano = feather_blend_with_pad(
            strips,
            overlap_target,
            TARGET_W,
            TARGET_H,
            FEATHER_STRENGTH
        )

        out_path = PANORAMA_DIR / f"pano_f{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), pano)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
