#!/usr/bin/env python3
"""
Perspective Cropping 과정 시각화 스크립트
=========================================

이 스크립트는 FoundPose 추론의 전처리 단계인 Perspective Cropping을 시각화합니다.

[Perspective Cropping 과정]
  1. 실제 BOP 테스트 이미지 입력
  2. CNOS-FastSAM 분할 탐지 결과에서 객체 마스크 로드
  3. 마스크의 2D 바운딩 박스 중심에 광학 축을 맞춘 가상 핀홀 카메라 구성
     - 가상 카메라 이미지 크기 = 템플릿 이미지 크기 (420 x 420)
     - 바운딩 박스를 정사각형으로 만들고 crop_rel_pad 여백 추가
  4. 원본 이미지를 가상 핀홀 카메라로 워핑(warp)하여 크롭 이미지 생성

[출력]
  visualization/image/perspective_cropping_viz.png
    - Panel 1: 원본 실제 이미지
    - Panel 2: CNOS 마스크 오버레이 (모든 탐지 결과를 1장에 표시)
    - Panel 3~: 각 탐지별 가상 핀홀 카메라 워핑 크롭 이미지

실행 방법:
  python run_pipeline.py visualization/code/visualize_perspective_cropping.py
  또는
  C:/Users/PC/miniconda3/envs/foundpose/python.exe visualization/code/visualize_perspective_cropping.py
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Windows에서 한글 폰트 설정
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass  # 폰트 설정 실패 시 기본 폰트 사용

# ---------------------------------------------------------------------------
# 경로 설정: REPO_ROOT 기준으로 FoundPose 유틸리티 임포트
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "external", "bop_toolkit"))
sys.path.insert(0, os.path.join(REPO_ROOT, "external", "dinov2"))

from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import calc_crop_box, construct_crop_camera, warp_image
from bop_toolkit_lib.pycoco_utils import rle_to_binary_mask

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
BOP_PATH = os.environ.get(
    "BOP_PATH",
    "D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets",
)
DATASET = "lmo"
SCENE_ID = 2       # LM-O 테스트 씬 번호
IMAGE_ID = 1001    # 시각화할 이미지 프레임 ID
OBJECT_LID = 1     # 대상 오브젝트 (LM-O object 1)

# Perspective Cropping 파라미터 (scripts/infer.py의 기본값과 동일)
CROP_SIZE = (420, 420)   # 가상 카메라 이미지 크기 (= 템플릿 이미지 크기)
CROP_REL_PAD = 0.2       # 바운딩 박스 여백 비율

MAX_DETECTIONS = 4  # 표시할 최대 탐지 수 (score 상위 N개)

# CNOS-FastSAM 탐지 결과 JSON
CNOS_JSON_PATH = os.path.join(
    BOP_PATH,
    "bop23_default_detections_for_task4",
    "cnos-fastsam",
    "cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json",
)

SCENE_DIR = os.path.join(BOP_PATH, DATASET, "test", f"{SCENE_ID:06d}")
OUTPUT_PATH = os.path.join(REPO_ROOT, "visualization", "image", "perspective_cropping_viz.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 탐지별 색상 (RGBA-like, [0,1] range)
COLORS = [
    np.array([1.0, 0.25, 0.25]),  # 빨강
    np.array([0.2,  0.85, 0.2 ]),  # 초록
    np.array([0.3,  0.5,  1.0 ]),  # 파랑
    np.array([1.0,  0.8,  0.0 ]),  # 노랑
]

# ---------------------------------------------------------------------------
# 1. 이미지 로드
# ---------------------------------------------------------------------------
image_path = os.path.join(SCENE_DIR, "rgb", f"{IMAGE_ID:06d}.png")
assert os.path.isfile(image_path), f"이미지 파일을 찾을 수 없습니다: {image_path}"

image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
H, W = image_rgb.shape[:2]
print(f"[1] 이미지 로드: {image_path}  ({W}x{H})")

# ---------------------------------------------------------------------------
# 2. 카메라 파라미터 로드 (scene_camera.json)
# ---------------------------------------------------------------------------
with open(os.path.join(SCENE_DIR, "scene_camera.json")) as f:
    scene_camera_json = json.load(f)

cam_json = scene_camera_json[str(IMAGE_ID)]
K = np.array(cam_json["cam_K"], dtype=np.float32).reshape(3, 3)
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

# LM-O scene_camera.json에는 extrinsics(R, t)가 없으므로 T_world_from_eye = Identity
orig_camera = PinholePlaneCameraModel(
    width=W,
    height=H,
    f=(fx, fy),
    c=(cx, cy),
    T_world_from_eye=np.eye(4, dtype=np.float32),
)
print(f"[2] 카메라: f=({fx:.2f}, {fy:.2f}), c=({cx:.2f}, {cy:.2f}), size=({W}x{H})")

# ---------------------------------------------------------------------------
# 3. CNOS 탐지 결과 로드 및 필터링
# ---------------------------------------------------------------------------
with open(CNOS_JSON_PATH) as f:
    all_preds = json.load(f)

# (scene_id, image_id, category_id)로 필터링 후 score 내림차순 정렬
preds = [
    p for p in all_preds
    if p["scene_id"] == SCENE_ID
    and p["image_id"] == IMAGE_ID
    and p["category_id"] == OBJECT_LID
]
preds = sorted(preds, key=lambda x: x["score"], reverse=True)[:MAX_DETECTIONS]
print(f"[3] CNOS 탐지: scene={SCENE_ID}, image={IMAGE_ID}, obj={OBJECT_LID} → {len(preds)}개")

assert len(preds) > 0, (
    f"탐지 결과가 없습니다. scene_id={SCENE_ID}, image_id={IMAGE_ID}, "
    f"category_id={OBJECT_LID} 설정을 확인하세요."
)

# ---------------------------------------------------------------------------
# 4. Perspective Cropping 수행
# ---------------------------------------------------------------------------
orig_image_float = image_rgb.astype(np.float32) / 255.0  # [0, 1] float

results = []
for i, pred in enumerate(preds):
    # 4-1. RLE → binary mask 디코딩
    mask = rle_to_binary_mask(pred["segmentation"]).astype(np.uint8)

    # 4-2. Morphological opening (고립 픽셀 제거)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    # 4-3. Amodal bbox [x, y, w, h] → AlignedBox2f
    bx, by, bw, bh = pred["bbox"]
    box_amodal = AlignedBox2f(left=bx, top=by, right=bx + bw, bottom=by + bh)

    # 4-4. 정사각형 크롭 박스 계산 (가장 긴 변 기준으로 정사각형)
    crop_box = calc_crop_box(box=box_amodal, make_square=True)

    # 4-5. 가상 핀홀 카메라 구성
    #   - 광학 축: 바운딩 박스 중심과 일치
    #   - 이미지 크기: CROP_SIZE (= 템플릿 이미지 크기)
    #   - crop_rel_pad: 바운딩 박스 주변 여백
    crop_camera = construct_crop_camera(
        box=crop_box,
        camera_model_c2w=orig_camera,
        viewport_size=CROP_SIZE,
        viewport_rel_pad=CROP_REL_PAD,
    )

    # 4-6. 이미지 워핑: 원본 → 가상 핀홀 카메라
    #   박스가 이미 viewport보다 크면 INTER_AREA(다운샘플링 품질 우수), 작으면 INTER_LINEAR
    interpolation = (
        cv2.INTER_AREA if crop_box.width >= crop_camera.width else cv2.INTER_LINEAR
    )
    cropped_image = warp_image(
        src_camera=orig_camera,
        dst_camera=crop_camera,
        src_image=orig_image_float,
        interpolation=interpolation,
    )

    # 4-7. 마스크도 동일하게 워핑 (INTER_NEAREST로 이진값 보존)
    cropped_mask = warp_image(
        src_camera=orig_camera,
        dst_camera=crop_camera,
        src_image=mask.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
    )

    results.append(
        {
            "score": pred["score"],
            "mask": mask,
            "box_amodal": box_amodal,
            "crop_box": crop_box,
            "crop_camera_f": crop_camera.f,
            "crop_camera_c": crop_camera.c,
            "cropped_image": cropped_image,
            "cropped_mask": cropped_mask,
        }
    )
    print(
        f"  Det {i+1}: score={pred['score']:.3f}, "
        f"bbox=[{bx:.0f},{by:.0f},{bw:.0f},{bh:.0f}], "
        f"crop_f=({crop_camera.f[0]:.1f},{crop_camera.f[1]:.1f})"
    )

# ---------------------------------------------------------------------------
# 5. 시각화
# ---------------------------------------------------------------------------
N = len(results)
N_COLS = max(N, 2)  # 하단 행 열 수 (최소 2)

fig = plt.figure(figsize=(5 * N_COLS, 10))
gs = GridSpec(
    nrows=2,
    ncols=N_COLS,
    figure=fig,
    hspace=0.35,
    wspace=0.08,
)

# ---- Panel 1: 원본 이미지 ----
ax0 = fig.add_subplot(gs[0, : N_COLS // 2])
ax0.imshow(image_rgb)
ax0.set_title(
    f"[1] 원본 이미지\n"
    f"Scene {SCENE_ID}, Frame {IMAGE_ID}  ({W}×{H})",
    fontsize=11,
    fontweight="bold",
)
ax0.axis("off")

# ---- Panel 2: CNOS 마스크 오버레이 ----
ax1 = fig.add_subplot(gs[0, N_COLS // 2 :])
overlay = orig_image_float.copy()
legend_patches = []
for i, res in enumerate(results):
    color = COLORS[i % len(COLORS)]
    mask = res["mask"]
    # 마스크 영역에 색상 오버레이 (반투명)
    alpha = 0.55
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask > 0,
            overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay[:, :, c],
        )
    # 마스크 윤곽선 (contour)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    cv2.drawContours(overlay_uint8, contours, -1, (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 2)
    overlay = overlay_uint8.astype(np.float32) / 255.0

    # 바운딩 박스
    box = res["box_amodal"]
    rect = plt.Rectangle(
        (box.left, box.top),
        box.width,
        box.height,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
        linestyle="--",
    )
    ax1.add_patch(rect)
    # 스코어 텍스트
    ax1.text(
        box.left + 2,
        box.top - 5,
        f"#{i+1} {res['score']:.2f}",
        color=color,
        fontsize=9,
        fontweight="bold",
    )
    legend_patches.append(
        mpatches.Patch(color=color, label=f"Det {i+1} (score={res['score']:.3f})")
    )

ax1.imshow(np.clip(overlay, 0, 1))
ax1.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.8)
ax1.set_title(
    f"[2] CNOS-FastSAM 분할 마스크 ({N}개 탐지)\n"
    f"Object LID={OBJECT_LID}",
    fontsize=11,
    fontweight="bold",
)
ax1.axis("off")

# ---- Panel 3~: 크롭 이미지 ----
for i, res in enumerate(results):
    ax = fig.add_subplot(gs[1, i])
    color = COLORS[i % len(COLORS)]

    crop_img = np.clip(res["cropped_image"], 0, 1)
    crop_mask = res["cropped_mask"]

    # 크롭 이미지에 마스크 영역 반투명 색상 오버레이
    display = crop_img.copy()
    alpha = 0.3
    for c in range(3):
        display[:, :, c] = np.where(
            crop_mask > 0.5,
            display[:, :, c] * (1 - alpha) + color[c] * alpha,
            display[:, :, c],
        )

    # 마스크 윤곽선
    crop_mask_u8 = (crop_mask > 0.5).astype(np.uint8)
    display_u8 = (np.clip(display, 0, 1) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(crop_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display_u8, contours, -1, (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 2)

    ax.imshow(display_u8)
    ax.set_title(
        f"[3-{i+1}] 가상 핀홀 워핑 크롭 (Det {i+1})\n"
        f"score={res['score']:.3f} | 크기={CROP_SIZE[0]}×{CROP_SIZE[1]}\n"
        f"가상 카메라 f=({res['crop_camera_f'][0]:.1f}, {res['crop_camera_f'][1]:.1f})",
        fontsize=9,
    )
    ax.axis("off")

# 사용하지 않는 하단 axes 숨김
for j in range(N, N_COLS):
    fig.add_subplot(gs[1, j]).axis("off")

fig.suptitle(
    f"Perspective Cropping 시각화  |  LM-O  Object {OBJECT_LID}\n"
    f"Scene {SCENE_ID}, Image {IMAGE_ID}  |  "
    f"crop_size={CROP_SIZE[0]}×{CROP_SIZE[1]}, crop_rel_pad={CROP_REL_PAD}",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n[완료] 시각화 저장: {OUTPUT_PATH}")
