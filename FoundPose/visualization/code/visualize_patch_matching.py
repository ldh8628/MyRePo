#!/usr/bin/env python3
"""
패치 매칭 시각화 스크립트
=========================

FoundPose의 2D-3D 대응 구축 단계인 패치 매칭을 시각화합니다.

[검증 대상 — 논문 설명]
  "검색된 템플릿에 대해, 크롭의 패치 디스크립터를 유클리드 거리 기준으로
   가장 가까운 디스크립터에 매칭"

[실제 구현 — corresp_util.py `cyclic_buddies_matching`]
  1. 쿼리(크롭) → 템플릿 방향  NN 검색 (유클리드)
  2. 템플릿 → 쿼리(크롭) 방향  NN 검색 (역방향)
  3. 사이클 거리(원점 → 매칭 → 역매칭 후 원점까지의 픽셀 거리) 계산
  4. 사이클 거리가 가장 작은 top-k 쌍을 최종 Best Buddy로 선택
  → 단순 NN에 상호 일관성(cyclic consistency) 필터를 추가한 구조

[시각화 출력 — visualization/image/patch_matching_viz.png]
  Row 0: 원본 이미지   — [크롭] [Top-1 템플릿] [Top-2] ... [Top-h]
  Row 1: PCA 특징 맵  — 동일한 PCA 색공간으로 시각화 (색이 같으면 디스크립터가 유사)
  Row 2: 매칭 결과    — [크롭 어트랙션 맵] [각 템플릿 패치를 매칭된 크롭 패치 색으로 표시]

실행:
  python run_pipeline.py visualization/code/visualize_patch_matching.py
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
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA as SklearnPCA

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "external", "bop_toolkit"))
sys.path.insert(0, os.path.join(REPO_ROOT, "external", "dinov2"))

from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import calc_crop_box, construct_crop_camera, warp_image
from utils import feature_util, knn_util, repre_util, projector_util
from utils.template_util import calc_tfidf, find_nearest_object_features
from utils.corresp_util import cyclic_buddies_matching
from bop_toolkit_lib.pycoco_utils import rle_to_binary_mask

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
BOP_PATH       = os.environ.get("BOP_PATH", "D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets")
DATASET        = "lmo"
SCENE_ID       = 2
IMAGE_ID       = 1001
OBJECT_LID     = 1
DETECTION_IDX  = 0

CROP_SIZE      = (420, 420)
CROP_REL_PAD   = 0.2
TOP_N_TEMPLATES = 5

REPRE_DIR    = os.path.join(REPO_ROOT, "outputs", "object_repre", DATASET, "v1", str(OBJECT_LID))
CNOS_JSON    = os.path.join(
    BOP_PATH, "bop23_default_detections_for_task4", "cnos-fastsam",
    "cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json",
)
SCENE_DIR    = os.path.join(BOP_PATH, DATASET, "test", f"{SCENE_ID:06d}")
OUTPUT_PATH  = os.path.join(REPO_ROOT, "visualization", "image", "patch_matching_viz.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[장치] {DEVICE}")

# ===========================================================================
# ① repre.pth 로드
# ===========================================================================
print("\n[1] repre.pth 로드")
repre = repre_util.load_object_repre(repre_dir=REPRE_DIR, tensor_device="cpu")
# 저장된 CUDA 텐서를 CPU로 강제 이동
repre.feat_cluster_centroids = repre.feat_cluster_centroids.cpu()
repre.feat_cluster_idfs      = repre.feat_cluster_idfs.cpu()
repre.template_descs         = repre.template_descs.cpu()
repre.feat_vectors           = repre.feat_vectors.cpu()
try:
    repre.feat_to_template_ids = repre.feat_to_template_ids.cpu()
except Exception:
    pass

# ===========================================================================
# ② 이미지 + 카메라 로드
# ===========================================================================
print("\n[2] 이미지 + 카메라")
image_path = os.path.join(SCENE_DIR, "rgb", f"{IMAGE_ID:06d}.png")
image_rgb  = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
H, W       = image_rgb.shape[:2]

with open(os.path.join(SCENE_DIR, "scene_camera.json")) as f:
    sc = json.load(f)
K = np.array(sc[str(IMAGE_ID)]["cam_K"], dtype=np.float32).reshape(3, 3)
orig_camera = PinholePlaneCameraModel(
    width=W, height=H,
    f=(K[0, 0], K[1, 1]), c=(K[0, 2], K[1, 2]),
    T_world_from_eye=np.eye(4, dtype=np.float32),
)

# ===========================================================================
# ③ CNOS 탐지 → Perspective Crop
# ===========================================================================
print("\n[3] Perspective Crop")
with open(CNOS_JSON) as f:
    all_preds = json.load(f)
preds = sorted(
    [p for p in all_preds if p["scene_id"] == SCENE_ID
     and p["image_id"] == IMAGE_ID and p["category_id"] == OBJECT_LID],
    key=lambda x: x["score"], reverse=True,
)
pred = preds[DETECTION_IDX]

orig_mask = rle_to_binary_mask(pred["segmentation"]).astype(np.uint8)
orig_mask = cv2.morphologyEx(orig_mask, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
bx, by, bw, bh = pred["bbox"]
crop_box    = calc_crop_box(AlignedBox2f(bx, by, bx+bw, by+bh), make_square=True)
crop_camera = construct_crop_camera(crop_box, orig_camera, CROP_SIZE, CROP_REL_PAD)

interp = cv2.INTER_AREA if crop_box.width >= crop_camera.width else cv2.INTER_LINEAR
crop_float  = warp_image(orig_camera, crop_camera, image_rgb.astype(np.float32)/255.0, interp)
crop_mask   = warp_image(orig_camera, crop_camera, orig_mask.astype(np.float32), cv2.INTER_NEAREST)
crop_mask_u8 = (crop_mask > 0.5).astype(np.uint8)
print(f"  score={pred['score']:.3f}, 마스크 픽셀={crop_mask_u8.sum()}")

# ===========================================================================
# ④ DINOv2 특징 추출 (크롭)
# ===========================================================================
print("\n[4] DINOv2 특징 추출 (크롭)")
extractor = feature_util.make_feature_extractor(repre.feat_opts.extractor_name)
extractor.eval()
if DEVICE == "cuda":
    extractor = extractor.cuda()

crop_tensor = torch.from_numpy(crop_float).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
with torch.no_grad():
    out = extractor(crop_tensor)
crop_feat_chw = out["feature_maps"][0].cpu()          # (C, fH, fW)
crop_feat_hwc = crop_feat_chw.permute(1,2,0).numpy()  # (fH, fW, C)
feat_C, feat_H, feat_W = crop_feat_chw.shape
PATCH_SIZE = CROP_SIZE[0] // feat_H
print(f"  특징 맵: ({feat_C} × {feat_H} × {feat_W}), 패치 크기={PATCH_SIZE}px")

# ===========================================================================
# ⑤ BoW + Top-h 템플릿 선택 (visualize_similar_templates.py와 동일)
# ===========================================================================
print("\n[5] BoW → Top-h 템플릿 선택")
grid_pts    = feature_util.generate_grid_points(CROP_SIZE, cell_size=1.0)
mask_tensor = torch.from_numpy(crop_mask_u8)
query_pts   = feature_util.filter_points_by_mask(grid_pts, mask_tensor)
query_feats = feature_util.sample_feature_map_at_points(
    crop_feat_chw, query_pts, CROP_SIZE).contiguous()

if query_feats.shape[1] != repre.feat_vectors.shape[1] and len(repre.feat_raw_projectors) > 0:
    query_feats_proj = projector_util.project_features(
        query_feats, repre.feat_raw_projectors).contiguous()
else:
    query_feats_proj = query_feats

vw_index = knn_util.KNN(k=repre.template_desc_opts.tfidf_knn_k,
                         metric=repre.template_desc_opts.tfidf_knn_metric)
vw_index.fit(repre.feat_cluster_centroids)
word_ids, word_dists = find_nearest_object_features(query_feats_proj, vw_index)
word_ids, word_dists = word_ids.cpu(), word_dists.cpu()

query_bow = calc_tfidf(word_ids, word_dists, repre.feat_cluster_idfs,
                        repre.template_desc_opts.tfidf_soft_assign,
                        repre.template_desc_opts.tfidf_soft_sigma_squared)
cos_sims  = F.cosine_similarity(repre.template_descs,
                                 query_bow.unsqueeze(0).expand_as(repre.template_descs))
top_scores, top_ids = torch.topk(cos_sims, k=TOP_N_TEMPLATES, sorted=True)
top_ids_np    = top_ids.numpy()
top_scores_np = top_scores.numpy()
for r, (tid, sc_) in enumerate(zip(top_ids_np, top_scores_np)):
    print(f"  Top-{r+1}: template #{tid}, cos_sim={sc_:.4f}")

# ===========================================================================
# ⑥ DINOv2 특징 추출 (각 템플릿 이미지)
# ===========================================================================
print("\n[6] DINOv2 특징 추출 (템플릿)")
tpl_feat_hwc  = {}   # tid → (fH, fW, C)  raw 384-dim
tpl_images    = {}   # tid → (420, 420, 3) uint8
tpl_masks_30  = {}   # tid → (fH, fW)     binary mask

for tid in top_ids_np:
    img = repre.templates[tid].cpu().permute(1,2,0).numpy()   # HWC uint8
    tpl_images[tid] = img
    t   = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        to = extractor(t)
    tpl_feat_hwc[tid] = to["feature_maps"][0].cpu().permute(1,2,0).numpy()  # (fH, fW, C)

    # 마스크: 템플릿 이미지에서 검은 배경(픽셀 max < 5) 제외
    non_black = (img.astype(np.float32).max(axis=2) > 5).astype(np.uint8)
    tpl_masks_30[tid] = cv2.resize(non_black, (feat_W, feat_H), interpolation=cv2.INTER_NEAREST)
    print(f"  template #{tid}: 마스크 패치 수={tpl_masks_30[tid].sum()}")

# 크롭 마스크도 30x30으로 다운스케일
crop_mask_30 = cv2.resize(crop_mask_u8, (feat_W, feat_H), interpolation=cv2.INTER_NEAREST)

# ===========================================================================
# ⑦ PCA → RGB 시각화 색공간 구성 (전체 이미지 공통)
# ===========================================================================
print("\n[7] PCA RGB 색공간 구성")
crop_feat_flat = crop_feat_hwc[crop_mask_30 > 0]   # (N_crop, 384)
tpl_feats_flat = [tpl_feat_hwc[t][tpl_masks_30[t] > 0] for t in top_ids_np]
all_feats_np   = np.concatenate([crop_feat_flat] + tpl_feats_flat, axis=0)  # (N_all, 384)

viz_pca     = SklearnPCA(n_components=3, random_state=42)
viz_pca.fit(all_feats_np)
pca_all     = viz_pca.transform(all_feats_np)
pca_min     = pca_all.min(axis=0)
pca_range   = pca_all.max(axis=0) - pca_min + 1e-8
print(f"  PCA fitted on {len(all_feats_np)} patches (크롭 + {TOP_N_TEMPLATES}개 템플릿)")
print(f"  분산 설명량: {viz_pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}% (3PC)")

def feat_to_rgb_map(feat_hwc, mask_hw):
    """(fH, fW, C) 특징 맵 → (fH, fW, 3) PCA RGB float [0,1]. 마스크 외 = 검정."""
    fH, fW = mask_hw.shape
    proj = viz_pca.transform(feat_hwc.reshape(-1, feat_hwc.shape[-1]))  # (fH*fW, 3)
    norm = np.clip((proj - pca_min) / pca_range, 0, 1)
    rgb  = norm.reshape(fH, fW, 3)
    rgb[mask_hw == 0] = 0.0
    return rgb

def upscale_rgb(img_hw3, size=(420, 420)):
    """float [0,1] 패치맵 → uint8 420x420 이미지 (최근접 보간)."""
    u8 = (np.clip(img_hw3, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(u8, size, interpolation=cv2.INTER_NEAREST)

# 크롭 PCA RGB
crop_pca_rgb_30  = feat_to_rgb_map(crop_feat_hwc, crop_mask_30)   # (fH, fW, 3)
crop_pca_rgb_420 = upscale_rgb(crop_pca_rgb_30)                    # (420, 420, 3) uint8

# 각 크롭 마스크 패치의 PCA 색상 (매칭 시 참조용)
crop_positions    = np.argwhere(crop_mask_30 > 0)          # (N_crop, 2) [row, col]
crop_pca_colors   = crop_pca_rgb_30[crop_mask_30 > 0]      # (N_crop, 3) float [0,1]

# ===========================================================================
# ⑧ PCA 투영(256-dim) → 패치 매칭 (유클리드 NN)
# ===========================================================================
print("\n[8] 패치 매칭 (유클리드 NN, 256-dim PCA 공간)")

def apply_proj(feat_np):
    """(N, 384) → (N, 256) 논문과 동일한 PCA 투영 적용."""
    t   = torch.from_numpy(feat_np).float()
    out = projector_util.project_features(t, repre.feat_raw_projectors)
    return out.numpy()

def l2_nn(query, ref):
    """(N_q, D) × (N_r, D) → (N_q,) nearest ref index for each query."""
    # |q - r|^2 = |q|^2 + |r|^2 - 2 q·r
    dists = (
        (query ** 2).sum(1, keepdims=True) +
        (ref   ** 2).sum(1, keepdims=True).T -
        2 * query @ ref.T
    )
    return dists.argmin(axis=1)

crop_feat_256    = apply_proj(crop_feat_flat)               # (N_crop, 256)

match_maps_420   = {}   # tid → (420, 420, 3) uint8 — 템플릿 패치를 매칭 크롭 색으로 채색
crop_attract_cnt = np.zeros(len(crop_positions), dtype=np.int32)  # 크롭 패치별 매칭 횟수

for tid in top_ids_np:
    m              = tpl_masks_30[tid]
    tpl_flat       = tpl_feat_hwc[tid][m > 0]    # (N_tpl, 384)
    tpl_feat_256   = apply_proj(tpl_flat)          # (N_tpl, 256)
    tpl_positions  = np.argwhere(m > 0)            # (N_tpl, 2)

    # 논문 설명: "크롭 패치 → 가장 가까운 템플릿 패치" (query2obj)
    crop2tpl       = l2_nn(crop_feat_256, tpl_feat_256)    # (N_crop,)
    # 역방향: "템플릿 패치 → 가장 가까운 크롭 패치" (obj2query)
    tpl2crop       = l2_nn(tpl_feat_256,  crop_feat_256)   # (N_tpl,)

    # 어트랙션 카운트 (N_tpl개 템플릿 패치가 각각 어떤 크롭 패치를 지목했는지)
    for ci in tpl2crop:
        crop_attract_cnt[ci] += 1

    # 매칭 맵: 각 템플릿 패치를 가장 가까운 크롭 패치의 PCA 색으로 채색
    match_30 = np.zeros((feat_H, feat_W, 3), dtype=np.float32)
    for i, (r, c) in enumerate(tpl_positions):
        match_30[r, c] = crop_pca_colors[tpl2crop[i]]
    match_maps_420[tid] = upscale_rgb(match_30)

    # 사이클 일관성 검증 (cyclic buddies: crop → tpl → crop 사이클)
    cycle_ids  = tpl2crop[crop2tpl]        # crop → tpl → 다시 crop
    cycle_ok   = (cycle_ids == np.arange(len(crop_positions)))  # 원점 복귀 여부
    print(f"  template #{tid}: cyclic buddy 비율 = "
          f"{cycle_ok.mean()*100:.1f}% "
          f"({cycle_ok.sum()}/{len(crop_positions)} 크롭 패치)")

# 크롭 어트랙션 맵
crop_attract_30 = np.zeros((feat_H, feat_W), dtype=np.float32)
for i, (r, c) in enumerate(crop_positions):
    crop_attract_30[r, c] = crop_attract_cnt[i]
if crop_attract_30.max() > 0:
    crop_attract_norm = crop_attract_30 / crop_attract_30.max()
else:
    crop_attract_norm = crop_attract_30

attract_color = plt.cm.YlOrRd(crop_attract_norm)[:, :, :3].astype(np.float32)
attract_color[crop_mask_30 == 0] = 0.0
crop_attract_420 = upscale_rgb(attract_color)   # (420, 420, 3)

# ===========================================================================
# ⑨ 시각화 Figure 생성
# ===========================================================================
print("\n[9] 시각화 생성")

N_COLS = 1 + TOP_N_TEMPLATES   # 크롭 + h개 템플릿
TEMPLATE_COLORS = ["#D62728", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B"]

fig, axes = plt.subplots(3, N_COLS, figsize=(3.8 * N_COLS, 12))

# ─── Row 0: 원본 이미지 ───────────────────────────────────────────────────

# 크롭 이미지 (마스크 윤곽선 포함)
ax = axes[0, 0]
crop_disp = (np.clip(crop_float, 0, 1) * 255).astype(np.uint8).copy()
cnts, _ = cv2.findContours(crop_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(crop_disp, cnts, -1, (30, 144, 255), 2)
ax.imshow(crop_disp)
ax.set_title(f"쿼리 크롭\nScene {SCENE_ID}, Frame {IMAGE_ID}\n(score={pred['score']:.3f})", fontsize=9, fontweight="bold")
ax.axis("off")

for rank, (tid, sc_) in enumerate(zip(top_ids_np, top_scores_np)):
    ax = axes[0, rank + 1]
    ax.imshow(tpl_images[tid])
    ax.set_title(f"Top-{rank+1} 템플릿 #{tid}\ncos_sim={sc_:.4f}",
                 fontsize=9, color=TEMPLATE_COLORS[rank], fontweight="bold")
    ax.axis("off")

# ─── Row 1: PCA 특징 맵 ──────────────────────────────────────────────────

ax = axes[1, 0]
ax.imshow(crop_pca_rgb_420)
ax.set_title("크롭 PCA 특징 맵\n(같은 색 = 유사한 디스크립터)", fontsize=9, fontweight="bold")
ax.axis("off")

for rank, tid in enumerate(top_ids_np):
    ax = axes[1, rank + 1]
    tpl_pca = feat_to_rgb_map(tpl_feat_hwc[tid], tpl_masks_30[tid])
    ax.imshow(upscale_rgb(tpl_pca))
    ax.set_title(f"템플릿 #{tid} PCA 특징 맵",
                 fontsize=9, color=TEMPLATE_COLORS[rank])
    ax.axis("off")

# ─── Row 2: 매칭 결과 ────────────────────────────────────────────────────

# 크롭 어트랙션 맵
ax = axes[2, 0]
ax.imshow(crop_attract_420)
# 색상 범례 (colorbar 대신 텍스트)
ax.set_title("크롭 어트랙션 맵\n(어두움=적게 매칭, 밝음=많이 매칭)\n"
             f"총 {TOP_N_TEMPLATES}개 템플릿의 패치들이 지목한 빈도",
             fontsize=8, fontweight="bold")
ax.axis("off")

for rank, tid in enumerate(top_ids_np):
    ax = axes[2, rank + 1]
    ax.imshow(match_maps_420[tid])
    ax.set_title(f"템플릿 #{tid} 매칭 맵\n"
                 f"(각 패치 → 가장 가까운 크롭 패치 색)\n"
                 f"크롭 PCA와 색이 같으면 매칭 성공",
                 fontsize=8, color=TEMPLATE_COLORS[rank])
    ax.axis("off")

# ─── Row 레이블 ──────────────────────────────────────────────────────────
for row, label in enumerate(["① 원본 이미지", "② PCA 패치 디스크립터", "③ 매칭 결과"]):
    axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold",
                             rotation=90, labelpad=6)

# ─── 설명 범례 ────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.04,
    "■ PCA 특징 맵: DINOv2 384차원 특징을 PCA 3성분(→RGB)으로 시각화. 모든 이미지가 동일한 색공간 사용.\n"
    "  → 크롭과 템플릿에서 같은 색 영역 = 유사한 패치 디스크립터 = 낮은 유클리드 거리\n"
    "■ 매칭 맵: 각 템플릿 패치를 256-dim PCA 공간에서 유클리드 최근접 크롭 패치의 색으로 채색.\n"
    "  → 매칭 맵 색상이 크롭 PCA 맵과 일치할수록 해당 영역의 대응이 정확함을 의미\n"
    "■ 실제 구현(cyclic_buddies): 크롭→템플릿 NN 후 역방향 NN으로 사이클 거리 계산, 상위 top-k 선택",
    ha="center", fontsize=8.5, color="#333333",
    transform=fig.transFigure,
    bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#CCCCCC", alpha=0.9),
)

fig.suptitle(
    f"패치 매칭 시각화  |  LM-O Object {OBJECT_LID}\n"
    f"[DINOv2 특징 → 256-dim PCA → 유클리드 NN 매칭 → Cyclic Buddy 필터]",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.tight_layout(h_pad=1.0, w_pad=0.4)
plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n[완료] {OUTPUT_PATH}")
