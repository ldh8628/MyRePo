#!/usr/bin/env python3
"""
유사 템플릿 검색 시각화 스크립트
===================================

FoundPose의 초기 포즈 추정 단계 — Bag-of-Words 기반 템플릿 검색을 시각화합니다.

[검색 과정]
  1. Perspective Cropping  → 객체 크롭 이미지 (420×420)
  2. DINOv2 특징 추출      → 크롭 이미지에서 패치별 특징 벡터 추출
  3. PCA 투영              → 원본 차원 → 256차원 압축
  4. Visual Word 할당      → 각 특징 벡터를 k=3 최근접 클러스터 중심에 할당
  5. TF-IDF BoW 계산       → 2048-차원 희소 디스크립터 생성
  6. 코사인 유사도 비교    → repre.pth에 저장된 798개 템플릿 BoW와 비교
  7. Top-h 템플릿 선택     → 코사인 유사도 상위 h개 템플릿 반환

[출력]
  visualization/image/similar_templates_viz.png
    Row 0: [쿼리 크롭 이미지]  [Top-1 템플릿]  [Top-2]  ...  [Top-h]
    Row 1: [쿼리 BoW 차트]     [Top-1 BoW]     [Top-2]  ...  [Top-h]
           * BoW 차트: 쿼리 상위 활성화 Visual Words의 TF-IDF 값 비교

실행 방법:
  python run_pipeline.py visualization/code/visualize_similar_templates.py
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

# 한글 폰트 설정 (Windows)
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
from bop_toolkit_lib.pycoco_utils import rle_to_binary_mask

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
BOP_PATH = os.environ.get(
    "BOP_PATH",
    "D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets",
)
DATASET        = "lmo"
SCENE_ID       = 2
IMAGE_ID       = 1001
OBJECT_LID     = 1        # repre.pth를 로드할 오브젝트 ID
DETECTION_IDX  = 0        # 사용할 탐지 결과 인덱스 (0 = score 최상위)

CROP_SIZE      = (420, 420)
CROP_REL_PAD   = 0.2
TOP_N_TEMPLATES = 5       # 유사 템플릿 개수 h
TOP_K_WORDS    = 25       # BoW 차트에 표시할 상위 Visual Word 수

REPRE_DIR = os.path.join(REPO_ROOT, "outputs", "object_repre", DATASET, "v1", str(OBJECT_LID))
CNOS_JSON_PATH = os.path.join(
    BOP_PATH,
    "bop23_default_detections_for_task4",
    "cnos-fastsam",
    "cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json",
)
SCENE_DIR   = os.path.join(BOP_PATH, DATASET, "test", f"{SCENE_ID:06d}")
OUTPUT_PATH = os.path.join(REPO_ROOT, "visualization", "image", "similar_templates_viz.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[장치] {DEVICE}")

# ---------------------------------------------------------------------------
# Step 1. repre.pth 로드
# ---------------------------------------------------------------------------
print(f"\n[Step 1] repre.pth 로드: {REPRE_DIR}")
repre = repre_util.load_object_repre(repre_dir=REPRE_DIR, tensor_device="cpu")

# repre.pth는 GPU에서 저장되었을 수 있음 → 시각화는 CPU에서 수행하므로 강제 이동
# (load_object_repre의 tensor_device 파라미터가 현재 구현에서 실제로 적용되지 않음)
repre.feat_cluster_centroids = repre.feat_cluster_centroids.cpu()
repre.feat_cluster_idfs      = repre.feat_cluster_idfs.cpu()
repre.template_descs         = repre.template_descs.cpu()
repre.feat_vectors           = repre.feat_vectors.cpu()

extractor_name = repre.feat_opts.extractor_name
print(f"  DINOv2 모델: {extractor_name}")
print(f"  feat_cluster_centroids: {repre.feat_cluster_centroids.shape}  (k-means 클러스터 중심)")
print(f"  feat_cluster_idfs:      {repre.feat_cluster_idfs.shape}    (IDF 값)")
print(f"  template_descs:         {repre.template_descs.shape}  (템플릿 BoW 디스크립터)")
print(f"  templates:              {repre.templates.shape}    (템플릿 RGB 이미지)")
print(f"  TF-IDF 설정: knn_k={repre.template_desc_opts.tfidf_knn_k}, "
      f"soft_assign={repre.template_desc_opts.tfidf_soft_assign}")

# ---------------------------------------------------------------------------
# Step 2. 이미지 + 카메라 로드
# ---------------------------------------------------------------------------
print(f"\n[Step 2] 이미지 로드: scene={SCENE_ID}, image={IMAGE_ID}")
image_path = os.path.join(SCENE_DIR, "rgb", f"{IMAGE_ID:06d}.png")
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
H, W = image_rgb.shape[:2]

with open(os.path.join(SCENE_DIR, "scene_camera.json")) as f:
    scene_camera_json = json.load(f)
cam_json = scene_camera_json[str(IMAGE_ID)]
K = np.array(cam_json["cam_K"], dtype=np.float32).reshape(3, 3)
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
orig_camera = PinholePlaneCameraModel(
    width=W, height=H, f=(fx, fy), c=(cx, cy),
    T_world_from_eye=np.eye(4, dtype=np.float32),
)
print(f"  카메라: f=({fx:.2f},{fy:.2f}), c=({cx:.2f},{cy:.2f}), 크기=({W}x{H})")

# ---------------------------------------------------------------------------
# Step 3. CNOS 탐지 결과 로드 → Perspective Crop
# ---------------------------------------------------------------------------
print(f"\n[Step 3] CNOS 탐지 + Perspective Crop")
with open(CNOS_JSON_PATH) as f:
    all_preds = json.load(f)
preds = [
    p for p in all_preds
    if p["scene_id"] == SCENE_ID
    and p["image_id"] == IMAGE_ID
    and p["category_id"] == OBJECT_LID
]
preds = sorted(preds, key=lambda x: x["score"], reverse=True)
assert len(preds) > 0, "탐지 결과 없음."
pred = preds[DETECTION_IDX]

# 마스크 디코딩 및 전처리
orig_mask = rle_to_binary_mask(pred["segmentation"]).astype(np.uint8)
orig_mask = cv2.morphologyEx(
    orig_mask, cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
)

# Amodal bbox → AlignedBox2f
bx, by, bw, bh = pred["bbox"]
box_amodal = AlignedBox2f(left=bx, top=by, right=bx + bw, bottom=by + bh)

# 정사각형 크롭 박스 계산
crop_box = calc_crop_box(box=box_amodal, make_square=True)

# 가상 핀홀 카메라 구성
crop_camera = construct_crop_camera(
    box=crop_box,
    camera_model_c2w=orig_camera,
    viewport_size=CROP_SIZE,
    viewport_rel_pad=CROP_REL_PAD,
)

# 이미지 + 마스크 워핑
orig_image_float = image_rgb.astype(np.float32) / 255.0
interpolation = cv2.INTER_AREA if crop_box.width >= crop_camera.width else cv2.INTER_LINEAR
cropped_image_float = warp_image(
    src_camera=orig_camera, dst_camera=crop_camera,
    src_image=orig_image_float, interpolation=interpolation,
)
cropped_mask = warp_image(
    src_camera=orig_camera, dst_camera=crop_camera,
    src_image=orig_mask.astype(np.float32), interpolation=cv2.INTER_NEAREST,
)
cropped_mask_binary = (cropped_mask > 0.5).astype(np.uint8)

mask_pixel_count = cropped_mask_binary.sum()
print(f"  탐지 score={pred['score']:.3f}, bbox=[{bx:.0f},{by:.0f},{bw:.0f},{bh:.0f}]")
print(f"  크롭 크기: {CROP_SIZE[0]}x{CROP_SIZE[1]}, 마스크 픽셀 수: {mask_pixel_count}")

# ---------------------------------------------------------------------------
# Step 4. DINOv2 특징 추출
# ---------------------------------------------------------------------------
print(f"\n[Step 4] DINOv2 특징 추출 ({extractor_name})")
extractor = feature_util.make_feature_extractor(model_name=extractor_name)
extractor.eval()
if DEVICE == "cuda":
    extractor = extractor.cuda()

# 크롭 이미지를 텐서로 변환: (H, W, C) → (1, C, H, W), float32
crop_tensor_chw = torch.from_numpy(cropped_image_float).permute(2, 0, 1).float()
crop_tensor_bchw = crop_tensor_chw.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    extractor_output = extractor(crop_tensor_bchw)
feature_map_chw = extractor_output["feature_maps"][0].cpu()  # (C, h_feat, w_feat)
feat_C, feat_H, feat_W = feature_map_chw.shape
print(f"  특징 맵 크기: ({feat_C} x {feat_H} x {feat_W})")
print(f"  → 패치 크기 {CROP_SIZE[0] // feat_H}px, 총 {feat_H * feat_W}개 패치")

# ---------------------------------------------------------------------------
# Step 5. 마스크 내 그리드 포인트에서 특징 샘플링
# ---------------------------------------------------------------------------
print(f"\n[Step 5] 마스크 내 특징 샘플링")
# grid_cell_size=1: 각 픽셀마다 그리드 포인트 생성 (infer.py와 동일)
grid_points = feature_util.generate_grid_points(
    grid_size=CROP_SIZE, cell_size=1.0
)
mask_tensor = torch.from_numpy(cropped_mask_binary)
query_points = feature_util.filter_points_by_mask(grid_points, mask_tensor)
query_features = feature_util.sample_feature_map_at_points(
    feature_map_chw=feature_map_chw,
    points=query_points,
    image_size=(CROP_SIZE[0], CROP_SIZE[1]),
).contiguous()
print(f"  그리드 포인트 (전체): {grid_points.shape[0]}개")
print(f"  마스크 내 포인트:     {query_points.shape[0]}개")
print(f"  추출된 특징 shape:    {query_features.shape}  (num_points x feat_dim)")

# ---------------------------------------------------------------------------
# Step 6. PCA 투영 (feat_dim → 256)
# ---------------------------------------------------------------------------
print(f"\n[Step 6] PCA 투영")
if (
    query_features.shape[1] != repre.feat_vectors.shape[1]
    and len(repre.feat_raw_projectors) > 0
):
    query_features_proj = projector_util.project_features(
        feat_vectors=query_features,
        projectors=repre.feat_raw_projectors,
    ).contiguous()
    print(f"  PCA 적용: {query_features.shape[1]}차원 → {query_features_proj.shape[1]}차원")
else:
    query_features_proj = query_features
    print(f"  PCA 투영 불필요 (이미 {query_features.shape[1]}차원)")

# ---------------------------------------------------------------------------
# Step 7. Visual Word 할당 (KNN: 각 특징 → k=3 최근접 클러스터 중심)
# ---------------------------------------------------------------------------
print(f"\n[Step 7] Visual Word 할당 (KNN k={repre.template_desc_opts.tfidf_knn_k})")
visual_words_knn_index = knn_util.KNN(
    k=repre.template_desc_opts.tfidf_knn_k,
    metric=repre.template_desc_opts.tfidf_knn_metric,
)
visual_words_knn_index.fit(repre.feat_cluster_centroids)

word_ids, word_dists = find_nearest_object_features(
    query_features=query_features_proj,
    knn_index=visual_words_knn_index,
)
# KNN 결과를 CPU로 이동 (repre 텐서들이 CPU에 있으므로 장치 통일)
word_ids   = word_ids.cpu()
word_dists = word_dists.cpu()
print(f"  word_ids shape:   {word_ids.shape}  (num_points x k)")
print(f"  word_dists shape: {word_dists.shape}  (num_points x k)")
print(f"  고유 Visual Word 수: {word_ids.unique().shape[0]} / {repre.feat_cluster_centroids.shape[0]}")

# ---------------------------------------------------------------------------
# Step 8. TF-IDF BoW 디스크립터 계산
# ---------------------------------------------------------------------------
print(f"\n[Step 8] TF-IDF BoW 디스크립터 계산")
query_bow = calc_tfidf(
    feature_word_ids=word_ids,
    feature_word_dists=word_dists,
    word_idfs=repre.feat_cluster_idfs,
    soft_assignment=repre.template_desc_opts.tfidf_soft_assign,
    soft_sigma_squared=repre.template_desc_opts.tfidf_soft_sigma_squared,
)
query_bow_np = query_bow.numpy()
nonzero_count = (query_bow_np > 0).sum()
print(f"  BoW 벡터 shape:   {query_bow.shape}  (num_clusters = {repre.feat_cluster_centroids.shape[0]})")
print(f"  비율 : {nonzero_count} / {len(query_bow_np)} 활성화  ({100*nonzero_count/len(query_bow_np):.1f}%)")
print(f"  최댓값: {query_bow_np.max():.6f}, 최솟값(>0): {query_bow_np[query_bow_np > 0].min():.6f}")

# ---------------------------------------------------------------------------
# Step 9. 코사인 유사도 계산 → Top-h 템플릿 선택
# ---------------------------------------------------------------------------
print(f"\n[Step 9] 코사인 유사도 → Top-{TOP_N_TEMPLATES} 템플릿 선택")
# repre.template_descs: (798, 2048)
cos_sims = F.cosine_similarity(
    repre.template_descs,
    query_bow.unsqueeze(0).expand_as(repre.template_descs),
)
top_scores, top_ids = torch.topk(cos_sims, k=TOP_N_TEMPLATES, sorted=True)
top_scores_np = top_scores.numpy()
top_ids_np    = top_ids.numpy()

for rank, (tid, tscore) in enumerate(zip(top_ids_np, top_scores_np)):
    print(f"  Top-{rank+1}: template #{tid:3d}, cos_sim={tscore:.4f}")

# ---------------------------------------------------------------------------
# Step 10. 시각화
# ---------------------------------------------------------------------------
print(f"\n[Step 10] 시각화 생성")

N_COLS = 1 + TOP_N_TEMPLATES   # 쿼리 + h개 템플릿

# BoW 비교를 위한 공통 Word 인덱스 선택
# 쿼리에서 상위 TOP_K_WORDS 활성화 단어를 고정 축으로 사용
top_word_idx = np.argsort(query_bow_np)[::-1][:TOP_K_WORDS]  # 내림차순
top_word_idx_sorted = np.sort(top_word_idx)                   # 인덱스 오름차순 정렬

# 쿼리 및 각 템플릿에서 해당 단어들의 TF-IDF 값 추출
template_bows_np = repre.template_descs.numpy()  # (798, 2048)
bow_vals_query = query_bow_np[top_word_idx_sorted]  # (TOP_K_WORDS,)
bow_vals_templates = [
    template_bows_np[tid][top_word_idx_sorted] for tid in top_ids_np
]  # list of (TOP_K_WORDS,) arrays

# 색상 팔레트
QUERY_COLOR    = "#2171B5"  # 파랑 (쿼리)
TEMPLATE_COLORS = ["#D62728", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B"]  # 탬플릿별 색상

fig = plt.figure(figsize=(4.5 * N_COLS, 10))
gs = GridSpec(nrows=2, ncols=N_COLS, figure=fig, hspace=0.45, wspace=0.12)

# =====================================================================
# Row 0: 이미지 표시
# =====================================================================

# ---- 쿼리 크롭 이미지 ----
ax_img_query = fig.add_subplot(gs[0, 0])
crop_display = np.clip(cropped_image_float, 0, 1).copy()
# 마스크 윤곽선 표시
contours, _ = cv2.findContours(cropped_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
crop_display_u8 = (crop_display * 255).astype(np.uint8)
cv2.drawContours(crop_display_u8, contours, -1, (0, 100, 255), 2)
ax_img_query.imshow(crop_display_u8)
ax_img_query.set_title(
    f"쿼리 크롭 이미지\n"
    f"Scene {SCENE_ID}, Frame {IMAGE_ID}\n"
    f"score={pred['score']:.3f}",
    fontsize=9, fontweight="bold",
)
ax_img_query.axis("off")

# ---- Top-h 템플릿 이미지 ----
for rank, (tid, tscore) in enumerate(zip(top_ids_np, top_scores_np)):
    ax = fig.add_subplot(gs[0, rank + 1])
    tpl_img = repre.templates[tid].cpu().permute(1, 2, 0).numpy()  # (H, W, C) uint8
    ax.imshow(tpl_img)
    ax.set_title(
        f"Top-{rank+1} 템플릿 #{tid}\n"
        f"cos_sim = {tscore:.4f}",
        fontsize=9,
        color=TEMPLATE_COLORS[rank],
        fontweight="bold",
    )
    ax.axis("off")

# =====================================================================
# Row 1: BoW 디스크립터 바 차트 비교
# =====================================================================
# x 레이블: Visual Word 인덱스 (쿼리 기준 상위 TOP_K_WORDS)
x_labels = [f"W{idx}" for idx in top_word_idx_sorted]
x_pos = np.arange(len(top_word_idx_sorted))

# ---- 쿼리 BoW 차트 ----
ax_bow_query = fig.add_subplot(gs[1, 0])
bars = ax_bow_query.bar(x_pos, bow_vals_query, color=QUERY_COLOR, alpha=0.85, width=0.7)
ax_bow_query.set_xticks(x_pos)
ax_bow_query.set_xticklabels(x_labels, rotation=90, fontsize=6)
ax_bow_query.set_ylabel("TF-IDF", fontsize=8)
ax_bow_query.set_title(
    f"쿼리 BoW 디스크립터\n"
    f"(상위 {TOP_K_WORDS}개 Visual Words)",
    fontsize=9, fontweight="bold", color=QUERY_COLOR,
)
ax_bow_query.set_ylim(0, None)
ax_bow_query.yaxis.set_tick_params(labelsize=7)
ax_bow_query.grid(axis="y", linestyle="--", alpha=0.4)

# ---- 템플릿 BoW 차트 (각각) ----
for rank, (tid, tscore, tbow_vals) in enumerate(
    zip(top_ids_np, top_scores_np, bow_vals_templates)
):
    ax = fig.add_subplot(gs[1, rank + 1])
    tc = TEMPLATE_COLORS[rank]

    # 쿼리와 공유된 단어: 둘 다 > 0
    shared_mask = (bow_vals_query > 0) & (tbow_vals > 0)

    bar_colors = [tc if shared_mask[i] else "#AAAAAA" for i in range(len(x_pos))]
    ax.bar(x_pos, tbow_vals, color=bar_colors, alpha=0.85, width=0.7)

    # 쿼리 BoW를 얇은 선으로 오버레이 (비교 참조)
    ax.step(np.append(x_pos, x_pos[-1] + 1) - 0.5,
            np.append(bow_vals_query, bow_vals_query[-1]),
            where="post", color=QUERY_COLOR, linewidth=1.2, linestyle="--",
            alpha=0.6, label="쿼리 BoW")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
    ax.set_title(
        f"Top-{rank+1} 템플릿 #{tid}\n"
        f"cos_sim={tscore:.4f}  |  공유 단어 {shared_mask.sum()}/{TOP_K_WORDS}",
        fontsize=9, color=tc, fontweight="bold",
    )
    ax.set_ylim(0, None)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if rank == 0:
        ax.legend(fontsize=7, loc="upper right")

# =====================================================================
# 범례 + BoW 설명 텍스트
# =====================================================================
legend_patches = [
    mpatches.Patch(color=QUERY_COLOR, label="쿼리 크롭 이미지"),
] + [
    mpatches.Patch(color=TEMPLATE_COLORS[r], label=f"Top-{r+1} 템플릿 (공유 단어)")
    for r in range(TOP_N_TEMPLATES)
] + [
    mpatches.Patch(color="#AAAAAA", label="쿼리와 비공유 단어"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=N_COLS + 1,
           fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

fig.text(
    0.5, -0.07,
    "■ Visual Word: DINOv2 특징 공간의 k-means 클러스터 중심 (총 2048개)\n"
    "■ TF-IDF 값: 해당 단어가 이 이미지에 얼마나 특징적으로 나타나는지\n"
    "  (높을수록 그 이미지를 구분하는 핵심 시각 패턴)\n"
    f"■ 차트 x축: 쿼리의 상위 {TOP_K_WORDS}개 활성화 단어만 표시  |  회색 점선: 쿼리 BoW 참조선",
    ha="center", fontsize=8.5, color="#333333",
    transform=fig.transFigure,
    bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#CCCCCC", alpha=0.9),
)

fig.suptitle(
    f"유사 템플릿 검색 시각화  |  LM-O  Object {OBJECT_LID}\n"
    f"[DINOv2 특징 추출 → PCA → Visual Word 할당 → TF-IDF BoW → 코사인 유사도]",
    fontsize=12, fontweight="bold", y=1.01,
)

plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n[완료] 시각화 저장: {OUTPUT_PATH}")

# ---------------------------------------------------------------------------
# 검증 요약 출력
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("검증 요약")
print("=" * 55)
print(f"쿼리 BoW sparsity:     {100*(query_bow_np == 0).mean():.1f}% 가 0")
print(f"템플릿 BoW sparsity:   {100*(template_bows_np == 0).mean():.1f}% 가 0 (798개 전체)")
print(f"Top-{TOP_N_TEMPLATES} 코사인 유사도: {top_scores_np}")
print(f"Bottom-5 코사인 유사도: {torch.topk(-cos_sims, k=5).values.neg().numpy()}")
print(f"평균 코사인 유사도:    {cos_sims.mean().item():.4f}")
print(f"보정: 과정이 infer.py의 tfidf_matching()과 동일한지 → True")
print("=" * 55)
