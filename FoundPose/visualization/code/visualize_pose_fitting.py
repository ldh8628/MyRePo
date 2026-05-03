#!/usr/bin/env python3
"""
포즈 피팅 시각화 스크립트
==========================

[논문 설명]
  1. 각 검색된 템플릿에 대해 2D-3D 대응점으로부터 포즈(R, t)를 추정
  2. EPnP 알고리즘 사용
  3. RANSAC: 최소 4개 대응점 무작위 샘플 → 포즈 계산 → inlier 수 계산 → 반복
  4. 최종 포즈 품질 = inlier(내재적 대응) 수
  5. h개 템플릿 중 inlier 수 최대 포즈를 최종 대략적 포즈로 선택

[두 질문에 대한 답]
  Q1. "4개 대응 세트에서 무엇을 반복적으로 한다는 건지?"
      → RANSAC 1 iteration: 전체 대응점 중 4개 무작위 샘플 → cv2.solvePnP로 포즈 계산
        → 이 포즈로 모든 대응점 재투영 → 재투영 오차 < 10px인 점 수(inlier) 계산
        → 1000번 반복 후 inlier 수 최대 포즈 채택
  Q2. "내재적 대응의 수란?"
      → 최종 채택 포즈로 3D→2D 재투영 시 재투영 오차 < 10px인 대응점의 수
        (pnp_util.py:79: quality = float(len(inliers)))

[실제 구현 — pnp_util.py]
  cv2.solvePnPRansac(..., flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=1000, reprojectionError=10.0)
  → RANSAC 완료 후 inlier들로 cv2.solvePnPRefineLM 추가 정밀화

[시각화 출력 — visualization/image/pose_fitting_viz.png]
  Row 0: RANSAC 과정 시뮬레이션
          [전체 대응점] [랜덤 4-pt 샘플 iter1] [iter2] [iter3] [최종 RANSAC 결과]
  Row 1: Inlier/Outlier 상세 분석 (Top-1 템플릿)
          [Inlier=초록 / Outlier=빨강] [재투영 오차 히스토그램] [재투영된 3D 모델 오버레이]
  Row 2: h개 템플릿 각각의 포즈 추정 결과 + 최선 포즈 하이라이트

실행:
  python run_pipeline.py visualization/code/visualize_pose_fitting.py
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
from utils.misc import calc_crop_box, construct_crop_camera, warp_image, get_intrinsic_matrix, tensor_to_array
from utils import feature_util, knn_util, repre_util, projector_util, pnp_util
from utils.template_util import calc_tfidf, find_nearest_object_features
from utils.corresp_util import cyclic_buddies_matching
from bop_toolkit_lib.pycoco_utils import rle_to_binary_mask

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
BOP_PATH        = os.environ.get("BOP_PATH", "D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets")
DATASET         = "lmo"
SCENE_ID        = 2
IMAGE_ID        = 1001
OBJECT_LID      = 1
DETECTION_IDX   = 0

CROP_SIZE       = (420, 420)
CROP_REL_PAD    = 0.2
TOP_N_TEMPLATES = 5

# PnP RANSAC 파라미터 (infer.py 기본값과 동일)
PNP_RANSAC_ITER     = 1000
PNP_INLIER_THRESH   = 10.0    # px
PNP_CONF            = 0.99
PNP_REFINE_LM       = True
PNP_MIN_CORRESP     = 6

# RANSAC 시뮬레이션용 파라미터
RANSAC_VIZ_ITERS    = 3       # Row 0에서 보여줄 개별 iteration 수
RANSAC_MIN_SAMPLE   = 4       # PnP 최소 대응점 수

REPRE_DIR   = os.path.join(REPO_ROOT, "outputs", "object_repre", DATASET, "v1", str(OBJECT_LID))
CNOS_JSON   = os.path.join(BOP_PATH, "bop23_default_detections_for_task4", "cnos-fastsam",
                            "cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json")
SCENE_DIR   = os.path.join(BOP_PATH, DATASET, "test", f"{SCENE_ID:06d}")
OUTPUT_PATH = os.path.join(REPO_ROOT, "visualization", "image", "pose_fitting_viz.png")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPLATE_COLORS = ["#D62728", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B"]

# ===========================================================================
# ① ~ ⑧: 이전 스크립트들과 동일한 전처리 파이프라인
# ===========================================================================
print("[1] repre.pth 로드")
repre = repre_util.load_object_repre(repre_dir=REPRE_DIR, tensor_device="cpu")
repre.feat_cluster_centroids = repre.feat_cluster_centroids.cpu()
repre.feat_cluster_idfs      = repre.feat_cluster_idfs.cpu()
repre.template_descs         = repre.template_descs.cpu()
repre.feat_vectors           = repre.feat_vectors.cpu()
try:
    repre.feat_to_template_ids = repre.feat_to_template_ids.cpu()
    repre.vertices             = repre.vertices.cpu()
except Exception:
    pass

print("[2] 이미지 + 카메라")
image_rgb = cv2.cvtColor(cv2.imread(os.path.join(SCENE_DIR, "rgb", f"{IMAGE_ID:06d}.png")),
                          cv2.COLOR_BGR2RGB)
H, W = image_rgb.shape[:2]
with open(os.path.join(SCENE_DIR, "scene_camera.json")) as f:
    sc = json.load(f)
K_orig = np.array(sc[str(IMAGE_ID)]["cam_K"], dtype=np.float32).reshape(3, 3)
orig_camera = PinholePlaneCameraModel(width=W, height=H,
    f=(K_orig[0,0], K_orig[1,1]), c=(K_orig[0,2], K_orig[1,2]),
    T_world_from_eye=np.eye(4, dtype=np.float32))

print("[3] Perspective Crop")
with open(CNOS_JSON) as f:
    all_preds = json.load(f)
pred = sorted([p for p in all_preds if p["scene_id"]==SCENE_ID and
               p["image_id"]==IMAGE_ID and p["category_id"]==OBJECT_LID],
              key=lambda x: x["score"], reverse=True)[DETECTION_IDX]
orig_mask = cv2.morphologyEx(rle_to_binary_mask(pred["segmentation"]).astype(np.uint8),
                              cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
bx, by, bw, bh = pred["bbox"]
crop_box    = calc_crop_box(AlignedBox2f(bx,by,bx+bw,by+bh), make_square=True)
crop_camera = construct_crop_camera(crop_box, orig_camera, CROP_SIZE, CROP_REL_PAD)
interp      = cv2.INTER_AREA if crop_box.width >= crop_camera.width else cv2.INTER_LINEAR
crop_float  = warp_image(orig_camera, crop_camera, image_rgb.astype(np.float32)/255.0, interp)
crop_mask_u8 = (warp_image(orig_camera, crop_camera, orig_mask.astype(np.float32),
                            cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
crop_img_u8 = np.clip(crop_float * 255, 0, 255).astype(np.uint8)
K_crop = get_intrinsic_matrix(crop_camera).astype(np.float32)

print("[4] DINOv2 특징 추출")
extractor = feature_util.make_feature_extractor(repre.feat_opts.extractor_name)
extractor.eval()
if DEVICE == "cuda":
    extractor = extractor.cuda()
with torch.no_grad():
    out = extractor(torch.from_numpy(crop_float).permute(2,0,1).float().unsqueeze(0).to(DEVICE))
crop_feat_chw = out["feature_maps"][0].cpu()
grid_pts  = feature_util.generate_grid_points(CROP_SIZE, cell_size=1.0)
q_pts     = feature_util.filter_points_by_mask(grid_pts, torch.from_numpy(crop_mask_u8))
q_feats   = feature_util.sample_feature_map_at_points(crop_feat_chw, q_pts, CROP_SIZE).contiguous()
if q_feats.shape[1] != repre.feat_vectors.shape[1] and len(repre.feat_raw_projectors) > 0:
    q_feats_proj = projector_util.project_features(q_feats, repre.feat_raw_projectors).contiguous()
else:
    q_feats_proj = q_feats

print("[5] BoW → Top-h 템플릿")
vw_idx = knn_util.KNN(k=repre.template_desc_opts.tfidf_knn_k, metric=repre.template_desc_opts.tfidf_knn_metric)
vw_idx.fit(repre.feat_cluster_centroids)
wids, wdists = find_nearest_object_features(q_feats_proj, vw_idx)
wids, wdists = wids.cpu(), wdists.cpu()
q_bow    = calc_tfidf(wids, wdists, repre.feat_cluster_idfs,
                       repre.template_desc_opts.tfidf_soft_assign,
                       repre.template_desc_opts.tfidf_soft_sigma_squared)
cos_sims = F.cosine_similarity(repre.template_descs, q_bow.unsqueeze(0).expand_as(repre.template_descs))
top_scores, top_ids = torch.topk(cos_sims, k=TOP_N_TEMPLATES, sorted=True)
top_ids_np    = top_ids.numpy()
top_scores_np = top_scores.numpy()

print("[6] 패치 매칭 (Cyclic Buddies) - top-h 템플릿별 대응점 생성")
q_knn = knn_util.KNN(k=1, metric="l2")
q_knn.fit(q_feats_proj.cpu())

corresps = []
for tid in top_ids_np:
    mask_t = (repre.feat_to_template_ids == int(tid)).cpu()
    feat_ids = torch.nonzero(mask_t).flatten()
    tpl_feats = repre.feat_vectors[feat_ids].cpu()

    tpl_knn = knn_util.KNN(k=1, metric="l2")
    tpl_knn.fit(tpl_feats)

    q_ids, obj_ids, dists, scores = cyclic_buddies_matching(
        query_points=q_pts.cpu(), query_features=q_feats_proj.cpu(),
        query_knn_index=q_knn, object_features=tpl_feats,
        object_knn_index=tpl_knn, top_k=300, debug=False,
    )
    corresps.append({
        "template_id":    tid,
        "template_score": top_scores_np[list(top_ids_np).index(tid)],
        "coord_2d":       q_pts[q_ids].cpu(),
        "coord_3d":       repre.vertices[feat_ids[obj_ids]].cpu(),
        "coord_conf":     scores.cpu(),
    })
    print(f"  template #{tid}: 대응점 수={len(q_ids)}")

# ===========================================================================
# ⑦ PnP RANSAC 포즈 추정 — 각 템플릿별
# ===========================================================================
print("\n[7] PnP RANSAC 포즈 추정")
poses = []
for c in corresps:
    if len(c["coord_2d"]) < PNP_MIN_CORRESP:
        print(f"  template #{c['template_id']}: 대응점 부족({len(c['coord_2d'])}), 스킵")
        poses.append(None)
        continue
    success, R, t, inliers, quality = pnp_util.estimate_pose(
        corresp=c, camera_c2w=crop_camera,
        pnp_type="opencv", pnp_ransac_iter=PNP_RANSAC_ITER,
        pnp_inlier_thresh=PNP_INLIER_THRESH,
        pnp_required_ransac_conf=PNP_CONF, pnp_refine_lm=PNP_REFINE_LM,
    )
    if success and inliers is not None:
        poses.append({"R": R, "t": t, "inliers": inliers.flatten(), "quality": quality,
                      "template_id": c["template_id"], "template_score": c["template_score"]})
        print(f"  template #{c['template_id']}: inliers={int(quality)}/{len(c['coord_2d'])} "
              f"(품질={quality:.0f})")
    else:
        poses.append(None)
        print(f"  template #{c['template_id']}: PnP 실패")

valid_poses = [(i, p) for i, p in enumerate(poses) if p is not None]
assert len(valid_poses) > 0, "모든 템플릿에서 포즈 추정 실패"
best_idx, best_pose = max(valid_poses, key=lambda x: x[1]["quality"])
print(f"\n  → 최선 포즈: template #{best_pose['template_id']}, "
      f"inliers={int(best_pose['quality'])}")

# ===========================================================================
# 헬퍼 함수
# ===========================================================================
def project_pts(pts_3d_np, R, t, K):
    """(N,3) 3D 모델 점 → (N,2) 2D 픽셀 좌표."""
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(pts_3d_np.astype(np.float32), rvec, t, K, None)
    return proj.squeeze(1)

def reproj_errors(pts2d, pts3d, R, t, K):
    """각 대응점의 재투영 오차(픽셀) 계산."""
    proj = project_pts(pts3d, R, t, K)
    return np.linalg.norm(pts2d - proj, axis=1)

def draw_pts_on_img(img, pts, colors, radii=None, alpha=1.0):
    """img(HWC uint8) 위에 점들을 색상으로 그림. 반환: 복사본."""
    out = img.copy().astype(np.float32)
    for i, (pt, col) in enumerate(zip(pts, colors)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
            continue
        r = int(radii[i]) if radii is not None else 3
        c_bgr = (int(col[0]*255), int(col[1]*255), int(col[2]*255))
        cv2.circle(out, (x, y), r, c_bgr, -1)
    return np.clip(out, 0, 255).astype(np.uint8)

def blend_model_pts(img, pts3d_all, R, t, K, dot_color=(0.0, 0.9, 0.5), step=8):
    """서브샘플링된 3D 모델 점들을 재투영하여 이미지에 오버레이."""
    sub_pts = pts3d_all[::step]
    proj    = project_pts(sub_pts, R, t, K)
    H, W    = img.shape[:2]
    out     = img.copy()
    for pt in proj:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(out, (x, y), 2, (int(dot_color[0]*255), int(dot_color[1]*255),
                                         int(dot_color[2]*255)), -1)
    return out

# 모든 대응점 numpy 변환 (Top-1 기준)
c0       = corresps[0]
pts2d_all = tensor_to_array(c0["coord_2d"]).astype(np.float32)   # (N, 2)
pts3d_all = tensor_to_array(c0["coord_3d"]).astype(np.float32)   # (N, 3)
conf_all  = tensor_to_array(c0["coord_conf"]).astype(np.float32) # (N,)
N_all     = len(pts2d_all)

model_verts = tensor_to_array(repre.vertices).astype(np.float32)  # (M, 3)

# ===========================================================================
# ⑧ RANSAC 시뮬레이션 — Row 0 전용
# ===========================================================================
def simulate_ransac_iter(pts2d, pts3d, K, seed):
    """RANSAC 1 iteration: 4개 무작위 샘플 → PnP → inlier 계산."""
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(pts2d), size=RANSAC_MIN_SAMPLE, replace=False)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            pts3d[idx].astype(np.float32), pts2d[idx].astype(np.float32),
            K, None, flags=cv2.SOLVEPNP_P3P,
        )
    except Exception:
        return None, None, None, idx
    if not ok:
        return None, None, None, idx
    R_, _  = cv2.Rodrigues(rvec)
    errors = reproj_errors(pts2d, pts3d, R_, tvec, K)
    inlier_mask = errors < PNP_INLIER_THRESH
    return R_, tvec, inlier_mask, idx

rng_seeds   = [42, 7, 123]
iter_results = []
for seed in rng_seeds:
    R_s, t_s, mask_s, samp_s = simulate_ransac_iter(pts2d_all, pts3d_all, K_crop, seed)
    iter_results.append((R_s, t_s, mask_s, samp_s))

# 최종 RANSAC inlier/outlier (best pose 기준)
if poses[0] is not None:
    final_inlier_ids = poses[0]["inliers"]
    final_inlier_mask = np.zeros(N_all, dtype=bool)
    final_inlier_mask[final_inlier_ids] = True
    R_best0, t_best0 = poses[0]["R"], poses[0]["t"]
else:
    final_inlier_mask = np.zeros(N_all, dtype=bool)
    R_best0 = t_best0 = None

# ===========================================================================
# ⑨ Figure 생성
# ===========================================================================
print("\n[8] 시각화 생성")

N_COLS_R0 = 1 + RANSAC_VIZ_ITERS + 1      # 전체 대응 + 3 iter + 최종
N_COLS_R1 = 3
N_COLS_R2 = TOP_N_TEMPLATES
MAX_COLS  = max(N_COLS_R0, N_COLS_R1, N_COLS_R2)

fig = plt.figure(figsize=(3.8 * MAX_COLS, 15))
gs  = GridSpec(3, MAX_COLS, figure=fig, hspace=0.45, wspace=0.10)

# ─── Row 0: RANSAC 과정 시뮬레이션 ──────────────────────────────────────

# Col 0: 전체 대응점
ax = fig.add_subplot(gs[0, 0])
colors_all = plt.cm.plasma(conf_all / (conf_all.max() + 1e-8))[:, :3]
img_all = draw_pts_on_img(crop_img_u8, pts2d_all, colors_all, radii=[3]*N_all)
ax.imshow(img_all)
ax.set_title(f"전체 대응점\n({N_all}개, Top-1 템플릿 #{top_ids_np[0]})\n색상=Cyclic Buddy 신뢰도",
             fontsize=8, fontweight="bold")
ax.axis("off")

# Col 1~3: 개별 RANSAC iteration 시뮬레이션
for col, (R_s, t_s, mask_s, samp_s) in enumerate(iter_results):
    ax = fig.add_subplot(gs[0, col + 1])
    img_iter = crop_img_u8.copy()
    # 모든 점을 회색으로
    for pt in pts2d_all:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < 420 and 0 <= y < 420:
            cv2.circle(img_iter, (x,y), 2, (150,150,150), -1)
    # 샘플 4개를 파란색 큰 점으로 표시
    for i in samp_s:
        x, y = int(round(pts2d_all[i,0])), int(round(pts2d_all[i,1]))
        if 0 <= x < 420 and 0 <= y < 420:
            cv2.circle(img_iter, (x,y), 7, (30,144,255), -1)
            cv2.circle(img_iter, (x,y), 7, (255,255,255), 1)
    # 포즈가 추정된 경우 inlier/outlier 표시
    n_inlier = mask_s.sum() if mask_s is not None else 0
    if R_s is not None and mask_s is not None:
        for i, (pt, is_in) in enumerate(zip(pts2d_all, mask_s)):
            if i in samp_s:
                continue
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < 420 and 0 <= y < 420:
                col_c = (0,200,0) if is_in else (220,50,50)
                cv2.circle(img_iter, (x,y), 2, col_c, -1)
    ax.imshow(img_iter)
    ax.set_title(f"RANSAC Iter #{col+1} (seed={rng_seeds[col]})\n"
                 f"파란 점=무작위 샘플 {RANSAC_MIN_SAMPLE}개\n"
                 f"inlier={n_inlier}/{N_all}",
                 fontsize=8)
    ax.axis("off")

# Col 4: 최종 RANSAC 결과
ax = fig.add_subplot(gs[0, RANSAC_VIZ_ITERS + 1])
img_final = crop_img_u8.copy()
for i, (pt, is_in) in enumerate(zip(pts2d_all, final_inlier_mask)):
    x, y = int(round(pt[0])), int(round(pt[1]))
    if 0 <= x < 420 and 0 <= y < 420:
        col_c = (0, 200, 0) if is_in else (220, 50, 50)
        cv2.circle(img_final, (x,y), 3, col_c, -1)
n_final_in = final_inlier_mask.sum()
ax.imshow(img_final)
ax.set_title(f"최종 RANSAC 결과\n"
             f"초록=inlier ({n_final_in}개)\n"
             f"빨강=outlier ({N_all - n_final_in}개)",
             fontsize=8, fontweight="bold")
ax.axis("off")

# 나머지 Row0 axes 숨기기
for c in range(N_COLS_R0, MAX_COLS):
    fig.add_subplot(gs[0, c]).axis("off")

# ─── Row 1: Inlier / Outlier 상세 분석 (Top-1 템플릿) ───────────────────

# Col 0: Inlier/Outlier 시각화
ax = fig.add_subplot(gs[1, 0])
img_io = crop_img_u8.copy()
for pt, is_in in zip(pts2d_all, final_inlier_mask):
    x, y = int(round(pt[0])), int(round(pt[1]))
    if 0 <= x < 420 and 0 <= y < 420:
        col_c = (0, 220, 0) if is_in else (220, 50, 50)
        cv2.circle(img_io, (x,y), 4, col_c, -1)
ax.imshow(img_io)
ax.set_title(f"내재적 대응 (Inliers) 시각화\n"
             f"초록=inlier (재투영 오차 < {PNP_INLIER_THRESH:.0f}px): {n_final_in}개\n"
             f"빨강=outlier: {N_all - n_final_in}개",
             fontsize=8, fontweight="bold")
legend_in  = mpatches.Patch(color="#00DC00", label=f"Inlier ({n_final_in}개)")
legend_out = mpatches.Patch(color="#DC3232", label=f"Outlier ({N_all-n_final_in}개)")
ax.legend(handles=[legend_in, legend_out], fontsize=7, loc="lower right")
ax.axis("off")

# Col 1: 재투영 오차 히스토그램
ax = fig.add_subplot(gs[1, 1])
if R_best0 is not None:
    errors = reproj_errors(pts2d_all, pts3d_all, R_best0, t_best0, K_crop)
    ax.hist(errors, bins=40, color="#4C8CB5", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(PNP_INLIER_THRESH, color="#D62728", linewidth=2.0, linestyle="--",
               label=f"inlier 임계값 {PNP_INLIER_THRESH:.0f}px")
    ax.set_xlabel("재투영 오차 (px)", fontsize=9)
    ax.set_ylabel("대응점 수", fontsize=9)
    ax.set_title(f"재투영 오차 히스토그램\n"
                 f"중앙값={np.median(errors):.1f}px | 평균={errors.mean():.1f}px\n"
                 f"inlier 기준: < {PNP_INLIER_THRESH:.0f}px",
                 fontsize=8)
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.tick_params(labelsize=8)
else:
    ax.text(0.5, 0.5, "포즈 추정 실패", ha="center", va="center", fontsize=10)
    ax.axis("off")

# Col 2: 재투영된 3D 모델 오버레이
ax = fig.add_subplot(gs[1, 2])
if R_best0 is not None:
    img_model = blend_model_pts(crop_img_u8, model_verts, R_best0, t_best0, K_crop,
                                 dot_color=(0.1, 0.85, 0.5), step=5)
    # inlier 점 다시 표시
    for pt, is_in in zip(pts2d_all, final_inlier_mask):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < 420 and 0 <= y < 420 and is_in:
            cv2.circle(img_model, (x,y), 4, (255,220,0), -1)
    ax.imshow(img_model)
    ax.set_title(f"3D 모델 재투영 (Top-1 추정 포즈)\n"
                 f"노란 점=inlier 대응 | 초록 점=3D 모델\n"
                 f"template #{top_ids_np[0]}, quality={int(poses[0]['quality']) if poses[0] else 'N/A'}",
                 fontsize=8, fontweight="bold")
else:
    ax.text(0.5, 0.5, "포즈 추정 실패", ha="center", va="center", fontsize=10)
    ax.set_title("3D 모델 재투영", fontsize=9)
ax.axis("off")

# 나머지 Row1 axes 숨기기
for c in range(N_COLS_R1, MAX_COLS):
    fig.add_subplot(gs[1, c]).axis("off")

# ─── Row 2: h개 템플릿 포즈 결과 + 최선 포즈 하이라이트 ─────────────────

for rank, (pose, c, tid, tsc) in enumerate(zip(poses, corresps, top_ids_np, top_scores_np)):
    ax = fig.add_subplot(gs[2, rank])
    tc = TEMPLATE_COLORS[rank]
    is_best = (rank == best_idx)

    if pose is not None:
        img_p = blend_model_pts(crop_img_u8, model_verts, pose["R"], pose["t"], K_crop,
                                 dot_color=(0.1, 0.85, 0.5), step=5)
        # inlier 점 표시
        pts2d_c = tensor_to_array(c["coord_2d"]).astype(np.float32)
        inlier_mask_c = np.zeros(len(pts2d_c), dtype=bool)
        inlier_mask_c[pose["inliers"]] = True
        for pt, is_in in zip(pts2d_c, inlier_mask_c):
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < 420 and 0 <= y < 420:
                cv2.circle(img_p, (x,y), 3, (255,220,0) if is_in else (180,50,50), -1)
        ax.imshow(img_p)
        title_str = (f"{'★ 최선 포즈 ★ ' if is_best else ''}"
                     f"Top-{rank+1} template #{tid}\n"
                     f"cos_sim={tsc:.3f} | inliers={int(pose['quality'])}\n"
                     f"대응점={len(pts2d_c)}")
        ax.set_title(title_str, fontsize=8, color=tc,
                     fontweight="bold" if is_best else "normal")
        # 최선 포즈에 테두리 강조
        if is_best:
            for spine in ax.spines.values():
                spine.set_edgecolor(tc)
                spine.set_linewidth(4)
    else:
        ax.imshow(crop_img_u8)
        ax.set_title(f"Top-{rank+1} template #{tid}\n포즈 추정 실패\n"
                     f"대응점={len(tensor_to_array(c['coord_2d']))}", fontsize=8, color=tc)
    ax.axis("off")

for c in range(TOP_N_TEMPLATES, MAX_COLS):
    fig.add_subplot(gs[2, c]).axis("off")

# ─── 하단 설명 ────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.04,
    f"■ RANSAC 1 iteration: 전체 대응 {N_all}개 중 {RANSAC_MIN_SAMPLE}개 무작위 샘플 → cv2.solvePnP → 재투영 오차 < {PNP_INLIER_THRESH:.0f}px 인 점 수 = inlier 수\n"
    f"■ 반복 횟수: {PNP_RANSAC_ITER}회 → inlier 최다 포즈 선택 → cv2.solvePnPRefineLM으로 추가 정밀화 (pnp_refine_lm=True)\n"
    f"■ 내재적 대응(Inlier): 추정 포즈로 3D→2D 재투영 시 오차 < {PNP_INLIER_THRESH:.0f}px인 대응점.  품질(Quality) = Inlier 수\n"
    f"■ 최선 포즈 선택: h={TOP_N_TEMPLATES}개 템플릿 중 Inlier 수(품질)가 가장 높은 포즈 → template #{best_pose['template_id']} (inliers={int(best_pose['quality'])})",
    ha="center", fontsize=8.5, color="#333333",
    transform=fig.transFigure,
    bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#CCCCCC", alpha=0.9),
)

fig.suptitle(
    f"PnP RANSAC 포즈 피팅 시각화  |  LM-O Object {OBJECT_LID}\n"
    f"[2D-3D 대응 → RANSAC(1000iter) → Inlier 기반 포즈 선택]",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.savefig(OUTPUT_PATH, dpi=110, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n[완료] {OUTPUT_PATH}")
