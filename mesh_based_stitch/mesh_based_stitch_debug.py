import os
import cv2 as cv
import numpy as np
import time


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize01(x: np.ndarray, eps=1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + eps)


def to_color_map(gray_float_01: np.ndarray, colormap=cv.COLORMAP_TURBO) -> np.ndarray:
    x = np.clip(gray_float_01, 0.0, 1.0)
    x8 = (x * 255.0).astype(np.uint8)
    return cv.applyColorMap(x8, colormap)


def draw_grid(img_bgr: np.ndarray, grid_step: int, color=(60, 60, 60), thickness=1) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for y in range(0, h, grid_step):
        cv.line(out, (0, y), (w - 1, y), color, thickness, cv.LINE_AA)
    for x in range(0, w, grid_step):
        cv.line(out, (x, 0), (x, h - 1), color, thickness, cv.LINE_AA)
    return out


def build_canvas_from_corners_and_sizes(corners, sizes):
    """
    corners: [(x0,y0), ...] in panorama coords
    sizes:   [(w,h), ...]
    returns:
      tx, ty, canvas_size (W,H), shifted_corners (x0+tx, y0+ty)
    """
    xs = []
    ys = []
    for (x, y), (w, h) in zip(corners, sizes):
        xs += [x, x + w]
        ys += [y, y + h]

    xmin = int(np.floor(min(xs)))
    ymin = int(np.floor(min(ys)))
    xmax = int(np.ceil(max(xs)))
    ymax = int(np.ceil(max(ys)))

    tx = -xmin
    ty = -ymin
    W = xmax - xmin
    H = ymax - ymin

    shifted = [(x + tx, y + ty) for (x, y) in corners]
    return tx, ty, (W, H), shifted


def paste_to_canvas(img, corner_xy, canvas_shape, dtype=None):
    """
    img: (h,w,3) or (h,w)
    corner_xy: (x0,y0) top-left on canvas
    canvas_shape: (H,W)
    """
    x0, y0 = corner_xy
    h, w = img.shape[:2]
    if dtype is None:
        dtype = img.dtype

    if img.ndim == 3:
        canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=dtype)
        canvas[y0:y0 + h, x0:x0 + w] = img
    else:
        canvas = np.zeros((canvas_shape[0], canvas_shape[1]), dtype=dtype)
        canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


# ============================================================
# Feature detection + matching (simple; used for mesh residual)
# ============================================================

def detect_and_match(img1_bgr, img2_bgr, ratio=0.75):
    """
    img1,img2: BGR (in same coordinate system, e.g., cylindrical warped)
    return pts1, pts2 (float32 Nx2) in their own image coordinates
    """
    if hasattr(cv, "SIFT_create"):
        det = cv.SIFT_create()
        norm = cv.NORM_L2
    else:
        det = cv.ORB_create(5000)
        norm = cv.NORM_HAMMING

    k1, d1 = det.detectAndCompute(img1_bgr, None)
    k2, d2 = det.detectAndCompute(img2_bgr, None)
    if d1 is None or d2 is None:
        raise RuntimeError("특징점 기술자 없음")

    matcher = cv.BFMatcher(norm, crossCheck=False)
    matches_knn = matcher.knnMatch(d1, d2, k=2)

    good = []
    for pair in matches_knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError("유효 매칭 수 부족")

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    return pts1, pts2, k1, k2, good


# ============================================================
# Mesh flow (same logic as your previous)
# ============================================================

def build_mesh_flow(
    matchesA_canvas,
    matchesB_canvas,
    roi_shape,
    grid_step=120,
    sigma=None,
    clip_ratio=0.5,
    smooth_sigma_ratio=0.5,
    border_damping=True,
    border_width_ratio=0.15,
    return_grid=False,
):
    """
    residual = A - B 를 가우시안 가중 평균으로 격자 변위 추정

    return_grid=True:
      dx,dy,grid_dx,grid_dy,grid_y,grid_x,damping_map
    """
    h, w = roi_shape
    residuals = matchesA_canvas - matchesB_canvas
    pts = matchesB_canvas

    ys = np.arange(0, h + 1, grid_step)
    xs = np.arange(0, w + 1, grid_step)
    if ys[-1] != h:
        ys = np.append(ys, h)
    if xs[-1] != w:
        xs = np.append(xs, w)

    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    Gy, Gx = grid_y.shape

    grid_dx = np.zeros((Gy, Gx), np.float32)
    grid_dy = np.zeros((Gy, Gx), np.float32)

    if len(pts) == 0:
        dx0 = np.zeros((h, w), np.float32)
        dy0 = np.zeros((h, w), np.float32)
        if return_grid:
            return dx0, dy0, grid_dx, grid_dy, grid_y, grid_x, np.ones((Gy, Gx), np.float32)
        return dx0, dy0

    if sigma is None:
        sigma = float(grid_step)

    for i in range(Gy):
        vy = grid_y[i, 0]
        for j in range(Gx):
            vx = grid_x[0, j]
            d2 = (pts[:, 0] - vx) ** 2 + (pts[:, 1] - vy) ** 2
            wgt = np.exp(-d2 / (2.0 * sigma * sigma)) + 1e-8
            wsum = np.sum(wgt)
            grid_dx[i, j] = np.sum(residuals[:, 0] * wgt) / wsum
            grid_dy[i, j] = np.sum(residuals[:, 1] * wgt) / wsum

    damping_map = np.ones((Gy, Gx), np.float32)

    if border_damping:
        border_w = int(w * border_width_ratio)
        border_h = int(h * border_width_ratio)
        border_size = max(border_w, border_h)

        for i in range(Gy):
            for j in range(Gx):
                y = float(grid_y[i, j])
                x = float(grid_x[0, j])
                dist_to_border = min(x, y, w - x, h - y)
                if dist_to_border < border_size:
                    damping = min(dist_to_border / max(border_size, 1.0), 1.0)
                    damping = damping ** 3
                    damping_map[i, j] = damping

        grid_dx *= damping_map
        grid_dy *= damping_map

    # 격자 -> 전체 해상도 보간
    dx = cv.resize(grid_dx, (w, h), interpolation=cv.INTER_CUBIC)
    dy = cv.resize(grid_dy, (w, h), interpolation=cv.INTER_CUBIC)

    # 과도 변형 클리핑
    mag = np.hypot(dx, dy)
    th = grid_step * float(clip_ratio)
    scale = np.ones_like(mag, np.float32)
    over = mag > th
    scale[over] = th / (mag[over] + 1e-6)
    dx *= scale
    dy *= scale

    # 스무딩
    s = max(0.1, grid_step * smooth_sigma_ratio)
    dx = cv.GaussianBlur(dx, (0, 0), s)
    dy = cv.GaussianBlur(dy, (0, 0), s)

    if return_grid:
        return dx, dy, grid_dx, grid_dy, grid_y, grid_x, damping_map
    return dx, dy


# ============================================================
# OpenCV stitching-style cylindrical warper + blender
# ============================================================

def opencv_cylindrical_warp_pair(imgs_bgr, work_megapix=0.6, debug=False, debug_dir="./result/debug_cyl"):
    """
    OpenCV detail 모듈 기반:
      - features
      - matching
      - camera estimation + bundle adjustment
      - cylindrical warper
    return:
      warped_imgs: [w0, w1]
      warped_masks: [m0, m1] (uint8 0/255)
      corners: [(x0,y0),(x1,y1)] in panorama coords
      sizes:   [(w,h),(w,h)]
      cameras: camera objects
      warper_scale: float
    """
    if not hasattr(cv, "detail"):
        raise RuntimeError("현재 OpenCV 빌드에 cv.detail 모듈이 없습니다.")

    ensure_dir(debug_dir)

    # 1) work scale (OpenCV stitching 예제 스타일)
    #    work_megapix: 특징추출/매칭을 위한 다운스케일
    full_sizes = [(im.shape[1], im.shape[0]) for im in imgs_bgr]
    work_scales = []
    imgs_work = []

    for im in imgs_bgr:
        h, w = im.shape[:2]
        scale = min(1.0, np.sqrt((work_megapix * 1e6) / (w * h + 1e-6)))
        work_scales.append(scale)
        if scale < 0.999:
            imw = cv.resize(im, (int(w * scale), int(h * scale)), interpolation=cv.INTER_LINEAR)
        else:
            imw = im.copy()
        imgs_work.append(imw)

    # 2) features (detail API)
    if hasattr(cv, "SIFT_create"):
        finder = cv.SIFT_create()
    else:
        finder = cv.ORB_create(5000)

    features = []
    for i, imw in enumerate(imgs_work):
        feat = cv.detail.computeImageFeatures2(finder, imw)
        features.append(feat)
        if debug:
            kp_vis = cv.drawKeypoints(imw, feat.keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imwrite(os.path.join(debug_dir, f"cyl_01_work_keypoints_{i}.jpg"), kp_vis)

    # 3) matcher
    matcher = cv.detail_BestOf2NearestMatcher(False, 0.3, 6, 6)  # try_gpu=False
    pairwise_matches = matcher.apply2(features)
    matcher.collectGarbage()

    # 4) estimate cameras
    estimator = cv.detail_HomographyBasedEstimator()
    ok, cameras = estimator.apply(features, pairwise_matches, None)
    if not ok:
        raise RuntimeError("카메라 추정 실패(HomographyBasedEstimator)")

    # --- FIX: BundleAdjusterRay는 내부에서 CV_32F를 기대하는 경우가 있음 ---
    for i in range(len(cameras)):
        # R을 float32로
        cameras[i].R = cameras[i].R.astype(np.float32)

        # K()는 getter라 직접 set은 못하지만, focal/ppx/ppy가 float64로 섞이는 경우 방지
        cameras[i].focal = float(cameras[i].focal)
        cameras[i].ppx = float(cameras[i].ppx)
        cameras[i].ppy = float(cameras[i].ppy)

        # 왜곡계수(있으면)도 float32로
        if hasattr(cameras[i], "k") and cameras[i].k is not None:
            cameras[i].k = np.array(cameras[i].k, dtype=np.float32)


    # 5) bundle adjustment
    adjuster = cv.detail_BundleAdjusterRay()
    adjuster.setConfThresh(1.0)
    ok, cameras = adjuster.apply(features, pairwise_matches, cameras)
    if not ok:
        raise RuntimeError("번들 조정 실패(BundleAdjusterRay)")

    # 6) wave correction(선택)
    #    두 장만이라 큰 영향은 적지만, OpenCV 흐름 그대로 적용
    try:
        rmats = [cam.R for cam in cameras]
        rmats = cv.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_HORIZ)
        for i in range(len(cameras)):
            cameras[i].R = rmats[i]
    except Exception:
        pass

    # 7) warper scale 결정 (보통 focal median)
    focals = [cam.focal for cam in cameras]
    warper_scale = float(np.median(focals))

    warper = cv.PyRotationWarper("cylindrical", warper_scale)

    warped_imgs = []
    warped_masks = []
    corners = []
    sizes = []

    for i, im_full in enumerate(imgs_bgr):
        # K는 full-res 기준으로 맞춰야 하므로 work_scale을 반영해 복원
        scale = work_scales[i]
        cam = cameras[i]
        K = cam.K().astype(np.float32)

        if scale > 1e-6:
            K[0, 0] /= scale
            K[1, 1] /= scale
            K[0, 2] /= scale
            K[1, 2] /= scale

        mask = np.full((im_full.shape[0], im_full.shape[1]), 255, np.uint8)

        # warp image
        corner, wimg = warper.warp(im_full, K, cam.R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        # warp mask
        _, wmask = warper.warp(mask, K, cam.R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)

        warped_imgs.append(wimg)
        warped_masks.append(wmask)
        corners.append((int(corner[0]), int(corner[1])))
        sizes.append((int(wimg.shape[1]), int(wimg.shape[0])))

        if debug:
            cv.imwrite(os.path.join(debug_dir, f"cyl_02_warped_img_{i}.jpg"), wimg)
            cv.imwrite(os.path.join(debug_dir, f"cyl_03_warped_mask_{i}.png"), wmask)

    return warped_imgs, warped_masks, corners, sizes, cameras, warper_scale


def create_exposure_compensator():
    # GAIN_BLOCKS가 일반적으로 가장 무난
    try:
        return cv.detail.ExposureCompensator_createDefault(cv.detail.ExposureCompensator_GAIN_BLOCKS)
    except Exception:
        return None


def create_seam_finder():
    """
    OpenCV 빌드에 따라 seam finder 클래스 이름이 다를 수 있어 fallback 처리.
    """
    # DP seam finder (COLOR) -> 있으면 우선 사용
    try:
        return cv.detail_DpSeamFinder("COLOR")
    except Exception:
        pass

    # GraphCut seam finder
    try:
        return cv.detail_GraphCutSeamFinder("COST_COLOR")
    except Exception:
        pass

    return None


def multiband_blend(warped_imgs, warped_masks, corners, blend_strength=5, debug=False, debug_dir="./result/debug_blend"):
    """
    warped_imgs: uint8 BGR
    warped_masks: uint8 0/255
    corners: list of (x,y) in panorama coords
    return blended panorama (uint8 BGR)
    """
    ensure_dir(debug_dir)

    # OpenCV python binding마다 상수명이 다름
    if hasattr(cv.detail, "Blender_MULTI_BAND"):
        blender_type = cv.detail.Blender_MULTI_BAND
    elif hasattr(cv.detail, "Blender_MULTIBAND"):
        blender_type = cv.detail.Blender_MULTIBAND
    else:
        raise RuntimeError("cv.detail Blender multiband 상수를 찾을 수 없습니다.")

    try:
        blender = cv.detail.Blender_createDefault(blender_type, False)
        if hasattr(blender, "setNumBands"):
            blender.setNumBands(int(np.clip(blend_strength, 1, 10)))
    except Exception as e:
        raise RuntimeError(f"Blender 생성 실패: {e}")

    sizes = [(im.shape[1], im.shape[0]) for im in warped_imgs]
    blender.prepare(corners, sizes)

    for i, (im, m, c) in enumerate(zip(warped_imgs, warped_masks, corners)):
        # blender는 보통 int16 입력을 기대
        im16 = im.astype(np.int16)

        # mask는 uint8 0/255 보장
        m8 = (m > 0).astype(np.uint8) * 255

        blender.feed(im16, m8, c)

        if debug:
            vis = im.copy()
            vis[m8 == 0] = 0
            cv.imwrite(os.path.join(debug_dir, f"blend_input_{i}.jpg"), vis)

    result, result_mask = blender.blend(None, None)
    result = np.clip(result, 0, 255).astype(np.uint8)

    if debug:
        cv.imwrite(os.path.join(debug_dir, "blend_result.jpg"), result)
        cv.imwrite(os.path.join(debug_dir, "blend_result_mask.png"), result_mask)

    return result


# ============================================================
# Cylindrical + Mesh residual + Seam + MultiBand blend
# ============================================================

def stitch_cylindrical_mesh_blend(
    path1: str,
    path2: str,
    grid_step: int = 120,
    sigma: float = None,
    clip_ratio: float = 0.5,
    smooth_sigma_ratio: float = 0.5,
    border_damping: bool = True,
    border_width_ratio: float = 0.15,
    work_megapix: float = 0.6,
    blend_strength: int = 5,
    debug: bool = True,
    out_dir: str = "./result",
):
    """
    1) OpenCV detail 기반 cylindrical warper로 두 이미지를 파노라마 좌표계로 워핑
    2) exposure compensation + seam finder로 mask 보정
    3) (워핑된 좌표계에서) img2를 img1에 더 잘 맞추도록 mesh residual warp
    4) MultiBand blender로 최종 합성
    """
    t0 = time.time()
    ensure_dir(out_dir)
    dbg_cyl = os.path.join(out_dir, "debug_cyl")
    dbg_mesh = os.path.join(out_dir, "debug_mesh")
    dbg_blend = os.path.join(out_dir, "debug_blend")
    ensure_dir(dbg_cyl)
    ensure_dir(dbg_mesh)
    ensure_dir(dbg_blend)

    img1 = cv.imread(path1)
    img2 = cv.imread(path2)
    if img1 is None or img2 is None:
        raise RuntimeError("이미지 읽기 실패")

    if debug:
        cv.imwrite(os.path.join(out_dir, "input_01_img1.jpg"), img1)
        cv.imwrite(os.path.join(out_dir, "input_02_img2.jpg"), img2)

    # ------------------------------------------------------------
    # (A) cylindrical warper (OpenCV stitching 계열)
    # ------------------------------------------------------------
    warped_imgs, warped_masks, corners, sizes, cameras, warper_scale = opencv_cylindrical_warp_pair(
        [img1, img2],
        work_megapix=work_megapix,
        debug=debug,
        debug_dir=dbg_cyl,
    )

    w1, w2 = warped_imgs
    m1, m2 = warped_masks
    c1, c2 = corners

    # ------------------------------------------------------------
    # (B) exposure compensation (선택)
    # ------------------------------------------------------------
    compensator = create_exposure_compensator()
    if compensator is not None:
        # feed는 (corners, images, masks)
        compensator.feed(corners, warped_imgs, warped_masks)

        warped_imgs_comp = []
        for i, (im, m, c) in enumerate(zip(warped_imgs, warped_masks, corners)):
            imc = im.copy()
            
            # OpenCV Python binding (4.12 conda)에서는 apply(index, corner, image, mask) 필요
            try:
                compensator.apply(i, c, imc, m)
            except cv.error:
                # 일부 빌드에서 apply(corner, image, mask)로 노출되는 경우도 있어 fallback
                compensator.apply(c, imc, m)

            warped_imgs_comp.append(imc)

        warped_imgs = warped_imgs_comp
        w1, w2 = warped_imgs

        if debug:
            cv.imwrite(os.path.join(dbg_cyl, "cyl_04_exposure_comp_0.jpg"), w1)
            cv.imwrite(os.path.join(dbg_cyl, "cyl_04_exposure_comp_1.jpg"), w2)

    # ------------------------------------------------------------
    # (C) seam finder (선택) - 마스크를 seam 기준으로 업데이트
    # ------------------------------------------------------------
    seam_finder = create_seam_finder()
    if seam_finder is not None:
        # seam finder는 float32 이미지를 요구하는 경우가 많음
        imgs_f = [im.astype(np.float32) for im in warped_imgs]
        masks = [m1.copy(), m2.copy()]
        seam_finder.find(imgs_f, corners, masks)
        m1, m2 = masks[0], masks[1]
        if debug:
            cv.imwrite(os.path.join(dbg_cyl, "cyl_05_seam_mask_0.png"), m1)
            cv.imwrite(os.path.join(dbg_cyl, "cyl_05_seam_mask_1.png"), m2)

    # ------------------------------------------------------------
    # (D) mesh residual warp (cylindrical warped 좌표계에서)
    #     - 두 warped image는 각각 corner가 다르므로 canvas로 올려서 dx/dy를 동일 좌표계에서 계산
    # ------------------------------------------------------------
    tx, ty, (CW, CH), shifted_corners = build_canvas_from_corners_and_sizes([c1, c2], sizes)
    sc1, sc2 = shifted_corners

    # canvas에 w1,w2 및 마스크 올리기
    canvas1 = paste_to_canvas(w1, sc1, (CH, CW))
    canvas2 = paste_to_canvas(w2, sc2, (CH, CW))
    mask1_canvas = paste_to_canvas(m1, sc1, (CH, CW), dtype=np.uint8)
    mask2_canvas = paste_to_canvas(m2, sc2, (CH, CW), dtype=np.uint8)

    if debug:
        cv.imwrite(os.path.join(dbg_mesh, "mesh_01_canvas1.jpg"), canvas1)
        cv.imwrite(os.path.join(dbg_mesh, "mesh_02_canvas2.jpg"), canvas2)
        cv.imwrite(os.path.join(dbg_mesh, "mesh_03_mask1_canvas.png"), mask1_canvas)
        cv.imwrite(os.path.join(dbg_mesh, "mesh_04_mask2_canvas.png"), mask2_canvas)

    # 매칭은 warped 이미지 좌표계에서 하고 -> canvas 좌표로 변환
    # (매칭 안정성을 위해: 검은 영역을 피하려고 mask를 적용한 이미지를 쓰는 게 좋지만,
    #  여기서는 간단히 w1,w2에서 수행)
    pts1, pts2, k1, k2, good = detect_and_match(w1, w2, ratio=0.75)

    # pts1는 w1 내부좌표 -> panorama 좌표: +c1 -> canvas 좌표: +c1+tx,ty
    # pts2도 동일
    pts1_canvas = pts1 + np.array([c1[0] + tx, c1[1] + ty], dtype=np.float32)
    pts2_canvas = pts2 + np.array([c2[0] + tx, c2[1] + ty], dtype=np.float32)

    if sigma is None:
        sigma = float(grid_step)

    dx, dy, grid_dx, grid_dy, grid_y, grid_x, damping_map = build_mesh_flow(
        matchesA_canvas=pts1_canvas,
        matchesB_canvas=pts2_canvas,
        roi_shape=(CH, CW),
        grid_step=grid_step,
        sigma=sigma,
        clip_ratio=clip_ratio,
        smooth_sigma_ratio=smooth_sigma_ratio,
        border_damping=border_damping,
        border_width_ratio=border_width_ratio,
        return_grid=True,
    )

    if debug:
        mag = np.hypot(dx, dy)
        cv.imwrite(os.path.join(dbg_mesh, "mesh_05_disp_mag.png"), to_color_map(normalize01(mag)))

        # 매칭 잔차 벡터 시각화
        vis = draw_grid(canvas1, grid_step)
        for (x, y) in pts1_canvas.astype(np.int32):
            cv.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1, cv.LINE_AA)  # A red
        for (x, y) in pts2_canvas.astype(np.int32):
            cv.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1, cv.LINE_AA)  # B blue
        for a, b in zip(pts2_canvas, pts1_canvas):
            cv.arrowedLine(
                vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                (0, 255, 255), 1, cv.LINE_AA, tipLength=0.2
            )
        cv.imwrite(os.path.join(dbg_mesh, "mesh_06_matches_residual_on_canvas.jpg"), vis)

    # canvas2(=img2)만 remap으로 보정
    X, Y = np.meshgrid(np.arange(CW, dtype=np.float32), np.arange(CH, dtype=np.float32))
    map_x = X - dx
    map_y = Y - dy

    canvas2_refined = cv.remap(
        canvas2,
        map_x,
        map_y,
        cv.INTER_CUBIC,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask2_refined = cv.remap(
        mask2_canvas,
        map_x,
        map_y,
        cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )

    if debug:
        cv.imwrite(os.path.join(dbg_mesh, "mesh_07_canvas2_refined.jpg"), canvas2_refined)
        cv.imwrite(os.path.join(dbg_mesh, "mesh_08_mask2_refined.png"), mask2_refined)

    # ------------------------------------------------------------
    # (E) MultiBand blender로 최종 합성
    #     blender는 "corner+image" 입력을 받는데, 우리는 이미 canvas로 통일했으므로
    #     corner는 (0,0)으로 feed하면 됨.
    # ------------------------------------------------------------
    blended = multiband_blend(
        warped_imgs=[canvas1, canvas2_refined],
        warped_masks=[mask1_canvas, mask2_refined],
        corners=[(0, 0), (0, 0)],
        blend_strength=blend_strength,
        debug=debug,
        debug_dir=dbg_blend,
    )

    elapsed = (time.time() - t0) * 1000.0
    if debug:
        cv.imwrite(os.path.join(out_dir, "final_blended.jpg"), blended)

    print("=" * 60)
    print("Cylindrical + Mesh + MultiBand Blend 완료")
    print(f"  canvas size: {CW} x {CH}")
    print(f"  grid_step={grid_step}, sigma={sigma}, clip_ratio={clip_ratio}")
    print(f"  blend_strength={blend_strength}, work_megapix={work_megapix}")
    print(f"  elapsed: {elapsed:.1f} ms")
    print("=" * 60)

    return blended


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    out = stitch_cylindrical_mesh_blend(
        path1="./images/left.jpg",
        path2="./images/right.jpg",
        grid_step=120,            # cylindrical 이후엔 과도한 메쉬를 피하려고 100~160 권장
        sigma=None,               # None이면 grid_step으로 자동
        clip_ratio=0.4,           # cylindrical 후에는 0.3~0.6 권장(왜곡 방지)
        smooth_sigma_ratio=0.5,
        border_damping=True,
        border_width_ratio=0.15,
        work_megapix=0.6,         # 특징/매칭용 다운스케일(너무 작으면 매칭 불안정)
        blend_strength=5,         # multiband bands 정도(1~10)
        debug=True,
        out_dir="./result",
    )
    cv.imwrite("./result/stitch_cyl_mesh_multiband.jpg", out)
    print("저장: ./result/stitch_cyl_mesh_multiband.jpg")
