import cv2 as cv
import numpy as np
import time

# 26.01.14 코드

def detect_and_match(img1, img2):
    """
    img1, img2에 대한 특징을 추출하고 매칭 시킨 pts1, pts2를 반환
    (pts1: img1 좌표, pts2: img2 좌표)
    """
    print("특징 검출 매칭")
    if hasattr(cv, "SIFT_create"):
        det = cv.SIFT_create()
        norm = cv.NORM_L2
    else:
        det = cv.ORB_create(5000)
        norm = cv.NORM_HAMMING

    k1, d1 = det.detectAndCompute(img1, None)
    k2, d2 = det.detectAndCompute(img2, None)

    if d1 is None or d2 is None:
        raise RuntimeError("특징점 기술자 없음")

    matcher = cv.BFMatcher(norm, crossCheck=False)
    matches = matcher.knnMatch(d1, d2, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError("유효 매칭 수 부족")
    print(f"    매칭점: {len(good)}")

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    return pts1, pts2


def compute_canvas_and_warps(img1, img2, H21):
    """
    img2를 img1 좌표계로 H21(= img2 -> img1)로 워핑하고,
    두 이미지가 함께 들어갈 캔버스 생성.
    """
    print("전역 워핑, 캔버스 생성")
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

    # img2 corners를 img1 좌표계로 변환
    corners2_in1 = cv.perspectiveTransform(corners2, H21)
    all_pts = np.vstack((corners1, corners2_in1))

    xmin, ymin = np.floor(all_pts.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(all_pts.max(axis=0).ravel()).astype(int)

    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    size = (int(xmax - xmin), int(ymax - ymin))  # (W, H)

    warped2 = cv.warpPerspective(img2, T @ H21, size)
    canvas1 = np.zeros((size[1], size[0], 3), dtype=img1.dtype)
    canvas1[ty : ty + h1, tx : tx + w1] = img1

    return canvas1, warped2, T, tx, ty, size


def build_mesh_flow(
    matchesA_canvas,
    matchesB_canvas,
    roi_shape,
    grid_step=120,
    sigma=None,
    clip_ratio=0.75,
    smooth_sigma_ratio=0.5,
    border_damping=True,
    border_width_ratio=0.15,
):
    """
    matchesA_canvas: 기준(목표) 점들
    matchesB_canvas: 워핑된 이미지 쪽 점들
    residual = A - B 를 가우시안 가중 평균으로 격자 변위 추정
    
    개선사항:
    - border_damping: 경계 영역의 변형을 점진적으로 억제
    - border_width_ratio: 경계로 간주할 영역 비율 (0.15 = 15%)
    """
    print("메쉬 플로우 추정")
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
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)

    if sigma is None:
        sigma = float(grid_step)

    # 각 격자점에 대해 가우시안 가중 평균으로 변위 계산
    for i in range(Gy):
        vy = grid_y[i, 0]
        for j in range(Gx):
            vx = grid_x[0, j]
            d2 = (pts[:, 0] - vx) ** 2 + (pts[:, 1] - vy) ** 2
            wgt = np.exp(-d2 / (2.0 * sigma * sigma)) + 1e-8
            wsum = np.sum(wgt)
            grid_dx[i, j] = np.sum(residuals[:, 0] * wgt) / wsum
            grid_dy[i, j] = np.sum(residuals[:, 1] * wgt) / wsum

    # ★ 경계 영역 변형 억제 (새로 추가)
    if border_damping:
        print(f"  경계 변형 억제 적용 (경계 비율: {border_width_ratio})")
        border_w = int(w * border_width_ratio)
        border_h = int(h * border_width_ratio)
        
        # 경계로부터의 거리 기반 damping 맵 생성
        damping_map = np.ones((Gy, Gx), np.float32)
        
        for i in range(Gy):
            for j in range(Gx):
                y = grid_y[i, j]
                x = grid_x[0, j]
                
                # 경계로부터의 최소 거리 계산
                dist_to_border = min(x, y, w - x, h - y)
                
                # 경계 영역 내부라면 damping 적용
                border_size = max(border_w, border_h)
                if dist_to_border < border_size:
                    # 0(경계)에서 1(내부)로 부드럽게 변화
                    damping = min(dist_to_border / max(border_size, 1.0), 1.0)
                    # 3차 함수로 부드러운 감쇠 (급격한 변화 방지)
                    damping = damping ** 3
                    damping_map[i, j] = damping
        
        # damping 적용
        grid_dx *= damping_map
        grid_dy *= damping_map

    # 격자를 전체 이미지 크기로 보간
    dx = cv.resize(grid_dx, (w, h), interpolation=cv.INTER_CUBIC)
    dy = cv.resize(grid_dy, (w, h), interpolation=cv.INTER_CUBIC)

    # 변위 크기 클리핑 (과도한 변형 방지)
    mag = np.hypot(dx, dy)
    th = grid_step * clip_ratio
    scale = np.ones_like(mag, np.float32)
    over = mag > th
    scale[over] = th / (mag[over] + 1e-6)
    dx *= scale
    dy *= scale

    # 가우시안 블러로 부드럽게
    s = max(0.1, grid_step * smooth_sigma_ratio)
    dx = cv.GaussianBlur(dx, (0, 0), s)
    dy = cv.GaussianBlur(dy, (0, 0), s)

    return dx, dy


def build_mesh_flow_adaptive(
    matchesA_canvas,
    matchesB_canvas,
    roi_shape,
    grid_step=120,
    sigma=None,
    base_clip_ratio=0.75,
    smooth_sigma_ratio=0.5,
    border_damping=True,
    border_width_ratio=0.15,
):
    """
    매칭점 밀도에 따라 적응적으로 클리핑하는 버전
    매칭점이 적은 영역(바깥쪽)은 더 보수적으로 변형
    """
    print("적응적 메쉬 플로우 추정")
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
    match_density = np.zeros((Gy, Gx), np.float32)

    if len(pts) == 0:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)

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
            
            # 매칭점 밀도 저장
            match_density[i, j] = wsum

    # 경계 damping (선택적)
    if border_damping:
        print(f"  경계 변형 억제 적용 (경계 비율: {border_width_ratio})")
        border_w = int(w * border_width_ratio)
        border_h = int(h * border_width_ratio)
        
        damping_map = np.ones((Gy, Gx), np.float32)
        
        for i in range(Gy):
            for j in range(Gx):
                y = grid_y[i, j]
                x = grid_x[0, j]
                dist_to_border = min(x, y, w - x, h - y)
                border_size = max(border_w, border_h)
                
                if dist_to_border < border_size:
                    damping = min(dist_to_border / max(border_size, 1.0), 1.0)
                    damping = damping ** 3
                    damping_map[i, j] = damping
        
        grid_dx *= damping_map
        grid_dy *= damping_map

    dx = cv.resize(grid_dx, (w, h), interpolation=cv.INTER_CUBIC)
    dy = cv.resize(grid_dy, (w, h), interpolation=cv.INTER_CUBIC)
    
    # 밀도 맵도 리사이즈
    density_map = cv.resize(match_density, (w, h), interpolation=cv.INTER_CUBIC)
    density_normalized = density_map / (density_map.max() + 1e-6)

    # ★ 적응적 클리핑: 매칭점이 적은 영역은 더 보수적으로
    print(f"  적응적 클리핑 적용 (기본 비율: {base_clip_ratio})")
    mag = np.hypot(dx, dy)
    # 밀도가 낮을수록 클리핑 임계값 감소 (0.3~1.0 범위)
    adaptive_th = grid_step * base_clip_ratio * (0.3 + 0.7 * density_normalized)
    scale = np.ones_like(mag, np.float32)
    over = mag > adaptive_th
    scale[over] = adaptive_th[over] / (mag[over] + 1e-6)
    dx *= scale
    dy *= scale

    s = max(0.1, grid_step * smooth_sigma_ratio)
    dx = cv.GaussianBlur(dx, (0, 0), s)
    dy = cv.GaussianBlur(dy, (0, 0), s)

    return dx, dy


def stitch_mesh_based(
    path1, 
    path2, 
    grid_step=80, 
    sigma=100, 
    mask_scale=1.4,
    use_adaptive=False,
    border_damping=True,
    border_width_ratio=0.15,
    clip_ratio=0.75,
):
    """
    개선된 메쉬 기반 이미지 스티칭
    
    Args:
        path1, path2: 이미지 경로
        grid_step: 메쉬 격자 크기 (작을수록 촘촘, 권장: 80-120)
        sigma: 가우시안 가중치 범위 (권장: 100)
        use_adaptive: 적응적 클리핑 사용 여부
        border_damping: 경계 변형 억제 여부
        border_width_ratio: 경계 영역 비율 (0.15 = 15%)
        clip_ratio: 변위 클리핑 비율
    """
    t0 = time.time()
    print("=" * 60)
    print("개선된 메쉬 기반 스티칭 실행")
    print(f"  grid_step={grid_step}, sigma={sigma}")
    print(f"  adaptive={use_adaptive}, border_damping={border_damping}")
    print("=" * 60)
    
    print("이미지 입력 확인")
    img1 = cv.imread(path1)
    img2 = cv.imread(path2)
    if img1 is None or img2 is None:
        raise RuntimeError("이미지 읽기 실패")

    pts1, pts2 = detect_and_match(img1, img2)

    print("호모그래피 추정(RANSAC)")
    H21, _ = cv.findHomography(pts2, pts1, cv.RANSAC, 2.0)
    if H21 is None:
        raise RuntimeError("호모그래피 실패")

    canvas1, warped2, T, tx, ty, size = compute_canvas_and_warps(img1, img2, H21)

    print("매칭점 캔버스 좌표 변환")
    # img1 점은 translation만
    pts1_canvas = pts1 + np.array([tx, ty], dtype=np.float32)

    # img2 점은 H21로 img1좌표로 보낸 뒤 translation
    pts2_in1 = cv.perspectiveTransform(pts2.reshape(-1, 1, 2), H21).reshape(-1, 2)
    pts2_canvas = pts2_in1 + np.array([tx, ty], dtype=np.float32)

    # 메쉬 플로우 계산 (개선된 버전)
    if use_adaptive:
        dx, dy = build_mesh_flow_adaptive(
            matchesA_canvas=pts1_canvas,
            matchesB_canvas=pts2_canvas,
            roi_shape=warped2.shape[:2],
            grid_step=grid_step,
            sigma=sigma,
            base_clip_ratio=clip_ratio,
            border_damping=border_damping,
            border_width_ratio=border_width_ratio,
        )
    else:
        dx, dy = build_mesh_flow(
            matchesA_canvas=pts1_canvas,
            matchesB_canvas=pts2_canvas,
            roi_shape=warped2.shape[:2],
            grid_step=grid_step,
            sigma=sigma,
            clip_ratio=clip_ratio,
            border_damping=border_damping,
            border_width_ratio=border_width_ratio,
        )

    print("지역 remap (INTER_CUBIC)")
    h, w = warped2.shape[:2]
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = X - dx
    map_y = Y - dy

    warped2_refined = cv.remap(
        warped2,
        map_x,
        map_y,
        cv.INTER_CUBIC,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 최종 합성
    print("이미지 합성")
    out = canvas1.copy()
    mask = (warped2_refined[:, :, 0] > 0) | (warped2_refined[:, :, 1] > 0) | (warped2_refined[:, :, 2] > 0)
    out[mask] = warped2_refined[mask]

    elapsed = (time.time() - t0) * 1000
    print("=" * 60)
    print(f"완료: {elapsed:.1f} ms")
    print("=" * 60)
    return out


if __name__ == "__main__":
    # 기본 버전 (경계 억제 적용)
    print("\n[1] 기본 버전 - 경계 변형 억제")
    out1 = stitch_mesh_based(
        "./images/left.jpg",
        "./images/right.jpg",
        grid_step=80,              # 더 촘촘한 메쉬
        sigma=100,                 # 가우시안 영향 범위
        use_adaptive=False,        # 기본 클리핑
        border_damping=True,       # 경계 억제 활성화
        border_width_ratio=0.15,   # 경계 15% 영역
        clip_ratio=0.75,
    )
    cv.imwrite("./result/stitch_mesh_basic.jpg", out1)
    print("저장: stitch_mesh_basic.jpg\n")

    # 적응적 버전 (매칭점 밀도 기반)
    print("\n[2] 적응적 버전 - 매칭점 밀도 기반 클리핑")
    out2 = stitch_mesh_based(
        "./images/left.jpg",
        "./images/right.jpg",
        grid_step=80,
        sigma=100,
        use_adaptive=True,         # 적응적 클리핑
        border_damping=True,
        border_width_ratio=0.15,
        clip_ratio=0.75,
    )
    cv.imwrite("./result/stitch_mesh_adaptive.jpg", out2)
    print("저장: stitch_mesh_adaptive.jpg\n")