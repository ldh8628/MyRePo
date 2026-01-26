import cv2 as cv
import numpy as np
import time

# 25.12월 기준 최신 개발한 코드

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
):
    """
    matchesA_canvas: 기준(목표) 점들
    matchesB_canvas: 워핑된 이미지 쪽 점들
    residual = A - B 를 가우시안 가중 평균으로 격자 변위 추정
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

    for i in range(Gy):
        vy = grid_y[i, 0]
        for j in range(Gx):
            vx = grid_x[0, j]
            d2 = (pts[:, 0] - vx) ** 2 + (pts[:, 1] - vy) ** 2
            wgt = np.exp(-d2 / (2.0 * sigma * sigma)) + 1e-8
            wsum = np.sum(wgt)
            grid_dx[i, j] = np.sum(residuals[:, 0] * wgt) / wsum
            grid_dy[i, j] = np.sum(residuals[:, 1] * wgt) / wsum

    dx = cv.resize(grid_dx, (w, h), interpolation=cv.INTER_CUBIC)
    dy = cv.resize(grid_dy, (w, h), interpolation=cv.INTER_CUBIC)

    mag = np.hypot(dx, dy)
    th = grid_step * clip_ratio
    scale = np.ones_like(mag, np.float32)
    over = mag > th
    scale[over] = th / (mag[over] + 1e-6)
    dx *= scale
    dy *= scale

    s = max(0.1, grid_step * smooth_sigma_ratio)
    dx = cv.GaussianBlur(dx, (0, 0), s)
    dy = cv.GaussianBlur(dy, (0, 0), s)

    return dx, dy


def stitch_mesh_based(path1, path2, grid_step=120, sigma=None, mask_scale=1.4):
    t0 = time.time()
    print("매쉬 기반 스티칭2 실행")
    print("이미지 입력 확인")

    img1 = cv.imread(path1)
    img2 = cv.imread(path2)
    if img1 is None or img2 is None:
        raise RuntimeError("이미지 읽기 실패")

    pts1, pts2 = detect_and_match(img1, img2)

    print("호모그래피 추정(RANSAC)")
    # 핵심 수정: img2 -> img1 로 맞춤
    H21, _ = cv.findHomography(pts2, pts1, cv.RANSAC, 2.0)
    if H21 is None:
        raise RuntimeError("호모그래피 실패")

    canvas1, warped2, T, tx, ty, size = compute_canvas_and_warps(img1, img2, H21)

    print("매칭점 캔버스 좌표")
    # img1 점은 translation만
    pts1_canvas = pts1 + np.array([tx, ty], dtype=np.float32)

    # img2 점은 H21로 img1좌표로 보낸 뒤 translation
    pts2_in1 = cv.perspectiveTransform(pts2.reshape(-1, 1, 2), H21).reshape(-1, 2)
    pts2_canvas = pts2_in1 + np.array([tx, ty], dtype=np.float32)

    dx, dy = build_mesh_flow(
        matchesA_canvas=pts1_canvas,
        matchesB_canvas=pts2_canvas,
        roi_shape=warped2.shape[:2],
        grid_step=grid_step,
        sigma=sigma,
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

    # 최소 합성(스티칭): refined warped2의 유효 픽셀을 canvas1에 덮어쓰기
    out = canvas1.copy()
    mask = (warped2_refined[:, :, 0] > 0) | (warped2_refined[:, :, 1] > 0) | (warped2_refined[:, :, 2] > 0)
    out[mask] = warped2_refined[mask]

    print(f"완료: {(time.time() - t0) * 1000:.1f} ms")
    return out


if __name__ == "__main__":
    out = stitch_mesh_based(
        "./images/left.jpg",
        "./images/right.jpg",
        grid_step=120,
        sigma=None,
        mask_scale=1.4,
    )
    cv.imwrite("stitch_mesh_result2.jpg", out)
    print("saved: stitch_mesh_result2.jpg")
