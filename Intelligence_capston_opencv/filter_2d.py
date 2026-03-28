import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 그레이스케일로 읽기
img_gray = cv2.imread('Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

# ── 1. 평균값 필터 (Averaging Filter) ──────────────────────────────
kernel_avg = np.ones((5, 5), np.float32) / 25
img_avg = cv2.filter2D(img_gray, -1, kernel_avg)

# ── 2. 샤프닝 필터 (Sharpening Filter) ────────────────────────────
kernel_sharp = np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]], np.float32)
img_sharp = cv2.filter2D(img_gray, -1, kernel_sharp)

# ── 3. 라플라시안 필터 (Laplacian Filter) ──────────────────────────
kernel_laplacian = np.array([[ 0,  1,  0],
                              [ 1, -4,  1],
                              [ 0,  1,  0]], np.float32)
img_laplacian = cv2.filter2D(img_gray, -1, kernel_laplacian)

# 결과 출력
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(img_gray,      cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(img_avg,       cmap='gray')
axes[1].set_title('Averaging Filter (5x5)')
axes[1].axis('off')

axes[2].imshow(img_sharp,     cmap='gray')
axes[2].set_title('Sharpening Filter')
axes[2].axis('off')

axes[3].imshow(img_laplacian, cmap='gray')
axes[3].set_title('Laplacian Filter')
axes[3].axis('off')

plt.tight_layout()
plt.show()
