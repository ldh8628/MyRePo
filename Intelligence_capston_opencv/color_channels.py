import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기 (BGR)
img_bgr = cv2.imread('Lenna.png')

# B, G, R 채널 분리
b, g, r = cv2.split(img_bgr)

# 각 채널을 컬러로 시각화하기 위한 빈 채널 생성
zeros = np.zeros_like(b)

img_b = cv2.merge([b, zeros, zeros])  # Blue 채널만
img_g = cv2.merge([zeros, g, zeros])  # Green 채널만
img_r = cv2.merge([zeros, zeros, r])  # Red 채널만

# BGR -> RGB 변환 (matplotlib 출력용)
img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_b_rgb = cv2.cvtColor(img_b,   cv2.COLOR_BGR2RGB)
img_g_rgb = cv2.cvtColor(img_g,   cv2.COLOR_BGR2RGB)
img_r_rgb = cv2.cvtColor(img_r,   cv2.COLOR_BGR2RGB)

# 채널별 통계 출력
print(f"[B 채널] min: {b.min()}, max: {b.max()}, mean: {b.mean():.2f}")
print(f"[G 채널] min: {g.min()}, max: {g.max()}, mean: {g.mean():.2f}")
print(f"[R 채널] min: {r.min()}, max: {r.max()}, mean: {r.mean():.2f}")

# matplotlib으로 출력
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(img_r_rgb)
axes[1].set_title('Red Channel')
axes[1].axis('off')

axes[2].imshow(img_g_rgb)
axes[2].set_title('Green Channel')
axes[2].axis('off')

axes[3].imshow(img_b_rgb)
axes[3].set_title('Blue Channel')
axes[3].axis('off')

plt.tight_layout()
plt.show()
