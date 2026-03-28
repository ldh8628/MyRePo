import cv2
import matplotlib.pyplot as plt

# 이미지 읽기 (BGR)
img_bgr = cv2.imread('Lenna.png')

# BGR -> HSV 변환
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# H, S, V 채널 분리
h, s, v = cv2.split(img_hsv)

# 채널별 통계 출력
print(f"[H 채널] min: {h.min()}, max: {h.max()}, mean: {h.mean():.2f}  (범위: 0 ~ 179)")
print(f"[S 채널] min: {s.min()}, max: {s.max()}, mean: {s.mean():.2f}  (범위: 0 ~ 255)")
print(f"[V 채널] min: {v.min()}, max: {v.max()}, mean: {v.mean():.2f}  (범위: 0 ~ 255)")

# matplotlib으로 출력
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 원본 (BGR -> RGB 변환)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
axes[0].imshow(img_rgb)
axes[0].set_title('Original (RGB)')
axes[0].axis('off')

# H 채널
axes[1].imshow(h, cmap='hsv')
axes[1].set_title('H (Hue)')
axes[1].axis('off')

# S 채널
axes[2].imshow(s, cmap='gray')
axes[2].set_title('S (Saturation)')
axes[2].axis('off')

# V 채널
axes[3].imshow(v, cmap='gray')
axes[3].set_title('V (Value)')
axes[3].axis('off')

plt.tight_layout()
plt.show()
