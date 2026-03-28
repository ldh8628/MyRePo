import cv2
import matplotlib.pyplot as plt

# 이미지를 BGR로 읽기
img_bgr = cv2.imread('Lenna.png')

# B, G, R 채널별 히스토그램 계산
hist_b = cv2.calcHist([img_bgr], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img_bgr], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([img_bgr], [2], None, [256], [0, 256])

# 채널별 통계 출력
b, g, r = cv2.split(img_bgr)
print(f"[B 채널] min: {b.min()}, max: {b.max()}, mean: {b.mean():.2f}")
print(f"[G 채널] min: {g.min()}, max: {g.max()}, mean: {g.mean():.2f}")
print(f"[R 채널] min: {r.min()}, max: {r.max()}, mean: {r.mean():.2f}")

# 결과 출력
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 원본 이미지 (BGR -> RGB 변환)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
axes[0].imshow(img_rgb)
axes[0].set_title('Original (BGR)')
axes[0].axis('off')

# B, G, R 히스토그램 한 그래프에 표시
axes[1].plot(hist_b, color='blue',  label='B channel')
axes[1].plot(hist_g, color='green', label='G channel')
axes[1].plot(hist_r, color='red',   label='R channel')
axes[1].set_title('BGR Histogram')
axes[1].set_xlabel('Pixel Value (0~255)')
axes[1].set_ylabel('Frequency')
axes[1].set_xlim([0, 256])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
