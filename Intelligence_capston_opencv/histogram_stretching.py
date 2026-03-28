import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 그레이스케일로 읽기
img_gray = cv2.imread('Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

# 히스토그램 스트레칭
# 공식: stretched = (pixel - min) / (max - min) * 255
p_min = img_gray.min()
p_max = img_gray.max()
img_stretched = ((img_gray.astype(np.float32) - p_min) / (p_max - p_min) * 255).astype(np.uint8)

# 스트레칭 전/후 히스토그램 계산
hist_before = cv2.calcHist([img_gray],      [0], None, [256], [0, 256])
hist_after  = cv2.calcHist([img_stretched], [0], None, [256], [0, 256])

# 통계 출력
print(f"[스트레칭 전] min: {img_gray.min()},      max: {img_gray.max()},      mean: {img_gray.mean():.2f}")
print(f"[스트레칭 후] min: {img_stretched.min()}, max: {img_stretched.max()}, mean: {img_stretched.mean():.2f}")

# 결과 출력 (2행 2열)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 원본 이미지
axes[0][0].imshow(img_gray, cmap='gray')
axes[0][0].set_title('Original')
axes[0][0].axis('off')

# 스트레칭 이미지
axes[0][1].imshow(img_stretched, cmap='gray')
axes[0][1].set_title('Stretched')
axes[0][1].axis('off')

# 원본 히스토그램
axes[1][0].plot(hist_before, color='black')
axes[1][0].fill_between(range(256), hist_before.flatten(), color='gray', alpha=0.5)
axes[1][0].axvline(x=p_min, color='blue',  linestyle='--', label=f'min={p_min}')
axes[1][0].axvline(x=p_max, color='red',   linestyle='--', label=f'max={p_max}')
axes[1][0].set_title('Histogram (Before)')
axes[1][0].set_xlabel('Pixel Value (0~255)')
axes[1][0].set_ylabel('Frequency')
axes[1][0].set_xlim([0, 256])
axes[1][0].legend()
axes[1][0].grid(True, alpha=0.3)

# 스트레칭 후 히스토그램
axes[1][1].plot(hist_after, color='black')
axes[1][1].fill_between(range(256), hist_after.flatten(), color='gray', alpha=0.5)
axes[1][1].set_title('Histogram (After Stretching)')
axes[1][1].set_xlabel('Pixel Value (0~255)')
axes[1][1].set_ylabel('Frequency')
axes[1][1].set_xlim([0, 256])
axes[1][1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
