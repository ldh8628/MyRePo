import cv2
import matplotlib.pyplot as plt

# 이미지를 그레이스케일로 읽기
img_gray = cv2.imread('Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

# 히스토그램 평활화 적용
img_eq = cv2.equalizeHist(img_gray)

# 평활화 전/후 히스토그램 계산
hist_before = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
hist_after  = cv2.calcHist([img_eq],   [0], None, [256], [0, 256])

# 통계 출력
print(f"[평활화 전] min: {img_gray.min()}, max: {img_gray.max()}, mean: {img_gray.mean():.2f}")
print(f"[평활화 후] min: {img_eq.min()},   max: {img_eq.max()},   mean: {img_eq.mean():.2f}")

# 결과 출력 (2행 2열)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 원본 이미지
axes[0][0].imshow(img_gray, cmap='gray')
axes[0][0].set_title('Original')
axes[0][0].axis('off')

# 평활화 이미지
axes[0][1].imshow(img_eq, cmap='gray')
axes[0][1].set_title('Equalized')
axes[0][1].axis('off')

# 원본 히스토그램
axes[1][0].plot(hist_before, color='black')
axes[1][0].fill_between(range(256), hist_before.flatten(), color='gray', alpha=0.5)
axes[1][0].set_title('Histogram (Before)')
axes[1][0].set_xlabel('Pixel Value (0~255)')
axes[1][0].set_ylabel('Frequency')
axes[1][0].set_xlim([0, 256])
axes[1][0].grid(True, alpha=0.3)

# 평활화 후 히스토그램
axes[1][1].plot(hist_after, color='black')
axes[1][1].fill_between(range(256), hist_after.flatten(), color='gray', alpha=0.5)
axes[1][1].set_title('Histogram (After)')
axes[1][1].set_xlabel('Pixel Value (0~255)')
axes[1][1].set_ylabel('Frequency')
axes[1][1].set_xlim([0, 256])
axes[1][1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
