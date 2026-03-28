import cv2
import matplotlib.pyplot as plt

# 이미지를 그레이스케일로 읽기
img_gray = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# 히스토그램 계산
# cv2.calcHist([이미지], [채널], 마스크, [빈 수], [범위])
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

# 이미지 정보 출력
print(f"이미지 크기: {img_gray.shape}")
print(f"최솟값: {img_gray.min()}, 최댓값: {img_gray.max()}, 평균: {img_gray.mean():.2f}")

# 결과 출력
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 그레이스케일 이미지
axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title('Grayscale Image')
axes[0].axis('off')

# 히스토그램
axes[1].plot(hist, color='black')
axes[1].fill_between(range(256), hist.flatten(), color='gray', alpha=0.5)
axes[1].set_title('Grayscale Histogram')
axes[1].set_xlabel('Pixel Value (0~255)')
axes[1].set_ylabel('Frequency')
axes[1].set_xlim([0, 256])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
