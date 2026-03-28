import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기 (BGR)
img_bgr = cv2.imread('Candies.png')

# BGR -> HSV 변환
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# HSV에서 빨간색은 색상(H)이 0~10 과 170~179 두 구간에 걸침
lower_red1 = np.array([  0, 100, 100])
upper_red1 = np.array([ 10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([179, 255, 255])

# 각 구간의 마스크 생성 후 합치기
mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask1, mask2)

# 마스크를 원본 이미지에 적용
result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_red)

# BGR -> RGB 변환 (matplotlib 출력용)
img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result,  cv2.COLOR_BGR2RGB)

# 추출 픽셀 수 출력
print(f"추출된 빨간색 픽셀 수: {np.count_nonzero(mask_red)}")

# 결과 출력
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(mask_red, cmap='gray')
axes[1].set_title('Red Mask')
axes[1].axis('off')

axes[2].imshow(result_rgb)
axes[2].set_title('Red Candy Extracted')
axes[2].axis('off')

plt.tight_layout()
plt.show()
