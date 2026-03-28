import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기 (BGR)
img_bgr = cv2.imread('Candies.png')

# BGR -> HSV 변환
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# OpenCV HSV 파란색 범위
# - 표준 HSV H: 240° 근방 → OpenCV H: 240/2 = 120 (범위: 100 ~ 130)
# - S, V: 충분히 채도/밝기가 있는 픽셀만 선택
lower_blue = np.array([100, 100, 50])
upper_blue = np.array([130, 255, 255])

# 마스크 생성
mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

# 마스크를 원본 이미지에 적용
result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_blue)

# BGR -> RGB 변환 (matplotlib 출력용)
img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result,  cv2.COLOR_BGR2RGB)

# 추출 픽셀 수 출력
print(f"파란색 H 범위: 100 ~ 130  (OpenCV 기준, 표준 240° 근방)")
print(f"추출된 파란색 픽셀 수: {np.count_nonzero(mask_blue)}")

# 결과 출력
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(mask_blue, cmap='gray')
axes[1].set_title('Blue Mask')
axes[1].axis('off')

axes[2].imshow(result_rgb)
axes[2].set_title('Blue Candy Extracted')
axes[2].axis('off')

plt.tight_layout()
plt.show()
