import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기 (BGR)
img_bgr = cv2.imread('Candies.png')

# B, G, R 채널 분리
b, g, r = cv2.split(img_bgr)

# 원본 RGB 변환
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Red 성분 임계값 목록
thresholds = [50, 150, 200, 250]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# 원본 출력
axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')

# 각 임계값별 Red 성분 추출
for i, thresh in enumerate(thresholds):
    # R 채널에서 thresh ~ 255 범위의 마스크 생성
    mask = cv2.inRange(r, thresh, 255)

    # 마스크를 원본 이미지에 적용
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    print(f"[Red >= {thresh:3d}] 추출된 픽셀 수: {np.count_nonzero(mask)}")

    axes[i + 1].imshow(result_rgb)
    axes[i + 1].set_title(f'Red >= {thresh}')
    axes[i + 1].axis('off')

plt.tight_layout()
plt.show()
