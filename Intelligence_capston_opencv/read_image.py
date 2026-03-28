import cv2
import matplotlib.pyplot as plt

# 이미지를 그레이스케일로 읽기
img_gray = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# 이미지 정보 출력
print(f"이미지 크기: {img_gray.shape}")
print(f"데이터 타입: {img_gray.dtype}")

# matplotlib으로 출력
plt.figure(figsize=(6, 6))
plt.imshow(img_gray, cmap='gray')
plt.title('Lenna.png (Grayscale)')
plt.axis('off')
plt.tight_layout()
plt.show()
