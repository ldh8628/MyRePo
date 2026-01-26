import os
import cv2 as cv
import torch

if __name__ == "__main__":
    img1 = cv.imread("./images/left.jpg")
    img2 = cv.imread("./images/right.jpg")

    scale_h = max(480, min(1080, img1.shape[0])) / img1.shape[0]
    img1 = cv.resize(img1, (int(img1.shape[1] * scale_h), int(img1.shape[0] * scale_h)), interpolation=cv.INTER_AREA)
    img2 = cv.resize(img2, (int(img2.shape[1] * scale_h), int(img2.shape[0] * scale_h)), interpolation=cv.INTER_AREA)

    os.makedirs(os.path.dirname("./images/DL"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"