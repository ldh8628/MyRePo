# -*- coding: utf-8 -*-
"""
필요 패키지: opencv-python, numpy
pip install opencv-python numpy

폴더 구조(예시):
opencv_number/
  ├─ common/
  │   └─ functions.py   (sigmoid, softmax)
  ├─ dataset/
  │   └─ mnist.py       (load_mnist)
  ├─ sample_weight.pkl  (학습된 가중치)
  └─ number_predict.py  (이 파일)

기능:
- 창에 5개의 입력칸(검정 배경)에 마우스로 흰색 펜으로 숫자(0~9) 쓰기
- [인식] 버튼: 5칸 숫자 예측
- [전체지움] 버튼: 초기화
- ESC 또는 q: 종료
"""

import sys
import os
import pickle
import numpy as np

# ---- 예제 코드 호환: 필요한 경우 부모 폴더 추가 (현재 구조에선 없어도 OK)
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

# ========== (선택) MNIST 데이터셋 로딩 및 정확도 확인 ==========
# 처음 한 번만 수행해도 됨. 실행 속도를 위해 normalize=False로 간단 출력.
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000,)

# -------- 추론용 유틸 --------
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(BASE_DIR, "sample_weight.pkl")
    with open(weight_path, "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

# (선택) 배치 정확도 출력
x, t = get_data()
network = init_network()
accuracy_cnt = 0
batch_size = 100
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # 대략 0.935 근처

# ========== OpenCV UI (5칸 손글씨 → 예측) ==========
import cv2

# --- 레이아웃/설정 ---
W, H = 900, 380
PAD = 20
BOX_SIZE = 120
NUM_BOXES = 5
PEN_THICK = 8         # 흰색 펜 두께
FONT = cv2.FONT_HERSHEY_SIMPLEX


def compute_layout():
    total_w = NUM_BOXES * BOX_SIZE + (NUM_BOXES - 1) * PAD
    start_x = (W - total_w) // 2
    y_top = PAD
    boxes = []
    for i in range(NUM_BOXES):
        x1 = start_x + i * (BOX_SIZE + PAD)
        y1 = y_top
        boxes.append((x1, y1, x1 + BOX_SIZE, y1 + BOX_SIZE))
    # 버튼
    btn_h, btn_w = 44, 160
    y_btn = y_top + BOX_SIZE + 2 * PAD
    x_btn1 = start_x
    x_btn2 = start_x + btn_w + PAD
    recog_btn = (x_btn1, y_btn, x_btn1 + btn_w, y_btn + btn_h)
    clear_btn = (x_btn2, y_btn, x_btn2 + btn_w, y_btn + btn_h)
    result_y = y_btn + btn_h + 40
    return boxes, recog_btn, clear_btn, result_y

BOXES, RECO_BTN, CLR_BTN, RESULT_Y = compute_layout()

# MNIST 톤: 검정 배경(0), 흰 글씨(255)
box_images = [np.zeros((BOX_SIZE, BOX_SIZE), np.uint8) for _ in range(NUM_BOXES)]
drawing = False
active_box = -1
last_pt = None
pred_text = ""
DILATE_ITERS = 0      # 0~1 권장: 과팽창 금지
CONF_TH = 0.55        # 신뢰도 임계값(낮으면 '-'로 처리)

# --- 전처리: 28x28 (0~1)로 변환해 (1x784) 플랫 ---
def preprocess_digit_to_mnist(sample_gray):
    """
    입력: 0=배경(검정), 255=글씨(흰색)인 120x120 회색 영상
    출력: (1,784) float32, 0~1 (MNIST 규격)
    절차:
      - Otsu 이진화
      - 가장 큰 컨투어 추출
      - 20x20로 비율 유지 축소
      - 28x28로 중앙 배치(여백 4px)
      - 질량중심(centroid)을 (14,14)로 이동
      - 가벼운 블러 → 0~1 정규화
    """
    import cv2
    import numpy as np

    # 안전 클리핑
    img = np.clip(sample_gray, 0, 255).astype(np.uint8)

    # 이진화 (글씨=255, 배경=0 유지)
    # 펜이 흰색이므로 그대로 Otsu
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 가장 큰 컨투어
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None

    x, y, w, h = cv2.boundingRect(c)
    roi = th[y:y+h, x:x+w]

    # 얇은 획 보정(과팽창 금지)
    if DILATE_ITERS > 0:
        roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                         iterations=DILATE_ITERS)

    # 20x20로 비율 유지 축소
    side = max(w, h)
    scale = 20.0 / side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    roi_small = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 28x28로 중앙 배치(여백 4px)
    canvas = np.zeros((28, 28), np.uint8)
    x0 = (28 - new_w) // 2
    y0 = (28 - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = roi_small

    # 질량중심을 (14,14)로 이동(스큐/오프셋 보정)
    m = cv2.moments(canvas, binaryImage=True)
    if m['m00'] != 0:
        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, M, (28, 28), flags=cv2.INTER_NEAREST, borderValue=0)

    # 가벼운 블러로 에일리어싱 완화
    canvas = cv2.GaussianBlur(canvas, (3,3), 0)

    # 0~1 정규화 + 플랫
    canvas = canvas.astype(np.float32) / 255.0
    flat = canvas.reshape(1, 784)
    return flat

def predict_boxes_with_network(network):
    """각 칸에 대해 확률 최댓값이 CONF_TH 미만이면 '-' 처리"""
    out = []
    for i in range(NUM_BOXES):
        sample = preprocess_digit_to_mnist(box_images[i])
        if sample is None:
            out.append('-')
            continue
        y = predict(network, sample)            # (1,10)
        probs = y[0]
        p = int(np.argmax(probs))
        if float(probs[p]) < CONF_TH:
            out.append('-')
        else:
            out.append(str(p))
    return ''.join(out)

# --- UI 그리기 ---
def draw_button(canvas, rect, label):
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
    cx = (x1 + x2)//2 - tw//2
    cy = (y1 + y2)//2 + th//2 - 4
    cv2.putText(canvas, label, (cx, cy), FONT, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

def draw_ui(canvas):
    canvas[:] = 235
    # 칸
    for i, (x1, y1, x2, y2) in enumerate(BOXES):
        cv2.rectangle(canvas, (x1-1, y1-1), (x2+1, y2+1), (120, 120, 120), 2)
        roi = cv2.cvtColor(box_images[i], cv2.COLOR_GRAY2BGR)
        canvas[y1:y2, x1:x2] = roi
        cv2.putText(canvas, f"{i+1}", (x1+4, y1+20), FONT, 0.6, (150,150,150), 1, cv2.LINE_AA)

    draw_button(canvas, RECO_BTN, "Recognize")
    draw_button(canvas, CLR_BTN,  "Clear All")

    cv2.putText(canvas, f"Result: {pred_text}", (BOXES[0][0], RESULT_Y),
                FONT, 1.0, (10,10,10), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Left drag = draw (white pen) | Right click = clear box",
                (BOXES[0][0], RESULT_Y+32), FONT, 0.6, (90,90,90), 1, cv2.LINE_AA)

def which_box(x, y):
    for idx, (x1, y1, x2, y2) in enumerate(BOXES):
        if x1 <= x < x2 and y1 <= y < y2:
            return idx
    return -1

def in_rect(pt, rect):
    x, y = pt
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def clear_box(i):
    box_images[i][:] = 0

def clear_all():
    for i in range(NUM_BOXES):
        clear_box(i)

# --- 마우스 콜백 ---
def on_mouse(event, x, y, flags, param):
    global drawing, active_box, last_pt, pred_text

    if event == cv2.EVENT_LBUTTONDOWN:
        b = which_box(x, y)
        if b >= 0:
            drawing = True
            active_box = b
            last_pt = (x, y)
        else:
            if in_rect((x, y), RECO_BTN):
                pred = predict_boxes_with_network(network)
                pred_text = pred
                print("예측:", pred)
            elif in_rect((x, y), CLR_BTN):
                clear_all()
                pred_text = ""
    elif event == cv2.EVENT_MOUSEMOVE and drawing and active_box >= 0:
        bx1, by1, bx2, by2 = BOXES[active_box]
        if bx1 <= x < bx2 and by1 <= y < by2:
            if last_pt is not None:
                p1 = (last_pt[0] - bx1, last_pt[1] - by1)
                p2 = (x - bx1, y - by1)
                cv2.line(box_images[active_box], p1, p2, 255, PEN_THICK, cv2.LINE_AA)  # 흰 펜
            last_pt = (x, y)
        else:
            last_pt = None
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        active_box = -1
        last_pt = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        b = which_box(x, y)
        if b >= 0:
            clear_box(b)

# --- 실행 루프 ---
if __name__ == "__main__":
    # 위에서 network 이미 초기화됨
    cv2.namedWindow("MNIST Draw (5 boxes)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("MNIST Draw (5 boxes)", on_mouse)

    canvas = np.full((H, W, 3), 235, np.uint8)
    while True:
        draw_ui(canvas)
        cv2.imshow("MNIST Draw (5 boxes)", canvas)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()
