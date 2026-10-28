# ExtrinsicCalib

동시에 녹화한 두 센서의 ROS 2 bag에서 두 센서 사이의 외부 파라미터(위치·자세)와 시간차를 구하는 프로그램이다. 지원하는 조합은 **RGB-D–RGB-D**와 **LiDAR–RGB-D** 두 가지다. 체커보드 같은 표지판은 쓰지 않고, 센서를 실은 로봇이 주행하며 녹화한 데이터만 쓴다. 브라우저 화면(로컬 웹 서버)과 명령줄 두 가지로 실행할 수 있다.

## 동작 방식

1. **센서별 자기 궤적:** RGB-D는 Open3D RGB-D 오도메트리(색 항)로, LiDAR는 KISS-ICP로 구한다.
2. **시간차:** 1초 동안 회전한 각도는 센서를 어디에 달았든 같다. 그래서 두 궤적의 회전량이 가장 잘 맞는 시간차 τ를 찾는다.
3. **움직임 맞추기(hand-eye):** 같은 구간에서 두 센서가 움직인 양 A, B가 `A X = X B`를 만족하는 X를 강건 최소제곱으로 푼다.
4. **바닥면 (평면 주행일 때):** 지면 로봇은 평면에서만 움직이므로 두 센서의 높이 차를 움직임으로는 알 수 없다. 그래서 각 센서가 본 바닥면으로 높이와 기울기를 정한다.
   - 카메라: 깊이 영상에서 RANSAC으로 바닥 평면을 찾는다.
   - LiDAR: 주행 전체 스캔을 쌓은 지도에서 바닥을 찾는다.
   - 위아래 방향: 카메라 바닥으로 정한다.
   - 바닥이 일관되게 보이는 장면이 30장 미만이면 바닥을 쓰지 않는다. 바닥 법선과 회전축이 3° 넘게 어긋날 때도 쓰지 않는다. 이때 높이 차는 정해지지 않으며(0으로 둠) 경고를 띄운다.
5. **점검:**
   - 데이터를 짝수·홀수 구간으로 나눠 따로 푼 결과의 차이
   - 1초 구간 움직임 비교 그림
   - LiDAR 점이 깊이 면에 붙는 비율과, LiDAR 점을 컬러 영상에 투영한 그림

결과는 `T_a_b`이다. 센서 B 좌표를 센서 A 좌표로 옮기는 4×4 행렬이며, A가 기준 좌표계다. `tau_s`는 B의 시각이 실제보다 늦은 양이므로 B의 실제 시각은 `stamp − tau_s`다.

## 입력 조건

- **동시 녹화:** 두 bag의 header stamp가 같은 시계이거나 수백 ms 이내로 맞아 있어야 한다. 남은 시간차는 ±0.5 s 안에서 자동으로 찾는다.
- **RGB-D bag:** 컬러 `sensor_msgs/Image`, **컬러에 정렬된** 깊이 Image(16UC1 mm 또는 32FC1 m), 컬러 `CameraInfo`가 있어야 한다. RealSense ROS 2 드라이버의 `align_depth` 출력이 여기에 해당한다.
- **LiDAR bag:** `sensor_msgs/PointCloud2`가 있어야 한다. 점마다 `time` 필드가 있으면 움직임 왜곡도 보정한다.
- **bag 형식:** rosbag2 `.db3` 폴더. ROS 설치 없이 읽는다. `.mcap`도 rosbags로 읽도록 해 두었지만 실제 파일로는 시험하지 않았다.
- **녹화 방법:** 1~3분 정도, 직진과 좌우 회전을 섞어 주행한다. 회전이 없으면 시간차와 방향을 정할 수 없다. 카메라에 바닥이 보여야 한다.

## 설치

Ubuntu 22.04, Python 3.11에서 확인했다. GPU는 필요 없다.

```bash
cd ExtrinsicCalib
conda env create -f environment.yml      # 또는: python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
conda activate extcalib
```

## 실행

### 브라우저 화면

```bash
cd ExtrinsicCalib
python -m extcalib.server --root /data/recordings      # http://localhost:8795
```

1. 센서 A의 종류(LiDAR / RGB-D)를 고른다.
2. 두 센서의 bag 폴더를 입력하거나 "찾기"로 고른 뒤 "확인"을 누른다. 토픽은 자동으로 채워진다.
3. "계산 시작"을 누른다.

결과 JSON은 `results/`에 저장되고 화면에서 내려받을 수도 있다. 다른 PC에서 접속하려면 `--host 0.0.0.0`을 준다.

### 명령줄

```bash
python -m extcalib --a /data/rec/lidar --a-type lidar --b /data/rec/camera --out lidar_camera.json
python -m extcalib --a /data/rec/cam_front --a-type rgbd --b /data/rec/cam_side --out front_side.json
```

토픽 이름이 RealSense 기본값과 다르면 `--points-topic`, `--color-topic`, `--depth-topic`, `--info-topic`으로 지정한다.

## 결과 읽기

| 항목 | 좋은 범위 | 벗어나면 |
|---|---|---|
| 시간차 일치 오차 | < 0.5° / 1 s | 한 센서의 오도메트리가 나쁨 |
| 시간차 확실성 (2등/1등) | > 1.5 | 회전이 적어 시간차가 덜 정해짐 |
| B / A 이동 거리 비 | 0.98 ~ 1.02 | 카메라 깊이 축척 문제 (D455 개체에서 6 % 짧게 찍힌 사례가 있음) |
| 짝수/홀수 구간 결과 차이 | < 2 cm, < 0.5° | 데이터가 부족하거나 일관되지 않음 |
| LiDAR 점이 깊이 면 3 cm 안 | > 0.25 | 그림으로 모서리를 확인할 것 |

## 확인한 결과

2026-09-01 실내 녹화(Husky A200, VLP-16, RealSense D455 두 대, 2분 32초, 8자 주행)로 시험했다. 기준값은 같은 녹화에서 다른 방법으로 구한 값이다. ORB-SLAM3 궤적 hand-eye + 바닥 높이와, LiDAR 겹침 최대 높이(±10 cm)를 썼다. 따라서 기준값도 정답은 아니다.

| 조합 | 시간차 | 기준값과 차이 | 짝수/홀수 차이 | 계산 시간 |
|---|---|---|---|---|
| LiDAR ← 정면 카메라 | +0.024 s | 5.1 cm · 1.0° | 0.13 cm · 0.05° | 126 s |
| LiDAR ← 측면 카메라 | +0.210 s | 3.5 cm · 1.5° | 0.35 cm · 0.14° | 119 s |
| 정면 ← 측면 카메라 | +0.202 s | 3.3 cm · 0.9° | 0.07 cm · 0.04° | 117 s |

- 측면 카메라 bag이 약 0.21 s 늦게 기록된 것을 시간차로 찾았다. 이 녹화에서 별도로 확인한 값은 0.208 s다.
- LiDAR를 거쳐 구한 두 카메라 사이 값과 두 카메라를 직접 맞춘 값은 0.7 cm·0.3° 차이였다.

두 번째로 2026-08-26 다른 장소의 어두운 녹화(같은 카메라 두 대)를 시험했다. 정면 카메라에서 일관된 바닥이 13장만 잡혀 바닥 단계를 쓰지 않았고, 높이 미정 경고가 떴다. 이 녹화의 URDF 값과 비교한 결과는 다음과 같다.

| 축 | URDF 값과 차이 |
|---|---|
| 수평 | 2.6 cm, 3.6 cm |
| 높이 | 9.3 cm (정해지지 않은 축, 경고 대상) |
| 회전 | 1.8° |

## 한계

- **평면 주행 가정:** 지면 로봇의 평면 주행에 맞춰 만들었다. 손에 들고 3차원으로 크게 움직인 녹화는 바닥 단계 없이 일반 hand-eye로 풀지만, 실제 데이터로는 시험하지 않았다.
- **실내 16선 LiDAR의 높이:** LiDAR 점을 깊이 면에 직접 맞추는 방식은 이 시험에서 높이를 40~50 cm 틀렸다. 벽만 많이 보이기 때문이다. 그래서 이 프로그램은 그 방식을 점검용으로만 쓰고, 높이는 바닥면으로 정한다.
- **어두운 녹화:** 특징이 적은 녹화에서는 카메라 오도메트리가 자주 끊긴다. 끊긴 구간은 계산에서 빠지지만, 끊김이 많으면 경고가 뜬다.
- **바닥이 잘 안 보이는 녹화:** 카메라가 위를 향하거나 바닥이 어두우면 높이 차를 구하지 못한다. 결과에 경고로 표시되며, 그때는 두 센서의 높이 차를 자로 재서 결과 행렬의 위아래 성분을 직접 고쳐야 한다.
- **센서 B:** 센서 B는 RGB-D여야 한다. LiDAR–LiDAR 조합은 지원하지 않는다.

## 구성

```
extcalib/
  bagio.py      rosbag2 읽기 (.db3는 sqlite 직접, .mcap은 rosbags)
  odometry.py   RGB-D 오도메트리, KISS-ICP
  handeye.py    시간차, A X = X B
  floor.py      바닥면으로 높이·기울기
  refine.py     깊이 면, LiDAR 누적 지도, LiDAR–깊이 점검
  pipeline.py   전체 순서, 경고, 그림
  server.py     로컬 웹 서버, web/index.html
  __main__.py   명령줄
```

사용한 라이브러리: [Open3D](https://github.com/isl-org/Open3D)(MIT), [KISS-ICP](https://github.com/PRBonn/kiss-icp)(MIT), [rosbags](https://gitlab.com/ternaris/rosbags)(Apache-2.0), OpenCV, SciPy, Matplotlib.
