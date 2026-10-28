# HuskyRoute

LiDAR 녹화(rosbag2)로 지도를 만들고, 그 지도에서 Husky의 주행 경로를 시험하는 브라우저 도구다. 지도는 hdl_graph_slam으로 만든다. 지도 위에 Husky를 놓고 목표를 찍으면 Nav2 전역 계획기가 낸 경로를 보여 준다. 로봇과 시뮬레이터는 필요 없다.

## 동작 방식

1. **3D 지도:** 같은 저장소의 [HDL-GRAPH-SLAM](../HDL-GRAPH-SLAM/README.md)(ROS 2 Humble)을 띄우고 bag을 `--clock`으로 재생한다. 끝나면 `save_map`·`dump_graph` 서비스로 점군 지도와 최적화된 키프레임을 저장한다.
2. **2D 격자:** 키프레임마다 점까지 광선을 그려 칸마다 맞음/지나감을 센다.
   - 바닥 위 일정 높이(기본 0.15~1.2 m)를 높이 칸으로 나눠 따로 센다. 그래서 낮은 장애물 위로 지나간 광선이 그 장애물을 지우지 않는다.
   - 키프레임 몇 개에서만 보인 장애물은 움직인 것(함께 걸은 사람 등)으로 보고 뺀다.
   - 결과는 Nav2 `map_server` 형식(PGM + YAML)이다.
3. **지나간 길 선호 격자:** 지도를 만들 때 로봇이 지나간 길(키프레임 위치)에서 먼 칸에 비용을 조금 더 준 격자도 함께 만든다. 켜면 계획기가 되도록 그 길을 따른다.
4. **경로 계획:** Nav2 `map_server` + `planner_server`를 로봇 없이 띄우고 `ComputePathToPose`에 출발 자세를 직접 준다. 계획기는 셋 중 하나를 고른다.
   - Smac Hybrid-A*: 차체 모양과 방향을 따지고, 후진 허용 여부를 정할 수 있다.
   - Smac 2D
   - NavFn
   - 차체 크기와 장애물 여유 반경은 화면에서 바꾼다.

계획에는 저장된 지도와 장애물 여유만 들어간다. 사람처럼 지도에 없는 실시간 장애물, 실제 주행 제어(MPPI 등), 바퀴 미끄러짐은 반영하지 않는다.

## 입력 조건

- **bag:** rosbag2 폴더(`metadata.yaml` + `.db3`)에 `sensor_msgs/PointCloud2` 토픽이 있어야 한다. VLP-16 `/velodyne_points`로 확인했다. `.mcap`도 재생은 되게 해 두었지만 시험하지 않았다.
- **LiDAR 자세:** LiDAR 좌표계를 로봇 기준(`base_link`)과 같다고 본다. 따라서 LiDAR가 거의 수평으로 달려 있어야 한다. 지도 원점은 첫 LiDAR 위치다.
- **바닥 높이:** LiDAR 아래에 가장 많이 모인 높이를 바닥으로 잡는다. LiDAR를 높이 달아 바닥이 잘 안 보이면 "바닥에서 LiDAR 높이"를 직접 넣는다.
- **녹화:** 지도로 쓸 공간을 천천히 한 바퀴 이상 돌며 녹화한다. 같은 곳을 다시 지나면 hdl_graph_slam이 고리를 닫아 지도가 반듯해진다.

## 설치

Ubuntu 22.04에서 확인했다. ROS 2·Nav2·hdl_graph_slam 빌드 도구를 모두 conda(RoboStack) 환경 하나에 넣으므로 시스템 ROS나 sudo가 필요 없다. GPU도 필요 없다.

이 저장소에서 `HuskyRoute`와 `HDL-GRAPH-SLAM` 두 폴더가 필요하다.

```bash
cd MyRePo
conda env create -f HuskyRoute/environment.yml
conda activate huskyroute

# hdl_graph_slam 빌드 (처음 한 번, 몇 분)
cd HDL-GRAPH-SLAM/hdlgraphslam_ws
colcon build --packages-up-to hdl_graph_slam --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_VGICP_CUDA=OFF -DBUILD_PYTHON_BINDINGS=OFF
cd ../..
```

`/opt/ros/...`의 시스템 ROS를 같은 터미널에서 source하지 않는다. 섞이면 빌드나 실행이 깨진다.

## 실행

### 브라우저 화면

```bash
conda activate huskyroute
source HDL-GRAPH-SLAM/hdlgraphslam_ws/install/setup.bash
cd HuskyRoute
python -m huskyroute.server --root /data/recordings      # http://localhost:8798
```

1. **① 지도 → 새 지도 만들기**
   - "찾기"로 bag 폴더를 고른다. 토픽과 지도 이름은 자동으로 채워진다.
   - "지도 만들기"를 누른다. 녹화 길이만큼 걸린다(재생 속도 2배면 절반). 지도는 `maps/<이름>/`에 저장된다.
2. **② 로봇·계획기:** 계획기, 차체 크기, 장애물 여유를 정하고 "적용"을 누른다. 기본값은 Husky A200(0.99 × 0.67 m)에 여유 1.5~2 cm를 더한 크기다.
3. **③ 경로 시험**
   - "Husky 위치"를 누르고 지도를 클릭한다. 누른 채 끌면 그 방향이 차체 방향이 된다.
   - 이어서 목표를 같은 방식으로 찍는다.
   - 경로, 길이, 후진 구간, 장애물에 가장 가까운 정도가 나온다. "주행 미리보기"로 차체가 경로를 따라가는 모습을 본다.

주소 끝의 `#map=…&s=x,y,방향&g=x,y,방향`이 장면을 기억하므로, 주소를 다시 열면 같은 시험이 나온다. "격자 다시 만들기"는 SLAM을 다시 돌리지 않고 장애물 높이 범위나 LiDAR 높이만 바꿔 2D 격자를 새로 만든다. 다른 PC에서 접속하려면 `--host 0.0.0.0`을 준다.

### 명령줄

```bash
python -m huskyroute map /data/recordings/lab/lidar --name lab --rate 2
python -m huskyroute route lab --start=0,0,0 --goal=-4,-3,180 --planner hybrid      # 음수로 시작하면 = 로 붙여 쓴다
```

## 결과 예시

Husky A200 + VLP-16으로 실내 방에서 두 기둥 둘레를 8자로 돈 녹화(152 s)로 확인했다. 노트북 CPU 기준이다.

| 단계 | 결과 |
|---|---|
| 지도 만들기 (1배속) | 3분 36초, 키프레임 116개, 주행 50.2 m, 바닥은 LiDAR 아래 0.70 m |
| 2D 격자 | 0.05 m 칸, 방의 벽과 기둥 두 개가 장애물로 나옴 |
| 계획기 시작 | Hybrid-A* 약 6 s, 2D·NavFn 1 s 안팎 |
| 경로 계산 | 방 안 5 m 거리에서 수 ms (Hybrid-A*) |

## 한계

- **전역 경로만 본다.** 실제 로봇은 지역 계획기·제어기가 이 경로를 따라가며 실시간 장애물을 피한다. 그 부분은 시험하지 않으므로, 여기서 경로가 나와도 실주행에서 막힐 수 있다.
- **지도 품질은 hdl_graph_slam에 좌우된다.** 고리가 닫히지 않은 긴 복도나 특징 없는 공간에서는 지도가 휘거나 겹칠 수 있다.
- **LiDAR 장착 자세를 따로 받지 않는다.** 기울어진 LiDAR는 바닥과 장애물 높이를 틀리게 만든다.
- 경로가 없을 때 이유는 출발·목표 칸의 비용으로만 짐작한다(지도 밖, 모르는 곳, 장애물 위). 중간이 막힌 경우는 "찾지 못함"으로만 나온다.
- 계획기와 지도 만들기는 각각 ROS 도메인 62, 61에서 돈다(`HUSKYROUTE_PLAN_DOMAIN`, `HUSKYROUTE_MAP_DOMAIN`으로 바꿈). 같은 PC의 다른 ROS 프로그램과 섞이지 않도록 이 PC 안에서만 통신한다.

## 지도 폴더

| 파일 | 내용 |
|---|---|
| `<이름>.pgm`, `<이름>.yaml` | Nav2 `map_server`용 격자 (그대로 실제 Nav2에 쓸 수 있음) |
| `<이름>_pref.png`, `<이름>_pref.yaml` | 지나간 길 선호 격자 (`mode: scale`, Nav2 costmap에 `trinary_costmap: false` 필요) |
| `<이름>_track.json` | 지도 만들 때 지나간 길 [x, y] |
| `map.pcd` | hdl_graph_slam 3D 점군 지도 |
| `dump/` | 키프레임 점군과 최적화된 자세 (격자 다시 만들기에 씀) |
| `info.json`, `hdl.log`, `play.log` | 만든 설정, 격자 요약, 실행 기록 |

## 파일 구성

```
huskyroute/
  mapping.py     bag 정보, hdl_graph_slam 실행과 저장
  grid.py        키프레임 -> 2D 격자, 지나간 길 선호 격자
  planner.py     Nav2 계획기 실행, 경로 요청, 비용지도
  launch/planner.launch.py   map_server + planner_server + lifecycle manager
  server.py      로컬 웹 서버
  web/index.html 화면
  __main__.py    명령줄
```
