# MyRePo 프로젝트 인덱스

이 저장소는 SLAM·6D pose estimation·영상 처리·패턴 인식·ROS 2와 소규모 응용 실험을 한곳에 모은 프로젝트 모음이다. 아래 표는 `scripts` 지원 디렉터리를 제외한 최상위 프로젝트 20개와 각 맞춤 README를 연결한다.

> 감사 기준: 2026-08-05, `main`의 `ee7bfd7` 커밋. 각 프로젝트는 서로 독립적이며 Python·CUDA·ROS 2 등 요구 환경이 다르므로 저장소 전체를 하나의 환경에 설치하는 구조가 아니다.

| 폴더 | 한 문장 설명 | README 상태 |
| --- | --- | --- |
| [CadPrompt](CadPrompt/README.md) | CAD 모델을 여러 시점에서 렌더링해 YOLO-World·YOLOE 프롬프트 후보를 직접 채점하고 추천하는 로컬 웹 도구입니다. | 맞춤 작성 |
| [DL_stitch](DL_stitch/README.md) | TensorFlow 1.x 기반 UDIS의 coarse alignment와 image reconstruction 2단계 영상 스티칭 코드입니다. | 이번 감사에서 맞춤 작성 |
| [ExtrinsicCalib](ExtrinsicCalib/README.md) | 동시 녹화한 RGB-D–RGB-D 또는 LiDAR–RGB-D bag에서 표지판 없이 두 센서의 외부 파라미터와 시간차를 구하는 로컬 웹 도구입니다. | 맞춤 작성 |
| [FoundPose](FoundPose/README.md) | DINOv2 특징과 3D template 검색으로 신규 물체의 6D pose를 추정하는 FoundPose 커스텀 구현입니다. | 맞춤 작성 |
| [HDL-GRAPH-SLAM](HDL-GRAPH-SLAM/README.md) | ROS 2 Humble에서 LiDAR keyframe과 pose graph optimization으로 3D 지도를 만드는 runtime bundle입니다. | 맞춤 작성 |
| [HuskyRoute](HuskyRoute/README.md) | LiDAR bag으로 hdl_graph_slam 지도를 만들고, 지도 위에 Husky를 놓아 Nav2 계획기 경로를 시험하는 로컬 웹 도구입니다. | 맞춤 작성 |
| [ImprovedFoundPose](ImprovedFoundPose/README.md) | portable 경로와 GT 없는 추론 등 로컬 개선을 적용한 FoundPose 최소 실행 배포본입니다. | 맞춤 작성 |
| [Intelligence_capston_opencv](Intelligence_capston_opencv/README.md) | OpenCV 영상 입출력·색공간·histogram·filtering을 독립 스크립트로 익히는 실습 모음입니다. | 이번 감사에서 맞춤 작성 |
| [ORB3-SLAM](ORB3-SLAM/README.md) | ORB-SLAM3 core와 ROS 2 RGB-D wrapper를 담은 재현 가능한 runtime bundle입니다. | 맞춤 작성 |
| [Pattern Recognition](Pattern%20Recognition/README.md) | ViT AutoEncoder의 reconstruction error로 금속 표면 불량 후보를 찾는 PyTorch 실험입니다. | 이번 감사에서 맞춤 작성 |
| [Pattern_recognition_project](Pattern_recognition_project/README.md) | TensorFlow/Keras로 선형회귀와 Iris 이진분류를 학습하는 Jupyter Notebook 모음입니다. | 이번 감사에서 맞춤 작성 |
| [RTAB-MAP](RTAB-MAP/README.md) | RTAB-Map core와 ROS 2 wrapper로 RGB-D graph SLAM을 실행하는 runtime bundle입니다. | 맞춤 작성 |
| [SAM-6D](SAM-6D/README.md) | 공식 SAM-6D를 설치해 RGB-D와 CAD로 6D pose를 추정하는 ROS 2 wrapper bundle입니다. | 맞춤 작성 |
| [ViT_AutoEncoder](ViT_AutoEncoder/README.md) | ViT AutoEncoder·PatchCore·DRAEM·CycleGAN을 비교하는 금속 표면 이상 탐지 실험 모음입니다. | 이번 감사에서 맞춤 작성 |
| [hangle_normalizer](hangle_normalizer/README.md) | KoNLPy 규칙과 로컬 Ollama LLM을 비교해 한국어 문장을 정규화하고 CER/WER를 평가합니다. | 이번 감사에서 맞춤 작성 |
| [mesh_based_stitch](mesh_based_stitch/README.md) | 원통 투영·지역 mesh warp·MultiBand blending으로 두 영상을 합성하는 OpenCV 파노라마 실험입니다. | 이번 감사에서 맞춤 작성 |
| [opencv_number](opencv_number/README.md) | OpenCV 캔버스에 쓴 최대 5자리 숫자를 번들 MNIST 신경망으로 분류하는 GUI 데모입니다. | 이번 감사에서 맞춤 작성 |
| [picamera](picamera/README.md) | Raspberry Pi Camera 영상에서 Haar cascade로 얼굴과 눈을 표시하는 실시간 하드웨어 데모입니다. | 이번 감사에서 맞춤 작성 |
| [project_bank](project_bank/README.md) | 로그인·입출금·거래내역 흐름을 보여주는 Flask 인메모리 모의 은행 웹앱입니다. | 이번 감사에서 맞춤 작성 |
| [ros2_image_subscribe](ros2_image_subscribe/README.md) | ROS 2의 6개 압축 카메라를 수집·파노라마 합성하고 YOLO 탐지까지 적용하는 prototype입니다. | 이번 감사에서 맞춤 작성 |

`맞춤 작성`은 기존부터 이 저장소의 실행 환경과 포함 범위를 설명하던 문서이고, `이번 감사에서 맞춤 작성`은 이번 작업에서 실제 Git tree와 진입점을 기준으로 새로 작성하거나 교체한 문서다. 이는 각 프로젝트 소스의 독창성이나 실행 성공을 판정하는 표기가 아니다.
