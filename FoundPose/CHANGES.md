# CHANGES.md — 오류 및 수정 기록

## [2026-03-25 16:30] Depth 이미지 검은 화면 — 원인 분석 및 판정

- **현상**: `outputs/templates/v1/lmo/1/depth/` 의 PNG 파일들이 일반 뷰어에서 모두 검정으로 표시됨
- **원인**: 렌더링 버그 아님 — **uint16 시각화 문제**
  - 렌더링 파이프라인: pyrender → depth(m) → ×1000 → depth(mm, float32) → `save_depth()` → `np.round().astype(uint16)` → 16-bit PNG
  - 실제 저장값: 0~917 (mm 단위, 물체가 약 0.917m 거리)
  - uint16 최대값 65535 대비 917 = **1.4%** → Windows 기본 뷰어에서 검정으로 보임
  - `gen_repre.py`는 `inout.load_depth()` → `float32(mm)`으로 정상 읽어 파이프라인에 문제 없음
- **수정 파일**: 없음 (파이프라인 코드 정상, BOP 표준 형식 준수)
- **수정 내용**: 정규화된 시각화 이미지(`debug_depth_fixed.png`) 생성하여 데이터 정상성 확인
  - Row 1: RGB 원본 / Row 2: uint16 raw (matplotlib auto-scale) / Row 3: plasma colormap + vmin/vmax 정규화
  - 물체 윤곽 및 깊이 그라디언트 정상 표시 확인
- **결론**: 검은 화면은 시각화 도구의 한계이며 실제 depth 데이터와 파이프라인은 올바름

## [2026-03-25 15:00] 환경 설정 — pip가 시스템 Python에 설치된 문제

- **오류**: `pip install torch ...` 명령 실행 시 conda 환경이 아닌 시스템 Python(`C:\Users\PC\AppData\Local\Programs\Python\Python311`)에 설치됨
- **원인**: `conda run -n foundpose pip install ...` 형식이 Windows Git Bash에서 시스템 pip를 호출함
- **수정 내용**: `conda run` 대신 conda 환경의 Python 실행파일을 직접 지정하는 방식으로 변경
  ```bash
  # 변경 전 (시스템 pip 호출됨)
  conda run -n foundpose pip install torch==2.3.0 ...
  # 변경 후 (conda 환경 pip 직접 호출)
  C:/Users/PC/miniconda3/envs/foundpose/python.exe -m pip install torch==2.3.0 ...
  ```

---

## [2026-03-25 15:10] CUDA 버전 불일치 — yaml은 CUDA 11.7, 시스템은 CUDA 12.6

- **오류**: `conda_foundpose_gpu.yaml`에 `pytorch-cuda=11.7.0` 명시되어 있으나 시스템 CUDA는 12.6
- **원인**: 야믈 파일이 CUDA 11.7 환경 기준으로 작성됨
- **수정 내용**: PyTorch를 CUDA 12.1 호환 버전으로 설치 (CUDA 12.6 드라이버는 12.1 빌드와 호환됨)
  ```bash
  pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
  ```
  - 설치 결과: `torch 2.3.0+cu121`, CUDA 사용 가능 확인 (RTX 3080)

---

## [2026-03-25 15:15] triangle 패키지 빌드 실패 — MSVC 필요

- **오류**: `pip install triangle==20200424` 실패 — "Microsoft Visual C++ 14.0 or greater is required"
- **원인**: `triangle` 패키지가 C 확장 모듈을 포함하며 Windows에서 MSVC 컴파일러가 필요
- **수정 내용**: conda-forge 채널에서 사전 빌드된 바이너리로 설치
  ```bash
  conda install -n foundpose -c conda-forge triangle -y
  ```
  - 설치된 버전: `triangle-1.6` (conda-forge)

---

## [2026-03-25 15:20] bop_toolkit 의존성 충돌 — opencv/imageio/numpy 버전 충돌

- **오류**: bop_toolkit 설치 후 opencv-python, imageio, numpy 버전이 변경됨
  - bop_toolkit requires: `opencv-python>=4.11.0.86`, `imageio>=2.35.1`, `numpy<2.0.0`
  - FoundPose yaml specifies: `opencv-python==4.5.5.62`, `imageio==2.34.1`, `numpy==1.26.4`
- **원인**: 최신 bop_toolkit이 FoundPose yaml 작성 이후 업데이트됨
- **수정 내용**: 두 요구사항을 모두 만족하는 버전으로 핀
  ```bash
  pip install numpy==1.26.4 opencv-python==4.11.0.86 imageio==2.35.1 pillow==10.3.0
  ```
  - numpy==1.26.4: 두 조건 모두 만족 (<2.0.0 이고 특정 버전 고정)
  - opencv==4.11.0.86: bop_toolkit 요구사항 우선 (FoundPose API는 하위 호환)
  - imageio==2.35.1: bop_toolkit 최소 요구사항 충족

---

## [2026-03-25 15:25] 서브모듈 미초기화 — repo가 ZIP으로 다운로드됨

- **오류**: `external/dinov2/` 및 `external/bop_toolkit/` 디렉토리가 비어 있음
- **원인**: 레포지토리가 `git clone --recurse-submodules` 없이 ZIP으로 다운로드됨
- **수정 내용**:
  1. DINOv2: PyTorch Hub 캐시(`C:/Users/PC/.cache/torch/hub/facebookresearch_dinov2_main/`)에서 복사
     ```bash
     cp -r C:/Users/PC/.cache/torch/hub/facebookresearch_dinov2_main/. d:/LDH_ws/foundpose-main/external/dinov2/
     ```
  2. bop_toolkit: GitHub에서 직접 클론
     ```bash
     git clone https://github.com/thodan/bop_toolkit.git d:/LDH_ws/foundpose-main/external/bop_toolkit
     ```

---

## [2026-03-25 15:30] PYTHONPATH 미적용 — Windows에서 bash export 미동작

- **오류**: `export PYTHONPATH=...` 후 `python scripts/gen_templates.py` 실행 시 `ModuleNotFoundError: No module named 'utils'`
- **원인**: Windows Git Bash에서 PYTHONPATH 환경변수가 Python 프로세스에 전달되지 않음
- **수정 내용**: `run_pipeline.py` 래퍼 스크립트 생성 — Python 실행 시 sys.path를 직접 삽입
  ```python
  # run_pipeline.py
  import sys, os
  REPO = "D:/LDH_ws/foundpose-main"
  sys.path.insert(0, REPO)
  sys.path.insert(0, f"{REPO}/external/bop_toolkit")
  sys.path.insert(0, f"{REPO}/external/dinov2")
  os.environ.setdefault("BOP_PATH", "D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets")
  os.environ.setdefault("BOP_OUTPUT_PATH", "D:/LDH_ws/foundpose-main/outputs")
  ...
  ```

---

## [2026-03-25 15:35] EGL OpenGL 백엔드 오류 — Windows에서 PYOPENGL_PLATFORM=egl 미지원

- **오류**: `PYOPENGL_PLATFORM=egl` 설정 시 `ImportError: Unable to load EGL library`
- **원인**: EGL은 Linux/Android 전용 OpenGL 백엔드. Windows는 WGL 사용
- **수정 내용**: `PYOPENGL_PLATFORM` 환경변수 설정하지 않음 → Windows 기본값(WGL) 사용
  - pyrender OffscreenRenderer가 Windows WGL 백엔드로 정상 동작 확인

---

## [2026-03-25 15:52] DINOv2 체크포인트 자동 다운로드 성공

- gen_repre 첫 실행 시 DINOv2 ViT-S14-reg 체크포인트(84.2MB) 자동 다운로드됨
- 저장 경로: `C:\Users\PC\.cache\torch\hub\checkpoints\dinov2_vits14_reg4_pretrain.pth`
- 이후 재실행 시 캐시에서 로드

---
