# foundpose_custom

> FoundPose 논문(arXiv:2311.18809) 기반의 커스텀 구현입니다.
> 원본 저자: Evin Pınar Örnek et al. (Meta Reality Labs / TU Munich)
> 원본 레포: <https://github.com/facebookresearch/foundpose>
> 이 레포는 원본을 기반으로 **Windows + CUDA 12.6 환경의 세팅 오류를 수정**하고 실행 가이드를 보완한 개인 포크입니다.

---

## 목차

- [개요](#개요)
- [사전 요구사항](#사전-요구사항)
- [설치](#설치)
- [데이터 준비](#데이터-준비)
- [온보딩 실행](#온보딩-실행)
- [결과 확인](#결과-확인)
- [수정 사항](#수정-사항)
- [참고](#참고)

---

## 개요

FoundPose는 DINOv2 foundation model을 활용한 **학습 없는(training-free) 6D 물체 포즈 추정** 방법입니다.
3D 메시 모델(.ply)만 있으면 새로운 물체를 빠르게 온보딩하고, 단일 RGB 이미지에서 포즈를 추정할 수 있습니다.

**핵심 파이프라인:**

1. **템플릿 렌더링** (`gen_templates.py`): 3D 모델에서 SO(3) 균일 샘플링으로 798장의 RGB-D 이미지 렌더링
2. **오브젝트 표현 생성** (`gen_repre.py`): DINOv2 ViT-S/14-reg (layer 9)로 패치 디스크립터 추출 → PCA 256차원 축소 → k-means 2048 클러스터 BoW → TF-IDF 디스크립터
3. **포즈 추정** (`infer.py`): TF-IDF 유사도로 상위 5개 템플릿 검색 → Cyclic buddies 특징 매칭 → PnP-RANSAC (OpenCV) 포즈 추정

검증 환경 (LM-O, Object 1):

| 단계 | 소요 시간 |
| --- | --- |
| 템플릿 렌더링 (798장) | ~800초 (~1초/장) |
| DINOv2 특징 추출 | ~24초 |
| PCA + k-means + TF-IDF | ~5초 |
| **총 온보딩** | **~14분/오브젝트** |

---

## 사전 요구사항

- **OS**: Windows 10/11 또는 Ubuntu 20.04 이상
- **GPU**: CUDA 지원 NVIDIA GPU (RTX 3080 이상 권장, VRAM 10GB+)
- **CUDA**: 12.x (드라이버 기준; PyTorch cu121 빌드로 호환)
- **Python**: 3.9
- **Conda**: Miniconda 또는 Anaconda
- **Git**: 서브모듈 클론용

---

## 설치

### 1. 레포지토리 클론

```bash
git clone https://github.com/ldh8628/foundpose_custom.git
cd foundpose_custom
```

### 2. bop_toolkit 클론 (필수)

`external/bop_toolkit/`은 서브모듈이므로 별도로 클론합니다:

```bash
git clone https://github.com/thodan/bop_toolkit.git external/bop_toolkit
```

### 3. Conda 가상환경 생성

```bash
conda create -n foundpose python=3.9 -y
```

### 4. PyTorch 설치 (CUDA 12.x 환경)

> ⚠️ `conda_foundpose_gpu.yaml`의 `pytorch-cuda=11.7.0`은 CUDA 12.x와 호환되지 않습니다.
> cu121 빌드를 직접 설치합니다.

Windows:
```bash
C:/Users/<USER>/miniconda3/envs/foundpose/python.exe -m pip install \
  torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

Linux/Mac:
```bash
conda activate foundpose
pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 5. 나머지 의존성 설치

Windows:
```bash
# 기본 과학 패키지 및 FoundPose 의존성
C:/Users/<USER>/miniconda3/envs/foundpose/python.exe -m pip install \
  numpy==1.26.4 scipy==1.13.0 pillow==10.3.0 matplotlib==3.8.4 \
  scikit-learn==1.5.0 scikit-image==0.22.0 kornia==0.7.2 \
  imageio==2.35.1 torchinfo==1.8.0 torchmetrics==0.10.3 \
  omegaconf pycocotools pypng==0.20220715.0 pytz==2024.1 webdataset==0.2.86

# FAISS GPU (conda-forge 사전빌드 사용)
conda install -n foundpose -c conda-forge faiss-gpu=1.8.0 -y

# triangle (MSVC 없이 설치 — conda-forge 사전빌드)
conda install -n foundpose -c conda-forge triangle -y

# 렌더링
C:/Users/<USER>/miniconda3/envs/foundpose/python.exe -m pip install \
  pyopengl==3.1.0 pyglet==2.0.15 pyrender==0.1.45

# OpenCV (bop_toolkit 요구사항 우선)
C:/Users/<USER>/miniconda3/envs/foundpose/python.exe -m pip install \
  opencv-python==4.11.0.86

# bop_toolkit editable 설치
C:/Users/<USER>/miniconda3/envs/foundpose/python.exe -m pip install \
  -e external/bop_toolkit
```

Linux/Mac (`conda activate foundpose` 후):
```bash
pip install numpy==1.26.4 scipy==1.13.0 pillow==10.3.0 matplotlib==3.8.4 \
  scikit-learn==1.5.0 scikit-image==0.22.0 kornia==0.7.2 \
  imageio==2.35.1 torchinfo==1.8.0 torchmetrics==0.10.3 \
  omegaconf pycocotools pypng==0.20220715.0 pytz==2024.1 webdataset==0.2.86
conda install -c conda-forge faiss-gpu=1.8.0 triangle -y
pip install pyopengl==3.1.0 pyglet==2.0.15 pyrender==0.1.45 opencv-python==4.11.0.86
pip install -e external/bop_toolkit
```

### 6. DINOv2 코드

이 레포에는 `external/dinov2/`가 이미 포함되어 있습니다 (코드만, 모델 가중치 제외).
모델 가중치는 첫 실행 시 `torch.hub`를 통해 자동으로 다운로드됩니다.

- 캐시 경로: `~/.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth` (84.2MB)

---

## 데이터 준비

### BOP 데이터셋 (LM-O 예시)

[BOP 공식 사이트](https://bop.felk.cvut.cz/datasets/)에서 다운로드합니다:

- `lmo_models.zip` — 3D 모델
- `lmo_test_all.zip` — 테스트 이미지 (포즈 추정 시 필요)

다운로드 후 아래 구조로 배치합니다:

```
<BOP_PATH>/
└── lmo/
    ├── camera.json
    ├── models/               ← obj_000001.ply ~ obj_000012.ply
    ├── models_eval/
    └── test/
```

### CNOS-FastSAM 탐지 결과 (포즈 추정 시 필요)

[BOP 2023 기본 탐지 결과](https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task4.zip) 다운로드 후:

```
<BOP_PATH>/bop23_default_detections_for_task4/cnos-fastsam/
└── cnos-fastsam_lmo-test_<hash>.json
```

### 환경 변수 설정

#### Linux/Mac — conda 활성화 스크립트에 영구 등록

```bash
# 파일: $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export REPO_PATH=/path/to/foundpose_custom
export BOP_PATH=/path/to/bop/datasets
export BOP_OUTPUT_PATH=$REPO_PATH/outputs
export PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2
```

#### Windows — `run_pipeline.py` 래퍼 사용 (권장)

Windows Git Bash에서는 `export PYTHONPATH=...`이 Python 프로세스에 제대로 전달되지 않습니다.
`run_pipeline.py`가 자동으로 경로를 주입합니다. **사용 전 경로를 본인 환경에 맞게 수정하세요:**

```python
# run_pipeline.py 상단 (수정 필요)
REPO = "D:/path/to/foundpose_custom"
os.environ.setdefault("BOP_PATH", "D:/path/to/bop/datasets")
os.environ.setdefault("BOP_OUTPUT_PATH", "D:/path/to/foundpose_custom/outputs")
```

---

## 온보딩 실행

> **Windows 사용자**: 아래 Linux/Mac 명령의 `python scripts/...` 대신
> `C:/Users/<USER>/miniconda3/envs/foundpose/python.exe run_pipeline.py scripts/...` 를 사용합니다.

### STEP 1: 템플릿 렌더링

```bash
# Linux/Mac
conda activate foundpose
python scripts/gen_templates.py --opts-path configs/gen_templates/lmo.json

# Windows
C:/Users/<USER>/miniconda3/envs/foundpose/python.exe run_pipeline.py \
  scripts/gen_templates.py \
  --opts-path configs/gen_templates/lmo.json
```

설정 파일: `configs/gen_templates/lmo.json`

- `object_lids`: `[1, 5, 6, 8, 9, 10, 11, 12]` — 처리할 LM-O 오브젝트 ID 목록
- `min_num_viewpoints`: `57`, `num_inplane_rotations`: `14` → 총 798장 렌더링
- `crop_size`: `[420, 420]` px

단일 오브젝트 테스트: `--opts-path configs/gen_templates/lmo_test1.json` (`object_lids: [1]`)

**출력**: `outputs/templates/v1/lmo/{obj_id}/`

- `rgb/`, `depth/`, `mask/` 폴더 (각 798장)
- `metadata.json` (카메라 파라미터 포함)

> ⚠️ depth PNG는 uint16 형식(BOP 표준)으로 저장되어 일반 뷰어에서 검게 보입니다. 데이터는 정상입니다.

---

### STEP 2: 오브젝트 표현 생성

```bash
# Linux/Mac
python scripts/gen_repre.py --opts-path configs/gen_repre/lmo.json

# Windows
C:/Users/<USER>/miniconda3/envs/foundpose/python.exe run_pipeline.py \
  scripts/gen_repre.py \
  --opts-path configs/gen_repre/lmo.json
```

설정 파일: `configs/gen_repre/lmo.json`

- `extractor_name`: `"dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1"`
- `pca_components`: `256`
- `cluster_num`: `2048` (Visual Words)

단일 오브젝트 테스트: `--opts-path configs/gen_repre/lmo_test1.json`

**출력**: `outputs/object_repre/lmo/v1/{obj_id}/repre.pth`

`repre.pth` 구성 (Object 1 기준):

- `vertices`: `(140266, 3)` — 3D 버텍스 위치
- `feat_vectors`: `(140266, 256)` — PCA 축소 특징 벡터
- `feat_cluster_centroids`: `(2048, 256)` — k-means 클러스터 중심
- `templates`: `(798, 3, 420, 420)` — 템플릿 RGB 이미지
- `template_descs`: `(798, 2048)` — TF-IDF 디스크립터
- **파일 크기**: ~553MB

---

### STEP 3: 포즈 추정 (inference)

> ⚠️ CNOS-FastSAM 탐지 결과가 필요합니다. 데이터 준비 섹션 참고.

```bash
# Linux/Mac
python scripts/infer.py --opts-path configs/infer/lmo.json

# Windows
C:/Users/<USER>/miniconda3/envs/foundpose/python.exe run_pipeline.py \
  scripts/infer.py \
  --opts-path configs/infer/lmo.json
```

설정 파일: `configs/infer/lmo.json`

- `match_top_n_templates`: `5` — 상위 N개 템플릿 검색
- `match_feat_matching_type`: `"cyclic_buddies"`
- `match_top_k_buddies`: `300`
- `pnp_type`: `"opencv"`, `pnp_ransac_iter`: `400`

**출력**: `outputs/infer/v1/lmo/` (BOP 포맷 JSON)

---

## 결과 확인

### LM-O 벤치마크 결과 (원본 논문 기준)

| Dataset | Published AR | Reproduced AR |
| ------- | ------------ | ------------- |
| LMO     | 34.0         | 33.7          |
| TUD-L   | 42.7         | 40.7          |

### DINOv2 패치 디스크립터 시각화

`dinov2_pca_visualization.png` (이 레포에 포함):

- ViT-L/14 layer 18 특징, 전경 패치에만 PCA 적용
- 50장 템플릿 × (RGB | PCA 색상) 그리드
- PC1=8.0%, PC2=5.3%, PC3=5.0% 분산 설명량
- 같은 물체 부위끼리 동일한 색상 → positional + semantic 정보 인코딩 확인

### Depth 렌더링 확인

`debug_depth_fixed.png` (이 레포에 포함):

- uint16 raw → plasma colormap 정규화 시각화
- 물체 윤곽 및 깊이 그라디언트 정상 확인

---

## 수정 사항

원본 레포 대비 변경 내역 요약 (상세 내용: [`CHANGES.md`](CHANGES.md)):

- **pip 경로 오류**: `conda run pip` → conda 환경 Python 직접 호출로 수정
- **CUDA 버전 불일치**: yaml의 `pytorch-cuda=11.7.0` → cu121 빌드로 대체 (CUDA 12.x 호환)
- **triangle 패키지**: MSVC 미설치 시 빌드 실패 → conda-forge 사전빌드로 대체
- **bop_toolkit 의존성 충돌**: opencv, imageio, numpy 버전을 두 요구사항을 모두 만족하는 버전으로 고정
- **서브모듈 미초기화**: ZIP 다운로드 시 submodule 비어있는 문제 → 수동 클론/복사 가이드 추가
- **PYTHONPATH 미적용**: Windows Git Bash에서 환경변수 미전달 → `run_pipeline.py` 래퍼 생성
- **EGL 백엔드 오류**: Windows는 EGL 미지원 → `PYOPENGL_PLATFORM` 미설정으로 WGL 기본값 사용
- **신규 설정 파일**: `configs/gen_templates/lmo_test1.json`, `configs/gen_repre/lmo_test1.json` (단일 오브젝트 테스트용)

---

## 참고

- 논문: [FoundPose: Unseen Object Pose Estimation with Foundation Features (ECCV 2024)](https://arxiv.org/abs/2311.18809)
- 원본 코드: [github.com/facebookresearch/foundpose](https://github.com/facebookresearch/foundpose)
- BOP 벤치마크: [bop.felk.cvut.cz](https://bop.felk.cvut.cz)
- DINOv2: [github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- bop_toolkit: [github.com/thodan/bop_toolkit](https://github.com/thodan/bop_toolkit)

---

## 라이선스

이 프로젝트는 [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/) 라이선스를 따릅니다.
상업적 사용이 금지됩니다.
