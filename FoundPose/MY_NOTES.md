# MY_NOTES.md — FoundPose 실행 노트

> 작성일: 2026-03-25
> 환경: Windows 11, CUDA 12.6, RTX 3080, Python 3.9 (conda `foundpose`)
> 목적: FoundPose 온보딩(object representation 생성) 단계 검증

---

## 1. 환경 구성

### 가상환경 생성
```bash
conda create -n foundpose python=3.9 -y
```

### PyTorch 설치 (CUDA 12.6 환경에서 cu121 빌드 사용)
```bash
C:/Users/PC/miniconda3/envs/foundpose/python.exe -m pip install \
  torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
```
> ⚠️ YAML 파일의 `pytorch-cuda=11.7.0`은 CUDA 12.x 환경에서 호환 안 됨. cu121 빌드로 대체.

### 나머지 패키지 설치
```bash
# 기본 과학 패키지
C:/Users/PC/miniconda3/envs/foundpose/python.exe -m pip install \
  numpy==1.26.4 scipy==1.13.0 pillow==10.3.0 matplotlib==3.8.4 \
  scikit-learn==1.5.0 scikit-image==0.22.0 kornia==0.7.2 \
  imageio==2.35.1 torchinfo==1.8.0 torchmetrics==0.10.3 \
  omegaconf pycocotools pypng==0.20220715.0 pytz==2024.1 webdataset==0.2.86

# FAISS (conda-forge 사전빌드 사용)
conda install -n foundpose -c conda-forge faiss-gpu=1.8.0 -y

# triangle (conda-forge, MSVC 없이 설치 가능)
conda install -n foundpose -c conda-forge triangle -y

# 렌더링 관련
C:/Users/PC/miniconda3/envs/foundpose/python.exe -m pip install \
  pyopengl==3.1.0 pyglet==2.0.15 pyrender==0.1.45

# OpenCV (bop_toolkit 요구사항 우선)
C:/Users/PC/miniconda3/envs/foundpose/python.exe -m pip install \
  opencv-python==4.11.0.86
```

---

## 2. 외부 의존성 설정

### DINOv2 (external/dinov2/)
레포지토리를 ZIP으로 다운로드했기 때문에 서브모듈이 비어 있음.
이미 PyTorch Hub 캐시에 DINOv2가 있으면 복사:
```bash
cp -r C:/Users/PC/.cache/torch/hub/facebookresearch_dinov2_main/. \
  d:/LDH_ws/foundpose-main/external/dinov2/
```
없으면 직접 클론:
```bash
git clone https://github.com/facebookresearch/dinov2.git \
  d:/LDH_ws/foundpose-main/external/dinov2
```

### bop_toolkit (external/bop_toolkit/)
```bash
git clone https://github.com/thodan/bop_toolkit.git \
  d:/LDH_ws/foundpose-main/external/bop_toolkit

# pip editable 설치
C:/Users/PC/miniconda3/envs/foundpose/python.exe -m pip install \
  -e d:/LDH_ws/foundpose-main/external/bop_toolkit
```

### BOP 데이터셋 (LM-O)
본 검증에서는 이미 다운로드된 데이터셋 사용:
```
D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets/lmo/
├── camera.json
├── models/          ← 3D .ply 파일 (obj_000001.ply ~ obj_000012.ply)
├── models_eval/
└── test/
```
공식 다운로드: https://bop.felk.cvut.cz/datasets/
(lmo_models.zip + lmo_test_all.zip 필요)

### CNOS-FastSAM 탐지 결과 (추론 시 필요)
```
D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets/bop23_default_detections_for_task4/cnos-fastsam/
└── cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json
```
공식 다운로드: https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task4.zip

---

## 3. 환경 변수 설정

실행 시 아래 환경변수가 필요합니다. PYTHONPATH가 Windows Git Bash에서 제대로 전달 안 될 수 있으므로 `run_pipeline.py` 래퍼 스크립트 사용 권장.

```bash
export BOP_PATH="D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets"
export BOP_OUTPUT_PATH="D:/LDH_ws/foundpose-main/outputs"
export PYTHONPATH="D:/LDH_ws/foundpose-main;D:/LDH_ws/foundpose-main/external/bop_toolkit;D:/LDH_ws/foundpose-main/external/dinov2"
```

또는 `run_pipeline.py` 래퍼를 사용:
```bash
C:/Users/PC/miniconda3/envs/foundpose/python.exe run_pipeline.py <script> [args...]
```

conda 환경 활성화 스크립트에 환경변수 영구 등록 방법 (README.md 참조):
```
# 파일 위치: C:/Users/PC/miniconda3/envs/foundpose/etc/conda/activate.d/env_vars.sh
# (없으면 생성)
export REPO_PATH=D:/LDH_ws/foundpose-main
export BOP_PATH=D:/LDH_ws/gigapose_custom/gigaPose_datasets/datasets
export PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2
```

---

## 4. 온보딩 파이프라인 실행 명령어

### STEP 1: 템플릿 렌더링 (gen_templates)
```bash
cd D:/LDH_ws/foundpose-main
C:/Users/PC/miniconda3/envs/foundpose/python.exe run_pipeline.py \
  scripts/gen_templates.py \
  --opts-path configs/gen_templates/lmo.json
```
- 단일 오브젝트 테스트용: `configs/gen_templates/lmo_test1.json` (object_lids: [1])
- 출력: `outputs/templates/v1/lmo/{obj_id}/`
  - rgb/, depth/, mask/ 폴더 (각 798장)
  - metadata.json (카메라 파라미터 포함)

### STEP 2: 오브젝트 표현 생성 (gen_repre)
```bash
C:/Users/PC/miniconda3/envs/foundpose/python.exe run_pipeline.py \
  scripts/gen_repre.py \
  --opts-path configs/gen_repre/lmo.json
```
- 단일 오브젝트 테스트용: `configs/gen_repre/lmo_test1.json`
- 출력: `outputs/object_repre/lmo/v1/{obj_id}/repre.pth`

### STEP 3: 포즈 추론 (inference) — 추론 시 CNOS 탐지 결과 필요
```bash
C:/Users/PC/miniconda3/envs/foundpose/python.exe run_pipeline.py \
  scripts/infer.py \
  --opts-path configs/infer/lmo.json
```
> ⚠️ 추론은 온보딩 이후 단계. BOP 테스트 이미지와 CNOS 탐지 결과 필요.

---

## 5. 검증 결과 (Object 1, LM-O)

| 단계 | 결과 | 소요 시간 |
|------|------|-----------|
| 템플릿 렌더링 (798장) | ✅ 성공 | ~800초 (~1초/장) |
| DINOv2 특징 추출 | ✅ 성공 | ~24초 (798 템플릿) |
| PCA (256차원 축소) | ✅ 성공 | ~0.5초 |
| k-means (2048 클러스터) | ✅ 성공 | ~2.3초 |
| TF-IDF 디스크립터 생성 | ✅ 성공 | ~1.8초 |
| repre.pth 저장 | ✅ 성공 | ~0.8초 |

**생성된 repre.pth 내용**:
- `vertices`: (140266, 3) — 3D 버텍스 위치
- `feat_vectors`: (140266, 256) — PCA 적용된 특징 벡터 (DINOv2 ViT-S14-reg, layer 9, token facet)
- `feat_cluster_centroids`: (2048, 256) — k-means 클러스터 중심 (Visual Words)
- `templates`: (798, 3, 420, 420) — 템플릿 RGB 이미지
- `template_descs`: (798, 2048) — TF-IDF 템플릿 디스크립터
- **파일 크기**: 553MB (논문 기준 234MB보다 큰 이유: 템플릿 이미지 798장 포함)

**DINOv2 모델**:
- 사용 모델: ViT-S/14 with registers (dinov2_vits14_reg4)
- 체크포인트 자동 다운로드: `~/.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth` (84.2MB)
- 논문에서는 ViT-L/14 (더 큰 모델) 사용 가능 — 정확도 향상 예상

---

## 6. GitHub 업로드 시 주의사항

1. **대용량 파일 제외**: `.gitignore`에 `outputs/`, `*.pth`, `data/` 등 추가 완료
2. **서브모듈 처리**:
   - `external/dinov2/`와 `external/bop_toolkit/`는 git submodule 또는 `.gitignore`로 처리
   - 원본 `.gitmodules` 파일이 있으므로 `git submodule` 사용 권장
3. **BOP 데이터셋 비공개**: 라이선스 문제로 데이터셋은 업로드 금지. MY_NOTES.md의 다운로드 경로 안내
4. **라이선스**: CC BY-NC 4.0 — 상업적 사용 금지
5. **환경 파일**: `conda_foundpose_gpu.yaml`에 CUDA 버전 불일치 주석 추가 권장

---

## 7. 전체 LM-O 온보딩 (8개 오브젝트)

`configs/gen_templates/lmo.json`과 `configs/gen_repre/lmo.json`을 사용하면
8개 오브젝트 전체 온보딩 가능 (object_lids: [1, 5, 6, 8, 9, 10, 11, 12]).

예상 소요 시간:
- gen_templates: 8 × ~800초 ≈ 1.8시간
- gen_repre: 8 × ~30초 ≈ 4분
- 총: 약 2시간

---

## 8. 수정된 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `run_pipeline.py` | **신규 생성** — PYTHONPATH 및 BOP 환경변수 자동 주입 래퍼 스크립트 |
| `configs/gen_templates/lmo_test1.json` | **신규 생성** — object 1만 테스트하는 설정 파일 |
| `configs/gen_repre/lmo_test1.json` | **신규 생성** — object 1만 테스트하는 설정 파일 |
| `run_foundpose.bat` | **신규 생성** — Windows 배치 파일 (선택적 사용) |
| `.gitignore` | **신규 생성** — 대용량 파일 및 출력 디렉토리 제외 규칙 |
| `CHANGES.md` | **신규 생성** — 오류 및 수정 기록 |
| `MY_NOTES.md` | **신규 생성** — 이 파일 |
| `debug_depth_fixed.png` | **신규 생성** — Depth 이미지 정규화 시각화 (데이터 정상성 확인용) |
| `dinov2_pca_visualization.png` | **신규 생성** — DINOv2 Layer 18 PCA 시각화 (논문 Fig.3 스타일) |

---

## 9. DINOv2 PCA 시각화 결과

- **사용 모델**: DINOv2 ViT-L/14 (registers 없음, `dinov2_vitl14_pretrain.pth` 캐시 사용)
- **사용 레이어**: Layer 18 (총 24 blocks 중 75% 지점)
- **템플릿 수**: 50장 (798장 중 균등 간격 샘플링, step ~16)
- **패치 수**: 50 × 900 = 45,000패치 (420×420px ÷ 14px = 30×30 grid)
- **PCA 분산 설명량**: PC1=8.0%, PC2=5.3%, PC3=5.0% (전경 패치만 사용)
- **색상 일관성 관찰**:
  - 전경(물체) 패치에만 PCA 적용 후 배경은 검정으로 설정 → 깔끔한 시각화
  - 물체의 각 부위(앞면/옆면/윗면)가 서로 다른 색상으로 구분됨 ✅
  - 같은 물체의 유사한 시점끼리 비슷한 색상 패턴을 보임 ✅
  - 논문 Fig.3과 유사하게 positional + semantic 정보가 혼합 인코딩됨 ✅
- **이미지 저장 경로**: `dinov2_pca_visualization.png`

**Depth 이미지 시각화 결과**:

- 검은 화면은 시각화 도구 문제 (uint16 1.4% 범위)이며 데이터는 올바름
- 정규화 시각화에서 물체 윤곽 및 깊이 그라디언트 정상 확인
- 이미지 저장 경로: `debug_depth_fixed.png`

---

## 10. GitHub 업로드 시 추가 주의사항

- **수정된 파일 목록**: CHANGES.md 기준 (총 8개 신규 생성, 원본 코드 수정 없음)
- **대용량 파일 (.exr, templates/, outputs/)**: `.gitignore`에 포함
- **`debug_depth_fixed.png`**: 커밋 **포함** 권장 (데이터 정상성 증거)
- **`dinov2_pca_visualization.png`**: 커밋 **포함** 권장 (논문 재현 결과물)
- **DINOv2 모델 체크포인트**: torch.hub 캐시에서 자동 다운로드, 레포에 포함하지 않음
- **원본 FoundPose 코드**: 수정하지 않았으므로 수정 주석 불필요
