# ImprovedFoundPose 실행 배포 보고서

이 폴더는 입력 데이터가 별도로 제공됐을 때 FoundPose의
`template 생성 → object representation 생성 → pose 추론 → BOP CSV 생성`
순서를 실행하는 최소 공유본입니다. 데이터, 가중치, 실행 산출물과 연구 중간
코드는 포함하지 않습니다.

## 1. 범위와 라이선스

기준 코드는 FoundPose revision
`3103473bfa3985fdf563bb31cefc0cae8f979ae0`입니다. 라이선스는
`CC BY-NC 4.0`이며 상업적 이용은 허용되지 않습니다. 기업의 평가·제품 적용
전에는 반드시 별도 권리 확인이 필요합니다. 자세한 출처와 변경 내역은
`NOTICE.md`, 정확한 포함/제외 범위는 `SHARE_MANIFEST.md`를 참조하십시오.

포함된 개선은 다음으로 제한됩니다.

1. image/mask 크기가 같을 때 `0:-0` slice로 빈 mask가 되던 오류 수정
2. 최신 Matplotlib의 `buffer_rgba()` 호환
3. 입력·출력 루트를 CLI 또는 환경변수로 받는 portable runner
4. GT 없는 추론을 위한 opt-in `require_gt=false`
5. BOP CSV 및 PLY 변환 도구의 하드코딩 경로 제거
6. representation 버전 선택, PnP 실패, 빈 detection mask의 안전 처리

Top-K 연구, projection reranking, 외부 template candidate, OnePose audit,
3DGS/GS2Mesh 파이프라인은 포함하지 않았습니다.

## 2. 설치

Linux와 NVIDIA GPU를 기준으로 합니다. Python 3.9 및 Git이 필요합니다.

```bash
cd ImprovedFoundPose
conda env create -f conda_foundpose_gpu.yaml
conda activate foundpose_gpu
bash scripts/setup_external.sh
python -m pip install -e external/bop_toolkit
```

설치 스크립트는 BOP Toolkit과 DINOv2를 `NOTICE.md`에 기록된 revision으로
checkout합니다. 가중치는 Git에 포함하지 않으며 DINOv2가 최초 실행 시
사용자 캐시로 다운로드합니다. 네트워크가 차단된 환경에서는 수신자가 동일
revision의 코드와 가중치를 승인된 내부 저장소로 별도 제공해야 합니다.
배포 환경은 PyTorch 2.3, torchvision 0.18, CUDA runtime 12.1 조합이며,
DINOv2는 PyTorch 기본 attention 경로를 사용하도록 xFormers를 비활성화합니다.

## 3. 입력 계약

LM-O 예제에서 `<BOP_PATH>`는 아래 구조를 가져야 합니다.

```text
<BOP_PATH>/
├── lmo/
│   ├── camera.json
│   ├── models/
│   │   ├── models_info.json
│   │   └── obj_000001.ply ...
│   ├── test_targets_bop19.json
│   └── test/<scene-id>/
│       ├── rgb/
│       ├── depth/
│       ├── scene_camera.json
│       ├── scene_gt.json
│       ├── scene_gt_info.json
│       └── mask_visib/
└── detections/cnos-fastsam/
    └── cnos-fastsam_lmo-test.json
```

기본 `configs/infer/lmo.json`은 평가 가능한 BOP 입력을 위해
`require_gt=true`입니다. GT가 없는 실제 입력은 복사한 config에서
`require_gt=false`로 바꾸면 `scene_gt.json`, `scene_gt_info.json`,
`mask_visib/`가 없어도 됩니다. 이 경우에도 현재 로더 계약상 RGB, depth,
scene camera, test targets, CNOS-FastSAM 탐지 JSON은 필요합니다.

입력과 출력은 저장소 밖에 두는 것을 권장합니다.

```bash
export BOP_PATH=/absolute/path/to/bop
export BOP_OUTPUT_PATH=/absolute/path/to/foundpose-output
```

## 4. 실제 실행 순서와 단계별 결과

모든 명령은 공유본 루트에서 실행합니다. 환경변수 대신 각 명령에
`--bop-path`와 `--output-path`를 전달할 수도 있습니다.

### 단계 1: 객체 template 생성

```bash
python run_pipeline.py gen_templates -- \
  --opts-path configs/gen_templates/lmo.json
```

결과: `<BOP_OUTPUT_PATH>/templates/v1/lmo/<obj-id>/`에 798-view RGB, depth,
mask와 `metadata.json`이 생성됩니다.

### 단계 2: object representation 생성

```bash
python run_pipeline.py gen_repre -- \
  --opts-path configs/gen_repre/lmo.json
```

결과: `<BOP_OUTPUT_PATH>/object_repre/lmo/v1/<obj-id>/repre.pth`에 DINOv2
feature, PCA, visual words, TF-IDF descriptor가 저장됩니다. 이 파일은
대용량 산출물이므로 Git에 추가하지 않습니다.

### 단계 3: pose 추론

```bash
python run_pipeline.py infer -- \
  --opts-path configs/infer/lmo.json
```

결과:
`<BOP_OUTPUT_PATH>/inference/lmo_v1/<obj-id>/estimated-poses.json`과 설정에
따른 시각화가 생성됩니다. GT가 없는 실행에서는 pose estimate는 저장되지만
GT 기반 평가 지표는 생성되지 않습니다.

### 단계 4: BOP submission CSV 생성

```bash
python run_pipeline.py prepare_bop_submission -- \
  --dataset lmo --version v1
```

결과:
`<BOP_OUTPUT_PATH>/inference/lmo_v1/coarse_lmo-estimated-poses.csv`가
생성됩니다.

## 5. 선택적 PLY 온보딩

재구성 PLY를 millimetre 단위로 scale하고 bounding-box 중심으로 옮기려면
모든 경로를 명시하여 실행합니다.

```bash
python -m pip install open3d
python tools/convert_ply_gs2mesh_to_foundpose.py \
  --source-ply /path/to/input.ply \
  --output-dir /path/to/bop-dataset/models \
  --obj-id 1 \
  --scale-to-mm 1000
```

도구는 `obj_000001.ply`, `models_info.json`, transform JSON을 만듭니다.
scale과 center만 처리하며 canonical rotation이나 원본 camera 좌표계와의
정렬은 보장하지 않습니다. 입력 좌표계를 사용자가 별도로 검증해야 합니다.

## 6. 검증 보고서

공유 직전 아래 명령을 실행합니다.

```bash
bash scripts/validate_release.sh
python -m pytest -q
python -m compileall -q scripts tools utils tests
```

이번 공유본에서 수행한 검증 결과는 다음과 같습니다.

| 검증 | 결과 |
| --- | --- |
| Python compile | PASS |
| 단위/정적 테스트 | PASS — 17개 |
| 내부 절대경로·secret·금지 파일·100MB 파일 | PASS — 검출 0개 |
| 공유본 크기 | PASS — 528 KiB |
| Git 추적 파일 검사 | PASS — `ImprovedFoundPose/` 아래 파일만 추적 |
| 실제 GPU 전체 파이프라인 | 이번 무입력 패키징 환경에서는 미실행 |

기존 작업 기록의 LM-O Object 1 기준 측정치는 template 약 800초, feature
추출 약 24초, PCA/k-means/TF-IDF 약 5초, 총 onboarding 약 14분입니다.
이는 기존 환경의 참고값이며 이번 최소 패키지에서 다시 benchmark한 결과가
아닙니다.

## 7. 제한사항

- 공식 FoundPose와 동일하게 3D mesh, camera 정보, 탐지 mask의 품질에
  의존합니다.
- GPU/driver/렌더링 backend 차이에 따른 전체 재현성은 정적 테스트만으로
  보장할 수 없습니다.
- 외부 dependency 설치와 DINOv2 weight 다운로드가 필요합니다.
- `require_gt=false`는 GT 기반 필터와 평가를 생략하므로 정확도 검증을
  별도로 수행해야 합니다.
- PLY 변환은 좌표축 회전을 추정하지 않습니다.
- 이 배포본은 연구 단계 후보 기법의 성능 개선을 주장하지 않습니다.
