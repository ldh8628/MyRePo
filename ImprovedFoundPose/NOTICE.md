# 출처 및 제3자 고지

이 배포본은 Meta Platforms, Inc.의
[FoundPose](https://github.com/facebookresearch/foundpose)를 기반으로 하며,
기준 revision은 `3103473bfa3985fdf563bb31cefc0cae8f979ae0`입니다.
원본 저작권 고지와 `CC BY-NC 4.0` 전문은 `LICENSE`에 보존했습니다.

## 이용 조건

`CC BY-NC 4.0`은 비상업적 이용만 허용합니다. 기업 간 공유가 곧 상업적
이용 허가를 의미하지 않습니다. 평가·제품 적용·고객 제공 전에 권리자 허가와
법무 검토가 필요합니다.

## 이 배포본의 변경

- 탐지 mask가 이미지와 같은 크기일 때 `0:-0` slice로 비는 문제 수정
- 최신 Matplotlib에서 `buffer_rgba()`를 사용하는 호환 경로 추가
- 로컬 절대경로 없이 입력·출력 루트를 받는 `run_pipeline.py` 추가
- ground truth가 없는 입력을 opt-in으로 처리하는 `require_gt` 옵션 추가
- GT-free 시각화, PnP 실패, 빈 mask와 representation 버전 선택 안전 처리
- BOP submission 생성기의 데이터셋·버전·출력 경로 하드코딩 제거
- PLY 변환기의 모든 입력/출력 경로를 필수 CLI 인자로 변경

## 외부 의존성

외부 코드는 이 폴더에 포함하지 않습니다. `scripts/setup_external.sh`가 아래
revision을 각각 원 저작권자의 저장소에서 설치하며, 설치된 각 저장소의
라이선스가 별도로 적용됩니다.

- `thodan/bop_toolkit`: `e7ba9f2389d48d721856df8147fb133c0145e99c`
- `facebookresearch/dinov2`: `e1277af2ba9496fbadf7aec6eba56e8d882d1e35`

DINOv2 모델 가중치도 포함하지 않으며 최초 실행 시 해당 프로젝트의 다운로드
및 라이선스 조건을 따릅니다.
