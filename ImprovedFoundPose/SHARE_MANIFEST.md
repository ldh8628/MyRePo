# 공유 매니페스트

## 포함 파일

| 범주 | 파일 |
| --- | --- |
| 실행 | `run_pipeline.py` |
| 핵심 단계 | `scripts/gen_templates.py`, `scripts/gen_repre.py`, `scripts/infer.py`, `scripts/prepare_bop_submission.py` |
| 설치/검증 | `scripts/setup_external.sh`, `scripts/validate_release.sh`, `conda_foundpose_gpu.yaml` |
| 설정 | `configs/gen_templates/lmo.json`, `configs/gen_repre/lmo.json`, `configs/infer/lmo.json` |
| 런타임 유틸 | `utils/*.py` 28개 |
| 선택 도구 | `tools/convert_ply_gs2mesh_to_foundpose.py` |
| 테스트 | `tests/test_*.py` |
| 문서/정책 | `README.md`, `NOTICE.md`, `SHARE_MANIFEST.md`, `LICENSE`, `.gitignore` |

`utils/`는 네 핵심 단계가 직접 또는 간접 import하는 upstream 폐쇄 집합입니다.
외부 라이브러리는 용량과 라이선스 경계를 분명히 하기 위해 vendoring하지 않고
고정 revision 설치 스크립트로 제공합니다.

## 포함하지 않은 입력

- BOP 데이터셋, RGB/depth 영상, camera/GT/탐지 JSON
- 객체 PLY/OBJ/STL
- DINOv2 및 기타 모델 가중치·체크포인트

## 포함하지 않은 산출물

- 렌더링 template
- `repre.pth`를 포함한 object representation
- 추론 JSON·시각화·PLY·BOP CSV
- 캐시, 로그, HTML, 이미지, 실험 보고서

## 의도적으로 제외한 연구 코드

- `scripts/research`, `configs/research`, `src/select_topk`, `research_pose`
- Top-K hypothesis export, projection-sanity reranking, 외부 candidate JSON 훅
- OnePose 전용 audit와 개인/사내 절대경로
- 3DGS 및 GS2Mesh 저장소 복제본과 그 학습·평가 산출물
- `_bmad*`, agent 설정, 내부 연구 문서

위 연구 항목은 FoundPose 기본 성능 개선으로 채택·검증된 기능이 아니므로
이 공유본의 기능 또는 성능 개선으로 주장하지 않습니다.
