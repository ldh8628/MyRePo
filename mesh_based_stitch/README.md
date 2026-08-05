# mesh_based_stitch

겹치는 두 영상을 전역 homography와 지역 mesh displacement로 정합하는 OpenCV 파노라마 실험이다. 가장 완전한 경로는 cylindrical projection, exposure compensation, seam finding, residual mesh warp와 MultiBand blending을 순서대로 수행한다.

## 파일과 실행

Python, OpenCV와 NumPy가 필요하다. `mesh_based_stitch_debug.py`는 SIFT와 OpenCV stitching detail API를 포함한 OpenCV build를 요구한다.

```bash
cd mesh_based_stitch

# 권장: 단계별 시각화를 포함한 전체 파이프라인
python3 mesh_based_stitch_debug.py

# 전역 homography + 기본/적응형 mesh 비교
python3 mesh_based_stitch.py

# 더 단순한 mesh 구현
python3 mesh_based_stitch2.py

# PyTorch/Kornia homography 학습 실험(기본 1,200 step)
python3 DL_stitch.py
```

네 스크립트 모두 `images/left.jpg`와 `images/right.jpg`를 입력으로 사용한다. `DL_stitch.py`는 PyTorch와 Kornia로 synthetic homography 학습 후 정합하는 별도 실험이며 실행할 때 기본 1,200 step을 다시 학습한다. 비슷한 이름의 `DL_based_stitch.py`는 미완성 파일이므로 실행 진입점으로 사용하지 않는다.

## 출력

- 전체 파이프라인: `result/stitch_cyl_mesh_multiband.jpg`
- 기본/적응형 비교: `result/stitch_mesh_basic.jpg`, `result/stitch_mesh_adaptive.jpg`
- 단계별 진단: `result/debug_cyl/`, `result/debug_mesh/`, `result/debug_blend/`
- PyTorch 실험: `pano.png`, `alignedB.png`, `ckpts/homography_net.pth`

저장소에는 위 입력과 대표 결과·중간 시각화가 함께 들어 있다.

## 제한

입력 경로와 parameter가 코드에 고정돼 있고 CLI는 없다. 두 영상에 충분한 overlap과 반복되지 않는 특징점이 없으면 SIFT matching 또는 homography 추정이 실패한다. `DL_based_stitch.py`는 영상을 읽고 device를 선택하는 부분에서 끝나는 미완성 파일이므로 실행 결과를 만들지 않는다.
