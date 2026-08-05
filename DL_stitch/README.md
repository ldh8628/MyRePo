# DL_stitch

이 폴더는 Nie et al.의 **Unsupervised Deep Image Stitching(UDIS)** 공개 구현을 보관한 TensorFlow 1.x 기반 2단계 영상 스티칭 실험이다. 원본 프로젝트와 논문은 [nie-lang/UnsupervisedDeepImageStitching](https://github.com/nie-lang/UnsupervisedDeepImageStitching), [IEEE TIP 논문](https://doi.org/10.1109/TIP.2021.3092828)에서 확인할 수 있다. 이 README는 원문을 복제한 것이 아니라 MyRePo에 실제로 포함된 파일과 실행 제약을 설명한다.

원 구현의 저자는 Lang Nie, Chunyu Lin, Kang Liao, Shuaicheng Liu, Yao Zhao이다. 이 폴더에는 별도 최상위 `LICENSE`가 추적돼 있지 않으므로 재배포·상업적 사용 전에는 반드시 upstream의 최신 이용 조건을 확인해야 한다.

## 포함 내용

- `ImageAlignment/`: homography를 추정하고 두 입력을 coarse alignment하는 1단계 코드
- `ImageReconstruction/`: 1단계의 warp와 mask를 받아 최종 영상을 복원하는 2단계 코드
- `ImageAlignment/Codes/data/testing/input1`, `input2`: 번호가 대응하는 테스트 영상 각 1,105장
- `ImageAlignment/Codes/checkpoints_homo`, `ImageReconstruction/Codes/checkpoints`: checkpoint 메타데이터와 index 파일

원본 구현이 명시한 환경은 Python 3.6, NumPy 1.18.1, TensorFlow 1.13.1, TensorLayer 1.8.0이다. 최신 TensorFlow/Python에서의 호환성은 보장되지 않는다.

## 입력과 실행

상대경로를 사용하므로 각 `Codes` 디렉터리에서 실행한다. GPU 번호와 입력 루트는 먼저 각 단계의 `constant.py`에서 환경에 맞게 수정해야 한다.

현재 Git의 checkpoint에는 TensorFlow 변수값인 `.data-*` shard가 빠져 있으므로 실행 전에 완전한 pretrained checkpoint를 받아야 한다.

- [공식 alignment checkpoint](https://drive.google.com/drive/folders/10SCpFs0J05korpK0sWeSUVGtcpUdPy1e?usp=sharing)를 `ImageAlignment/Codes/checkpoints_homo/`에 배치한다.
- [공식 reconstruction checkpoint](https://drive.google.com/drive/folders/1ogfVFveLjvbnf_XRKAMjG5WaUzxz_zrb?usp=sharing)를 `ImageReconstruction/Codes/checkpoints/`에 배치한다.
- 다른 위치에 둘 경우 각 `inference.py`의 restore 경로를 함께 수정한다.

```bash
cd DL_stitch/ImageAlignment/Codes

# 정량 평가
python inference.py

# 2단계 입력용 warp와 mask 생성
python output_inference.py

cd ../../ImageReconstruction/Codes
python inference.py
```

학습이 필요하면 1단계에서 `python train_H.py`, 2단계에서 `python train.py`를 실행한다. 합성 MS-COCO 입력 생성 절차는 `ImageAlignment/ImageAlignment.md`, 재구성 입력과 VGG19 준비 절차는 `ImageReconstruction/ImageReconstruction.md`를 참고한다.

## 출력

`output_inference.py`는 `ImageAlignment/output/{training,testing}/` 아래에 `warp1`, `warp2`, `mask1`, `mask2` 영상을 만든다. 재구성의 기본 설정은 정확히 `../../ImageAlignment/output/training`과 `../../ImageAlignment/output/testing`을 읽으므로 위 실행 순서와 디렉터리를 유지하면 복사나 symlink 없이 2단계로 연결된다. 실제 출력 경로와 checkpoint 경로는 스크립트에 고정돼 있으므로 실행 위치를 바꾸면 함께 수정해야 한다.

## 확인된 제한

- 포함된 checkpoint에는 `.index`와 `.meta`는 있지만 TensorFlow 변수값을 담는 `.data-*` shard가 없어 현재 Git 내용만으로 pretrained inference를 복원할 수 없다.
- 1단계와 2단계는 하나의 명령으로 실행되지 않지만, 기본 경로를 유지하면 1단계 출력이 2단계 입력으로 바로 사용된다.
- `constant.py`의 GPU 번호와 상대 입력 경로가 하드코딩돼 있다.
- 이 저장소는 포함 파일의 실행 성공이나 논문 성능을 별도로 재현했다고 주장하지 않는다.
