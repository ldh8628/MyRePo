# ViT_AutoEncoder

금속 표면 이상 탐지를 비교하기 위한 PyTorch 실험 모음이다. ViT AutoEncoder와 PatchCore baseline, random/CycleGAN anomaly를 사용하는 DRAEM 변형, 평가·시각화 스크립트를 함께 보관한다.

## 주요 파일

- `vit_ae.py`: reconstruction-error 기반 ViT AutoEncoder
- `Patchcore.py`: pretrained CNN feature bank와 nearest-neighbor 기반 PatchCore 실험
- `cyclegan_train_infer.py`: normal↔defect CycleGAN 학습과 합성 defect 생성
- `draem_basic.py`: random synthetic anomaly 기반 기본 DRAEM
- `draem_hybrid_gan.py`, `draem_with_gan.py`: CycleGAN과 random anomaly를 조합한 변형
- `draem_hybrid_split.py`: train/validation/test를 명시적으로 분리하는 통합 진입점
- `draem_eval_*.py`: checkpoint의 image-level score, threshold와 confusion matrix 평가

`ros2_image_subscribe/`는 이 이상 탐지 코드와 직접 연결되지 않은 멀티카메라 파노라마 실험의 별도 snapshot이다.

## 입력과 대표 실행

`draem_hybrid_split.py`는 다음 구조를 기대한다.

```text
train_data/
├── train/normal/
├── validation/{normal,defect}/
└── test/{normal,defect}/
```

Python과 PyTorch, torchvision, OpenCV, Pillow, NumPy, scikit-learn, Matplotlib, tqdm이 필요하다.

```bash
cd ViT_AutoEncoder

python3 draem_hybrid_split.py \
  --data_root ./train_data \
  --train

python3 draem_hybrid_split.py \
  --data_root ./train_data \
  --eval_val --eval_test
```

기본 GAN checkpoint는 `G_AB.pth`이며 찾지 못하면 이 스크립트는 random anomaly 경로를 사용한다. 파일은 현재 Git에 포함돼 있지만 training provenance와 checksum은 문서화돼 있지 않다.

## 출력과 제한

학습 checkpoint는 `ckpts_draem_hybrid_split/`, 평가 metric과 image별 score CSV는 기본적으로 `eval_split/`에 생성된다. 학습에는 최소 한 batch를 만들 만큼의 정상 영상이 필요하고, class별 평가에는 normal·defect가 모두 비어 있지 않아야 한다. 저장소에는 `eval_draem_*`, `eval_threshold`, `eval_val_test`의 과거 결과 파일과 일부 ViT/CycleGAN 가중치가 있지만, 원본 `train_data/`와 해당 DRAEM checkpoint는 포함돼 있지 않다. 따라서 기록된 평가값을 현재 Git 내용만으로 재현할 수 없으며 데이터 분할과 checkpoint provenance를 확인하지 않은 채 성능 주장에 사용하면 안 된다.
