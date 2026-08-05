# Pattern Recognition

정상 금속 표면만으로 ViT AutoEncoder를 학습하고 reconstruction error를 이용해 불량 후보를 찾는 PyTorch 실험이다. 구현은 `vit_ae.py` 한 파일에 dataset 분할, 학습, threshold 추정, 평가와 heatmap 저장까지 포함한다.

## 입력

스크립트는 다음 구조를 기대한다.

```text
data/
├── normal/
└── defect/
```

현재 Git tree에는 `data/normal/` 영상 75장만 있고 `data/defect/`는 없다. 정상 영상만으로 reconstruction 학습은 가능하지만 의미 있는 정상/불량 metric을 얻으려면 두 class에 Pillow가 읽을 수 있는 영상을 각각 준비해야 한다.

## 실행

Python과 PyTorch, torchvision, OpenCV, Pillow, NumPy, Matplotlib, scikit-learn, tqdm이 필요하다.

```bash
cd "Pattern Recognition"
python3 vit_ae.py
```

기본 설정은 256×256 입력, patch 16, 50 epochs, batch size 16이다. CUDA가 있으면 자동으로 사용하며 `outputs_vit_ae/vit_ae.pth`가 이미 있으면 학습 대신 해당 가중치를 읽는다.

기존 `outputs_vit_ae/vit_ae.pth`가 손상됐거나 현재 코드와 호환되지 않으면 load 단계에서 실패한다. 이 경우 파일을 다른 곳으로 이동하거나 이름을 바꾼 뒤 다시 실행해 새 checkpoint를 학습한다.

## 출력과 제한

- 최적 validation 모델: `outputs_vit_ae/vit_ae.pth`
- 선별된 anomaly heatmap overlay: `outputs_vit_ae/vis/`
- ACC, precision, recall, F1: 표준 출력

데이터 경로와 hyperparameter는 `CFG`에 고정돼 있고 명령행 인자는 없다. 현재 포함 데이터만 사용하면 defect score 분포와 defect 성능을 평가할 수 없으므로 출력 수치를 일반적인 불량 검출 성능으로 해석하면 안 된다.
