# opencv_number

OpenCV 창의 다섯 입력칸에 마우스로 쓴 숫자를 MNIST용 3층 fully-connected network로 분류하는 GUI 데모다. 추론은 NumPy로 직접 구현돼 있으며 별도 deep-learning runtime을 사용하지 않는다.

## 포함 자산

- `number_predict.py`: 5칸 drawing UI, MNIST 전처리와 추론
- `sample_weight.pkl`: `W1~W3`, `b1~b3`를 담은 학습 가중치
- `dataset/mnist.py`, `dataset/mnist.pkl`: MNIST loader와 로컬 dataset cache
- `common/`: sigmoid, softmax와 신경망 학습 유틸리티

## 실행과 조작

Python, NumPy와 OpenCV GUI가 필요하다.

> `sample_weight.pkl`과 `dataset/mnist.pkl`은 역직렬화 시 임의 코드를 실행할 수 있는 pickle 파일이다. 신뢰할 수 있는 checkout의 파일만 사용해야 하며, 이 폴더에는 해당 파일의 원본 출처·license·checksum이 문서화돼 있지 않다.

```bash
cd opencv_number
python3 number_predict.py
```

각 검은 칸에서 왼쪽 버튼을 끌어 흰색 숫자를 그리고 오른쪽 버튼으로 해당 칸을 지운다. `Recognize`를 눌러 다섯 자리 결과를 표시하고 `Clear All`로 전체를 초기화한다. `Esc` 또는 `q`로 종료한다.

## 입출력과 제한

시작 시 로컬 MNIST test set으로 전체 batch 정확도를 계산해 표준 출력하고, 사용자가 그린 획은 28×28로 crop·중앙 정렬한 뒤 추론한다. 최고 확률이 0.55보다 낮은 칸은 `-`로 표시한다. 결과와 입력 획은 파일로 저장하지 않는다. 디스플레이가 없는 환경에서는 사용할 수 없고, MNIST와 크게 다른 필체·획 두께에 대한 성능은 보장되지 않는다.
