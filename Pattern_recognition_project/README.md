# Pattern_recognition_project

TensorFlow/Keras로 선형회귀와 이진 logistic regression의 기본 흐름을 학습하는 Jupyter Notebook 모음이다. 각 notebook은 데이터 생성 또는 scikit-learn 내장 dataset 로드부터 학습 곡선과 예측 시각화까지 독립적으로 실행된다.

## Notebook 구성

- `linear_regression.ipynb`: 잡음을 더한 `y = 3x + 2` 합성 데이터의 단일 선형층 회귀
- `linear_regression_california(feat1).ipynb`: California Housing의 중간소득 한 특성으로 주택가격 회귀
- `linear_regression_california(feat8).ipynb`: 8개 특성 표준화, train/test 분할과 다변량 회귀
- `logistic_regression.ipynb`: Iris의 앞 두 특성으로 Setosa 대 나머지를 분류하고 decision boundary 표시

## 실행

Python, Jupyter, TensorFlow, NumPy, Matplotlib, scikit-learn이 필요하다.

```bash
cd Pattern_recognition_project
jupyter lab
```

Jupyter에서 원하는 notebook을 열고 위에서 아래로 모든 cell을 실행한다. California Housing notebook은 `fetch_california_housing()`이 로컬 cache에 없으면 scikit-learn dataset을 내려받기 위한 네트워크가 필요하다.

## 출력과 제한

학습 loss, 가중치, 정확도와 그래프는 notebook cell 출력으로 남고 별도의 모델 파일은 생성하지 않는다. 고정된 실험용 코드이며 dependency version, seed와 재현 환경이 별도로 명세돼 있지 않아 저장된 cell 결과와 재실행 값이 다를 수 있다.
