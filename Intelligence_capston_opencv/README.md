# Intelligence_capston_opencv

OpenCV의 영상 입출력, 색공간, histogram, filtering을 작은 독립 스크립트로 실습하는 예제 모음이다. 각 파일이 별도 프로그램이며 통합 진입점은 없다.

## 주요 예제

- `read_image.py`, `color_channels.py`, `hsv_channels.py`: 영상 읽기와 채널 분리
- `histogram_gray.py`, `histogram_bgr.py`: grayscale/BGR histogram 표시
- `histogram_equalization.py`, `histogram_stretching.py`: 명암 대비 보정
- `filter_2d.py`: 평균·sharpen·Laplacian kernel 비교
- `red_extract.py`, `red_candy_extract.py`, `blue_candy_extract.py`: RGB 또는 HSV 범위 기반 사탕 색상 추출
- `camera_capture.py`, `play_video.py`: webcam과 동영상 프레임 표시
- `download_video.py`: KaggleHub의 lane-detection 예제 영상 다운로드

## 실행

Python, OpenCV, NumPy, Matplotlib이 필요하다. `download_video.py`만 `kagglehub`가 추가로 필요하다.

```bash
cd Intelligence_capston_opencv
python3 read_image.py
python3 filter_2d.py
python3 red_candy_extract.py
python3 camera_capture.py

# Kaggle cache에 예제 영상 다운로드
python3 download_video.py
```

정적 영상 예제는 현재 폴더의 `Lenna.png`, `Hawkes.jpg`, `Candies.png`를 상대경로로 읽는다. `download_video.py`는 Kaggle cache 경로만 출력하고 파일을 이 폴더로 복사하지 않는다. 출력된 디렉터리에서 사용할 MP4를 골라 `test_video.mp4`라는 이름으로 현재 폴더에 복사한 다음 `python3 play_video.py`를 실행한다. Webcam 예제는 OpenCV 장치 번호 0을 사용한다.

## 출력과 제한

결과는 OpenCV 또는 Matplotlib 창에 표시되며 파일로 저장하지 않는다. 명령행 인자가 없고 입력 이름·색상 임계값이 코드에 고정돼 있다. GUI가 없는 환경에서는 창 표시 예제가 동작하지 않으며 `q`로 종료하는 영상 예제는 카메라 또는 동영상 decoder가 필요하다.
