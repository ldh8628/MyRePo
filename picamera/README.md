# picamera

Raspberry Pi Camera의 실시간 프레임에서 OpenCV Haar cascade로 얼굴을 찾고, 얼굴 내부에서 눈이 하나 이상 검출된 경우에만 얼굴 영역을 표시하는 하드웨어 데모다.

## 환경과 실행

Raspberry Pi OS의 libcamera stack, Picamera2, OpenCV와 화면 출력 환경이 필요하다. 일반 노트북 webcam용 코드가 아니다.

```bash
cd picamera
python3 haarcascades_eye_face.py
```

스크립트는 Picamera2 preview를 640×480 `RGB888`로 설정하고 OpenCV 설치에 포함된 `haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`을 사용한다. 실행 창에서 `q`를 누르면 종료한다.

## 입출력과 제한

입력은 연결된 Pi Camera의 live frame이며 출력은 검출 문구와 사각형을 그린 화면이다. 영상이나 검출 좌표는 저장하지 않는다. 눈이 검출되지 않은 얼굴을 무시하므로 눈 감김, 모자, 각도와 조명에 민감하며 Haar cascade 결과는 현대적인 얼굴 인식 또는 신원 확인 기능이 아니다.
