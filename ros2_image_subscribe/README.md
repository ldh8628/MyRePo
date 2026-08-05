# ros2_image_subscribe

ROS 2의 다중 압축 카메라 영상을 JPEG로 수집하고 cylindrical panorama로 합성하며, 선택적으로 YOLOv8 또는 YOLO-NAS 객체 탐지·간단한 tracking을 적용하는 prototype 모음이다.

## 권장 offline 흐름

ROS 2 `rclpy`, `sensor_msgs`, OpenCV와 NumPy가 필요하다. 관련 스크립트는 `ROS_DOMAIN_ID=20`과 `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`를 내부에서 설정하므로 별도 `export`는 필요 없다. 다른 값을 쓰려면 각 스크립트의 설정을 수정한다.

```bash
source /opt/ros/humble/setup.bash
cd ros2_image_subscribe

# /cam0 ~ /cam5에서 각 100장을 순번 파일명으로 저장
python3 ros2_image_sucscribe2.py

# 여섯 카메라의 공통 frame을 panorama로 합성
python3 create_panorama.py
```

입력 topic은 `/cam0/image_raw/compressed`부터 `/cam5/image_raw/compressed`까지이며 message type은 `sensor_msgs/msg/CompressedImage`, QoS reliability는 BEST_EFFORT다. 캡처 결과는 `frames_compressed/camX/camX_fNNNN.jpg`, 합성 결과는 `frames_compressed/panorama/pano_fNNNN.jpg`에 저장된다.

## 탐지·실시간 변형

```bash
# 저장된 6×100 frame을 panorama+YOLO-NAS+tracking으로 처리
python3 ros2_image_pano_opti.py

# cam1~cam3 live panorama에 YOLOv8을 적용해 publish
python3 3image_panorama_yolo.py

# cam0~cam5 live panorama+YOLO-NAS publish
python3 ros2_image_pano_opti_pub

# 탐지 없는 cam0~cam5 live panorama publish
python3 ros2_image_panorama
```

offline 탐지에는 ONNX Runtime과 `yolo_nas_s.onnx`, 3-camera 예제에는 Ultralytics와 `yolov8n.pt`가 필요하다. 두 model 파일은 현재 저장소에 포함돼 있다. 일부 ONNX 스크립트는 파일이 없으면 [Hugging Face의 FP16 model](https://huggingface.co/hr16/yolo-nas-fp16/resolve/main/yolo_nas_s_fp16.onnx)을 자동 다운로드하지만, 이 폴더에는 model version·checksum·provenance가 별도로 문서화돼 있지 않다. Live 결과는 `/panorama/image_raw/compressed`로 publish한다. 확장자가 없는 `ros2_image_pano_opti_pub`과 `ros2_image_panorama`는 위와 같이 `python3`로 실행한다.

## 확인된 제한

- 비슷한 이름의 `ros2_image_subscribe.py`는 timestamp 파일명을 사용해 `create_panorama.py`가 요구하는 `camX_fNNNN.jpg` 형식과 호환되지 않는다. offline 파이프라인에는 오타가 있는 `ros2_image_sucscribe2.py`를 사용한다.
- 해상도 640×360, intrinsic/distortion, 카메라 순서, overlap과 출력 크기가 코드에 고정돼 있다.
- 여섯 카메라 frame은 timestamp 동기화가 아니라 카메라별 수신 순번으로 묶이므로 움직임이 빠르거나 전송 지연이 다르면 seam이 어긋날 수 있다.
- ROS package manifest와 dependency lockfile이 없는 직접 실행 prototype이며 저장소의 sample frame과 결과는 일반화된 성능 검증이 아니다.
