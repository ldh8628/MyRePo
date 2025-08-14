from picamera2 import Picamera2
import cv2
import time

# Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Picamera2 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(2)  # 카메라 워밍업

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]      # 얼굴 내부 영역(그레이)
        roi_color = frame[y:y+h, x:x+w]    # 얼굴 내부 영역(컬러)

        # 눈 검출 (얼굴 내부에서만)
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 1:  # 눈이 하나 이상 인식되면 얼굴로 간주
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # 눈이 감겨있거나 모자, 조명 등으로 눈이 안 보이는 경우 무시
            cv2.putText(frame, "No eyes detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 영상 출력
    cv2.imshow("Face + Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
