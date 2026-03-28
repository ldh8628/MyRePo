import cv2

# 비디오 파일 열기
cap = cv2.VideoCapture('test_video.mp4')

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

# 비디오 정보 출력
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"해상도: {width} x {height}")
print(f"총 프레임 수: {total_frames}")
print("재생 중... (종료: 'q' 키)")

while True:
    ret, frame = cap.read()

    # 영상 끝에 도달하면 종료
    if not ret:
        print("재생 완료")
        break

    cv2.imshow('test_video.mp4', frame)

    # 'q' 키를 누르면 종료, FPS에 맞게 대기
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        print("사용자가 종료했습니다.")
        break

cap.release()
cv2.destroyAllWindows()
