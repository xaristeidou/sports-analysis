import cv2
from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")

video_source = "assets/running.mp4"
cap = cv2.VideoCapture(video_source)


while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(
            source=frame,
            conf=0.5
        )

        annotated_frame = results[0].plot(boxes=False)


        cv2.imshow("Sports analysis", annotated_frame)
        key = cv2.waitKey(1)

        if key == 113:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
