import cv2
from ultralytics import YOLO


model = YOLO("yolov8m-pose.pt")


video_source = "assets/video_path.mp4"
cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(
            source=frame,
            conf = 0.25
        )

        annotated_frame = results[0].plot()
    