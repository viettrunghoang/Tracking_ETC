from ultralytics import YOLO
from utils import draw_detections, format_predictions, match_detections_with_tracks 
from BYTEtracker.byte_tracker import BYTETracker
import numpy as np
import cv2

class BYTETrackerArgs:
  track_thresh: float = 0.5 #0.25
  track_buffer: int = 30
  match_thresh: float = 0.8
  aspect_ratio_thresh: float = 3.0
  min_box_area: float = 1.0
  mot20: bool = False

byte_tracker = BYTETracker(BYTETrackerArgs)

cap = cv2.VideoCapture('traffic_8.mp4')
model = YOLO("yolo-Weights/yolov8n.pt") #car detecter

while True:
    success, img = cap.read()
    #detect car with yolo
    detections = model(img)[0]

    # create a new list of detection with tracker_id attribute.
    detections_with_tracker = []
    for detection in detections:
      detection.tracker_id = ""
      detections_with_tracker.append(detection)

    output_results = format_predictions(detections_with_tracker, with_conf=True)
    
    if output_results.size > 0:
        tracks = byte_tracker.update(
            output_results=output_results,
            img_info=img.shape,
            img_size=img.shape
        )

        detections_with_tracker = match_detections_with_tracks(detections_with_tracker, tracks)
    else:
        tracks = []


    # get trackers with ByteTrack
    # tracks = byte_tracker.update(
    #     output_results=format_predictions(detections_with_tracker, with_conf=True),
    #     img_info=img.shape,
    #     img_size=img.shape
    # )

    # set tracker_id in yolo detections
    # detections_with_tracker = match_detections_with_tracks(detections_with_tracker, tracks)

    # annotate the frame
    image = draw_detections(img, detections_with_tracker, True)

    img = cv2.resize(image, (0, 0), fx=0.35, fy=0.35)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
