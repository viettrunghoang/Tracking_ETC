from SMtracker.mc_SMILEtrack import SMILEtrack
from onemetric.cv.utils.iou import box_iou_batch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from tqdm import TqdmSynchronisationWarning
import numpy as np
import cv2

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def draw_detections(image, detections, draw_tacker_id: bool = False):
  image = image.copy()
  for pred in detections:
    bbox = pred.boxes.xyxy.int().tolist()[0]
    cls_ind = pred.boxes.cls.int().item()
    cls = classNames[cls_ind]
    cv2.rectangle(img=image, pt1=tuple(bbox[:2]), pt2=tuple(bbox[2:]), color=(255, 0, 0), thickness=3)
    if draw_tacker_id:
      cv2.putText(image, str(pred.tracker_id), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 3)
    else:
      cv2.putText(image, cls, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 3)

  return image

class SMILETrackArgs:
  track_high_thresh: float = 0.5    
  track_low_thresh: float = 0.1
  new_track_thresh: float = 0.6
  track_buffer: int = 30
  match_thresh: float = 0.8     #
  proximity_thresh: float = 0.5
  appearance_thresh: float = 0.25
  with_reid: bool = False
  cmc_method: str = 'sparseOptFlow'
  name: str = 'exp'
  ablation: bool = False
  mot20: bool = False

  # initiate tracker
smile_tracker = SMILEtrack(SMILETrackArgs)


def format_predictions(predictions, with_conf: bool = True, with_track: bool=True):
    """
    Định dạng phát hiện yolo thành định dạng ByteTracke: (x1, y1, x2, y2, conf, cls)
    """
    frame_detections = []
    for pred in predictions:
        bbox = pred.boxes.xyxy.int().tolist()[0]
        conf = pred.boxes.conf.item()
        cls = pred.boxes.cls.item()
        if with_track:
            detection = bbox + [conf, cls]
        else:
            if with_conf:
                detection = bbox + [conf]
            else:
                detection = bbox

        frame_detections.append(detection)
    return np.array(frame_detections, dtype=float)

def match_detections_with_tracks(detections, tracks):
    """
    Find which tracker corresponds to yolo detections and set the tracker_id.
    We compute the iou between the detection and trackers.
    """
    detections_bboxes = format_predictions(detections, with_conf=False, with_track=False)
    tracks_bboxes = np.array([track.tlbr for track in tracks], dtype=float)

    if detections_bboxes.shape[0] == 0 or tracks_bboxes.shape[0] == 0:
        print("No valid bounding boxes found for detections or tracks.")
        return detections

    iou = box_iou_batch(tracks_bboxes, detections_bboxes)
    track2detection = np.argmax(iou, axis=1)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            detections[detection_index].tracker_id = tracks[tracker_index].track_id

    return detections

cap = cv2.VideoCapture('traffic_8.mp4')

while True:
    success, img = cap.read()

    detections = model(img, verbose=False)[0]

    detections_with_tracker = []
    for detection in detections:
      detection.tracker_id = ""
      detections_with_tracker.append(detection)

    output_results = format_predictions(detections_with_tracker, with_conf=True)
    
    if output_results.size > 0:
        tracks = smile_tracker.update(
        output_results=format_predictions(detections_with_tracker, with_conf=True, with_track=True),
        img=img
        )

        detections_with_tracker = match_detections_with_tracks(detections_with_tracker, tracks)
    else:
        tracks = []

    image = draw_detections(img, detections_with_tracker, True)
    image = cv2.resize(image, (0, 0), fx=0.35, fy=0.35)
    cv2.imshow('Webcam', image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
