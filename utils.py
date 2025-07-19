import cv2
import numpy as np
from onemetric.cv.utils.iou import box_iou_batch
from config import colors, classNames, color_ranges

def draw_detections(image, detections, draw_tacker_id: bool = False):
  image = image.copy()
  for pred in detections:
    bbox = pred.boxes.xyxy.int().tolist()[0]
    cls_ind = pred.boxes.cls.int().item()
    cls = classNames[cls_ind]
    if (cls != 'car') and (cls != 'bus') and (cls != 'truck'):
      color = (255, 255, 0)
    else: 
      color=colors[cls]
    cv2.rectangle(img=image, pt1=tuple(bbox[:2]), pt2=tuple(bbox[2:]), color=color, thickness=3)
    if draw_tacker_id:
      cv2.putText(image, str(pred.tracker_id), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
    else:
      cv2.putText(image, cls, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

  return image

def format_predictions(predictions, with_conf: bool = True):
  """
  Format yolo detection to ByteTracke format: (x1, y1, x2, y2, conf)
  """
  frame_detections = []
  for pred in predictions:
      bbox = pred.boxes.xyxy.int().tolist()[0]
      conf = pred.boxes.conf.item()
      if with_conf:
        detection = bbox + [conf]
      else:
        detection = bbox

      frame_detections.append(detection)
  return np.array(frame_detections, dtype=float)

# def match_detections_with_tracks(detections, tracks):
#   """
#   Find which tracker corresponds to yolo detections and set the tracker_id.
#   We compute the iou between the detection and trackers.
#   """
#   detections_bboxes = format_predictions(detections, with_conf=False)
#   tracks_bboxes = np.array([track.tlbr for track in tracks], dtype=float)
#   iou = box_iou_batch(tracks_bboxes, detections_bboxes)
#   track2detection = np.argmax(iou, axis=1)

#   for tracker_index, detection_index in enumerate(track2detection):
#     if iou[tracker_index, detection_index] != 0:
#       detections[detection_index].tracker_id = tracks[tracker_index].track_id
#   return detections

def match_detections_with_tracks(detections, tracks):
    """
    Find which tracker corresponds to yolo detections and set the tracker_id.
    We compute the iou between the detection and trackers.
    """
    detections_bboxes = format_predictions(detections, with_conf=False)
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

def detect_car_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    max_area = 0
    dominant_color = None
    
    for color, ranges in color_ranges.items():
        mask = cv2.inRange(hsv_image, np.array(ranges[0]), np.array(ranges[1]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                dominant_color = color
    return dominant_color