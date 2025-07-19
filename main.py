from ultralytics import YOLO
from config import * 
from utils import *
import numpy as np
import cv2
import math 
import os
import time  # Thêm import time để tính FPS

MODEL_LICENSE_PLATE = 'Fast_ETC/license_plate_detector.pt'
cap = cv2.VideoCapture('traffic_50.mp4')
model = YOLO("yolo-Weights/yolov8s.pt")
license_plate_detector = YOLO(MODEL_LICENSE_PLATE)

frame = 0
prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    frame += 1
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    results = model.track(img)

    # coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            cls = int(box.cls[0])
            if (classNames[cls] != 'car') and (classNames[cls] != 'bus') and (classNames[cls] != 'truck'):
                continue

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # detect car color
            image_car = img[y1:y2, x1:x2]
            car_color = detect_car_color(image_car)
            license_detections = license_plate_detector(image_car)[0]

            if len(license_detections.boxes.cls.tolist()) != 0:
                x3, y3, x4, y4 = license_detections.boxes[0].xyxy[0]
                x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)
                cv2.rectangle(img, (x3 + x1, y3 + y1), (x4 + x1, y4 + y1), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            print("Class name -->", classNames[cls])
            print(img.shape)

            # object details
            org = [x1, y1]
            color_org = [x1, y1 + 30]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            cv2.putText(img, car_color, color_org, font, fontScale, color, thickness)

    cv2.putText(img, f'FPS: {fps:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    img = cv2.resize(img, (0, 0), fx=0.35, fy=0.35)
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
