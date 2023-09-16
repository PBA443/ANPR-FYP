import cv2
import torch
import numpy as np
from tracker import *
from ultralytics import YOLO
import pandas as pd
import math
import time

maximum_speed = 40
cy1 = 350
cy2 = 410
cy3 = 350
cy4 = 407
offset = 6
offset_2 = 6
go_down = {}
go_up = {}
go_down_time = {}
go_up_time = {}
down_counter = []
up_counter = []
down_counter_time = []
up_counter_time = []
model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture('veh2.mp4')
count = 0
tracker = Tracker()
tracked_vehicle_id = set()

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

classNames = ["person",'bicycle','car','motorcycle',
                'airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench',
                'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
                'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass',
                'cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
                'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
                'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier'
              ]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 600))
    results = model(frame)
    list = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1),int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[0] - 3
            if 'car' in class_name:
                list.append([x1,y1,x2,y2])

        bbox_id =tracker.update(list)
        for bbox in bbox_id:
                x3,y3,x4,y4,id = bbox
                cx=(x3+x4)//2
                cy=(y3+y4)//2
                #cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                #cv2.putText(frame, str(id), (x3, y3 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

#wrong way detection
                if cy4 < (cy+offset) and cy4 > (cy-offset) and cx > 556 and cx < 880:
                    go_up[id] = cy
                if id in go_up:
                    if cy3 < (cy + offset) and cy3 > (cy - offset):
                        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                        cv2.putText(frame, str(id), (x3, y3 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                        if up_counter.count(id)==0:
                            up_counter.append(id)
                            #take numberplate

                if cy1 < (cy+offset_2) and cy1 > (cy-offset_2) and cx > 313 and cx < 524:
                    go_down[id] = cy
                if id in go_down:
                    if cy2 < (cy + offset_2) and cy2 > (cy - offset_2):
                        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                        cv2.putText(frame, str(id), (x3, y3 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                        if down_counter.count(id) == 0:
                            down_counter.append(id)
                            # take numberplate

#speed violation detection
                if cy3 < (cy + offset) and cy3 > (cy - offset):
                    go_down_time[id] = time.time()
                if id in go_down_time:
                    if cy4 < (cy + offset) and cy4 > (cy - offset):
                        end_down_time = time.time() - go_down_time[id]
                        # cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                        # cv2.putText(frame, str(id), (x3, y3 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                        if down_counter_time.count(id) == 0:
                            down_counter_time.append(id)
                            distance = 10
                            down_speed = (distance / end_down_time) * 3.6
                            cv2.putText(frame, str(down_speed), (x3, y3 - 2), 0, 1, (255, 255, 255), 1, cv2.LINE_AA)
                            # if down_speed > maximum_speed:
                                # detect number plate

                if cy2 < (cy + offset_2) and cy2 > (cy - offset_2):
                    go_up_time[id] = time.time()
                if id in go_up_time:
                    if cy1 < (cy + offset_2) and cy1 > (cy - offset_2):
                        end_up_time = time.time() - go_up_time[id]
                        # cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                        # cv2.putText(frame, str(id), (x3, y3 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                        if up_counter_time.count(id) == 0:
                            up_counter_time.append(id)
                            distance = 10
                            up_speed = (distance / end_up_time) * 3.6
                            cv2.putText(frame, str(up_speed), (x3, y3 - 2), 0, 1, (255, 255, 255), 1, cv2.LINE_AA)
                            # if up_speed > maximum_speed:
                            # detect licence plate


    cv2.line(frame, (313,cy1), (524,cy1), (255,255,0), 1)
    cv2.putText(frame, ('line 1'), (315, 352), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
    cv2.line(frame, (193, cy2), (515, cy2), (255, 255, 0), 1)
    cv2.putText(frame,('line 2'), (198, cy2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
    cv2.line(frame, (545, cy3), (750, cy3), (255, 255, 0), 1)
    cv2.putText(frame, ('line 3'), (550, cy3), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
    cv2.line(frame, (556, cy4), (880, cy4), (255, 255, 0), 1)
    cv2.putText(frame,('line 4'), (561, cy4), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
    down_vehicle_count = len(down_counter)
    up_vehicle_count = len(up_counter)
    cv2.putText(frame, ('right side wrongway = ')+str(up_vehicle_count), (29,16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),2)
    cv2.putText(frame, ('left side wrongway  = ') + str(down_vehicle_count), (34,30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()