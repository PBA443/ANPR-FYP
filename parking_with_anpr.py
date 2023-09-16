import numpy as np
from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import  get_car, read_license_plate , write_csv
import torch

#variables
frame_results = {}
output_result = {}
vehicles = [1,2,3]
v_tracker = Sort()
#anpr_area1 = [(496,346),(201,394),(651,461),(781,348)]
#anpr_area1 = [(227,444),(178,547),(501,547),(504,444)]
anpr_area1 = [(6,312),(6,596),(1013,596),(1013,312)]

#load model
v_model = YOLO('yolov8l.pt')
p_model = YOLO('last.pt')

#load video
cap = cv2.VideoCapture('dewata1.mp4')

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)


#read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 600))
    if ret:
        frame_results[frame_nmr] = {}
        v_results = v_model(frame)[0]
        print(v_results)
        v_detections = []
        for detection in v_results.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = detection
            c_x = x1+ (x2 - x1)/2
            c_y = y1+(y2 - y1)/2
            anp_track_results = cv2.pointPolygonTest(np.array(anpr_area1, np.int32), ((c_x, c_y)), False)
            if (class_id in vehicles) and anp_track_results >= 0:
                v_detections.append([x1, y1, x2, y2, score])
                cv2.circle(frame, (int(c_x), int(c_y)), 2, (0, 0, 255), -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                print(detection)
                print(v_detections)

                #track vehicle
                v_track_id = v_tracker.update(np.asarray(v_detections))

                #detect license plate
                p_results = p_model(frame)[0]

                for p_result in p_results.boxes.data.tolist():

                    x1,y1,x2,y2,score,class_id = p_result

                    #assign license plate for car
                    px1,py1,px2,py2,car_id = get_car(p_result,v_track_id)
                    pc_x = x1 + (x2 - x1) / 2
                    pc_y = y1 + (y2 - y1) / 2

                    #crop license plae
                    plate_crop=None
                    if (pc_x > 227 and pc_x < 504 and pc_y < 547 and pc_y > 444):
                        plate_crop = frame[int(y1):int(y2),int(x1):int(x2),:]

                    #process license pale
                        plate_crop_gray = cv2.cvtColor(plate_crop,cv2.COLOR_BGR2GRAY)
                        _,plate_crop_thresh = cv2.threshold(plate_crop_gray,64,255,cv2.THRESH_BINARY_INV)


                        cv2.imshow('original',plate_crop)
                        cv2.imshow('thresh', plate_crop_thresh)

                        #read licence plate numbers
                        plate_text,plate_confident_val = read_license_plate(plate_crop_thresh)

                        if plate_text is not None:
                            frame_results[frame_nmr][car_id] = {'car': {'bbox' : [px1,py1,px2,py2]},
                                                                'license_plate' : {'bbox': [x1,y1,x2,y2],'text': [plate_text],'bbox_score': [score], 'text_score': [plate_confident_val]}}



    cv2.polylines(frame, [np.array(anpr_area1, np.int32)], True, (255, 255, 0), 1)
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

write_csv(frame_results, 'D:/yolov8/new_test.csv')
cap.release()
cv2.destroyAllWindows()

