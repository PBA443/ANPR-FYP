import cv2
import numpy as np
from ultralytics import YOLO
from utills import get_limits
from PIL import Image
import os
import time
from util import read_license_plate
import pyrebase
from firebase_admin import storage
from firebase_admin import db
from datetime import datetime

front_light_violation1 = 'images/front_traffic_light/number_plate'
if not os.path.exists(front_light_violation1):
    os.makedirs(front_light_violation1)
front_light_violation2 = 'images/front_traffic_light/frame'
if not os.path.exists(front_light_violation2):
    os.makedirs(front_light_violation2)
front_light_violation3 = 'images/front_traffic_light/vehicle'
if not os.path.exists(front_light_violation3):
    os.makedirs(front_light_violation3)


def create_mj(id_mapping, plate_mapping, vehicle_class):
    mj = {}  # Create the 'mj' dictionary
    vls = vehicle_class
    for i, (vid, box) in enumerate(id_mapping.items()):  # Use 'enumerate' to get both index 'i' and value 'vid'
        x1, y1, x2, y2, conf1 = box
        vehicle = vid

        if not plate_mapping:
            mj[vehicle] = (x1, y1, x2, y2, conf1, vls[i])  # Use index 'i' to access 'vehicle_class'

        else:
            for pid, pbox in plate_mapping.items():
                x3, y3, x4, y4, conf2 = pbox
                if x1 < x3 and x4 < x2 and y1 < y3 and y4 < y2:
                    mj[vehicle] = (
                        x1, y1, x2, y2, conf1, x3, y3, x4, y4, conf2, vls[i])  # Use index 'i' to access 'vehicle_class'
                else:
                    mj[vehicle] = (x1, y1, x2, y2, conf1, vls[i])  # Use index 'i' to access 'vehicle_class'
    return mj


def track_plate_id(plate):
    input = plate
    for res in input:
        if res.id != None:
            track_id = res.id.int().cpu().tolist()
            return track_id


def newassign_ids_and_coordinates_to_tracked_objects(boxes, v_track_id, v_det_conf):
    if v_track_id is None:
        return {}
    if len(boxes) != len(v_track_id) or len(boxes) != len(v_det_conf):
        raise ValueError("Number of boxes, IDs, and detection confidences must be the same.")
    id_coord_mapping = {}
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box
        id_mapping = v_track_id[i]
        det_conf = v_det_conf[i]  # Get the detection confidence value
        # Include the detection confidence value in the tuple
        id_coord_mapping[id_mapping] = (x1, y1, x2, y2, det_conf)
    return id_coord_mapping


def detect_green_color(frame):
    traffic_signal = [(345, 116), (344, 183), (372, 184), (372, 113)]
    green = [0, 0, 255]
    plate_crop = frame[116:183, 345:372, :]
    hsvImg = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    lower_limit, upper_limit = get_limits(color=green)
    mask = cv2.inRange(hsvImg, lower_limit, upper_limit)
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        plate_crop = cv2.rectangle(plate_crop, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.polylines(frame, [np.array(traffic_signal, np.int32)], True, (0, 0, 255), 2)
    return bbox is not None


# firebase
config = {
    'apiKey': "AIzaSyD_NFy6pD8-gnQJUMqKbPF8qux7IYz8H_g",
    'authDomain': "research-project-811a8.firebaseapp.com",
    'databaseURL': "https://research-project-811a8-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "research-project-811a8",
    'storageBucket': "research-project-811a8.appspot.com",
    'messagingSenderId': "792324726693",
    'appId': "1:792324726693:web:94ef9d2bc5d1558724924c",
    'measurementId': "G-77N9TYF9N3",
    'serviceAccount': "credentials.json",
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()

# set location
location = "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d15857.823186462296!2d80.08955465947969!3d6.4637958678341905!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3ae22d305126dacb%3A0xb503df7f647ef4cd!2sWelipenna%20Interchange!5e0!3m2!1sen!2slk!4v1693501904033!5m2!1sen!2slk"

# get time
now = datetime.now()
current_time = now.strftime("%H:%M:%S")

# output variables
output_video_path = "output_video.mp4"
output_video_fps = 30  # Choose the desired frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the output video
# Initialize the video writer
out = cv2.VideoWriter(output_video_path, fourcc, output_video_fps, (1020, 600))

# variables
offset = 6
go_up = {}
red_light = {}
red_light_counter = []
up_counter = []
red_light_violation_area = [(380, 481), (129, 594), (912, 593), (754, 481)]
vehicle_types = [2]
down = {}
down_counter_direction = []
model = YOLO('yolov8l.pt')
p_model = YOLO('last.pt')
video_path = "Front.mp4"
cap = cv2.VideoCapture(video_path)
light_violated_count = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = cv2.resize(frame, (1020, 600))
    if success:
        # Run YOLOv8 tracking on the frame to detect vehicles
        results = model.track(frame, persist=True)
        # Get the boxes, class and id of vehicles
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        vehicle_class = results[0].boxes.cls.int().cpu().tolist()
        v_track_id = results[0].boxes.id.int().cpu().tolist()
        v_con_level = results[0].boxes.conf.cpu().tolist()
        vehicle_det_conf = [int(value * 100) for value in v_con_level]
        # Run YOLOv8 tracking on the frame to detect number plates
        plate_result = p_model.track(frame, persist=True)
        # Get the boxes, class, and ID of number plates
        p_boxes = plate_result[0].boxes.xyxy.int().cpu().tolist()
        p_track_id = track_plate_id(plate_result[0].boxes)
        p_con_level = plate_result[0].boxes.conf.cpu().tolist()
        plate_det_conf = [int(value * 100) for value in p_con_level]

        # Call assign_ids_and_coordinates_to_tracked_objects function to assign IDs and coordinates to tracked number plates
        id_mapping = newassign_ids_and_coordinates_to_tracked_objects(boxes, v_track_id, vehicle_det_conf)

        # Call assign_ids_and_coordinates_to_tracked_objects function to assign IDs and coordinates to tracked number plates
        plate_mapping = newassign_ids_and_coordinates_to_tracked_objects(p_boxes, p_track_id, plate_det_conf)

        # tracked vehicle with number plate coordinates
        mj = create_mj(id_mapping, plate_mapping, vehicle_class)
        detected_green = detect_green_color(frame)
        image_counter = 0

        for id_, value, in mj.items():
            x1, y1, x2, y2, conf1, x3, y3, x4, y4, conf2, clsid = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            v_id = 0
            c_x, c_y, pc_x, pc_y = 0, 0, 0, 0
            if len(value) == 6:
                x1, y1, x2, y2, conf1, clsid = value
                v_id = id_
                c_x = x1 + (x2 - x1) // 2
                c_y = y1 + (y2 - y1) // 2
            elif len(value) == 11:
                x1, y1, x2, y2, conf1, x3, y3, x4, y4, conf2, clsid = value
                v_id = id_
                c_x = x1 + (x2 - x1) // 2
                c_y = y1 + (y2 - y1) // 2
                pc_x = x3 + (x3 - x4) // 2
                pc_y = y3 + (y3 - y4) // 2

            # traffic light violation detection
            red_light_violation_detection = cv2.pointPolygonTest(np.array(red_light_violation_area, np.int32),
                                                                 ((c_x, c_y)), False)
            if detected_green == True:
                cv2.polylines(frame, [np.array(red_light_violation_area, np.int32)], True, (0, 0, 255), 1)
                if red_light_violation_detection >= 0 and conf1 > 60:

                    cv2.circle(frame, (c_x, c_y), 2, (0, 0, 255), -1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 0), 2)
                    # cv2.putText(frame, str(v_id), (x1, y1 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    frame_image_filename = os.path.join(front_light_violation2, f'front_traffic_light_{v_id}.jpg')
                    image_trafficlight_frame = os.path.join(front_light_violation2, f'front_traffic_light_{v_id}.jpg')
                    data_frontredlight = {"violation": "redlightfront_violation", "ID": v_id, "Time": current_time,
                                          "Location": location}
                    cv2.imwrite(frame_image_filename, frame)
                    storage.child(image_trafficlight_frame).put(image_trafficlight_frame)
                    db.child("Violation_Details_front").child(v_id).set(data_frontredlight)
                    red_light[v_id] = c_y
                    noi_frontredlight = {"NOI": len(red_light)}
                    db.child("NumberOf_frontredlight_violation").set(noi_frontredlight)
                    # vehicle
                    RLV_vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    RLV_vehicle_image_filename = os.path.join(front_light_violation3, f'front_traffic_light_{v_id}.jpg')
                    image_trafficlight_vehicle = os.path.join(front_light_violation3, f'front_traffic_light_{v_id}.jpg')
                    cv2.imwrite(RLV_vehicle_image_filename, RLV_vehicle_crop)
                    storage.child(image_trafficlight_vehicle).put(image_trafficlight_vehicle)

                    # plate
                    RLV_plate_crop = frame[int(y3):int(y4), int(x3):int(x4), :]
                    RLV_plate_image_filename = os.path.join(front_light_violation1, f'front_traffic_light_{v_id}.jpg')
                    image_trafficlight_plate = os.path.join(front_light_violation1, f'front_traffic_light_{v_id}.jpg')
                    if RLV_plate_crop is not None and not RLV_plate_crop.size == 0:
                        cv2.imwrite(RLV_plate_image_filename, RLV_plate_crop)
                        storage.child(image_trafficlight_plate).put(image_trafficlight_plate)
                        # cv2.imshow('red light violated vehicle plates', RLV_plate_crop)
                        RLV_plate_crop_gray = cv2.cvtColor(RLV_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, plate_crop_thresh = cv2.threshold(RLV_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        plate_text, plate_confident_val = read_license_plate(plate_crop_thresh)
                        print(plate_text)
        cv2.imshow("YOLOv8 Tracking", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
# Release the video capture object and close the display window
out.release()
cap.release()
cv2.destroyAllWindows()
