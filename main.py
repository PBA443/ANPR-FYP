import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from util import read_license_plate
import time
import easyocr
import os

def create_mj(id_mapping, plate_mapping, vehicle_class):
    mj = {}  # Create the 'mj' dictionary
    vls = vehicle_class
    for i, (vid, box) in enumerate(id_mapping.items()):  # Use 'enumerate' to get both index 'i' and value 'vid'
        x1, y1, x2, y2,conf1 = box
        vehicle = vid

        if not plate_mapping:
            mj[vehicle] = (x1, y1, x2, y2, conf1, vls[i])  # Use index 'i' to access 'vehicle_class'

        else:
            for pid, pbox in plate_mapping.items():
                x3, y3, x4, y4,conf2 = pbox
                if x1 < x3 and x4 < x2 and y1 < y3 and y4 < y2:
                    mj[vehicle] = (x1, y1, x2, y2,conf1, x3, y3, x4, y4,conf2, vls[i])  # Use index 'i' to access 'vehicle_class'
                else:
                    mj[vehicle] = (x1, y1, x2, y2,conf1, vls[i])  # Use index 'i' to access 'vehicle_class'
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

parking_violate1 = 'images/parking_violation/number_plate'
if not os.path.exists(parking_violate1):
    os.makedirs(parking_violate1)

parking_violate2 = 'images/parking_violation/violation'
if not os.path.exists(parking_violate2):
    os.makedirs(parking_violate2)

direct_violate1 = 'images/direction_violation/number_plate'
if not os.path.exists(direct_violate1):
    os.makedirs(direct_violate1)

direct_violate2 = 'images/direction_violation/violation'
if not os.path.exists(direct_violate2):
    os.makedirs(direct_violate2)

speed_violate1 = 'images/direction_violation/number_plate'
if not os.path.exists(direct_violate1):
    os.makedirs(direct_violate1)

speed_violate2 = 'images/direction_violation/violation'
if not os.path.exists(direct_violate2):
    os.makedirs(direct_violate2)

#variables
track_history = defaultdict(lambda: [])
vehicle_types = [1,2,3,5,7]
parking_violation_area = [(609,165),(863,480),(1013,451),(717,169)]
direction_violation_detection_area = [(291,167),(77,470),(948,465),(660,175)]
offset = 6
go_up = {}
down_counter = []
down_time = {}

down = {}
down_counter_direction = []
#get_time = {}
down_counter_time = []
model = YOLO('yolov8n.pt')  # Load the YOLOv8 model
p_model = YOLO('last.pt')   # Load the YOLOv8 model number plate
video_path = "dewata1.mp4"  # load video file
cap = cv2.VideoCapture(video_path)
reader = easyocr.Reader(['en'])
saved_images_for_v_id = {}
vehicle_entry_time = {}

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read() # Read a frame from the video
    frame = cv2.resize(frame, (1020, 600))
    veh = {}
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

        #tracked vehicle with number plate coordinates
        mj = create_mj(id_mapping, plate_mapping,vehicle_class)
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

        #parking violation
            anp_track_results = cv2.pointPolygonTest(np.array(parking_violation_area, np.int32), ((c_x, c_y)), False)
            if (clsid in vehicle_types) and anp_track_results >= 0 and conf1 >60:
                if id_ not in vehicle_entry_time:
                    vehicle_entry_time[id_] = time.time()
                if time.time() - vehicle_entry_time[id_] >= 5:
                    # Save the parking violation image
                    cv2.circle(frame, (c_x, c_y), 2, (0, 0, 255), -1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 0), 2)
                    cv2.putText(frame, str(v_id), (x1, y1 - 2), 0, 1, (255, 255, 255), thickness=1,lineType=cv2.LINE_AA)
                    frame_image_filename = os.path.join(parking_violate2, f'parking_{v_id}.jpg')
                    cv2.imwrite(frame_image_filename, frame)
                    plate_crop = None
                    if (pc_x > 6 and pc_x < 1013 and pc_y < 596 and pc_y > 312):
                        plate_crop = frame[int(y3):int(y4), int(x3):int(x4), :]
                        cv2.imshow('plate', plate_crop)
                        plate_image_filename = os.path.join(parking_violate1, f'parking_{v_id}.jpg')
                        cv2.imwrite(plate_image_filename, plate_crop)
                        saved_images_for_v_id[v_id] = True

        #direction violation
            if 545 < (c_y + offset) and 545 > (c_y - offset) and c_x > 17 and c_x < 1017 and (clsid in vehicle_types) and conf1>60:
                down[v_id] = time.time()
            if v_id in down:
                if 470 < (c_y + offset) and 470 > (c_y - offset):
                    get_down = time.time() - down[v_id]
                    distance = 10
                    down_speed = (distance / get_down) * 3.6
                    if (down_speed > 0) and down_counter_direction.count(v_id) == 0:
                        down_counter_direction.append(v_id)
                for val in down_counter_direction:
                    for key, value in mj.items():
                        if val == key:
                            sx1, sy1, sx2, sy2, sconf1, sx3, sy3, sx4, sy4, sconf2, sclsid = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            sv_id = 0
                            sc_x, sc_y, spc_x, spc_y = 0, 0, 0, 0
                            if len(value) == 6:
                                sx1, sy1, sx2, sy2, sconf1, sclsid = value
                                sv_id = val
                                sc_x = sx1 + (sx2 - sx1) // 2
                                sc_y = sy1 + (sy2 - sy1) // 2
                            elif len(value) == 11:
                                sx1, sy1, sx2, sy2, sconf1, sx3, sy3, sx4, sy4, sconf2, sclsid = value
                                sv_id = val
                                sc_x = sx1 + (sx2 - sx1) // 2
                                sc_y = sy1 + (sy2 - sy1) // 2
                                spc_x = sx3 + (sx3 - sx4) // 2
                                spc_y = sy3 + (sy3 - sy4) // 2
                                # crop license plate
                                license_plate_crop = frame[int(sy3):int(sy4), int(sx3):int(sx4), :]
                                cv2.imshow("frame", license_plate_crop)
                                frame_image_filename = os.path.join(direct_violate2, f'direction_{v_id}.jpg')
                                cv2.imwrite(frame_image_filename, frame)
                                plate_image_filename = os.path.join(direct_violate1, f'direction_{v_id}.jpg')
                                cv2.imwrite(plate_image_filename, license_plate_crop)
                            cv2.circle(frame, (sc_x, sc_y), 2, (0, 0, 255), -1)
                            cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (255, 0, 0), 2)
                            cv2.rectangle(frame, (int(sx3), int(sy3)), (int(sx4), int(sy4)), (255, 0, 0), 2)

        #speed violation
            if 167 < (c_y + offset) and 167 > (c_y - offset) and c_x > 291 and c_x < 660 and (clsid in vehicle_types) and conf1>60:
                down_time[v_id] = time.time()
            if v_id in down:
                get_time = time.time() - down[v_id]
                if 312 < (c_y + offset) and 312 > (c_y - offset):
                    distance = 10
                    down_speed = (distance / get_time) * 3.6
                    if (down_speed > 40) and down_counter_time.count(v_id) == 0:
                        down_counter_time.append(v_id)
                for val in down_counter_time:
                    for key, value in mj.items():
                        if val == key:
                            sx1, sy1, sx2, sy2, sconf1, sx3, sy3, sx4, sy4, sconf2, sclsid = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            sv_id = 0
                            sc_x, sc_y, spc_x, spc_y = 0, 0, 0, 0
                            if len(value) == 6:
                                sx1, sy1, sx2, sy2, sconf1, sclsid = value
                                sv_id = val
                                sc_x = sx1 + (sx2 - sx1) // 2
                                sc_y = sy1 + (sy2 - sy1) // 2
                            elif len(value) == 11:
                                sx1, sy1, sx2, sy2, sconf1, sx3, sy3, sx4, sy4, sconf2, sclsid = value
                                sv_id = val
                                sc_x = sx1 + (sx2 - sx1) // 2
                                sc_y = sy1 + (sy2 - sy1) // 2
                                spc_x = sx3 + (sx3 - sx4) // 2
                                spc_y = sy3 + (sy3 - sy4) // 2
                                # crop license plate
                                speedplate = frame[int(sy3):int(sy4), int(sx3):int(sx4), :]
                                cv2.imshow("speed_violation", speedplate)
                                frame_image_filename = os.path.join(speed_violate2, f'speed_{v_id}.jpg')
                                cv2.imwrite(frame_image_filename, frame)
                                plate_image_filename = os.path.join(speed_violate1, f'speed_{v_id}.jpg')
                                cv2.imwrite(plate_image_filename, speedplate)
                            cv2.circle(frame, (sc_x, sc_y), 2, (0, 0, 255), -1)
                            cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (255, 0, 0), 2)
                            cv2.rectangle(frame, (int(sx3), int(sy3)), (int(sx4), int(sy4)), (255, 0, 0), 2)

        # parking violation detection area
        cv2.polylines(frame, [np.array(parking_violation_area, np.int32)], True, (0, 0, 255), 1)

        #direction vilation detection area
        cv2.line(frame, (17, 545), (1017, 557), (255, 0, 0), 1)
        cv2.putText(frame, ('line 1'), (17, 545), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.line(frame, (77, 470), (948, 465), (255, 0, 0), 1)
        cv2.putText(frame, ('line 2'), (77, 470), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

        # speed violation detection area
        cv2.line(frame, (291, 167), (660, 167), (255, 255, 0), 1)
        cv2.putText(frame, ('line 3'), (291, 167), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        cv2.line(frame, (6, 312), (1013, 312), (255, 255, 0), 1)
        cv2.putText(frame, ('line 4'), (6, 312), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

        # Display the annotated frame


        cv2.imshow("Parking Violation", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
