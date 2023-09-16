import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from util import read_license_plate

def create_mj(id_mapping, plate_mapping, vehicle_class):
    mj = {}  # Create the 'mj' dictionary
    vls = vehicle_class
    for i, (vid, box) in enumerate(id_mapping.items()):  # Use 'enumerate' to get both index 'i' and value 'vid'
        x1, y1, x2, y2 = box
        vehicle = vid

        if not plate_mapping:
            mj[vehicle] = (x1, y1, x2, y2, vls[i])  # Use index 'i' to access 'vehicle_class'

        else:
            for pid, pbox in plate_mapping.items():
                x3, y3, x4, y4 = pbox
                if x1 < x3 and x4 < x2 and y1 < y3 and y4 < y2:
                    mj[vehicle] = (x1, y1, x2, y2, x3, y3, x4, y4, vls[i])  # Use index 'i' to access 'vehicle_class'
                else:
                    mj[vehicle] = (x1, y1, x2, y2, vls[i])  # Use index 'i' to access 'vehicle_class'
    return mj
def track_plate_id(plate):
    input = plate
    for res in input:
        if res.id != None:
            track_id = res.id.int().cpu().tolist()
            return track_id
def newassign_ids_and_coordinates_to_tracked_objects(boxes, v_track_id):
    # Check if the number of boxes matches the number of IDs
    if v_track_id is None:
        return {}
    if len(boxes) != len(v_track_id):
        raise ValueError("Number of boxes and number of IDs must be the same.")
    # Create a dictionary to store the assigned IDs and coordinates for each tracked object
    id_coord_mapping = {}
    # Assign IDs and coordinates to each tracked object using a loop
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box
        id_mapping = v_track_id[i]
        id_coord_mapping[id_mapping] = (x1, y1, x2, y2)

    return id_coord_mapping

#variables
track_history = defaultdict(lambda: [])
vehicle_types = [1,2,3,5,7]
anpr_area1 = [(6,312),(6,596),(1013,596),(1013,312)]
offset = 6
go_down = {}
down_counter = []
model = YOLO('yolov8l.pt')  # Load the YOLOv8 model
p_model = YOLO('last.pt')   # Load the YOLOv8 model number plate
video_path = "dewata1.mp4"  # load video file
cap = cv2.VideoCapture(video_path)
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('YOLOv8 Tracking')
cv2.setMouseCallback('YOLOv8 Tracking', POINTS)


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

        # Run YOLOv8 tracking on the frame to detect number plates
        plate_result = p_model.track(frame, persist=True)
        # Get the boxes, class, and ID of number plates
        p_boxes = plate_result[0].boxes.xyxy.int().cpu().tolist()
        p_track_id = track_plate_id(plate_result[0].boxes)

        # Call assign_ids_and_coordinates_to_tracked_objects function to assign IDs and coordinates to tracked number plates
        id_mapping = newassign_ids_and_coordinates_to_tracked_objects(boxes, v_track_id)
        # Call assign_ids_and_coordinates_to_tracked_objects function to assign IDs and coordinates to tracked number plates
        plate_mapping = newassign_ids_and_coordinates_to_tracked_objects(p_boxes, p_track_id)

        #tracked vehicle with number plate coordinates
        mj = create_mj(id_mapping, plate_mapping,vehicle_class)

        for id_, value, in mj.items():
            x1, y1, x2, y2, x3, y3, x4, y4, clsid = 0, 0, 0, 0, 0, 0, 0, 0, 0
            v_id = 0
            c_x, c_y, pc_x, pc_y = 0, 0, 0, 0
            if len(value) == 5:
                x1, y1, x2, y2, clsid = value
                v_id = id_
                c_x = x1 + (x2 - x1) // 2
                c_y = y1 + (y2 - y1) // 2
            elif len(value) == 9:
                x1, y1, x2, y2, x3, y3, x4, y4, clsid = value
                v_id = id_
                c_x = x1 + (x2 - x1) // 2
                c_y = y1 + (y2 - y1) // 2
                pc_x = x3 + (x3 - x4) // 2
                pc_y = y3 + (y3 - y4) // 2


        #parking violation
            '''anp_track_results = cv2.pointPolygonTest(np.array(anpr_area1, np.int32), ((c_x, c_y)), False)
            if (clsid in vehicle_types) and anp_track_results >= 0:
                cv2.circle(frame, (c_x, c_y), 2, (0, 0, 255), -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 0), 2)
                cv2.putText(frame, str(v_id), (x1, y1 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                plate_crop = None
                #vehicle_crop = None
                #crop if the number pate is withing the relavant area
                if (pc_x > 6 and pc_x < 1013 and pc_y < 596 and pc_y > 312):
                    plate_crop = frame[int(y3):int(y4), int(x3):int(x4), :]
                    cv2.imshow('plate', plate_crop)

                #if need to crop vehicle use below code
                #elif (c_x > 6 and c_x < 1013 and c_y < 596 and c_y > 312):
                    #vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    #cv2.imshow('vehicle', vehicle_crop)
'''
            anp_track_results = cv2.pointPolygonTest(np.array(anpr_area1, np.int32), ((c_x, c_y)), False)
        #direction violation
            if 130 < (c_y+offset) and 130 > (c_y-offset) and c_x > 100 and c_x < 1013 :
                go_down[v_id] = c_y
            if v_id in go_down:
                if 312 < (c_y+offset) and (clsid in vehicle_types):
                    cv2.circle(frame, (c_x, c_y), 2, (0, 0, 255), -1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 0), 2)
                    vehicle_crop = None
                    if (pc_x > 6 and pc_x < 1013 and pc_y < 596 and pc_y > 312):
                        plate_crop = frame[int(y3):int(y4), int(x3):int(x4), :]
                        cv2.imshow('plate', plate_crop)

        cv2.polylines(frame, [np.array(anpr_area1, np.int32)], True, (255, 255, 0), 1)
        cv2.line(frame, (274,130), (644, 130), (255, 255, 0), 1)
        cv2.putText(frame, ('line 1'), (274,130), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        cv2.line(frame, (100,312), (810,312), (255, 255, 0), 1)
        cv2.putText(frame, ('line 2'), (100,312), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
