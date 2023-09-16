from ultralytics import YOLO

def assign_ids_to_boxes(boxes, v_track_id):
    x1, y1, x2, y2 = boxes
    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2
    newbox = cx,cy
    # Check if the number of boxes matches the number of IDs
    if len(boxes) != len(v_track_id):
        raise ValueError("Number of boxes and number of IDs must be the same.")

    # Create a dictionary to store the assigned IDs for each box
    id_mapping = {}

    # Assign IDs to each box using a loop
    for i in range(len(boxes)):
        box = boxes[i]
        id_mapping[tuple(box)] = v_track_id[i]

    return id_mapping


def assign_ids_to_boxes_center(boxes, v_track_id):
    # Check if the number of boxes matches the number of IDs
    if len(boxes) != len(v_track_id):
        raise ValueError("Number of boxes and number of IDs must be the same.")

    # Create a dictionary to store the assigned IDs for each box's center
    id_mapping = {}

    # Assign IDs to each box's center using a loop
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box
        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2
        id_mapping[(cx, cy)] = v_track_id[i]

    return id_mapping


def assign_ids_to_boxes_center_with_y2(boxes, v_track_id):
    # Check if the number of boxes matches the number of IDs
    if len(boxes) != len(v_track_id):
        raise ValueError("Number of boxes and number of IDs must be the same.")

    # Create a dictionary to store the assigned IDs and y2 values for each box's center
    id_y2_mapping = {}

    # Assign IDs and y2 values to each box's center using a loop
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box
        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2
        id_y2_mapping[(cx, cy)] = (v_track_id[i], y2)

    return id_y2_mapping

def print_ids_for_center_boxes_above_y2_threshold(id_y2_mapping):
    for center, (id, y2) in id_y2_mapping.items():
        if y2 > 312:
            print(f"Box with center {center} has ID {id} and y2 value {y2} (above y2 threshold)")


def assign_ids_and_coordinates_to_tracked_objects(boxes, v_track_id):
    # Check if the number of boxes matches the number of IDs
    if len(boxes) != len(v_track_id):
        raise ValueError("Number of boxes and number of IDs must be the same.")

    # Create a dictionary to store the assigned IDs and coordinates for each tracked object
    id_coord_mapping = {}

    # Assign IDs and coordinates to each tracked object using a loop
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box
        id_mapping = v_track_id[i]
        id_coord_mapping[tuple(box)] = (id_mapping, x1, y1, x2, y2)

    return id_coord_mapping

def track_plate_id(plate):
    input = plate
    for res in input:
        if res.id != None:
            track_id = res.id.int().cpu().tolist()
            return track_id


def assign_vehicle_id_to_number_plate(p_boxes, p_track_id, boxes, v_track_id):
    # Check if the number of p_boxes matches the number of p_track_id
    if len(p_boxes) != len(p_track_id):
        raise ValueError("Number of p_boxes and number of p_track_id must be the same.")

    # Create a copy of v_track_id and boxes to update
    updated_v_track_id = v_track_id.copy()
    updated_boxes = boxes.copy()

    # Iterate through each number plate and compare with corresponding vehicle
    for i in range(len(p_boxes)):
        p_box = p_boxes[i]
        px1, py1, px2, py2 = p_box

        for j in range(len(boxes)):
            box = boxes[j]
            x1, y1, x2, y2 = box

            # Check if the number plate is inside the vehicle box
            if (px1 > x1) and (px2 < x2) and (py1 > y1) and (py2 < y2):
                updated_v_track_id[j] = p_track_id[i]

    return updated_v_track_id, updated_boxes, p_boxes

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2 = license_plate
    foundIt = False
    for j in range(len(vehicle_track_ids)):

        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]
    return -1, -1, -1, -1, -1

def func0(id_mapping,plate_mapping):
    foundIt = False
    px1, py1, px2, py2 = plate_mapping
    for j in range(len(id_mapping)):
        xcar1, ycar1, xcar2, ycar2 = id_mapping[j]
        if px1 > xcar1 and py1 > ycar1 and px2 < xcar2 and py2 < ycar2:
            car_indx = j
            foundIt = True
            break
    if foundIt == True:
        new_d = id_mapping.copy()
        for key, value in plate_mapping.items():
            if key in new_d:
                new_d[key] += value
            else:
                new_d[key] = value

        return new_d




