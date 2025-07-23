# detection.py

import pandas as pd
from moduls.TrackingLogic import is_person_inside_vehicle
# Pretpostavka je da se funkcija `is_person_inside_vehicle` nalazi u nekom pomoćnom fajlu, npr. `utils.py`
# from utils import is_person_inside_vehicle

def filter_detections(results):
    """
    Obrađuje rezultate modela, filtrira osobe koje su u vozilima i vraća listu detekcija za tracker.
    """
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a).astype("float")

    vehicle_boxes = []
    raw_person_detections = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, conf, cls = int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4], int(row[5])
        # Klase za vozila: 1: car, 2: motorcycle, 3: airplane, 5: bus, 7: truck
        if cls in [1, 2, 3, 5, 7]:
            vehicle_boxes.append([x1, y1, x2, y2])
        # Klasa za osobu: 0
        elif cls == 0:
            raw_person_detections.append({'box': [x1, y1, x2, y2], 'conf': conf})

    detections_for_tracker = []
    for p_det in raw_person_detections:
        x1, y1, x2, y2 = p_det['box']
        conf = p_det['conf']
        is_inside = any(is_person_inside_vehicle([x1, y1, x2, y2], v_box) for v_box in vehicle_boxes)
        if not is_inside:
            detections_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
            
    return detections_for_tracker