import cv2
import pandas as pd
import torch
import numpy as np
from ultralytics import YOLO

def init_model(config):
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    half = device != 'cpu'
    
    model_path = config['model']['path']
    model = YOLO(model_path)
    model.to(device)
    model.fuse()
    

    if half:
        model.half()
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    return model

def is_person_inside_vehicle(p_box, v_box, threshold=0.3):
    x_overlap = max(0, min(p_box[2], v_box[2]) - max(p_box[0], v_box[0]))
    y_overlap = max(0, min(p_box[3], v_box[3]) - max(p_box[1], v_box[1]))
    overlap_area = x_overlap * y_overlap
    if overlap_area == 0: 
        return False
    person_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
    if person_area == 0: 
        return False
    if (overlap_area / person_area) > threshold: 
        return True
    person_center = ((p_box[0] + p_box[2]) // 2, (p_box[1] + p_box[3]) // 2)
    if (v_box[0] <= person_center[0] <= v_box[2] and v_box[1] <= person_center[1] <= v_box[3]): 
        return True
    return False

def process_frame(frame, model, tracker, config, args):
    processing_size = 1920
    h, w = frame.shape[:2]
    scale = processing_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    
    with torch.no_grad():
        results = model.predict(resized_frame, 
                              classes=config['model']['classes'],
                              conf=config['model']['conf_threshold'],
                              iou=config['model']['iou_threshold'],
                              imgsz=processing_size,
                              verbose=False,
                              device=model.device)

    if args.show_boxes:
        annotated_frame = results[0].plot()
        if scale != 1.0:
            annotated_frame = cv2.resize(annotated_frame, (w, h))
    else:
        annotated_frame = frame.copy()

    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a).astype("float")
    
    if scale != 1.0:
        px.iloc[:, :4] = px.iloc[:, :4] / scale

    vehicle_boxes = []
    raw_person_detections = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, conf, cls = int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4], int(row[5])
        if cls in [1, 2, 3, 5, 7]:
            vehicle_boxes.append([x1, y1, x2, y2])
        elif cls == 0:
            raw_person_detections.append({'box': [x1, y1, x2, y2], 'conf': conf})

    detections_for_tracker = []
    for p_det in raw_person_detections:
        x1, y1, x2, y2 = p_det['box']
        conf = p_det['conf']
        is_inside = any(is_person_inside_vehicle([x1, y1, x2, y2], v_box) for v_box in vehicle_boxes)
        if not is_inside:
            detections_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
    return annotated_frame, tracks, vehicle_boxes