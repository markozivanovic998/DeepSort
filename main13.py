# main13.py
import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cvzone
import os
import argparse
import torch
import math
import time
import threading
import yaml

from moduli.rtsp_stream import ThreadedRTSPStream
from moduli.frame_processor import process_frame, init_model
from moduli.visualization import RealTimeVisualizer
from moduli.ThreadVideoCapture import ThreadedVideoCapture
from moduli.TrackingLogic import is_left_side, is_person_inside_vehicle, process_straight_line, process_diagonal_line, process_circle_crossing
from moduli.CliParser import parse_arguments
from moduli.ModelLoader import initialize_model
from moduli.ConfigLoader import load_config
from moduli.VideoSource import get_video_source

args = parse_arguments()

perspective = args.perspective
left_shift = args.left
right_shift = args.right
top_shift = args.top
bottom_shift = args.bottom
diag_shift1 = args.diag_shift1
diag_shift2 = args.diag_shift2

args = parse_arguments()
app_config, rtsp_url = load_config()

OUTPUT_VIDEO_PATH = app_config['video']['output_path']

model,device = initialize_model(app_config['model']['path'])

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]

cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('RGB', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('RGB', RGB)

# OVA LINIJA SADA RADI ISPRAVNO ZBOG IZMENE U VideoSource.py
video_source, frame = get_video_source(args, app_config['video']['input_path'], rtsp_url)

# Provera da li je inicijalizacija uspela
if video_source is None or frame is None:
    print("Nije moguće otvoriti video izvor. Program se gasi.")
    exit()

# Ovi brojači moraju biti deklarisani pre while petlje
counter_down, counter_up, counter_left, counter_right = [], [], [], []
counter_diag1, counter_diag2, counter_diag3, counter_diag4 = [], [], [], []

# Kreiranje instance vizualizatora ako je `--plot` argument prisutan
visualizer = None
if args.plot:
    print("Kreiranje instance RealTimeVisualizer-a...")
    visualizer = RealTimeVisualizer()

frame_height, frame_width = frame.shape[:2]
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
tracker = DeepSort(max_age=30)
count = 0

USE_CIRCLES = args.circle > 0

if USE_CIRCLES:
    frame_center = (frame_width // 2, frame_height // 2)
    circle_radii = [(i + 1) * args.radius for i in range(args.circle)]
    circle_counters = [[] for _ in range(args.circle)]
else:
    if perspective in ['front', 'worm']:
        line_up = int(frame_height * 0.4) + top_shift
        line_down = int(frame_height * 0.6) + bottom_shift
        cx1 = int(frame_width * 0.3)
        cx2 = int(frame_width * 0.7)
    elif perspective == 'side':
        line_left = left_shift if left_shift > 0 else int(frame_width * 0.3)
        line_right = frame_width - right_shift if right_shift > 0 else int(frame_width * 0.7)
        cy1 = int(frame_height * 0.4)
        cy2 = int(frame_height * 0.45)
    else:
        line_up = int(frame_height * 0.4) + top_shift
        line_down = int(frame_height * 0.6) + bottom_shift
        line_left = int(frame_width * 0.3) + left_shift
        line_right = int(frame_width * 0.7) - right_shift

    start_diag1 = (0 + diag_shift1, 0)
    end_diag1 = (frame_width, frame_height - diag_shift1)
    start_diag2 = (frame_width - diag_shift2, 0)
    end_diag2 = (0, frame_height - diag_shift2)
    start_diag3 = (0 + diag_shift1, frame_height)
    end_diag3 = (frame_width, 0 + diag_shift1)
    start_diag4 = (frame_width - diag_shift2, frame_height)
    end_diag4 = (0, 0 + diag_shift2)

person_state = {}
passed_any_line = set()
y_start = int(frame_height * 0.1)
margin_x = int(frame_width * 0.02)
y_step = int(frame_height * 0.05)
scale = max(1, frame_width // 1000)
thickness = max(1, frame_width // 1000)


try:
    while True:
        # 1. Čitanje frejma na početku svake iteracije
        grabbed, frame = video_source.read()

        # 2. Provera da li je frejm uspešno pročitan
        if not grabbed:
            if args.RTSP:
                # Ako je RTSP stream, sačekaj i pokušaj ponovo
                print("Nije moguće dobiti frame sa RTSP streama. Pokušavam ponovo...")
                time.sleep(0.5) 
                continue
            else:
                # Ako je video fajl, došli smo do kraja
                print("Kraj video fajla.")
                break

        count += 1
        results = model.predict(frame, classes=[0, 1, 2, 3, 5, 7], conf=0.10, iou=0.1, imgsz=1920, verbose=False, device=device)

        if args.show_boxes:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()

        a = results[0].boxes.data.cpu()
        px = pd.DataFrame(a).astype("float")

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
        active_people = sum(1 for track in tracks if track.is_confirmed())
        cvzone.putTextRect(annotated_frame, f"Total: {active_people}", (margin_x, y_start - y_step),
                            scale=scale, thickness=thickness, colorT=(0, 0, 0), colorR=(255, 255, 255))
        
        if args.plot and visualizer:
            counters = {
                'up': counter_up, 'right': counter_right, 'diag1': counter_diag1, 'diag3': counter_diag3,
                'down': counter_down, 'left': counter_left, 'diag2': counter_diag2, 'diag4': counter_diag4
            }
            entered = len(counters['up']) + len(counters['right']) + len(counters['diag1']) + len(counters['diag3'])
            exited = len(counters['down']) + len(counters['left']) + len(counters['diag2']) + len(counters['diag4'])

            visualizer.update_data(total_people=active_people, entered=entered, exited=exited, frame_size=(frame_width, frame_height), tracks=tracks)
            visualizer.update_display()
            
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x3, y3, x4, y4 = map(int, track.to_ltrb())
            cx, cy = int((x3 + x4) // 2), int((y3 + y4) // 2)

            if not args.show_boxes:
                cv2.circle(annotated_frame, (cx, cy), 2, (0, 255, 0), 3)
                cv2.putText(annotated_frame, f'ID {track_id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.circle(annotated_frame, (cx, cy), 2, (0, 255, 0), 3)
                cv2.putText(annotated_frame, f'ID {track_id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if track_id in passed_any_line:
                continue

            if USE_CIRCLES:
                for i, r in enumerate(circle_radii):
                    process_circle_crossing(track_id, cx, cy, frame_center, r, i, circle_counters[i], person_state, passed_any_line)
            else:
                if not args.diag:
                    if perspective in ['front', 'worm'] or args.all:
                        process_straight_line(track_id, cy, line_down, 'down', 'above', counter_down, person_state, passed_any_line)
                        process_straight_line(track_id, cy, line_up, 'up', 'below', counter_up, person_state, passed_any_line)
                    if perspective == 'side' or args.all:
                        process_straight_line(track_id, cx, line_right, 'right', 'left', counter_right, person_state, passed_any_line)
                        process_straight_line(track_id, cx, line_left, 'left', 'right', counter_left, person_state, passed_any_line)
                if args.diag or args.all:
                    process_diagonal_line(track_id, cx, cy, start_diag1, end_diag1, 'diag1', counter_diag1, person_state, passed_any_line)
                    process_diagonal_line(track_id, cx, cy, start_diag2, end_diag2, 'diag2', counter_diag2, person_state, passed_any_line)
                    process_diagonal_line(track_id, cx, cy, start_diag3, end_diag3, 'diag3', counter_diag3, person_state, passed_any_line)
                    process_diagonal_line(track_id, cx, cy, start_diag4, end_diag4, 'diag4', counter_diag4, person_state, passed_any_line)

        # Ostatak koda za iscrtavanje je isti
        # ...
        ukupna_suma = 0
        if USE_CIRCLES:
            for r in circle_radii:
                cv2.circle(annotated_frame, frame_center, r, (255, 100, 0), 3)
            for i, counter in enumerate(circle_counters):
                count_val = len(counter)
                cvzone.putTextRect(annotated_frame, f"Circle {i+1}: {count_val}",
                                    (margin_x, y_start + i * y_step),
                                    scale=scale, thickness=thickness, colorT=(255, 255, 255), colorR=(0, 0, 0))
                ukupna_suma += count_val
        else:
            if (perspective in ['front', 'worm'] and not args.diag) or args.all:
                cv2.line(annotated_frame, (0, line_down), (frame_width, line_down), (0, 0, 255), 4)
                cv2.line(annotated_frame, (0, line_up), (frame_width, line_up), (255, 0, 0), 4)
            if (perspective == 'side' and not args.diag) or args.all:
                cv2.line(annotated_frame, (line_left, 0), (line_left, frame_height), (75, 0, 130), 4)
                cv2.line(annotated_frame, (line_right, 0), (line_right, frame_height), (0, 255, 255), 4)
            if args.diag or args.all:
                cv2.line(annotated_frame, start_diag1, end_diag1, (0, 100, 0), 3)
                cv2.putText(annotated_frame, "diag1", (start_diag1[0] + 10, start_diag1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.line(annotated_frame, start_diag2, end_diag2, (0, 165, 255), 3)
                cv2.putText(annotated_frame, "diag2", (start_diag2[0] - 80, start_diag2[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.line(annotated_frame, start_diag3, end_diag3, (0, 100, 0), 3)
                cv2.putText(annotated_frame, "diag3", (start_diag3[0] + 10, start_diag3[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.line(annotated_frame, start_diag4, end_diag4, (0, 165, 255), 3)
                cv2.putText(annotated_frame, "diag4", (start_diag4[0] - 80, start_diag4[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            y_offset_diag = 0
            if (not args.diag) or args.all:
                if perspective in ['front', 'worm'] or args.all:
                    cvzone.putTextRect(annotated_frame, f"Down: {len(counter_down)}", (margin_x, y_start), scale=scale, thickness=thickness, colorT=(255, 255, 255), colorR=(0, 0, 255))
                    cvzone.putTextRect(annotated_frame, f"Up: {len(counter_up)}", (margin_x, y_start + y_step), scale=scale, thickness=thickness, colorT=(255, 255, 255), colorR=(255, 0, 0))
                    ukupna_suma += len(counter_down) + len(counter_up)
                    y_offset_diag += 2
                if perspective == 'side' or args.all:
                    cvzone.putTextRect(annotated_frame, f"Left: {len(counter_left)}", (margin_x, y_start + y_offset_diag * y_step), scale=scale, thickness=thickness, colorT=(255, 255, 255), colorR=(75, 0, 130))
                    cvzone.putTextRect(annotated_frame, f"Right: {len(counter_right)}", (margin_x, y_start + (y_offset_diag + 1) * y_step), scale=scale, thickness=thickness, colorT=(255, 255, 255), colorR=(0, 255, 255))
                    ukupna_suma += len(counter_left) + len(counter_right)
                    y_offset_diag += 2
            if args.all or args.diag:
                cvzone.putTextRect(annotated_frame, f"Diag1: {len(counter_diag1)}", (margin_x, y_start + y_offset_diag * y_step), scale=scale, thickness=thickness, colorT=(0, 0, 0), colorR=(255, 255, 255))
                cvzone.putTextRect(annotated_frame, f"Diag2: {len(counter_diag2)}", (margin_x, y_start + (y_offset_diag + 1) * y_step), scale=scale, thickness=thickness, colorT=(0, 0, 0), colorR=(255, 255, 255))
                cvzone.putTextRect(annotated_frame, f"Diag3: {len(counter_diag3)}", (margin_x, y_start + (y_offset_diag + 2) * y_step), scale=scale, thickness=thickness, colorT=(0, 0, 0), colorR=(255, 255, 255))
                cvzone.putTextRect(annotated_frame, f"Diag4: {len(counter_diag4)}", (margin_x, y_start + (y_offset_diag + 3) * y_step), scale=scale, thickness=thickness, colorT=(0, 0, 0), colorR=(255, 255, 255))
                ukupna_suma += len(counter_diag1) + len(counter_diag2) + len(counter_diag3) + len(counter_diag4)

        y_suma_pos = frame_height - (y_step // 2) - 5
        cvzone.putTextRect(annotated_frame, f"Suma: {ukupna_suma}",
                           (margin_x, y_suma_pos),
                           scale=scale * 1.5,
                           thickness=thickness + 1,
                           colorT=(255, 255, 255), colorR=(0, 128, 0))


        cv2.imshow('RGB', annotated_frame)
        out.write(annotated_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Program prekinut od strane korisnika.")
finally:
    print("Zatvaranje resursa...")
    if 'video_source' in locals() and video_source is not None:
        if hasattr(video_source, 'stop'):
            video_source.stop()
        elif hasattr(video_source, 'release'):
            video_source.release()
    
    if 'out' in locals() and out.isOpened():
        out.release()
    cv2.destroyAllWindows()