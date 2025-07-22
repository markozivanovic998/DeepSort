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
from moduli.lineConfig import initialize_parameters
from moduli.detection import filter_detections
from moduli.counting import update_crossings
from moduli.drawing import draw_zones_and_stats, draw_track_annotations

args = parse_arguments()

perspective = args.perspective
left_shift = args.left
right_shift = args.right
top_shift = args.top
bottom_shift = args.bottom
diag_shift1 = args.diag_shift1
diag_shift2 = args.diag_shift2
circle = args.circle
radius= args.radius

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

video_source, frame = get_video_source(args, app_config['video']['input_path'], rtsp_url)
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
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, app_config['video']['default_fps'], (frame_width, frame_height))
tracker = DeepSort(max_age=app_config['tracker']['max_age'])
count = 0

params = initialize_parameters(frame_width, frame_height, args)

try:
    while True:
        grabbed, frame = video_source.read()
        if not grabbed:
            if args.RTSP:
                print("Nije moguće dobiti frame sa RTSP streama. Pokušavam ponovo...")
                time.sleep(0.5) 
                continue
            else:
                print("Kraj video fajla.")
                break

        count += 1
        results = model.predict(frame, classes=app_config['model']['classes'], conf=app_config['model']['conf_threshold'], iou=app_config['model']['iou_threshold'], imgsz=app_config['model']['imgsz'], verbose=False, device=device)

        if args.show_boxes:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()


        detections_for_tracker = filter_detections(results)

        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        update_crossings(tracks,params,args,perspective)
        
                # 5. Iscrtavanje anotacija, zona i statistike
        if not args.show_boxes:
            draw_track_annotations(annotated_frame, tracks)
        draw_zones_and_stats(annotated_frame, params, args,perspective)

        active_people = sum(1 for track in tracks if track.is_confirmed())
        ui = params['ui']
        cvzone.putTextRect(annotated_frame, f"Total: {active_people}", (ui['margin_x'], ui['y_start'] - ui['y_step']),
                            scale=ui['scale'], thickness=ui['thickness'], colorT=(0, 0, 0), colorR=(255, 255, 255))
        
        if args.plot and visualizer:
            entered = len(params['counters'].get('up', [])) + len(params['counters'].get('right', [])) # Dodati i dijagonale ako treba
            exited = len(params['counters'].get('down', [])) + len(params['counters'].get('left', [])) # Dodati i dijagonale
            visualizer.update_data(total_people=active_people, entered=entered, exited=exited, frame_size=(frame_width, frame_height), tracks=tracks)
            visualizer.update_display()

        # Prikaz frejma
        cv2.imshow("Object Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_source.release()
    cv2.destroyAllWindows()