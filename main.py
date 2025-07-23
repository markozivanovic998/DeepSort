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
from collections import Counter, defaultdict 

from moduls.rtsp_stream import ThreadedRTSPStream
from moduls.frame_processor import process_frame, init_model
from moduls.visualization import RealTimeVisualizer
from moduls.ThreadVideoCapture import ThreadedVideoCapture
from moduls.TrackingLogic import is_left_side, is_person_inside_vehicle, process_straight_line, process_diagonal_line, process_circle_crossing
from moduls.CliParser import parse_arguments
from moduls.ModelLoader import initialize_model
from moduls.ConfigLoader import load_config
from moduls.VideoSource import get_video_source
from moduls.lineConfig import initialize_parameters
from moduls.detection import filter_detections
from moduls.counting import update_crossings
from moduls.drawing import draw_zones_and_stats, draw_track_annotations
from moduls.Metrics import MetricsTracker

args = parse_arguments()
app_config, rtsp_url = load_config()

perspective = args.perspective
left_shift = args.left
right_shift = args.right
top_shift = args.top
bottom_shift = args.bottom
diag_shift1 = args.diag_shift1
diag_shift2 = args.diag_shift2
circle = args.circle
radius = args.radius

OUTPUT_VIDEO_PATH = app_config['video']['output_path']

model, device = initialize_model(app_config['model']['path'])
video_source, frame = get_video_source(args, app_config['video']['input_path'], rtsp_url)

if video_source is None or frame is None:
    print("Nije moguće otvoriti video izvor. Program se gasi.")
    exit()

frame_height, frame_width = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, app_config['video']['default_fps'], (frame_width, frame_height))
tracker = DeepSort(max_age=app_config['tracker']['max_age'])

visualizer = None
if args.plot:
    print("Kreiranje instance RealTimeVisualizer-a...")
    visualizer = RealTimeVisualizer()

params = initialize_parameters(frame_width, frame_height, args)
params['crossings_by_class'] = defaultdict(lambda: defaultdict(int))

metrics_tracker = MetricsTracker()
count = 0
previous_track_ids = set() 

stats_interval = 60 
last_stats_time = time.time()

try:
    while True:
        metrics_tracker.start_frame()

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
        
        detection_start_time = time.time()
        results = model.predict(frame, classes=app_config['model']['classes'], conf=app_config['model']['conf_threshold'], iou=app_config['model']['iou_threshold'], imgsz=app_config['model']['imgsz'], verbose=False, device=device)
        detection_time = time.time() - detection_start_time
        
        boxes = results[0].boxes
        avg_confidence = boxes.conf.mean().item() if len(boxes.conf) > 0 else 0.0
        class_indices = boxes.cls.cpu().numpy().astype(int)
        detections_by_class = {model.names[i]: count for i, count in Counter(class_indices).items()}

        if args.show_boxes:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()

        detections_for_tracker = filter_detections(results)
        
        tracking_start_time = time.time()
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
        tracking_time = time.time() - tracking_start_time
        
        current_track_ids = {track.track_id for track in tracks}
        newly_initiated_tracks_count = len(current_track_ids - previous_track_ids)
        previous_track_ids = current_track_ids

        update_crossings(tracks, params, args, perspective)
        
        if not args.show_boxes:
            draw_track_annotations(annotated_frame, tracks)
        draw_zones_and_stats(annotated_frame, params, args, perspective)

        active_people = sum(1 for track in tracks if track.is_confirmed())
        ui = params['ui']
        cvzone.putTextRect(annotated_frame, f"Total: {active_people}", (ui['margin_x'], ui['y_start'] - ui['y_step']),
                            scale=ui['scale'], thickness=ui['thickness'], colorT=(0, 0, 0), colorR=(255, 255, 255))
        
        if args.plot and visualizer:
            entered = len(params['counters'].get('up', [])) + len(params['counters'].get('right', []))
            exited = len(params['counters'].get('down', [])) + len(params['counters'].get('left', []))
            visualizer.update_data(total_people=active_people, entered=entered, exited=exited, frame_size=(frame_width, frame_height), tracks=tracks)
            visualizer.update_display()
        
        detections_count = len(detections_for_tracker)
        active_tracks_count = len(tracks)
        metrics_tracker.end_frame(
            detections_count=detections_count,
            active_tracks_count=active_tracks_count,
            detection_time=detection_time,
            tracking_time=tracking_time,
            new_tracks_count=newly_initiated_tracks_count,
            avg_confidence=avg_confidence,
            detections_by_class=detections_by_class
        )

        current_time = time.time()
        if args.show_stats and (current_time - last_stats_time) >= stats_interval:
            print(f"\n--- PERIODIČNI IZVEŠTAJ STATISTIKE ({time.strftime('%H:%M:%S')}) ---")
            metrics_tracker.update_final_counts(params)
            metrics_tracker.print_summary()
            print("--- KRAJ IZVEŠTAJA (stream se nastavlja) ---\n")
            last_stats_time = current_time

        cv2.imshow("Object Tracking", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    if args.show_stats:
        print("\n[INFO] Obrada završena. Generisanje FINALNOG izveštaja o metrikama...")
        metrics_tracker.update_final_counts(params)
        metrics_tracker.print_summary()
        metrics_tracker.save_summary_to_json("metrics_report.json")
    else:
        print("\n[INFO] Program završen. Za prikaz statistike, pokrenite sa --show-stats opcijom.")

    video_source.release()
    cv2.destroyAllWindows()
    if out:
        out.release()
    print("[INFO] Svi resursi su oslobođeni.")