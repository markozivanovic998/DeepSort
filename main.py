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
import numpy as np

# Uvoz lokalnih modula
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
from moduls.Evaluation import EvaluationMetrics
from moduls.behavior_analysis import BehaviorAnalytics

# --- INICIJALIZACIJA ---
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

eval_metrics_50 = EvaluationMetrics(num_classes=len(model.names), iou_threshold=0.50)
eval_metrics_75 = EvaluationMetrics(num_classes=len(model.names), iou_threshold=0.75)

behavior_analyzer = None
if args.show_behavior:
    print("[INFO] Analiza ponašanja aktivirana.")
    analysis_zones = {
        'Leva Zona': np.array([
            [0, int(frame_height * 0.5)], 
            [int(frame_width * 0.25), int(frame_height * 0.5)], 
            [int(frame_width * 0.25), frame_height], 
            [0, frame_height]
        ], np.int32),
        'Centar': np.array([
            [int(frame_width * 0.4), int(frame_height * 0.4)],
            [int(frame_width * 0.6), int(frame_height * 0.4)],
            [int(frame_width * 0.6), int(frame_height * 0.6)],
            [int(frame_width * 0.4), int(frame_height * 0.6)]
        ], np.int32)
    }

    behavior_analyzer = BehaviorAnalytics(
        frame_shape=frame.shape, 
        zones=analysis_zones,
        loitering_threshold=15.0, # Prag u sekundama za "zadržavanje"
        speed_threshold=150.0,    # Prag brzine u pikselima/sekundi
        direction_change_threshold=100.0 # Prag za promenu smera u stepenima
    )


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
        
        if metrics_tracker.total_frames_processed == 1:
            gflops = app_config['model'].get('gflops', 0.0)
            metrics_tracker.set_gflops(gflops)

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

        # === USLOVNO AŽURIRANJE STANJA ANALITIKE PONAŠANJA ===
        if args.show_behavior and behavior_analyzer:
            behavior_analyzer.update(tracks)
        # ========================================================
        
        update_crossings(tracks, params, args, perspective)
        
        if not args.show_boxes:
            draw_track_annotations(annotated_frame, tracks)
            
        draw_zones_and_stats(annotated_frame, params, args, perspective)

        # === USLOVNO ISCRTAVANJE REZULTATA ANALITIKE PONAŠANJA ===
        if args.show_behavior and behavior_analyzer:
            # Prvo iscrtaj heatmapu kao pozadinu
            annotated_frame = behavior_analyzer.draw_heatmap(annotated_frame, alpha=0.4)
            # Zatim iscrtaj informacije o analizi (zone, anomalije, itd.)
            annotated_frame = behavior_analyzer.draw_analytics(annotated_frame)
        # ===============================================================

        active_people = sum(1 for track in tracks if track.is_confirmed())
        ui = params['ui']
        cvzone.putTextRect(annotated_frame, f"Total: {active_people}", (ui['margin_x'], ui['y_start'] - ui['y_step']),
                            scale=ui['scale'], thickness=ui['thickness'], colorT=(0, 0, 0), colorR=(255, 255, 255))
        
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
        
        eval_summary = {
            "Precision_Recall_F1_at_IoU_0.50": eval_metrics_50.calculate_metrics(),
            "Precision_Recall_F1_at_IoU_0.75": eval_metrics_75.calculate_metrics()
        }
        
        metrics_tracker.update_final_counts(params)
        metrics_tracker.set_evaluation_summary(eval_summary)
        
        metrics_tracker.print_summary()
        metrics_tracker.save_summary_to_json("metrics_report.json")
    else:
        print("\n[INFO] Program završen. Za prikaz statistike, pokrenite sa --show-stats opcijom.")

    if video_source:
        video_source.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("[INFO] Svi resursi su oslobođeni.")

