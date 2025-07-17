#!/usr/bin/env python3

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU

import cv2
import os
import cvzone
import argparse
import time
import yaml
import traceback
import torch
import logging
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort

from modules.frame_processor import init_model, process_frame
from modules.rtsp_stream import ThreadedRTSPStream
from modules.video_capture import ThreadedVideoCapture
from modules.line_processor import process_straight_line, process_diagonal_line, process_circle_crossing
from modules.counter_display import draw_lines, draw_circles, display_counters
from modules.visualization import RealTimeVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open('../config/app_config.yaml') as f: 
        app_config = yaml.safe_load(f)
    with open('../config/rtsp_config.yaml') as f: 
        rtsp_config = yaml.safe_load(f)
    with open('../config/line_config.yaml') as f: 
        line_config = yaml.safe_load(f)
    return app_config, rtsp_config, line_config

def setup_cli():
    parser = argparse.ArgumentParser(description='Analiza videa za brojanje ljudi')
    
    line_group = parser.add_argument_group('Opcije za linije')
    line_group.add_argument('--perspective', type=str, default='front', 
                          choices=['front', 'side', 'worm', 'top'],
                          help='Perspektiva kamere')
    line_group.add_argument('--left', type=int, default=0, 
                          help='Pomeranje leve linije')
    line_group.add_argument('--right', type=int, default=0, 
                          help='Pomeranje desne linije')
    line_group.add_argument('--top', type=int, default=0, 
                          help='Pomeranje gornje linije')
    line_group.add_argument('--bottom', type=int, default=0, 
                          help='Pomeranje donje linije')
    line_group.add_argument('--all', action='store_true',
                          help='Prikaži sve linije')
    line_group.add_argument('--diag', action='store_true',
                          help='Koristi samo dijagonale')
    line_group.add_argument('--show_boxes', action='store_true', 
                          help='Prikaži bounding box-ove')
    line_group.add_argument('--diag_shift1', type=int, default=0, 
                          help='Pomeranje dijagonale 1 i 3')
    line_group.add_argument('--diag_shift2', type=int, default=0, 
                          help='Pomeranje dijagonale 2 i 4')
    line_group.add_argument('--RTSP', action='store_true', 
                          help='Koristi RTSP stream')

    circle_group = parser.add_argument_group('Opcije za krugove')
    circle_group.add_argument('--circle', type=int, default=0, 
                            help='Broj krugova')
    circle_group.add_argument('--radius', type=int, default=50, 
                            help='Razmak između krugova')

    viz_group = parser.add_argument_group('Vizualizacione opcije')
    viz_group.add_argument('--plot', action='store_true',
                         help='Prikaži dashboard sa statistikama u realnom vremenu')
    viz_group.add_argument('--headless', action='store_true',
                         help='Pokreće bez GUI (korisno za server)')
    viz_group.add_argument('--reset_stats', action='store_true',
                         help='Resetuje sve statističke podatke')
    
    return parser.parse_args()

def initialize_model(app_config):
    logger.info("Initializing model...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        app_config['device'] = device
        
        # Dodatna GPU diagnostika
        if device == 'cuda':
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"Device name: {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True
        
        model = init_model(app_config)
        
        # Prebacivanje modela na GPU
        model = model.to(device)
        logger.info(f"Model moved to: {next(model.parameters()).device}")
        
        # Ne koristimo half-precision za sada
        # model.half()
        
    except RuntimeError as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise
    return model

def setup_video_capture(args, app_config, rtsp_config):
    logger.info("Initializing video capture...")
    if args.RTSP:
        rtsp_url = f"rtsp://{rtsp_config['rtsp']['username']}:{rtsp_config['rtsp']['password']}@" \
                  f"{rtsp_config['rtsp']['ip']}:{rtsp_config['rtsp']['port']}{rtsp_config['rtsp']['path']}"
        logger.info(f"Connecting to RTSP stream: {rtsp_url}")
        video_capture = ThreadedRTSPStream(rtsp_url)
    else:
        input_path = app_config['video']['input_path']
        logger.info(f"Using local video: {input_path}")
        video_capture = ThreadedVideoCapture(input_path)
    
    video_capture.start()
    time.sleep(2.0)  # Give more time to initialize
    return video_capture

def initialize_video_writer(frame, app_config):
    frame_height, frame_width = frame.shape[:2]
    codecs = ['mp4v', 'avc1', 'h264', 'x264', 'MJPG', 'XVID']
    
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            app_config['video']['output_path'],
            fourcc,
            app_config['video']['default_fps'],
            (frame_width, frame_height))
        if out.isOpened():
            logger.info(f"Using codec: {codec}")
            return out, frame_width, frame_height
        out.release()
    raise RuntimeError("Nijedan codec nije uspio otvoriti VideoWriter")

def setup_gui(args, frame_width, frame_height):
    if not args.headless:
        try:
            cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('RGB', frame_width // 2, frame_height // 2)
            return True
        except Exception as e:
            logger.warning(f"GUI not available: {str(e)}. Continuing in headless mode")
            return False
    return False

def main():
    video_capture = None
    out = None
    frame_queue = deque(maxlen=2)
    visualizer = None
    gui_enabled = False
    
    try:
        args = setup_cli()
        app_config, rtsp_config, line_config = load_config()
        
        model = initialize_model(app_config)
        tracker = DeepSort(max_age=app_config['tracker']['max_age'])
        
        if args.plot:
            visualizer = RealTimeVisualizer()
            if args.reset_stats:
                visualizer.reset_data()
        
        video_capture = setup_video_capture(args, app_config, rtsp_config)
        grabbed, frame = video_capture.read()
        if not grabbed or frame is None:
            raise RuntimeError("Failed to read initial frame")
        
        out, frame_width, frame_height = initialize_video_writer(frame, app_config)
        logger.info(f"Video dimensions: {frame_width}x{frame_height}")
        
        gui_enabled = setup_gui(args, frame_width, frame_height)
        
        # Display configuration
        y_start = int(frame_height * line_config['display']['y_start'])
        margin_x = int(frame_width * line_config['display']['margin_x'])
        y_step = int(frame_height * line_config['display']['y_step'])
        scale = max(1, frame_width // 1000)
        thickness = max(1, frame_width // 1000)
        
        # Initialize counters
        person_state = {}
        passed_any_line = set()
        counters = {
            'down': [], 'up': [], 'left': [], 'right': [],
            'diag1': [], 'diag2': [], 'diag3': [], 'diag4': []
        }
        circle_counters = [[] for _ in range(args.circle)] if args.circle > 0 else []
        
        if args.circle > 0:
            frame_center = (frame_width // 2, frame_height // 2)
            circle_radii = [(i + 1) * args.radius for i in range(args.circle)]
        else:
            perspective = args.perspective
            line_positions = line_config['line_positions'][perspective]
            
            line_up = int(frame_height * line_positions['up']) + args.top
            line_down = int(frame_height * line_positions['down']) + args.bottom
            line_left = args.left if args.left > 0 else int(frame_width * line_positions['left'])
            line_right = frame_width - args.right if args.right > 0 else int(frame_width * line_positions['right'])
            
            diag_lines = [
                ((0 + args.diag_shift1, 0), (frame_width, frame_height - args.diag_shift1)),
                ((frame_width - args.diag_shift2, 0), (0, frame_height - args.diag_shift2)),
                ((0 + args.diag_shift1, frame_height), (frame_width, 0 + args.diag_shift1)),
                ((frame_width - args.diag_shift2, frame_height), (0, 0 + args.diag_shift2))
            ]
        
        logger.info("Starting processing loop...")
        last_frame_time = time.time()
        frame_count = 0
        skip_frames = 0  # For handling HEVC errors
        
        while True:
            start_time = time.time()
            frame_count += 1
            
            grabbed, frame = video_capture.read()
            if not grabbed or frame is None:
                logger.warning("No frame received. Skipping...")
                time.sleep(0.1)
                continue
                
            # Skip frames to recover from HEVC errors
            if skip_frames > 0:
                skip_frames -= 1
                continue
                
            annotated_frame = frame.copy()
            
            # GPU sinkronizacija
            if app_config['device'] == 'cuda':
                torch.cuda.synchronize()
                
            results = process_frame(frame, model, tracker, app_config, args)
            
            if results is None:
                continue
                
            annotated_frame, tracks, _ = results
            
            # Process tracks
            active_people = 0
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                active_people += 1
                track_id = track.track_id
                x3, y3, x4, y4 = map(int, track.to_ltrb())
                cx, cy = int((x3 + x4) // 2), int((y3 + y4) // 2)
                
                if not args.show_boxes:
                    cv2.circle(annotated_frame, (cx, cy), 2, (0, 255, 0), 3)
                    cv2.putText(annotated_frame, f'ID {track_id}', (x3, y3 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                if track_id in passed_any_line:
                    continue
                
                if args.circle > 0:
                    for i, r in enumerate(circle_radii):
                        process_circle_crossing(track_id, cx, cy, frame_center, r, i,
                                              circle_counters[i], person_state, passed_any_line)
                else:
                    if not args.diag:
                        if args.perspective in ['front', 'worm'] or args.all:
                            process_straight_line(track_id, cy, line_down, 'down', 'above',
                                                counters['down'], person_state, passed_any_line)
                            process_straight_line(track_id, cy, line_up, 'up', 'below',
                                                counters['up'], person_state, passed_any_line)
                        if args.perspective == 'side' or args.all:
                            process_straight_line(track_id, cx, line_right, 'right', 'left',
                                                counters['right'], person_state, passed_any_line)
                            process_straight_line(track_id, cx, line_left, 'left', 'right',
                                                counters['left'], person_state, passed_any_line)
                    
                    if args.diag or args.all:
                        for i, (start, end) in enumerate(diag_lines, start=1):
                            process_diagonal_line(track_id, cx, cy, start, end, f'diag{i}',
                                                counters[f'diag{i}'], person_state, passed_any_line)
            
            # Update visualization
            if args.plot and visualizer:
                entered = len(counters['up']) + len(counters['right']) + len(counters['diag1']) + len(counters['diag3'])
                exited = len(counters['down']) + len(counters['left']) + len(counters['diag2']) + len(counters['diag4'])
                
                visualizer.update_data(
                    total_people=active_people,
                    entered=entered,
                    exited=exited,
                    frame_size=(frame_width, frame_height),
                    tracks=tracks
                )
                visualizer.update_display()
            
            # Display counters
            cvzone.putTextRect(
                annotated_frame, f"Total: {active_people}",
                (margin_x, y_start - y_step),
                scale=scale, thickness=thickness,
                colorT=(0, 0, 0), colorR=(255, 255, 255))
            
            if args.circle > 0:
                draw_circles(annotated_frame, frame_center, circle_radii)
                for i, counter in enumerate(circle_counters):
                    count_val = len(counter)
                    cvzone.putTextRect(
                        annotated_frame, f"Circle {i+1}: {count_val}",
                        (margin_x, y_start + i * y_step),
                        scale=scale, thickness=thickness,
                        colorT=(255, 255, 255), colorR=(0, 0, 0))
            else:
                draw_lines(
                    annotated_frame, args.perspective, args,
                    line_up, line_down, line_left, line_right,
                    *[line for diag in diag_lines for line in diag],
                    frame_width, frame_height)
                
                annotated_frame = display_counters(
                    annotated_frame, args.perspective, args,
                    counters,  # Pass dictionary instead of individual counters
                    frame_width, frame_height,
                    y_start, y_step, margin_x, scale, thickness)
            
            # Display and recording
            if gui_enabled:
                cv2.imshow('RGB', annotated_frame)
            
            if out:
                frame_queue.append(annotated_frame)
                if len(frame_queue) >= 2:
                    out.write(frame_queue.popleft())
            
            # Control speed
            elapsed = time.time() - start_time
            target_delay = max(1, int((1000/video_capture.fps) - elapsed*1000))
            
            if gui_enabled:
                key = cv2.waitKey(target_delay) & 0xFF
                if key == ord('q'):
                    logger.info("User requested exit")
                    break
                elif key == ord('r') and args.plot and visualizer:
                    logger.info("Resetting statistics...")
                    visualizer.reset_data()
            
            # Handle HEVC errors by skipping frames
            if frame_count % 30 == 0 and args.RTSP:
                skip_frames = 2
            
            # Maintain real FPS
            while time.time() - last_frame_time < (1.0 / video_capture.fps):
                time.sleep(0.001)
            last_frame_time = time.time()
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        traceback.print_exc()
    finally:
        logger.info("Cleaning up resources...")
        if video_capture is not None:
            video_capture.stop()
            video_capture.release()
        if out is not None:
            while len(frame_queue) > 0:
                out.write(frame_queue.popleft())
            out.release()
        if visualizer is not None:
            visualizer.close()
        if gui_enabled:
            cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Exit complete")

if __name__ == "__main__":
    main()