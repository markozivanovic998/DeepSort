# drawing.py

import cv2
import cvzone

def draw_track_annotations(frame, tracks):
    """
    Iscrtava anotacije za svaki praćeni objekat (ID i centralnu tačku).
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x3, y3, x4, y4 = map(int, track.to_ltrb())
        cv2.circle(frame, (int((x3 + x4) / 2), int((y3+y4) / 2)), 2, (0, 255, 0), 3)
        cv2.putText(frame, f'ID {track_id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def draw_zones_and_stats(frame, params, args, perspective):
    """
    Iscrtava zone za brojanje (linije/krugove) i prikazuje statistiku.
    """
    ui = params['ui']
    zones = params['zones']
    counters = params['counters']
    frame_height, frame_width, _ = frame.shape

    if params['USE_CIRCLES']:
        for r in zones['circle_radii']:
            cv2.circle(frame, zones['frame_center'], r, (255, 100, 0), 3)
        for i, counter in enumerate(counters['circles']):
            cvzone.putTextRect(frame, f"Circle {i+1}: {len(counter)}",
                               (ui['margin_x'], ui['y_start'] + i * ui['y_step']),
                               scale=ui['scale'], thickness=ui['thickness'], colorT=(255, 255, 255), colorR=(0, 0, 0))
    else:
        y_offset_diag = 0
        if (perspective in ['front', 'worm'] and not args.diag) or args.all:
            cv2.line(frame, (0, zones['line_down']), (frame_width, zones['line_down']), (0, 0, 255), 4)
            cv2.line(frame, (0, zones['line_up']), (frame_width, zones['line_up']), (255, 0, 0), 4)
            #cvzone.putTextRect(frame, f"Down: {len(counters['down'])}", (ui['margin_x'], ui['y_start']), **ui, colorR=(0, 0, 255))
            #cvzone.putTextRect(frame, f"Up: {len(counters['up'])}", (ui['margin_x'], ui['y_start'] + ui['y_step']), **ui, colorR=(255, 0, 0))
            y_offset_diag += 2
        
        if (perspective == 'side' and not args.diag) or args.all:
            cv2.line(frame, (zones['line_left'], 0), (zones['line_left'], frame_height), (75, 0, 130), 4)
            cv2.line(frame, (zones['line_right'], 0), (zones['line_right'], frame_height), (0, 255, 255), 4)
            cvzone.putTextRect(frame, f"Left: {len(counters['left'])}", (ui['margin_x'], ui['y_start'] + y_offset_diag * ui['y_step']), **ui, colorR=(75, 0, 130))
            cvzone.putTextRect(frame, f"Right: {len(counters['right'])}", (ui['margin_x'], ui['y_start'] + (y_offset_diag + 1) * ui['y_step']), **ui, colorR=(0, 255, 255))
            y_offset_diag += 2

        if args.diag or args.all:
            # Implementacija za dijagonalne linije i brojače po potrebi...
            cv2.line(frame, zones['start_diag1'], zones['end_diag1'], (0, 100, 0), 3)
            # ... ostatak iscrtavanja za dijagonale