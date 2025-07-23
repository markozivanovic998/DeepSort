
# counting.py
# from utils import process_circle_crossing, process_straight_line, process_diagonal_line
from moduls.TrackingLogic import process_straight_line, process_diagonal_line, process_circle_crossing

def update_crossings(tracks, params, args, perspective):
    """
    Ažurira brojače na osnovu kretanja praćenih objekata preko definisanih zona.
    """
    zones = params['zones']
    counters = params['counters']
    person_state = params['state']['person_state']
    passed_any_line = params['state']['passed_any_line']

    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        x3, y3, x4, y4 = map(int, track.to_ltrb())
        cx, cy = int((x3 + x4) // 2), int((y3 + y4) // 2)

        if params['USE_CIRCLES']:
            for i, r in enumerate(zones['circle_radii']):
                process_circle_crossing(track_id, cx, cy, zones['frame_center'], r, i, counters['circles'][i], person_state, passed_any_line)
        else:
            if not args.diag:
                if perspective in ['front', 'worm'] or args.all:
                    process_straight_line(track_id, cy, zones['line_down'], 'down', 'above', counters['down'], person_state, passed_any_line)
                    process_straight_line(track_id, cy, zones['line_up'], 'up', 'below', counters['up'], person_state, passed_any_line)
                if perspective == 'side' or args.all:
                    process_straight_line(track_id, cx, zones['line_right'], 'right', 'left', counters['right'], person_state, passed_any_line)
                    process_straight_line(track_id, cx, zones['line_left'], 'left', 'right', counters['left'], person_state, passed_any_line)
            
            if args.diag or args.all:
                process_diagonal_line(track_id, cx, cy, zones['start_diag1'], zones['end_diag1'], 'diag1', counters['diag1'], person_state, passed_any_line)
                process_diagonal_line(track_id, cx, cy, zones['start_diag2'], zones['end_diag2'], 'diag2', counters['diag2'], person_state, passed_any_line)
                process_diagonal_line(track_id, cx, cy, zones['start_diag3'], zones['end_diag3'], 'diag3', counters['diag3'], person_state, passed_any_line)
                process_diagonal_line(track_id, cx, cy, zones['start_diag4'], zones['end_diag4'], 'diag4', counters['diag4'], person_state, passed_any_line)