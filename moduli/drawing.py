import cv2
import cvzone
import numpy as np

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
    Iscrtava zone za brojanje (linije/krugove) i prikazuje statistiku, uključujući sume za dijagonale i ukupan zbir.
    """
    ui = params['ui']
    zones = params['zones']
    counters = params['counters']
    frame_height, frame_width, _ = frame.shape

    # Inicijalizacija ukupne sume i početne Y pozicije za ispis statistike
    total_sum = 0
    y_stat_pos = ui['y_start']

    # Kreiranje rečnika sa stilovima koji su validni za putTextRect.
    # Ovo sprečava prosleđivanje nevažećih argumenata kao što je 'y_start'.
    text_style = {
        'scale': ui.get('scale', 1.5),      # Koristi vrednost iz ui ili podrazumevanu
        'thickness': ui.get('thickness', 2),
        'font': ui.get('font', cv2.FONT_HERSHEY_SIMPLEX),
        'colorT': ui.get('colorT', (255, 255, 255)) # Podrazumevana bela boja teksta
    }


    if params['USE_CIRCLES']:
        for r in zones['circle_radii']:
            cv2.circle(frame, zones['frame_center'], r, (255, 100, 0), 3)
        for i, counter in enumerate(counters['circles']):
            count = len(counter)
            total_sum += count
            cvzone.putTextRect(frame, f"Circle {i+1}: {count}",
                               (ui['margin_x'], y_stat_pos),
                               **text_style, colorR=(0, 0, 0))
            y_stat_pos += ui['y_step']
    else:
        # Iscrtavanje horizontalnih linija i njihovih brojača
        if (perspective in ['front', 'worm'] and not args.diag) or args.all:
            cv2.line(frame, (0, zones['line_down']), (frame_width, zones['line_down']), (0, 0, 255), 4)
            cv2.line(frame, (0, zones['line_up']), (frame_width, zones['line_up']), (255, 0, 0), 4)
            
            count_down = len(counters.get('down', []))
            count_up = len(counters.get('up', []))
            total_sum += count_down + count_up
            
            cvzone.putTextRect(frame, f"Down: {count_down}", (ui['margin_x'], y_stat_pos), **text_style, colorR=(0, 0, 255))
            y_stat_pos += ui['y_step']
            cvzone.putTextRect(frame, f"Up: {count_up}", (ui['margin_x'], y_stat_pos), **text_style, colorR=(255, 0, 0))
            y_stat_pos += ui['y_step']
        
        # Iscrtavanje vertikalnih linija i njihovih brojača
        if (perspective == 'side' and not args.diag) or args.all:
            cv2.line(frame, (zones['line_left'], 0), (zones['line_left'], frame_height), (75, 0, 130), 4)
            cv2.line(frame, (zones['line_right'], 0), (zones['line_right'], frame_height), (0, 255, 255), 4)

            count_left = len(counters.get('left', []))
            count_right = len(counters.get('right', []))
            total_sum += count_left + count_right

            cvzone.putTextRect(frame, f"Left: {count_left}", (ui['margin_x'], y_stat_pos), **text_style, colorR=(75, 0, 130))
            y_stat_pos += ui['y_step']
            cvzone.putTextRect(frame, f"Right: {count_right}", (ui['margin_x'], y_stat_pos), **text_style, colorR=(0, 255, 255))
            y_stat_pos += ui['y_step']

        # Iscrtavanje dijagonalnih linija i njihovih brojača
        if args.diag or args.all:
            shift_x, shift_y = args.diag_shift1, args.diag_shift2
            center_x, center_y = frame_width // 2, frame_height // 2

            point_top = (center_x, center_y - shift_y)
            point_bottom = (center_x, center_y + shift_y)
            point_left = (center_x - shift_x, center_y)
            point_right = (center_x + shift_x, center_y)

            cv2.line(frame, point_top, point_left, (144, 238, 144), 3)
            cv2.line(frame, point_top, point_right, (173, 255, 47), 3)
            cv2.line(frame, point_bottom, point_left, (138, 43, 226), 3)
            cv2.line(frame, point_bottom, point_right, (255, 105, 180), 3)

            count_tl = len(counters.get('diag_tl', []))
            count_tr = len(counters.get('diag_tr', []))
            count_bl = len(counters.get('diag_bl', []))
            count_br = len(counters.get('diag_br', []))
            total_sum += count_tl + count_tr + count_bl + count_br

            cvzone.putTextRect(frame, f"Diag TL: {count_tl}", (ui['margin_x'], y_stat_pos), **text_style, colorR=(144, 238, 144))
            y_stat_pos += ui['y_step']
            cvzone.putTextRect(frame, f"Diag TR: {count_tr}", (ui['margin_x'], y_stat_pos), **text_style, colorR=(173, 255, 47))
            y_stat_pos += ui['y_step']
            cvzone.putTextRect(frame, f"Diag BL: {count_bl}", (ui['margin_x'], y_stat_pos), **text_style, colorR=(138, 43, 226))
            y_stat_pos += ui['y_step']
            cvzone.putTextRect(frame, f"Diag BR: {count_br}", (ui['margin_x'], y_stat_pos), **text_style, colorR=(255, 105, 180))
            y_stat_pos += ui['y_step']

    # Iscrtavanje ukupne sume na dnu ekrana
    sum_y_pos = frame_height - ui['y_start'] - 20 
    cvzone.putTextRect(frame, f"Sum: {total_sum}",
                       (ui['margin_x'], sum_y_pos),
                       **text_style, colorR=(0, 0, 0))