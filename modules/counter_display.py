import cv2
import cvzone

def draw_lines(annotated_frame, perspective, args, line_up, line_down, line_left, line_right,
              start_diag1, end_diag1, start_diag2, end_diag2,
              start_diag3, end_diag3, start_diag4, end_diag4,
              frame_width, frame_height):
    
    if (perspective in ['front', 'worm'] and not args.diag) or args.all:
        cv2.line(annotated_frame, (0, line_down), (frame_width, line_down), (0, 0, 255), 4)
        cv2.line(annotated_frame, (0, line_up), (frame_width, line_up), (255, 0, 0), 4)
    
    if (perspective == 'side' and not args.diag) or args.all:
        cv2.line(annotated_frame, (line_left, 0), (line_left, frame_height), (75, 0, 130), 4)
        cv2.line(annotated_frame, (line_right, 0), (line_right, frame_height), (0, 255, 255), 4)
    
    if args.diag or args.all:
        cv2.line(annotated_frame, start_diag1, end_diag1, (0, 100, 0), 3)
        cv2.putText(annotated_frame, "diag1", (start_diag1[0] + 10, start_diag1[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.line(annotated_frame, start_diag2, end_diag2, (0, 165, 255), 3)
        cv2.putText(annotated_frame, "diag2", (start_diag2[0] - 80, start_diag2[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.line(annotated_frame, start_diag3, end_diag3, (0, 100, 0), 3)
        cv2.putText(annotated_frame, "diag3", (start_diag3[0] + 10, start_diag3[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.line(annotated_frame, start_diag4, end_diag4, (0, 165, 255), 3)
        cv2.putText(annotated_frame, "diag4", (start_diag4[0] - 80, start_diag4[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return annotated_frame

def draw_circles(annotated_frame, frame_center, circle_radii):
    for r in circle_radii:
        cv2.circle(annotated_frame, frame_center, r, (255, 100, 0), 3)
    return annotated_frame

def display_counters(annotated_frame, perspective, args,
                    counters_dict, frame_width, frame_height,
                    y_start, y_step, margin_x, scale, thickness):
    
    # Extract counters from dictionary
    counter_up = counters_dict['up']
    counter_down = counters_dict['down']
    counter_left = counters_dict['left']
    counter_right = counters_dict['right']
    counter_diag1 = counters_dict['diag1']
    counter_diag2 = counters_dict['diag2']
    counter_diag3 = counters_dict['diag3']
    counter_diag4 = counters_dict['diag4']
    
    ukupna_suma = 0
    y_offset_diag = 0
    
    if not args.diag or args.all:
        if perspective in ['front', 'worm'] or args.all:
            cvzone.putTextRect(annotated_frame, f"Down: {len(counter_down)}", 
                              (margin_x, y_start), scale=scale, thickness=thickness, 
                              colorT=(255, 255, 255), colorR=(0, 0, 255))
            cvzone.putTextRect(annotated_frame, f"Up: {len(counter_up)}", 
                              (margin_x, y_start + y_step), scale=scale, thickness=thickness, 
                              colorT=(255, 255, 255), colorR=(255, 0, 0))
            ukupna_suma += len(counter_down) + len(counter_up)
            y_offset_diag += 2
        
        if perspective == 'side' or args.all:
            cvzone.putTextRect(annotated_frame, f"Left: {len(counter_left)}", 
                              (margin_x, y_start + y_offset_diag * y_step), 
                              scale=scale, thickness=thickness, 
                              colorT=(255, 255, 255), colorR=(75, 0, 130))
            cvzone.putTextRect(annotated_frame, f"Right: {len(counter_right)}", 
                              (margin_x, y_start + (y_offset_diag + 1) * y_step), 
                              scale=scale, thickness=thickness, 
                              colorT=(255, 255, 255), colorR=(0, 255, 255))
            ukupna_suma += len(counter_left) + len(counter_right)
            y_offset_diag += 2
    
    if args.all or args.diag:
        cvzone.putTextRect(annotated_frame, f"Diag1: {len(counter_diag1)}", 
                          (margin_x, y_start + y_offset_diag * y_step), 
                          scale=scale, thickness=thickness, 
                          colorT=(0, 0, 0), colorR=(255, 255, 255))
        cvzone.putTextRect(annotated_frame, f"Diag2: {len(counter_diag2)}", 
                          (margin_x, y_start + (y_offset_diag + 1) * y_step), 
                          scale=scale, thickness=thickness, 
                          colorT=(0, 0, 0), colorR=(255, 255, 255))
        cvzone.putTextRect(annotated_frame, f"Diag3: {len(counter_diag3)}", 
                          (margin_x, y_start + (y_offset_diag + 2) * y_step), 
                          scale=scale, thickness=thickness, 
                          colorT=(0, 0, 0), colorR=(255, 255, 255))
        cvzone.putTextRect(annotated_frame, f"Diag4: {len(counter_diag4)}", 
                          (margin_x, y_start + (y_offset_diag + 3) * y_step), 
                          scale=scale, thickness=thickness, 
                          colorT=(0, 0, 0), colorR=(255, 255, 255))
        ukupna_suma += len(counter_diag1) + len(counter_diag2) + len(counter_diag3) + len(counter_diag4)

    y_suma_pos = frame_height - (y_step // 2) - 5
    cvzone.putTextRect(annotated_frame, f"Suma: {ukupna_suma}",
                       (margin_x, y_suma_pos),
                       scale=scale * 1.5,
                       thickness=thickness + 1,
                       colorT=(255, 255, 255), colorR=(0, 128, 0))
    
    return annotated_frame