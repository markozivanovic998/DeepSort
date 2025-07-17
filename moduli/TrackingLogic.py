import math

def is_left_side(p1, p2, pt):
    line_vec = (p2[0] - p1[0], p2[1] - p1[1])
    pt_vec = (pt[0] - p1[0], pt[1] - p1[1])
    cross = line_vec[0] * pt_vec[1] - line_vec[1] * pt_vec[0]
    return cross > 0

def is_person_inside_vehicle(p_box, v_box, threshold=0.3):
    x_overlap = max(0, min(p_box[2], v_box[2]) - max(p_box[0], v_box[0]))
    y_overlap = max(0, min(p_box[3], v_box[3]) - max(p_box[1], v_box[1]))
    overlap_area = x_overlap * y_overlap
    if overlap_area == 0: return False
    person_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
    if person_area == 0: return False
    if (overlap_area / person_area) > threshold: return True
    person_center = ((p_box[0] + p_box[2]) // 2, (p_box[1] + p_box[3]) // 2)
    if (v_box[0] <= person_center[0] <= v_box[2] and v_box[1] <= person_center[1] <= v_box[3]): return True
    return False

def process_straight_line(track_id, current_pos, line_coord, line_type, target_direction_indicator, counter_list, person_state_dict, passed_any_line_set):
    if track_id in passed_any_line_set: return
    current_zone = ''
    if line_type == 'down': current_zone = 'above' if current_pos < line_coord else 'below'
    elif line_type == 'up': current_zone = 'below' if current_pos > line_coord else 'above'
    elif line_type == 'right': current_zone = 'left' if current_pos < line_coord else 'right'
    elif line_type == 'left': current_zone = 'right' if current_pos > line_coord else 'left'
    if track_id not in person_state_dict: person_state_dict[track_id] = {}
    if line_type not in person_state_dict[track_id]: person_state_dict[track_id][line_type] = current_zone
    previous_zone = person_state_dict[track_id][line_type]
    if previous_zone == target_direction_indicator and current_zone != target_direction_indicator:
        if track_id not in counter_list:
            counter_list.append(track_id)
            passed_any_line_set.add(track_id)
        del person_state_dict[track_id][line_type]
    else:
        person_state_dict[track_id][line_type] = current_zone

def process_diagonal_line(track_id, cx, cy, start_diag, end_diag, line_name, counter_list, person_state_dict, passed_any_line_set):
    if track_id in passed_any_line_set: return
    current_side = is_left_side(start_diag, end_diag, (cx, cy))
    if track_id not in person_state_dict: person_state_dict[track_id] = {}
    if line_name not in person_state_dict[track_id]: person_state_dict[track_id][line_name] = current_side
    previous_side = person_state_dict[track_id][line_name]
    if current_side != previous_side:
        if track_id not in counter_list:
            counter_list.append(track_id)
            passed_any_line_set.add(track_id)
        del person_state_dict[track_id][line_name]
    else:
        person_state_dict[track_id][line_name] = current_side

def process_circle_crossing(track_id, cx, cy, center, radius, circle_index, counter_list, person_state_dict, passed_any_line_set):
    if track_id in passed_any_line_set:
        return
    distance = math.sqrt((cx - center[0])**2 + (cy - center[1])**2)
    current_state = 'inside' if distance < radius else 'outside'
    circle_name = f'circle_{circle_index}'
    if track_id not in person_state_dict:
        person_state_dict[track_id] = {}
    if circle_name not in person_state_dict[track_id]:
        person_state_dict[track_id][circle_name] = current_state
    previous_state = person_state_dict[track_id][circle_name]
    if current_state != previous_state:
        if track_id not in counter_list:
            counter_list.append(track_id)
            passed_any_line_set.add(track_id)
        del person_state_dict[track_id][circle_name]
    else:
        person_state_dict[track_id][circle_name] = current_state