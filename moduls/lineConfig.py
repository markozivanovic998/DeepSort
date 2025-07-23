# config.py

def initialize_parameters(frame_width, frame_height, args):
    """
    Inicijalizuje sve potrebne parametre, zone za brojanje i promenljive stanja.
    """
    params = {}

    # Glavne postavke
    params['USE_CIRCLES'] = args.circle > 0
    perspective = args.perspective

    # UI parametri za iscrtavanje teksta
    params['ui'] = {
        'y_start': int(frame_height * 0.1),
        'margin_x': int(frame_width * 0.02),
        'y_step': int(frame_height * 0.05),
        'scale': max(1, frame_width // 1000),
        'thickness': max(1, frame_width // 1000)
    }

    # Definicija zona za brojanje (krugovi ili linije)
    zones = {}
    if params['USE_CIRCLES']:
        zones['frame_center'] = (frame_width // 2, frame_height // 2)
        zones['circle_radii'] = [(i + 1) * args.radius for i in range(args.circle)]
        counters = {'circles': [[] for _ in range(args.circle)]}
    else:
        # Učitavanje vrednosti iz konfiguracije sa podrazumevanim vrednostima
        top_shift = args.left
        bottom_shift = args.bottom
        left_shift = args.left
        right_shift = args.right
        diag_shift1 = args.diag_shift1
        diag_shift2 = args.diag_shift2

        if perspective in ['front', 'worm']:
            zones['line_up'] = int(frame_height * 0.4) + top_shift
            zones['line_down'] = int(frame_height * 0.6) + bottom_shift
        elif perspective == 'side':
            zones['line_left'] = left_shift if left_shift > 0 else int(frame_width * 0.3)
            zones['line_right'] = frame_width - right_shift if right_shift > 0 else int(frame_width * 0.7)
        else: # 'all' or default
            zones['line_up'] = int(frame_height * 0.4) + top_shift
            zones['line_down'] = int(frame_height * 0.6) + bottom_shift
            zones['line_left'] = int(frame_width * 0.3) + left_shift
            zones['line_right'] = int(frame_width * 0.7) - right_shift

        zones['start_diag1'] = (0 + diag_shift1, 0)
        zones['end_diag1'] = (frame_width, frame_height - diag_shift1)
        zones['start_diag2'] = (frame_width - diag_shift2, 0)
        zones['end_diag2'] = (0, frame_height - diag_shift2)
        zones['start_diag3'] = (0 + diag_shift1, frame_height)
        zones['end_diag3'] = (frame_width, 0 + diag_shift1)
        zones['start_diag4'] = (frame_width - diag_shift2, frame_height)
        zones['end_diag4'] = (0, 0 + diag_shift2)
        
        counters = {
            'up': [], 'down': [], 'left': [], 'right': [],
            'diag1': [], 'diag2': [], 'diag3': [], 'diag4': []
        }

    params['zones'] = zones
    params['counters'] = counters
    
    # Stanja za praćenje
    params['state'] = {
        'person_state': {},
        'passed_any_line': set()
    }
    
    return params