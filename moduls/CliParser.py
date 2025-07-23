import argparse

def parse_arguments():
    """
    Konfiguriše i parsira argumente komandne linije.
    """
    parser = argparse.ArgumentParser(description='Analiza videa iz različitih perspektiva')
    
    # --- Grupa za linije i dijagonale ---
    line_group = parser.add_argument_group('Opcije za linije i dijagonale')
    line_group.add_argument('--perspective', type=str, default='front', choices=['front', 'side', 'worm', 'top'],
                        help='Perspektiva kamere (npr. front, side, worm, top)')
    line_group.add_argument('--left', type=int, default=0, help='Pomeranje leve linije u pikselima (od leve ivice za side)')
    line_group.add_argument('--right', type=int, default=0, help='Pomeranje desne linije u pikselima (od desne ivice za side)')
    line_group.add_argument('--top', type=int, default=0, help='Pomeranje gornje linije u pikselima')
    line_group.add_argument('--bottom', type=int, default=0, help='Pomeranje donje linije u pikselima')
    line_group.add_argument('--all', action='store_true',
                        help='Crtaj sve linije (left, right, top, bottom) i sve brojače bez obzira na perspektivu')
    line_group.add_argument('--diag', action='store_true',
                        help='Crtaj samo dijagonale i broji prelazak ljudi preko njih')
    line_group.add_argument('--show_boxes', action='store_true', help='Prikaži bounding box-ove oko osoba')
    line_group.add_argument('--diag_shift1', type=int, default=0, help='Pomeranje dijagonale 1 i 3')
    line_group.add_argument('--diag_shift2', type=int, default=0, help='Pomeranje dijagonale 2 i 4')
    line_group.add_argument('--RTSP', action='store_true', help='Koristi RTSP live streaming umesto lokalnog videa')
    line_group.add_argument('--plot', action='store_true',
                             help='Prikaži dashboard sa statistikama u realnom vremenu')

    # --- Grupa za koncentrične krugove ---
    circle_group = parser.add_argument_group('Opcije za koncentrične krugove')
    circle_group.add_argument('--circle', type=int, default=0, help='Broj koncentričnih krugova za crtanje. Ova opcija isključuje sve ostale linije.')
    circle_group.add_argument('--radius', type=int, default=50, help='Razmak (radijus) između krugova u pikselima.')

    parser.add_argument('--show-stats', action='store_true', help='Prikazuje konačnu statistiku performansi na kraju izvršavanja.')
    
    # Parsiraj argumente i vrati ih
    return parser.parse_args()