# perspective.py

import cv2
import numpy as np

def apply_perspective_transform(frame, perspective_type: str, width: int, height: int):
    """
    Primenjuje transformaciju perspektive na ulazni frejm.

    Args:
        frame (np.ndarray): Ulazni frejm (slika).
        perspective_type (str): Tip perspektive ('top', 'side', 'worm').
        width (int): Širina frejma.
        height (int): Visina frejma.

    Returns:
        np.ndarray: Transformisani frejm. Ako je tip nepoznat ili 'front', vraća originalni frejm.
    """
    
    # Ako je perspektiva 'front' (podrazumevana) ili nije zadata, ne radi ništa.
    if not perspective_type or perspective_type.lower() == 'front':
        return frame

    pts1 = None
    pts2 = None

    # 1. Top-Down (Ptičja) perspektiva
    # Uzima trapezoid sa dna slike (koji predstavlja put ili tlo) i ispravlja ga u pravougaonik.
    if perspective_type.lower() == 'top':
        # Izvorne tačke (trapezoid u originalnoj slici)
        # Ove vrednosti treba podesiti prema tvojoj kameri!
        src_points = [
            [width * 0.45, height * 0.6],  # Gore levo
            [width * 0.55, height * 0.6],  # Gore desno
            [width * 0.1, height],         # Dole levo
            [width * 0.9, height]          # Dole desno
        ]
        pts1 = np.float32(src_points)

        # Odredišne tačke (pravougaonik u izlaznoj slici)
        dst_points = [
            [0, 0],                        # Gore levo
            [width, 0],                    # Gore desno
            [0, height],                   # Dole levo
            [width, height]                # Dole desno
        ]
        pts2 = np.float32(dst_points)

    # 2. Side (Bočna) perspektiva
    # "Ispravlja" pogled sa strane. Korisno ako kamera gleda niz ulicu pod uglom.
    elif perspective_type.lower() == 'side':
        # Izvorne tačke (npr. leva strana scene)
        src_points = [
            [width * 0.2, height * 0.2],   # Gore levo
            [width * 0.7, height * 0.1],   # Gore desno
            [0, height],                   # Dole levo
            [width, height * 0.8]          # Dole desno
        ]
        pts1 = np.float32(src_points)

        # Odredišne tačke (pravougaonik)
        dst_points = [
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ]
        pts2 = np.float32(dst_points)

    # 3. Worm's-Eye (Žablja) perspektiva
    # Simulira pogled odozdo, izobličavajući sliku da izgleda kao da se gleda ka gore.
    elif perspective_type.lower() == 'worm':
        # Izvorne tačke (pravougaonik u donjem delu slike)
        src_points = [
            [0, height * 0.5],
            [width, height * 0.5],
            [0, height],
            [width, height]
        ]
        pts1 = np.float32(src_points)

        # Odredišne tačke (trapezoid koji se širi ka gore)
        dst_points = [
            [-width * 0.15, 0],  # Pomeramo van okvira da dobijemo efekat širenja
            [width * 1.15, 0],
            [0, height],
            [width, height]
        ]
        pts2 = np.float32(dst_points)

    # Ako su tačke definisane, izvrši transformaciju
    if pts1 is not None and pts2 is not None:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped_frame = cv2.warpPerspective(frame, matrix, (width, height))
        return warped_frame

    # Ako tip perspektive nije prepoznat, vrati originalni frejm
    return frame