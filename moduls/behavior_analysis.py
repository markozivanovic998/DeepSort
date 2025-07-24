import cv2
import numpy as np
import time
import math
from collections import defaultdict
import cvzone

class BehaviorAnalytics:
    """
    Modul za analizu ponašanja objekata na osnovu podataka o praćenju.
    
    Implementira sledeće funkcionalnosti:
    - Zadržavanje po zoni (Dwell Time)
    - Detekcija promene smera kretanja
    - Izračunavanje brzine kretanja
    - Generisanje heatmape kretanja
    - Detekcija anomalija (prebrzo kretanje, besciljno zadržavanje)
    """

    def __init__(self, frame_shape, zones=None, loitering_threshold=10.0, speed_threshold=100.0, direction_change_threshold=90.0):
        """
        Inicijalizacija modula.

        Args:
            frame_shape (tuple): Dimenzije frejma (height, width, channels).
            zones (dict): Rečnik sa definisanim zonama. Ključ je ime zone, vrednost je NumPy niz tačaka (poligon).
                          Primer: {'restricted_area': np.array([[100, 100], [400, 100], [400, 400], [100, 400]], np.int32)}
            loitering_threshold (float): Vreme u sekundama nakon kojeg se boravak u zoni smatra besciljnim zadržavanjem (loitering).
            speed_threshold (float): Prag brzine u pikselima po sekundi za detekciju prebrzog kretanja.
            direction_change_threshold (float): Prag promene ugla u stepenima za detekciju nagle promene smera.
        """
        self.frame_height, self.frame_width = frame_shape[:2]
        self.zones = zones if zones else {}
        self.loitering_threshold = loitering_threshold
        self.speed_threshold = speed_threshold
        self.direction_change_threshold = direction_change_threshold

        # Struktura za skladištenje podataka o svakom praćenom objektu
        self.track_data = defaultdict(lambda: {
            'positions': [],        # Lista (x, y, timestamp) tuple-ova za istoriju kretanja
            'in_zone': None,        # Trenutna zona u kojoj se objekat nalazi
            'entry_time': None,     # Vreme ulaska u zonu
            'total_dwell_time': defaultdict(float), # Ukupno vreme zadržavanja po zonama
            'speed': 0.0,           # Trenutna brzina
            'anomalies': set()      # Skup detektovanih anomalija ('LOITERING', 'FAST_SPEED', 'DIRECTION_CHANGE')
        })

        # Heatmapa kretanja
        self.heatmap = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)

    def _calculate_centroid(self, bbox):
        """Pomoćna funkcija za računanje centroida (središnje tačke) bounding box-a."""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(self, tracks):
        """
        Ažurira stanje analitike sa novim podacima o praćenju za trenutni frejm.

        Args:
            tracks (list): Lista praćenih objekata iz DeepSORT-a. Svaki objekat mora imati
                           atribute `track_id` i `to_tlbr()`.
        """
        current_time = time.time()
        active_track_ids = {track.track_id for track in tracks if track.is_confirmed()}

        # Očisti podatke za objekte koji više nisu aktivni
        for track_id in list(self.track_data.keys()):
            if track_id not in active_track_ids:
                del self.track_data[track_id]

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            centroid = self._calculate_centroid(bbox)
            data = self.track_data[track_id]
            
            # Resetuj anomalije za trenutni frejm
            data['anomalies'].clear()
            
            # Dodaj trenutnu poziciju i vreme u istoriju
            data['positions'].append((*centroid, current_time))
            if len(data['positions']) > 50: # Ograniči dužinu istorije radi efikasnosti
                data['positions'].pop(0)

            # 🔥 1. Ažuriranje Heatmape
            cv2.circle(self.heatmap, centroid, 15, 1, thickness=-1)

            history = data['positions']
            
            # 🏃‍♂️ 2. Analiza Brzine kretanja
            if len(history) >= 2:
                p1, t1 = history[-2][:2], history[-2][2]
                p2, t2 = history[-1][:2], history[-1][2]
                delta_t = t2 - t1
                if delta_t > 0:
                    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    speed = distance / delta_t  # Brzina u pikselima po sekundi
                    data['speed'] = speed
                    if speed > self.speed_threshold:
                        data['anomalies'].add('Prebrzo kretanje')

            # 🔄 3. Analiza Promene smera kretanja
            if len(history) >= 3:
                p1, p2, p3 = history[-3][:2], history[-2][:2], history[-1][:2]
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                angle1 = math.atan2(v1[1], v1[0])
                angle2 = math.atan2(v2[1], v2[0])
                angle_diff = abs(math.degrees(angle1 - angle2))
                
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff > self.direction_change_threshold:
                    data['anomalies'].add('Promena smera')

            # 🎯 4. Analiza Zadržavanja po zoni i Anomalija (Loitering)
            current_zone = None
            for name, polygon in self.zones.items():
                if cv2.pointPolygonTest(polygon, centroid, False) >= 0:
                    current_zone = name
                    break
            
            if current_zone != data['in_zone']:
                if data['in_zone'] is not None: # Ako je napustio prethodnu zonu
                    data['total_dwell_time'][data['in_zone']] += current_time - data['entry_time']
                
                data['entry_time'] = current_time if current_zone is not None else None
                data['in_zone'] = current_zone
            
            if data['in_zone'] is not None:
                dwell_duration = current_time - data['entry_time']
                if dwell_duration > self.loitering_threshold:
                    data['anomalies'].add('Zadrzavanje')

    def draw_heatmap(self, frame, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """Iscrtava heatmapu preko datog frejma."""
        if np.max(self.heatmap) > 0:
            norm_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colored_heatmap = cv2.applyColorMap(norm_heatmap, colormap)
            return cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)
        return frame

    def draw_analytics(self, frame):
        """Iscrtava sve analitičke podatke (zone, tekstualne informacije) na frejmu."""
        # Iscrtaj definisane zone
        for name, polygon in self.zones.items():
            cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, name, (polygon[0][0], polygon[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Iscrtaj informacije za svaki praćeni objekat
        for track_id, data in self.track_data.items():
            if not data['positions']:
                continue
            
            cx, cy = data['positions'][-1][:2]
            info_lines = [f"ID: {track_id}", f"Brzina: {data['speed']:.1f} px/s"]

            if data['in_zone']:
                dwell_time = (time.time() - data['entry_time']) + data['total_dwell_time'][data['in_zone']]
                info_lines.append(f"Zona: {data['in_zone']} ({dwell_time:.1f}s)")
            
            if data['anomalies']:
                info_lines.append(f"Anomalije: {', '.join(data['anomalies'])}")

            # Iscrtaj tekstualne informacije pored objekta
            for i, line in enumerate(info_lines):
                y_pos = cy + 20 + i * 18
                cvzone.putTextRect(frame, line, (cx + 10, y_pos), scale=0.9, thickness=1, 
                                   colorR=(50, 50, 50),
                                   offset=5)
        return frame