import cv2
import numpy as np
import time
import math
from collections import defaultdict
from itertools import combinations
import cvzone
import json # 📝 LOGOVANJE: Uvoz potrebnih biblioteka
import os

class BehaviorAnalytics:
    """
    Modul za analizu ponašanja objekata na osnovu podataka o praćenju.
    
    Nova, pojednostavljena inicijalizacija omogućava odabir svrhe korišćenja (use case)
    za automatsko podešavanje parametara heatmape.
    
    📝 LOGOVANJE: Dodata je mogućnost logovanja događaja i generisanja statističkog izveštaja.
    """

    # 🔥 PREDEFINISANE POSTAVKE ZA HEATMAPU NA OSNOVU SVRHE KORIŠĆENJA
    HEATMAP_PRESETS = {
        # Namena: (reset_interval u sekundama, decay_factor)
        'detekcija_guzvi':          (30, 0.95),   # Brz reset, brzo bleđenje
        'ponasanje_1h':             (300, 0.99),  # Reset na 5 min, sporo bleđenje
        'dugorocna_analiza':        (0, 1.0),     # Bez reseta, bez bleđenja
        'optimizacija_proizvoda':   (600, 0.995), # Reset na 10 min, vrlo sporo bleđenje
        'kontrola_tokova':          (90, 0.98),   # Reset na 1.5 min, srednje bleđenje
        'pracenje_zadrzavanja':     (240, 0.99),  # Reset na 4 min, sporo bleđenje
        'sumnjivo_ponasanje':       (60, 0.97),   # Reset na 1 min, umereno bleđenje
        'sportska_analitika':       (2700, 0.998),# Reset na 45 min (poluvreme)
        'interaktivne_instalacije': (15, 0.90),   # Veoma brz reset i bleđenje
        'smart_city':               (60, 0.96),   # Reset na 1 min, brzo bleđenje
    }

    def __init__(self, 
                 frame_shape, 
                 zones=None, 
                 loitering_threshold=10.0, 
                 speed_threshold=100.0, 
                 direction_change_threshold=90.0,
                 # Napredne detekcije
                 idle_speed_threshold=15.0,
                 idle_time_threshold=5.0,
                 zone_hopping_time=10.0,
                 zone_hopping_count=3,
                 cyclic_path_points=20,
                 cyclic_path_distance_threshold=25,
                 group_distance_threshold=60,
                 linear_path_angle_threshold=15.0,
                 # 🔥 NOVI, JEDNOSTAVNIJI NAČIN PODEŠAVANJA HEATMAPE
                 heatmap_use_case=None,
                 heatmap_reset_interval=0,
                 heatmap_decay_factor=0.99,
                 # 📝 LOGOVANJE: Novi parametri za fajlove
                 log_file="behavior_log.json",
                 report_file="behavior_report.json"
                ):
        """
        Inicijalizacija modula.
        
        Args:
            heatmap_use_case (str, optional): Naziv predefinisane postavke za heatmapu.
            log_file (str, optional): Putanja do fajla za logovanje događaja. Ako je None, logovanje je isključeno.
            report_file (str, optional): Putanja do fajla za finalni statistički izveštaj.
        """
        self.frame_height, self.frame_width = frame_shape[:2]
        self.zones = zones if zones else {}
        self.loitering_threshold = loitering_threshold
        self.speed_threshold = speed_threshold
        self.direction_change_threshold = direction_change_threshold
        
        self.idle_speed_threshold = idle_speed_threshold
        self.idle_time_threshold = idle_time_threshold
        self.zone_hopping_time = zone_hopping_time
        self.zone_hopping_count = zone_hopping_count
        self.cyclic_path_points = cyclic_path_points
        self.cyclic_path_distance_threshold = cyclic_path_distance_threshold
        self.group_distance_threshold = group_distance_threshold
        self.linear_path_angle_threshold = linear_path_angle_threshold

        if heatmap_use_case and heatmap_use_case in self.HEATMAP_PRESETS:
            interval, decay = self.HEATMAP_PRESETS[heatmap_use_case]
            self.heatmap_reset_interval = interval
            self.heatmap_decay_factor = decay
            print(f"Heatmap konfigurisan za namenu: '{heatmap_use_case}' (Reset: {interval}s, Bleđenje: {decay})")
        else:
            self.heatmap_reset_interval = heatmap_reset_interval
            self.heatmap_decay_factor = heatmap_decay_factor

        self.heatmap_decay_factor = np.clip(self.heatmap_decay_factor, 0.9, 1.0)
        self.heatmap = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        self.last_heatmap_reset_time = time.time()
        
        self.track_data = defaultdict(lambda: {
            'positions': [], 'in_zone': None, 'entry_time': None,
            'total_dwell_time': defaultdict(float), 'speed': 0.0,
            'anomalies': set(), 'behaviors': set(), 'zone_history': [],
            'movement_state': 'Mirovanje', 'idle_start_time': None
        })
        self.track_groups = {}
        
        # 📝 LOGOVANJE: Inicijalizacija sistema za logovanje i statistiku
        self.log_file = log_file
        self.report_file = report_file
        self.seen_track_ids = set()
        self.last_group_state = []
        
        # Brisanje starog log fajla pri pokretanju
        if self.log_file and os.path.exists(self.log_file):
            os.remove(self.log_file)
            
        self.report_data = {
            'opste_statistike': {
                'ukupno_pracenih_objekata': 0,
                'trenutno_aktivnih_objekata': 0
            },
            'detektovane_anomalije': defaultdict(int),
            'detektovani_obrasci': defaultdict(int),
            'statistike_zona': defaultdict(lambda: {'broj_ulazaka': 0, 'ukupno_vreme_zadrzavanja_s': 0.0}),
            'statistike_grupa': {
                'trenutni_broj_grupa': 0
            }
        }

    # 📝 LOGOVANJE: Nova metoda za upisivanje događaja u log fajl
    def _log_event(self, event_type, data):
        """Upisuje jedan događaj u JSON log fajl."""
        if not self.log_file:
            return
        
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'event': event_type,
            'data': data
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')
        except IOError as e:
            print(f"Greška prilikom pisanja u log fajl: {e}")

    # 📝 LOGOVANJE: Nova metoda za čuvanje finalnog izveštaja
    def save_report(self):
        """Čuva akumulirane statističke podatke u JSON fajl izveštaja."""
        if not self.report_file:
            print("Putanja za fajl izveštaja nije podešena. Izveštaj neće biti sačuvan.")
            return

        # Ažuriranje finalnih vrednosti pre čuvanja
        # Dodavanje vremena zadržavanja za objekte koji su još uvek u zoni na kraju
        current_time = time.time()
        for track_id, data in self.track_data.items():
            if data['in_zone'] and data['entry_time']:
                duration = current_time - data['entry_time']
                self.report_data['statistike_zona'][data['in_zone']]['ukupno_vreme_zadrzavanja_s'] += duration

        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                json.dump(self.report_data, f, ensure_ascii=False, indent=4)
            print(f"📊 Statistički izveštaj je sačuvan u fajl: {self.report_file}")
        except IOError as e:
            print(f"Greška prilikom čuvanja izveštaja: {e}")

    def _calculate_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(self, tracks):
        current_time = time.time()
        
        # 📝 LOGOVANJE: Ažuriranje broja aktivnih objekata za izveštaj
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]
        self.report_data['opste_statistike']['trenutno_aktivnih_objekata'] = len(confirmed_tracks)

        if self.heatmap_reset_interval > 0 and (current_time - self.last_heatmap_reset_time) > self.heatmap_reset_interval:
            self.heatmap.fill(0)
            self.last_heatmap_reset_time = current_time
            print(f"[{time.strftime('%H:%M:%S')}] Heatmap resetovan.")
            # 📝 LOGOVANJE: Beleženje reseta heatmape
            self._log_event('heatmap_resetovan', {'vreme_reseta': time.strftime('%Y-%m-%d %H:%M:%S')})
        
        self.heatmap *= self.heatmap_decay_factor

        active_track_ids = {track.track_id for track in confirmed_tracks}
        for track_id in list(self.track_data.keys()):
            if track_id not in active_track_ids:
                del self.track_data[track_id]

        for track in confirmed_tracks:
            track_id = track.track_id
            bbox = track.to_tlbr()
            centroid = self._calculate_centroid(bbox)
            data = self.track_data[track_id]
            
            # 📝 LOGOVANJE: Detekcija i logovanje novog objekta
            if track_id not in self.seen_track_ids:
                self.seen_track_ids.add(track_id)
                self.report_data['opste_statistike']['ukupno_pracenih_objekata'] += 1
                self._log_event('novi_objekat_detektovan', {'track_id': track_id, 'pozicija': centroid})

            data['anomalies'].clear()
            data['behaviors'].clear()
            
            data['positions'].append((*centroid, current_time))
            if len(data['positions']) > 100: data['positions'].pop(0)

            cv2.circle(self.heatmap, centroid, 20, 1, thickness=-1)

            history = data['positions']
            
            if len(history) >= 2:
                p1, t1 = history[-2][:2], history[-2][2]
                p2, t2 = history[-1][:2], history[-1][2]
                delta_t = t2 - t1
                if delta_t > 0:
                    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    speed = distance / delta_t
                    data['speed'] = speed
                    
                    if speed > self.speed_threshold: data['anomalies'].add('Prebrzo kretanje')
                    
                    if speed < self.idle_speed_threshold: data['movement_state'] = 'Mirovanje'
                    elif speed < self.speed_threshold: data['movement_state'] = 'Hodanje'
                    else: data['movement_state'] = 'Trčanje'
            
            if data['movement_state'] == 'Mirovanje':
                if data['idle_start_time'] is None: data['idle_start_time'] = current_time
                elif (current_time - data['idle_start_time']) > self.idle_time_threshold: data['anomalies'].add('Stajanje na mestu')
            else:
                data['idle_start_time'] = None

            if len(history) >= 3:
                p1, p2, p3 = history[-3][:2], history[-2][:2], history[-1][:2]
                angle_diff = self._calculate_angle(p1, p2, p3)
                if angle_diff > self.direction_change_threshold: data['anomalies'].add('Promena smera')

            current_zone = self._get_current_zone(centroid)
            self._update_zone_data(track_id, current_zone, current_time)
            self._detect_zone_hopping(track_id, current_time)
            self._detect_path_patterns(track_id)
            
            # 📝 LOGOVANJE: Ažuriranje statistike i logovanje stanja objekta
            if data['anomalies']:
                for anomaly in data['anomalies']: self.report_data['detektovane_anomalije'][anomaly] += 1
            if data['behaviors']:
                for behavior in data['behaviors']: self.report_data['detektovani_obrasci'][behavior] += 1
            
            self._log_event('stanje_objekta', {
                'track_id': track_id,
                'pozicija': centroid,
                'brzina_px_s': round(data['speed'], 2),
                'stanje_kretanja': data['movement_state'],
                'anomalije': list(data['anomalies']),
                'obrasci': list(data['behaviors']),
                'zona': data['in_zone']
            })

        self._update_group_detection(active_track_ids)

    def _calculate_angle(self, p1, p2, p3):
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        angle_diff = abs(math.degrees(angle1 - angle2))
        return 360 - angle_diff if angle_diff > 180 else angle_diff

    def _get_current_zone(self, centroid):
        for name, polygon in self.zones.items():
            if cv2.pointPolygonTest(polygon, centroid, False) >= 0:
                return name
        return None
        
    def _update_zone_data(self, track_id, current_zone, current_time):
        data = self.track_data[track_id]
        previous_zone = data['in_zone']

        if current_zone != previous_zone:
            # Događaj izlaska iz prethodne zone
            if previous_zone is not None and data['entry_time'] is not None:
                duration = current_time - data['entry_time']
                data['total_dwell_time'][previous_zone] += duration
                # 📝 LOGOVANJE
                self.report_data['statistike_zona'][previous_zone]['ukupno_vreme_zadrzavanja_s'] += duration
                self._log_event('izlaz_iz_zone', {'track_id': track_id, 'zona': previous_zone, 'trajanje_s': round(duration, 2)})

            # Događaj ulaska u novu zonu
            if current_zone is not None:
                data['zone_history'].append((current_zone, current_time))
                if len(data['zone_history']) > 10: data['zone_history'].pop(0)
                # 📝 LOGOVANJE
                self.report_data['statistike_zona'][current_zone]['broj_ulazaka'] += 1
                self._log_event('ulaz_u_zonu', {'track_id': track_id, 'zona': current_zone})
            
            data['entry_time'] = current_time if current_zone is not None else None
            data['in_zone'] = current_zone

        if data['in_zone'] is not None:
            dwell_duration = current_time - data['entry_time']
            if dwell_duration > self.loitering_threshold: data['anomalies'].add('Zadrzavanje')

    def _detect_zone_hopping(self, track_id, current_time):
        data = self.track_data[track_id]
        recent_hops = [hop for hop in data['zone_history'] if current_time - hop[1] < self.zone_hopping_time]
        if len(set(hop[0] for hop in recent_hops)) >= self.zone_hopping_count:
            data['anomalies'].add('Ucestala promena zona')

    def _detect_path_patterns(self, track_id):
        data = self.track_data[track_id]
        history = data['positions']
        if len(history) >= self.cyclic_path_points:
            start_point = history[-self.cyclic_path_points][:2]
            end_point = history[-1][:2]
            distance = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            if distance < self.cyclic_path_distance_threshold: data['behaviors'].add('Ciklicno kretanje')
        if len(history) >= 15:
            p_start = history[-15][:2]
            p_mid = history[-8][:2]
            p_end = history[-1][:2]
            angle_change = self._calculate_angle(p_start, p_mid, p_end)
            if angle_change < self.linear_path_angle_threshold: data['behaviors'].add('Linearni prolaz')

    def _update_group_detection(self, active_track_ids):
        self.track_groups.clear()
        if len(active_track_ids) < 2: 
            if self.last_group_state: # Ako su postojale grupe, a sada ne, loguj promenu
                self.last_group_state = []
                self.report_data['statistike_grupa']['trenutni_broj_grupa'] = 0
                self._log_event('azuriranje_grupa', {'grupe': []})
            return

        positions = {tid: self.track_data[tid]['positions'][-1][:2] for tid in active_track_ids if self.track_data[tid]['positions']}
        pairs = []
        for id1, id2 in combinations(positions.keys(), 2):
            dist = math.sqrt((positions[id1][0] - positions[id2][0])**2 + (positions[id1][1] - positions[id2][1])**2)
            if dist < self.group_distance_threshold: pairs.append({id1, id2})

        groups = []
        if pairs:
            while len(pairs) > 0:
                current_group = pairs.pop(0)
                i = 0
                while i < len(pairs):
                    if not current_group.isdisjoint(pairs[i]):
                        current_group.update(pairs.pop(i))
                        i = 0
                    else: i += 1
                groups.append(list(current_group))
        
        for i, group in enumerate(groups):
            group_id = f"G{i+1}"
            for track_id in group: self.track_groups[track_id] = group_id
        
        # 📝 LOGOVANJE: Beleženje promene u stanju grupa
        sorted_groups = sorted([sorted(g) for g in groups])
        if sorted_groups != self.last_group_state:
            self.report_data['statistike_grupa']['trenutni_broj_grupa'] = len(sorted_groups)
            self._log_event('azuriranje_grupa', {'broj_grupa': len(sorted_groups), 'grupe': sorted_groups})
            self.last_group_state = sorted_groups

    def draw_heatmap(self, frame, alpha=0.5, colormap=cv2.COLORMAP_JET):
        if np.max(self.heatmap) > 0:
            norm_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colored_heatmap = cv2.applyColorMap(norm_heatmap, colormap)
            return cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)
        return frame

    def draw_analytics(self, frame):
        for name, polygon in self.zones.items():
            cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(frame, name, (polygon[0][0], polygon[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        for track_id, data in self.track_data.items():
            if not data['positions']: continue
            cx, cy = data['positions'][-1][:2]
            info_lines = [ f"ID: {track_id}", f"Brzina: {data['speed']:.1f} px/s", f"Stanje: {data['movement_state']}" ]
            if track_id in self.track_groups: info_lines[0] += f" | Grupa: {self.track_groups[track_id]}"
            if data['in_zone']:
                dwell_time = (time.time() - data['entry_time']) + data['total_dwell_time'][data['in_zone']]
                info_lines.append(f"Zona: {data['in_zone']} ({dwell_time:.1f}s)")
            if data['behaviors']: info_lines.append(f"Obrasci: {', '.join(data['behaviors'])}")
            if data['anomalies']:
                anomalies_str = f"Anomalije: {', '.join(data['anomalies'])}"
                y_pos = cy + 20 + len(info_lines) * 20
                cvzone.putTextRect(frame, anomalies_str, (cx + 10, y_pos), scale=0.9, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 200), offset=5)
            for i, line in enumerate(info_lines):
                y_pos = cy + 20 + i * 20
                cvzone.putTextRect(frame, line, (cx + 10, y_pos), scale=0.9, thickness=1, colorR=(50, 50, 50), offset=5)
        return frame