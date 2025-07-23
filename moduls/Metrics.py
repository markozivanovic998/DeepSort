import time
import json
import collections
from collections import defaultdict # NOVO

class MetricsTracker:
    """
    Klasa za praćenje različitih metrika performansi i operacija
    za aplikaciju za praćenje objekata.
    """
    def __init__(self, fps_window_size: int = 100):
        """
        Inicijalizuje MetricsTracker.

        Args:
            fps_window_size (int): Broj poslednjih frejmova koji se koriste za 
                                   izračunavanje pokretnog proseka FPS-a.
        """
        self.start_time = time.time()
        self.end_time = None
        
        # Metrike frejmova
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        self._frame_start_time = None
        self._frame_times = collections.deque(maxlen=fps_window_size)
        
        # NOVO: Detaljnije merenje vremena
        self.total_detection_time = 0.0
        self.total_tracking_time = 0.0
        
        # Metrike detekcije i praćenja
        self.total_detections = 0
        self.total_tracks_updated = 0
        self.peak_concurrent_tracks = 0
        self.total_tracks_initiated = 0 # NOVO
        self.total_detection_confidence = 0.0 # NOVO
        
        # NOVO: Metrike po klasi
        self.detections_per_class = defaultdict(int)
        
        # Poslovna logika (konačni brojevi)
        self.final_crossings = {}
        self.final_crossings_per_class = defaultdict(lambda: defaultdict(int)) # NOVO

    def start_frame(self):
        """
        Beleži početak obrade novog frejma.
        """
        self._frame_start_time = time.time()
    
    # NOVO: Ažurirana metoda `end_frame`
    def end_frame(self, detections_count: int, active_tracks_count: int, 
                  detection_time: float, tracking_time: float,
                  new_tracks_count: int = 0, avg_confidence: float = 0.0, 
                  detections_by_class: dict = None):
        """
        Beleži kraj obrade frejma i ažurira metrike.
        """
        if self._frame_start_time is None:
            return

        frame_processing_time = time.time() - self._frame_start_time
        self._frame_times.append(frame_processing_time)
        self.total_processing_time += frame_processing_time
        self.total_frames_processed += 1
        
        # Ažuriraj detaljna vremena
        self.total_detection_time += detection_time
        self.total_tracking_time += tracking_time
        
        # Ažuriraj statistiku detekcije i praćenja
        self.total_detections += detections_count
        self.total_tracks_updated += active_tracks_count
        self.total_tracks_initiated += new_tracks_count # NOVO
        self.total_detection_confidence += avg_confidence * detections_count # NOVO
        
        if active_tracks_count > self.peak_concurrent_tracks:
            self.peak_concurrent_tracks = active_tracks_count
        
        # Ažuriraj metrike po klasi
        if detections_by_class:
            for class_name, count in detections_by_class.items():
                self.detections_per_class[class_name] += count
            
        self._frame_start_time = None

    # NOVO: Ažurirana metoda `update_final_counts`
    def update_final_counts(self, params: dict):
        """
        Ažurira konačne brojeve prelaza iz stanja glavne aplikacije.
        """
        # Ukupni prelasci
        for direction, track_ids in params.get('counters', {}).items():
            self.final_crossings[direction] = len(track_ids)
        
        # Prelasci po klasi
        crossings_by_class = params.get('crossings_by_class', {})
        for direction, class_counts in crossings_by_class.items():
            for class_name, count in class_counts.items():
                self.final_crossings_per_class[direction][class_name] = count


    def get_summary(self) -> dict:
        """
        Generiše rečnik sa sažetkom svih praćenih metrika.
        """
        self.end_time = time.time()
        total_runtime = self.end_time - self.start_time
        
        # Proveravamo deljenje sa nulom
        if self.total_frames_processed == 0:
            return {"error": "No frames processed."}

        avg_processing_time_ms = (self.total_processing_time / self.total_frames_processed) * 1000
        avg_detection_time_ms = (self.total_detection_time / self.total_frames_processed) * 1000 # NOVO
        avg_tracking_time_ms = (self.total_tracking_time / self.total_frames_processed) * 1000 # NOVO
        
        overall_avg_fps = self.total_frames_processed / total_runtime if total_runtime > 0 else 0
        moving_avg_fps = len(self._frame_times) / sum(self._frame_times) if sum(self._frame_times) > 0 else 0
        
        avg_confidence = self.total_detection_confidence / self.total_detections if self.total_detections > 0 else 0 # NOVO
        
        summary = {
            "run_summary": {
                "total_runtime_seconds": round(total_runtime, 2),
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)),
            },
            "performance_metrics": {
                "total_frames_processed": self.total_frames_processed,
                "overall_average_fps": round(overall_avg_fps, 2),
                "moving_average_fps_last_100_frames": round(moving_avg_fps, 2),
                "average_total_processing_time_ms": round(avg_processing_time_ms, 2),
                "average_detection_time_ms": round(avg_detection_time_ms, 2), # NOVO
                "average_tracking_time_ms": round(avg_tracking_time_ms, 2), # NOVO
            },
            "detection_and_tracking": {
                "total_detections": self.total_detections,
                "average_detections_per_frame": round(self.total_detections / self.total_frames_processed, 2),
                "average_detection_confidence": round(avg_confidence, 3), # NOVO
                "peak_concurrent_tracks": self.peak_concurrent_tracks,
                "total_tracks_initiated": self.total_tracks_initiated, # NOVO
                "average_active_tracks_per_frame": round(self.total_tracks_updated / self.total_frames_processed, 2)
            },
            "class_based_metrics": { # NOVO
                "total_detections_by_class": dict(self.detections_per_class),
                "final_crossings_by_class": {k: dict(v) for k, v in self.final_crossings_per_class.items()},
            },
            "counting_results": self.final_crossings
        }
        return summary

    def print_summary(self):
        # Ova metoda ostaje ista, samo će ispisati više podataka
        summary = self.get_summary()
        print("\n" + "="*50)
        print("====== IZVEŠTAJ O METRIKAMA ======")
        print("="*50)
        
        for category, metrics in summary.items():
            print(f"\n--- {category.replace('_', ' ').title()} ---")
            if isinstance(metrics, dict):
                 for key, value in metrics.items():
                    print(f"{key.replace('_', ' ').title():<40}: {value}")
            else:
                 print(metrics)
                
        print("\n" + "="*50)

    def save_summary_to_json(self, filepath: str = "metrics_report.json"):
        # Ova metoda ostaje ista
        summary = self.get_summary()
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"\n[INFO] Izveštaj o metrikama sačuvan u {filepath}")
        except IOError as e:
            print(f"\n[GREŠKA] Nije moguće sačuvati izveštaj: {e}")