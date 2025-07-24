import time
import json
import collections
from collections import defaultdict

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
        
        # Detaljnije merenje vremena
        self.total_detection_time = 0.0
        self.total_tracking_time = 0.0
        
        # Metrike detekcije i praćenja
        self.total_detections = 0
        self.total_tracks_updated = 0
        self.peak_concurrent_tracks = 0
        self.total_tracks_initiated = 0
        self.total_detection_confidence = 0.0
        
        # Metrike po klasi
        self.detections_per_class = defaultdict(int)
        
        # Poslovna logika (konačni brojevi)
        self.final_crossings = {}
        self.final_crossings_per_class = defaultdict(lambda: defaultdict(int))

        # NOVO: Mesto za čuvanje evaluacionih metrika i GFLOPs
        self.evaluation_summary = None
        self.gflops = 0.0

    def start_frame(self):
        """Beleži početak obrade novog frejma."""
        self._frame_start_time = time.time()
    
    def end_frame(self, detections_count: int, active_tracks_count: int, 
                  detection_time: float, tracking_time: float,
                  new_tracks_count: int = 0, avg_confidence: float = 0.0, 
                  detections_by_class: dict = None):
        """Beleži kraj obrade frejma i ažurira metrike."""
        if self._frame_start_time is None:
            return

        frame_processing_time = time.time() - self._frame_start_time
        self._frame_times.append(frame_processing_time)
        self.total_processing_time += frame_processing_time
        self.total_frames_processed += 1
        
        self.total_detection_time += detection_time
        self.total_tracking_time += tracking_time
        
        self.total_detections += detections_count
        self.total_tracks_updated += active_tracks_count
        self.total_tracks_initiated += new_tracks_count
        self.total_detection_confidence += avg_confidence * detections_count
        
        if active_tracks_count > self.peak_concurrent_tracks:
            self.peak_concurrent_tracks = active_tracks_count
        
        if detections_by_class:
            for class_name, count in detections_by_class.items():
                self.detections_per_class[class_name] += count
            
        self._frame_start_time = None

    def update_final_counts(self, params: dict):
        """Ažurira konačne brojeve prelaza iz stanja glavne aplikacije."""
        for direction, track_ids in params.get('counters', {}).items():
            self.final_crossings[direction] = len(track_ids)
        
        crossings_by_class = params.get('crossings_by_class', {})
        for direction, class_counts in crossings_by_class.items():
            for class_name, count in class_counts.items():
                self.final_crossings_per_class[direction][class_name] = count

    # NOVO: Metode za postavljanje dodatnih podataka
    def set_evaluation_summary(self, eval_summary: dict):
        """Postavlja sažetak evaluacionih metrika."""
        self.evaluation_summary = eval_summary
        
    def set_gflops(self, gflops: float):
        """Postavlja GFLOPs vrednost modela."""
        self.gflops = gflops

    def get_summary(self) -> dict:
        """Generiše rečnik sa sažetkom svih praćenih metrika."""
        self.end_time = time.time()
        total_runtime = self.end_time - self.start_time
        
        if self.total_frames_processed == 0:
            return {"error": "No frames processed."}

        avg_processing_time_ms = (self.total_processing_time / self.total_frames_processed) * 1000
        avg_detection_time_ms = (self.total_detection_time / self.total_frames_processed) * 1000
        avg_tracking_time_ms = (self.total_tracking_time / self.total_frames_processed) * 1000
        
        overall_avg_fps = self.total_frames_processed / total_runtime if total_runtime > 0 else 0
        moving_avg_fps = len(self._frame_times) / sum(self._frame_times) if sum(self._frame_times) > 0 else 0
        
        avg_confidence = self.total_detection_confidence / self.total_detections if self.total_detections > 0 else 0
        
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
                "model_gflops": self.gflops,
                "average_total_processing_time_ms": round(avg_processing_time_ms, 2),
                "average_detection_time_ms": round(avg_detection_time_ms, 2),
                "average_tracking_time_ms": round(avg_tracking_time_ms, 2),
            },
            "detection_and_tracking_stats": {
                "total_detections": self.total_detections,
                "average_detections_per_frame": round(self.total_detections / self.total_frames_processed, 2),
                "average_detection_confidence": round(avg_confidence, 3),
                "peak_concurrent_tracks": self.peak_concurrent_tracks,
                "total_tracks_initiated": self.total_tracks_initiated,
                "average_active_tracks_per_frame": round(self.total_tracks_updated / self.total_frames_processed, 2)
            },
            "class_based_stats": {
                "total_detections_by_class": dict(self.detections_per_class),
                "final_crossings_by_class": {k: dict(v) for k, v in self.final_crossings_per_class.items()},
            },
            "counting_results": self.final_crossings
        }
        
        # Dodavanje evaluacionih metrika ako postoje
        if self.evaluation_summary:
            summary['evaluation_metrics (zahteva Ground Truth)'] = self.evaluation_summary
        
        return summary

    def print_summary(self):
        """Štampa formatirani sažetak metrika na konzolu."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("====== FINALNI IZVEŠTAJ O METRIKAMA ======")
        print("="*60)
        
        for category, metrics in summary.items():
            print(f"\n--- {category.replace('_', ' ').title()} ---")
            if isinstance(metrics, dict):
                 for key, value in metrics.items():
                    if isinstance(value, dict):
                        print(f"{key.replace('_', ' ').title():<40}:")
                        for sub_key, sub_value in value.items():
                            print(f"  - {sub_key.replace('_', ' ').title():<36}: {sub_value}")
                    else:
                        print(f"{key.replace('_', ' ').title():<40}: {value}")
            else:
                 print(metrics)
                
        print("\n" + "="*60)

    def save_summary_to_json(self, filepath: str = "metrics_report.json"):
        """Čuva sažetak metrika u JSON fajl."""
        summary = self.get_summary()
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            print(f"\n[INFO] Izveštaj o metrikama sačuvan u {filepath}")
        except IOError as e:
            print(f"\n[GREŠKA] Nije moguće sačuvati izveštaj: {e}")