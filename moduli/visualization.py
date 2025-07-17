import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Wedge, Circle
from collections import deque
from datetime import datetime, timedelta
import heapq
import logging
import time

logger = logging.getLogger(__name__)

class RealTimeVisualizer:
    def __init__(self, max_data_points=300, window_name="Analytics Dashboard"):
        self.max_data_points = max_data_points
        self.window_name = window_name
        self.last_update_time = time.time()
        
        # Create figure with multiple subplots
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = plt.figure(figsize=(18, 14), dpi=100)
        self.gs = self.fig.add_gridspec(4, 4)
        
        # Define subplots
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])  # Total people
        self.ax2 = self.fig.add_subplot(self.gs[0, 1])  # Entered/Exited
        self.ax3 = self.fig.add_subplot(self.gs[0, 2])  # Density gauge
        self.ax4 = self.fig.add_subplot(self.gs[0, 3])  # Current metrics
        self.ax5 = self.fig.add_subplot(self.gs[1, 0])  # Trend line
        self.ax6 = self.fig.add_subplot(self.gs[1, 1])  # Histogram
        self.ax7 = self.fig.add_subplot(self.gs[1, 2])  # Top/Bottom periods
        self.ax8 = self.fig.add_subplot(self.gs[2, 0:2])  # Heatmap
        self.ax9 = self.fig.add_subplot(self.gs[2, 2:])  # Dwell time
        self.ax10 = self.fig.add_subplot(self.gs[3, :])  # Alarms
        
        plt.tight_layout(pad=3.0)
        
        # Initialize data
        self.reset_data()
        
        # Alarm thresholds
        self.density_threshold = 80  # %
        self.people_threshold = 40   # absolute number
        
        # Initialize colorbar attribute
        self.cbar = None 
        
        # Create OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 1400)
        
    def reset_data(self):
        """Initialize all data structures"""
        self.timestamps = deque(maxlen=self.max_data_points)
        self.people_count = deque(maxlen=self.max_data_points)
        self.entered_count = deque(maxlen=self.max_data_points)
        self.exited_count = deque(maxlen=self.max_data_points)
        self.density = deque(maxlen=self.max_data_points)
        
        # For trend analysis
        self.trend_window = 10
        self.moving_avg = deque(maxlen=self.max_data_points)
        
        # For dwell time
        self.entry_times = {}  # {track_id: entry_time}
        self.stay_durations = deque(maxlen=100)
        
        # For top/bottom periods
        self.peak_periods = []
        self.low_periods = []
        
        # For heatmap
        self.heatmap_data = np.zeros((10, 10))
        self.heatmap_size = (10, 10)
        
        # For alarms
        self.alarms = deque(maxlen=10)
        
        # For performance tracking
        self.frame_times = deque(maxlen=100)
        
    def update_data(self, total_people, entered=0, exited=0, frame_size=(1920, 1080), tracks=None):
        """Update all metrics with new data"""
        start_time = time.time()
        current_time = datetime.now()
        
        # Basic data
        self.timestamps.append(current_time)
        self.people_count.append(total_people)
        self.entered_count.append(entered)
        self.exited_count.append(exited)
        
        # Density (% of scene filled)
        density = (total_people / 50) * 100  # Assume max 50 people in scene
        self.density.append(min(density, 100))
        
        # Trend line (moving average)
        if len(self.people_count) >= self.trend_window:
            window = list(self.people_count)[-self.trend_window:]
            self.moving_avg.append(sum(window) / len(window))
        
        # Average dwell time
        self._update_stay_duration(tracks, current_time)
        
        # Top/bottom periods
        self._update_peak_periods(total_people, current_time)
        
        # Movement heatmap
        self._update_heatmap(tracks, frame_size)
        
        # Check alarm thresholds
        self._check_alarms(total_people, density, current_time)
        
        # Track frame processing time
        self.frame_times.append(time.time() - start_time)
    
    def _update_stay_duration(self, tracks, current_time):
        """Update people dwell time data"""
        if tracks is None:
            return
            
        current_ids = {track.track_id for track in tracks}
        
        # Update dwell time for people who left
        exited_ids = set(self.entry_times.keys()) - current_ids
        for track_id in exited_ids:
            stay_duration = (current_time - self.entry_times[track_id]).total_seconds()
            self.stay_durations.append(stay_duration)
            del self.entry_times[track_id]
        
        # Add new people
        for track in tracks:
            if track.track_id not in self.entry_times:
                self.entry_times[track.track_id] = current_time
    
    def _update_peak_periods(self, total_people, current_time):
        """Identify periods with most/least people"""
        if len(self.people_count) < 5:  # Min 5 measurements for period
            return
            
        # Add current period to appropriate list
        # Note: heapq is a min-heap, so for peak periods, we store negative values
        # For peak periods (top 5 highest counts)
        if not self.peak_periods or total_people >= -self.peak_periods[0][0]:
            heapq.heappush(self.peak_periods, (-total_people, current_time))
            if len(self.peak_periods) > 5:
                heapq.heappop(self.peak_periods)
        # For low periods (top 5 lowest counts)
        if not self.low_periods or total_people <= self.low_periods[0][0]:
            heapq.heappush(self.low_periods, (total_people, current_time))
            if len(self.low_periods) > 5:
                heapq.heappop(self.low_periods)
    
    def _update_heatmap(self, tracks, frame_size):
        """Update people movement heatmap"""
        if tracks is None or frame_size[0] == 0 or frame_size[1] == 0:
            return
            
        # Apply decay to existing heatmap data
        self.heatmap_data *= 0.7
        
        # Normalize positions within heatmap grid
        grid_x = np.linspace(0, frame_size[0], self.heatmap_size[0] + 1)
        grid_y = np.linspace(0, frame_size[1], self.heatmap_size[1] + 1)
        
        for track in tracks:
            try:
                x1, y1, x2, y2 = track.to_ltrb()
            except AttributeError:
                if hasattr(track, 'bbox'):
                    x1, y1, x2, y2 = track.bbox
                else:
                    logger.warning("Track object missing bbox information")
                    continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Find corresponding cell in heatmap grid
            cell_x = np.searchsorted(grid_x, cx) - 1
            cell_y = np.searchsorted(grid_y, cy) - 1
            
            if 0 <= cell_x < self.heatmap_size[0] and 0 <= cell_y < self.heatmap_size[1]:
                self.heatmap_data[cell_y, cell_x] += 1
    
    def _check_alarms(self, total_people, density, current_time):
        """Check if alarm thresholds are exceeded"""
        if total_people > self.people_threshold:
            self.alarms.append((current_time, f"EXCEEDED: {total_people} people in scene!"))
        elif density > self.density_threshold:
            self.alarms.append((current_time, f"HIGH DENSITY: {density:.1f}% occupancy!"))
    
    def _create_gauge(self, current_value, angle):
        """Create a gauge visualization for density"""
        min_val, max_val = 0, 100
        threshold = self.density_threshold
        
        # Draw threshold marker
        threshold_angle = 180 * (threshold - min_val) / (max_val - min_val)
        self.ax3.add_patch(Wedge((0.5, 0.4), 0.4, threshold_angle - 2, threshold_angle + 2, 
                                 width=0.01, color='red', alpha=0.7))
        
        # Draw colored arcs for the gauge itself
        self.ax3.add_patch(Wedge((0.5, 0.4), 0.4, 0, angle, width=0.2, 
                                 color=plt.cm.viridis(current_value/100)))
        self.ax3.add_patch(Wedge((0.5, 0.4), 0.4, angle, 180, width=0.2, 
                                 color='lightgray', alpha=0.3))
        
        # Draw needle
        needle_length = 0.35
        rad = np.deg2rad(angle)
        x = 0.5 + needle_length * np.cos(rad)
        y = 0.4 + needle_length * np.sin(rad)
        self.ax3.plot([0.5, x], [0.4, y], color='black', linewidth=2)
        
        # Draw center circle
        self.ax3.add_patch(Circle((0.5, 0.4), 0.05, color='black'))
        
        # Add text displaying the current value
        self.ax3.text(0.5, 0.25, f'{current_value:.1f}%', 
                      ha='center', va='center', fontsize=14, 
                      fontweight='bold', color='black')
        
        # Set gauge properties
        self.ax3.set_xlim(0, 1)
        self.ax3.set_ylim(0, 0.8)
        self.ax3.axis('off')
        self.ax3.set_title('Density Gauge', fontsize=10)
    
    def update_display(self):
        """Update all graphs and display dashboard"""
        if not self.timestamps or time.time() - self.last_update_time < 0.1:
            return
            
        self.last_update_time = time.time()
        start_time = time.time()
        
        # Convert timestamps to matplotlib-compatible format
        timestamps_numeric = mdates.date2num(self.timestamps)
        
        # 1. Total people
        self.ax1.clear()
        self.ax1.plot(timestamps_numeric, self.people_count, 'b-', label='Total people', linewidth=2)
        self.ax1.set_title('Total People in Scene', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Time', fontsize=10)
        self.ax1.set_ylabel('People Count', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # 2. Entered/Exited
        self.ax2.clear()
        bar_width = 0.0001
        self.ax2.bar(timestamps_numeric, self.entered_count, width=bar_width, color='g', label='Entered')
        self.ax2.bar(timestamps_numeric, [-x for x in self.exited_count], width=bar_width, color='r', label='Exited')
        self.ax2.set_title('People Entered/Exited', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Time', fontsize=10)
        self.ax2.set_ylabel('Count', fontsize=10)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # 3. Density Gauge
        self.ax3.clear()
        if self.density:
            current_density = self.density[-1]
            min_val, max_val = 0, 100
            angle = 180 * (current_density - min_val) / (max_val - min_val)
            angle = min(max(angle, 0), 180)
            self._create_gauge(current_density, angle)
        
        # 4. Current Metrics
        self.ax4.clear()
        self.ax4.axis('off')
        
        if self.people_count:
            metrics = [
                f"Current People: {self.people_count[-1]}",
                f"Entered: {self.entered_count[-1] if self.entered_count else 0}",
                f"Exited: {self.exited_count[-1] if self.exited_count else 0}",
                f"Density: {self.density[-1]:.1f}%",
                f"Avg Dwell: {np.mean(self.stay_durations) if self.stay_durations else 0:.1f}s",
                f"FPS: {1/np.mean(self.frame_times) if self.frame_times else 0:.1f}",
                f"Alarms: {len(self.alarms)}"
            ]
            
            for i, metric in enumerate(metrics):
                self.ax4.text(0.1, 0.85 - i*0.12, metric, fontsize=12, 
                              fontweight='bold', transform=self.ax4.transAxes)
        
        # 5. Trend line
        self.ax5.clear()
        if len(self.moving_avg) > 0:
            ma_timestamps = timestamps_numeric[-len(self.moving_avg):]
            self.ax5.plot(ma_timestamps, self.moving_avg, 'c-', linewidth=2, label='Moving avg')
        self.ax5.plot(timestamps_numeric, self.people_count, 'b:', alpha=0.5, label='Actual')
        self.ax5.set_title(f'Trend Line ({self.trend_window}-measurement avg)', fontsize=12, fontweight='bold')
        self.ax5.set_xlabel('Time', fontsize=10)
        self.ax5.set_ylabel('People Count', fontsize=10)
        self.ax5.legend(loc='upper right')
        self.ax5.grid(True, alpha=0.3)
        self.ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # 6. Histogram
        self.ax6.clear()
        if len(self.people_count) > 5:
            self.ax6.hist(self.people_count, bins=10, color='skyblue', edgecolor='black', alpha=0.8)
            self.ax6.set_title('People Count Distribution', fontsize=12, fontweight='bold')
            self.ax6.set_xlabel('People Count', fontsize=10)
            self.ax6.set_ylabel('Frequency', fontsize=10)
            self.ax6.grid(True, alpha=0.3)
        
        # 7. Top/Bottom periods
        self.ax7.clear()
        top_counts = []
        low_counts = []
        top_times = []
        low_times = []

        if self.peak_periods:
            top_5 = sorted(self.peak_periods, key=lambda x: (-x[0], x[1])) 
            top_counts = [-count for count, _ in top_5[:3]]
            top_times = [ts.strftime("%H:%M") for _, ts in top_5[:3]]

        if self.low_periods:
            low_5 = sorted(self.low_periods, key=lambda x: (x[0], x[1]))
            low_counts = [count for count, _ in low_5[:3]]
            low_times = [ts.strftime("%H:%M") for _, ts in low_5[:3]]

        y_pos_top = np.arange(len(top_counts))
        y_pos_low = np.arange(len(low_counts)) + len(top_counts) + 0.5 if top_counts else np.arange(len(low_counts))

        if top_counts or low_counts:
            if top_counts:
                bars_top = self.ax7.barh(y_pos_top, top_counts, color='r', alpha=0.7, label='Peak')
                for i, (count, time_label) in enumerate(zip(top_counts, top_times)):
                    self.ax7.text(count, y_pos_top[i], f' {time_label}', va='center', fontweight='bold', color='black')
            
            if low_counts:
                bars_low = self.ax7.barh(y_pos_low, low_counts, color='g', alpha=0.7, label='Low')
                for i, (count, time_label) in enumerate(zip(low_counts, low_times)):
                    self.ax7.text(count, y_pos_low[i], f' {time_label}', va='center', fontweight='bold', color='black')

            self.ax7.set_yticks(np.concatenate([y_pos_top, y_pos_low]))
            self.ax7.set_yticklabels(['Peak 1', 'Peak 2', 'Peak 3'][:len(top_counts)] + 
                                     ['Low 1', 'Low 2', 'Low 3'][:len(low_counts)])
            
            self.ax7.set_title('Top/Bottom Periods', fontsize=12, fontweight='bold')
            self.ax7.set_xlabel('People Count', fontsize=10)
            self.ax7.grid(True, alpha=0.3)
            self.ax7.invert_yaxis()
            if top_counts and low_counts:
                self.ax7.legend(loc='lower right')
        
        # 8. Heatmap - FIXED
        self.ax8.clear()
        if self.cbar is None:
            dummy_im = self.ax8.imshow(np.zeros((10,10)), cmap='viridis', interpolation='gaussian', origin='lower', aspect='auto')
            self.cbar = self.fig.colorbar(dummy_im, ax=self.ax8, 
                                           orientation='vertical', fraction=0.046, pad=0.04)
            self.cbar.set_label('Movement Intensity', fontsize=9)
            self.cbar.ax.set_visible(False)

        if np.sum(self.heatmap_data) > 0:
            heatmap_normalized = self.heatmap_data / np.max(self.heatmap_data)
            img_heatmap = self.ax8.imshow(heatmap_normalized, cmap='viridis', interpolation='gaussian', origin='lower', aspect='auto')
            self.ax8.set_title('People Movement Heatmap', fontsize=12, fontweight='bold')
            self.ax8.set_xticks([])
            self.ax8.set_yticks([])
            
            self.cbar.mappable.set_array(img_heatmap.get_array())
            self.cbar.mappable.set_clim(img_heatmap.get_clim())
            self.cbar.ax.set_visible(True)
        else:
            self.cbar.ax.set_visible(False)
            self.ax8.text(0.5, 0.5, "No heatmap data yet", 
                  horizontalalignment='center', verticalalignment='center', 
                  transform=self.ax8.transAxes, fontsize=10, color='gray')
    
        # 9. Dwell Time Distribution - FIXED
        self.ax9.clear()
        if self.stay_durations:
            stay_minutes = [t/60 for t in self.stay_durations if t > 0]
            
            if stay_minutes:
                # Definiranje binova za do 120 minuta (2 sata)
                bins = np.linspace(0, 120, 25)  # 25 binova od 0 do 120 minuta
                
                self.ax9.hist(stay_minutes, bins=bins, color='purple', alpha=0.7, edgecolor='black')
                self.ax9.set_title('Dwell Time Distribution', fontsize=12, fontweight='bold')
                self.ax9.set_xlabel('Minutes', fontsize=10)
                self.ax9.set_ylabel('Frequency', fontsize=10)
                self.ax9.grid(True, alpha=0.3)
                
                # Prikaz prosječnog vremena boravka
                avg_time = np.mean(stay_minutes)
                self.ax9.axvline(avg_time, color='r', linestyle='--', 
                                 label=f'Avg: {avg_time:.1f} min')
                self.ax9.legend()
                
                # Postavljanje limita za prikaz do 120 minuta
                self.ax9.set_xlim(0, 120)
            else:
                self.ax9.text(0.5, 0.5, "No dwell time data yet", 
                                 transform=self.ax9.transAxes, fontsize=10, color='gray')
        else:
            self.ax9.text(0.5, 0.5, "No dwell time data yet", 
                              transform=self.ax9.transAxes, fontsize=10, color='gray')

        # 10. Alarms
        self.ax10.clear()
        self.ax10.set_title('Recent Alarms', fontsize=12, fontweight='bold', color='red')
        self.ax10.set_facecolor('#FFF0F0')
        self.ax10.set_xticks([])
        self.ax10.set_yticks([])
        
        if self.alarms:
            alarm_text = "\n".join([f"{ts.strftime('%H:%M:%S')}: {msg}" 
                                            for ts, msg in list(self.alarms)[-3:]])
            self.ax10.text(0.02, 0.5, alarm_text, fontsize=11, 
                              verticalalignment='center', color='darkred',
                              bbox=dict(boxstyle='round', facecolor='#FFE4E4', alpha=0.8))
        else:
            self.ax10.text(0.5, 0.5, "No active alarms", 
                              fontsize=12, ha='center', va='center', 
                              color='green', fontweight='bold')
        
        # Auto-rotate date labels
        for ax in [self.ax1, self.ax2, self.ax5]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        
        # Add super title
        self.fig.suptitle(f"Real-time Analytics Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                           fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Convert to OpenCV image
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        
        # Get RGB image buffer
        buf = canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8)
        width, height = canvas.get_width_height()
        img = img.reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Display FPS on image
        fps = 1/np.mean(self.frame_times) if self.frame_times else 0
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow(self.window_name, img)
    
    def close(self):
        """Close window and release resources"""
        cv2.destroyWindow(self.window_name)
        plt.close(self.fig)

# Example usage
if __name__ == "__main__":
    visualizer = RealTimeVisualizer()
    
    # Simulate data updates
    try:
        total_people = 0
        entered_total = 0
        exited_total = 0
        
        class DummyTrack:
            def __init__(self, track_id, bbox):
                self.track_id = track_id
                self.bbox = bbox
            
            def to_ltrb(self):
                return self.bbox

        dummy_tracks = [
            DummyTrack(1, (100, 100, 120, 120)),
            DummyTrack(2, (200, 200, 220, 220))
        ]

        while True:
            # Generate simulated data
            change = np.random.randint(-2, 3)
            
            new_entered = max(0, change)
            new_exited = max(0, -change)
            
            total_people = max(0, total_people + new_entered - new_exited)
            
            entered_total += new_entered
            exited_total += new_exited

            simulated_tracks = []
            for i in range(total_people):
                if i < len(dummy_tracks):
                    x1, y1, x2, y2 = dummy_tracks[i].bbox
                    x_move = np.random.randint(-10, 11)
                    y_move = np.random.randint(-10, 11)
                    simulated_tracks.append(DummyTrack(dummy_tracks[i].track_id, 
                                                       (max(0, x1 + x_move), max(0, y1 + y_move), 
                                                        min(640, x2 + x_move), min(480, y2 + y_move))))
                else:
                    new_x = np.random.randint(0, 600)
                    new_y = np.random.randint(0, 400)
                    simulated_tracks.append(DummyTrack(100 + i, (new_x, new_y, new_x + 20, new_y + 20)))
            
            dummy_tracks = simulated_tracks 

            # Update visualizer
            visualizer.update_data(total_people, new_entered, new_exited, frame_size=(640, 480), tracks=simulated_tracks)
            visualizer.update_display()
            
            # Exit on ESC key
            if cv2.waitKey(10) == 27:
                break
                
            time.sleep(0.05)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Cleaning up resources...")
        visualizer.close()


if __name__ == "__main__":
    visualizer = RealTimeVisualizer()

    for _ in range(30):
        visualizer.update_data(total_people=np.random.randint(10, 40),
                               entered=np.random.randint(0, 5),
                               exited=np.random.randint(0, 5),
                               frame_size=(640, 480),
                               tracks=[
                                   DummyTrack(i, (
                                       np.random.randint(0, 600),
                                       np.random.randint(0, 400),
                                       np.random.randint(600, 640),
                                       np.random.randint(400, 480)
                                   )) for i in range(np.random.randint(5, 10))
                               ])
        visualizer.update_display()
        time.sleep(0.05)

    cv2.waitKey(0)
    visualizer.close()