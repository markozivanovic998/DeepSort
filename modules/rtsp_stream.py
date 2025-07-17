import threading
import time
import cv2
import logging

logger = logging.getLogger(__name__)

class ThreadedRTSPStream:
    def __init__(self, src: str, reconnect_interval=5, max_reconnect_attempts=10):
        self.src = src
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_attempts = 0
        self.stream = None
        self.fps = 30
        self.grabbed = False
        self.frame = None
        self.started = False
        self.read_lock = threading.Lock()
        self.last_read_time = time.time()
        self._open_stream()
        
    def _open_stream(self):
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        for backend in backends:
            try:
                self.stream = cv2.VideoCapture(self.src, backend)
                self.stream.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                self.stream.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                
                if self.stream.isOpened():
                    self.fps = self.stream.get(cv2.CAP_PROP_FPS)
                    if self.fps <= 0:
                        self.fps = 30
                    self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    logger.info(f"Stream opened with backend: {backend}, FPS: {self.fps}")
                    self.reconnect_attempts = 0
                    return
                else:
                    logger.warning(f"Failed to open stream with backend {backend}")
            except Exception as e:
                logger.error(f"Error opening stream with backend {backend}: {str(e)}")
        
        logger.error("All backends failed to open stream")
        raise IOError(f"Unable to open video stream at {self.src}")

    def _reconnect(self):
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnect attempts reached")
            return False
            
        logger.warning(f"Reconnecting... attempt {self.reconnect_attempts+1}/{self.max_reconnect_attempts}")
        time.sleep(self.reconnect_interval)
        
        try:
            if self.stream:
                self.stream.release()
            self._open_stream()
            return True
        except Exception as e:
            self.reconnect_attempts += 1
            logger.error(f"Reconnect failed: {str(e)}")
            return False

    def start(self):
        if self.started:
            logger.warning("ThreadedRTSPStream already started.")
            return None
            
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if not self.stream or not self.stream.isOpened():
                if not self._reconnect():
                    break
                    
            grabbed = False
            try:
                grabbed = self.stream.grab()
            except Exception as e:
                logger.error(f"Grab error: {str(e)}")
                
            if not grabbed:
                logger.warning("Frame grab failed")
                if not self._reconnect():
                    break
                continue
                
            try:
                retrieved, frame = self.stream.retrieve()
            except Exception as e:
                logger.error(f"Retrieve error: {str(e)}")
                retrieved = False
                frame = None
                
            if not retrieved or frame is None:
                logger.warning("Frame retrieve failed")
                if not self._reconnect():
                    break
                continue
                
            with self.read_lock:
                self.grabbed = retrieved
                self.frame = frame
                self.last_read_time = time.time()
            
            # Maintain FPS
            sleep_time = max(0, 1.0 / self.fps - (time.time() - self.last_read_time))
            time.sleep(sleep_time)

    def read(self):
        with self.read_lock:
            if self.frame is None:
                return False, None
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def release(self):
        self.stop()
        if self.stream is not None and self.stream.isOpened():
            self.stream.release()
        self.stream = None