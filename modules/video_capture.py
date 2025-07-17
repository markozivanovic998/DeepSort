import threading
import time
import cv2
import queue

class ThreadedVideoCapture:
    def __init__(self, src: str):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError(f"Unable to open video file at {src}")
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    def start(self):
        if self.started:
            print("Warning: ThreadedVideoCapture already started.")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if self.stream.isOpened():
                grabbed, frame = self.stream.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time.sleep(1.0 / self.fps)  # Read at half the frame rate to maintain buffer

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def release(self):
        self.stop()
        self.stream.release()