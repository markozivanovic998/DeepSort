import threading
import time
import cv2

class ThreadedRTSPStream:
    def __init__(self, src: str):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.stream.isOpened():
            raise IOError(f"Unable to open video stream at {src}")
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            print("Warning: ThreadedRTSPStream already started.")
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
            time.sleep(0.001)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def release(self):
        self.stream.release()