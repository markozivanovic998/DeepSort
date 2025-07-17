import cv2
import threading
import queue


class ThreadedVideoWriter:
    """
    Video writer koji koristi poseban thread za upisivanje frame-ova u video.
    Omogućava brži rad glavnog procesa (npr. model inference).
    """

    def __init__(self, output_path, fourcc, fps, frame_size, queue_size=128):
        self.out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Pozadinski thread koji čita iz queue-a i upisuje u fajl."""
        while True:
            if self.stopped and self.queue.empty():
                break
            frame = self.queue.get()
            if frame is not None:
                self.out.write(frame)

    def write(self, frame):
        """Dodaje frame u queue."""
        self.queue.put(frame)

    def stop(self):
        """Zaustavlja writer i zatvara fajl."""
        self.stopped = True
        self.thread.join()
        self.out.release()
