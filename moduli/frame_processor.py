from ultralytics import YOLO
#from src import utils

def init_model(model_path: str) -> YOLO:
    return YOLO(model_path)

# In frame_processor.py
def process_frame(frame, model, tracker, track_history, draw_bounding_boxes=True):
    frame = utils.resizeImage(frame)
    result = model.track(frame, persist=True, tracker=tracker, classes=0, verbose=False)[0]
    
    if draw_bounding_boxes:
        frame = utils.draw_tracking_info(frame, result, track_history)
    
    return frame

