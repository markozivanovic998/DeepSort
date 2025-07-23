import torch
from ultralytics import YOLO

def initialize_model(model_path):
    """Proverava dostupnost GPU-a i učitava YOLO model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Koristim uređaj: {device}")
    
    model = YOLO(model_path)
    model.to(device)
    
    return model, device