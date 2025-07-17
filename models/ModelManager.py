from models.CSRNetModel.CSRNetModel import CSRNetModel
from models.CDETRModel.CDETRModel import CDETRModel
# Kasnije ovde lako dodajemo i druge modele:
# from models.DeepSortModel.DeepSortModel import DeepSortModel

# Glavna klasa za upravljanje svim modelima.
# Centralizuje pozive i sakriva detalje implementacije.
class ModelManager:
    def __init__(self):
        # Ovde registrujemo sve modele koji postoje u sistemu
        self.models = {
            'CSRNet': CSRNetModel(),
            'CDETR': CDETRModel()
            # 'DeepSort': DeepSortModel(),    <-- lako se dodaje
        }

    # Jedinstveni interfejs za image predikciju
    def predict_image(self, model_name, image):
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")
        return model.predict_image(image)

    # Jedinstveni interfejs za video predikciju
    def predict_video(self, model_name, video_path, **kwargs):
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")
        return model.predict_video(video_path, **kwargs)

