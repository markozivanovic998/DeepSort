from abc import ABC, abstractmethod

# Apstraktna bazna klasa za sve modele.
# Svaki konkretan model mora naslediti ovu klasu i implementirati metode.
class BaseModel(ABC):

    # Metoda za učitavanje modela (lazy load).
    @abstractmethod
    def load(self):
        pass

    # Metoda za predikciju na slici.
    @abstractmethod
    def predict_image(self, image):
        pass

    # Metoda za predikciju na videu.
    @abstractmethod
    def predict_video(self, video_path):
        pass
