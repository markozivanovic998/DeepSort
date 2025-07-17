import os
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy.ndimage import gaussian_filter
import io
import matplotlib.pyplot as plt
from PIL import Image

from models.BaseModel import BaseModel
from .network.csrnet import CSRNet  # relativni import


# Absolutna putanja do CSRNetModel.py
base_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(base_dir, 'weights', 'weights.pth')


class CSRNetModel(BaseModel):
    """
    Konkretna implementacija modela za CSRNet sa dodatnim vizualizacijama.
    """

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load(self):
        if self.model is None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights not found at: {weights_path}")

            self.model = CSRNet()
            checkpoint = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()

    def generate_heatmap_image(self, density_map):
        """
        Generiše heatmap sliku iz density map-e (PIL.Image).
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(density_map, cmap='jet', interpolation='bilinear')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close()
        buf.seek(0)
        heatmap_img = Image.open(buf)
        return heatmap_img

    def overlay_heatmap_on_image(self, original_image, density_map, alpha=0.5):
        """
        Preklapa heatmap na originalnu sliku.
        """
        # Normalize density map
        norm_density = (density_map / (density_map.max() + 1e-8)) * 255
        norm_density = np.clip(norm_density, 0, 255).astype(np.uint8)
        density_resized = Image.fromarray(norm_density).resize(original_image.size)
        density_colored = plt.cm.jet(np.array(density_resized) / 255)[:, :, :3]  # RGB

        density_colored_img = Image.fromarray((density_colored * 255).astype(np.uint8))
        blended = Image.blend(original_image.convert("RGB"), density_colored_img, alpha=alpha)
        return blended

    def predict_image(self, image):
        """
        Predikcija sa generisanjem dodatnih vizualizacija.
        """
        self.load()
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        count = output.sum().item()

        # Obrada density map-e
        density_map = output.squeeze().cpu().numpy()
        density_map_smoothed = gaussian_filter(density_map, sigma=3)

        # Generiši heatmap sliku (PIL.Image)
        heatmap_img = self.generate_heatmap_image(density_map_smoothed)

        # Generiši overlay
        overlay_img = self.overlay_heatmap_on_image(image, density_map_smoothed)

        return {
            'count': count,
            'heatmap_img': heatmap_img,
            'overlay_img': overlay_img
        }

    def predict_video(self, video_path):
        raise NotImplementedError("CSRNet ne podržava video obradu.")
