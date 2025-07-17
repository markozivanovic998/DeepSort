import os
from types import SimpleNamespace
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import time
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter

from models.BaseModel import BaseModel
from models.common.video_writer import ThreadedVideoWriter
from models.CDETRModel.network import build_model
from models.CDETRModel.util import misc as utils
from models.common.config_loader import ConfigLoader



class CDETRModel(BaseModel):
    """
    CDETR model za people counting na video snimcima.
    """

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Fiksne dimenzije za video (kao u originalnom modelu)
        self.width = 1024
        self.height = 768
        self.crop_size = 256
        self.num_w = self.width // self.crop_size
        self.num_h = self.height // self.crop_size

        # Putanja do težina i konfiguracije
        base_dir = os.path.dirname(__file__)
        self.weights_path = os.path.join(base_dir, 'weights', 'video_model.pth')
        self.config_path = os.path.join(base_dir, 'cdetr_config.yaml')

    def load(self):
        """Lazy-load modela (učitava samo prvi put)."""
        if self.model is not None:
            return

        loader = ConfigLoader.from_yaml(self.config_path)
        args = loader.to_namespace()

        model, _, _ = build_model(args)
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        if os.path.isfile(self.weights_path):
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            # Preimenovanje ključeva zbog različitih oznaka
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace('bbox', 'point')
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            print(f"✅ Loaded DETR model from {self.weights_path}")
        else:
            raise FileNotFoundError(f"❌ Checkpoint not found at {self.weights_path}")

        self.model = model.eval()

    def predict_video(self, video_path, start_frame=0, end_frame=None, progress_callback=None):
        """
        Glavna metoda za analizu video fajla.
        Obrada po frame-u, prikaz rezultata i snimanje novog videa.
        """
        self.load()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        # Dodajemo broj frame-ova za progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Ako nije zadat end_frame, idemo do kraja
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
        current_frame = 0

        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"detr_output_{timestamp}.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = ThreadedVideoWriter(output_path, fourcc, 30, (self.width * 2, self.height * 2))

        # Prebacujemo se na start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        processed_frames = 0
        total_to_process = end_frame - start_frame

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            if progress_callback:
                progress_callback(current_frame / total_frames)

            frame = cv2.resize(frame, (self.width, self.height))
            ori_frame = frame.copy()

            # Priprema ulaza
            image = self.to_tensor(frame)
            image = self.normalize(image).to(self.device)
            image = image.view(3, self.num_h, self.crop_size, self.width).view(
                3, self.num_h, self.crop_size, self.num_w, self.crop_size
            )
            image = image.permute(0, 1, 3, 2, 4).contiguous().view(
                3, self.num_w * self.num_h, self.crop_size, self.crop_size
            ).permute(1, 0, 2, 3)

            # Inference
            with torch.no_grad():
                outputs = self.model(image)

            out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
            topk_points = topk_indexes // out_logits.shape[2]
            out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
            out_point = out_point * self.crop_size
            value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)

            point_map, density_map, frame, count = self._generate_maps(value_points, frame)

            res1 = np.hstack((ori_frame, point_map))
            res2 = np.hstack((density_map, frame))
            final_frame = np.vstack((res1, res2))

            cv2.putText(final_frame, f"Count: {count}", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            writer.write(final_frame)
            current_frame += 1

        cap.release()
        writer.stop()
        return output_path

    def _generate_maps(self, out_pointes, frame):
        """
        Generiše prikaz detekcija i gustine za dati frame.
        """
        kpoint_list = []

        for i in range(len(out_pointes)):
            out_value = out_pointes[i].squeeze(0)[:, 0].cpu().numpy()
            out_point = out_pointes[i].squeeze(0)[:, 1:3].cpu().numpy().tolist()
            k = np.zeros((self.crop_size, self.crop_size))

            for j in range(len(out_point)):
                if out_value[j] < 0.25:
                    break
                x, y = int(out_point[j][0]), int(out_point[j][1])
                if 0 <= x < self.crop_size and 0 <= y < self.crop_size:
                    k[x, y] = 1

            kpoint_list.append(k)

        kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
        kpoint = kpoint.view(self.num_h, self.num_w, self.crop_size, self.crop_size)\
            .permute(0, 2, 1, 3).contiguous().view(self.num_h, self.crop_size, self.width).view(self.height, self.width).cpu().numpy()

        density_map = gaussian_filter(kpoint.copy(), 6)
        density_map = density_map / np.max(density_map) * 255 if np.max(density_map) > 0 else density_map
        density_map = density_map.astype(np.uint8)
        density_map = cv2.applyColorMap(density_map, 2)

        pred_coor = np.nonzero(kpoint)
        count = len(pred_coor[0])

        point_map = np.zeros((self.height, self.width, 3), dtype="uint8") + 255
        for i in range(count):
            w, h = int(pred_coor[1][i]), int(pred_coor[0][i])
            cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)
            cv2.circle(frame, (w, h), 3, (0, 255, 50), -1)

        return point_map, density_map, frame, count

    def predict_image(self, image):
        """
        Obrada slike (PIL Image) — kao single frame obrada iz videa.
        """
        self.load()

        # Pretvaranje u OpenCV format (ako je PIL)
        if hasattr(image, 'convert'):
            image = np.array(image.convert("RGB"))
        else:
            image = np.array(image)

        # Resize kao kod videa
        frame = cv2.resize(image, (self.width, self.height))
        ori_frame = frame.copy()

        # Priprema ulaza
        tensor_image = self.to_tensor(frame)
        tensor_image = self.normalize(tensor_image).to(self.device)
        tensor_image = tensor_image.view(3, self.num_h, self.crop_size, self.width).view(
            3, self.num_h, self.crop_size, self.num_w, self.crop_size
        )
        tensor_image = tensor_image.permute(0, 1, 3, 2, 4).contiguous().view(
            3, self.num_w * self.num_h, self.crop_size, self.crop_size
        ).permute(1, 0, 2, 3)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor_image)

        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        topk_points = topk_indexes // out_logits.shape[2]
        out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
        out_point = out_point * self.crop_size
        value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)

        point_map, density_map, frame_with_points, count = self._generate_maps(value_points, ori_frame)

        # Vraćamo rezultat
        return {
            'count': count,
            'frame_with_points': frame_with_points,
            'density_map': density_map
        }

