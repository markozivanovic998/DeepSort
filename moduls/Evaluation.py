import numpy as np
from collections import defaultdict

class EvaluationMetrics:
    """
    Klasa za izračunavanje metrika evaluacije detekcije objekata kao što su
    Preciznost, Opoziv, F1-score i osnova za mAP (mean Average Precision).

    ⚠️ Ova klasa zahteva Ground Truth (GT) podatke za poređenje sa predikcijama.
    """
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        """
        Inicijalizuje kalkulator metrika.

        Args:
            num_classes (int): Ukupan broj klasa koje model detektuje.
            iou_threshold (float): Prag za Intersection over Union (IoU) koji određuje
                                   da li je detekcija True Positive.
        """
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        # Stats: Sadrži rečnike za TP, FP, FN za svaku klasu
        self.stats = [{'TP': 0, 'FP': 0, 'FN': 0} for _ in range(num_classes)]

    def _calculate_iou(self, box1, box2):
        """Proračunava Intersection over Union (IoU) za dva bounding box-a."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def process_frame(self, predictions: list, ground_truths: list):
        """
        Obrađuje jedan frejm, poredeći predikcije sa ground truth podacima.

        Args:
            predictions (list): Lista predikcija. Format: [[x1, y1, x2, y2, conf, class_id], ...]
            ground_truths (list): Lista ground truth box-ova. Format: [[x1, y1, x2, y2, class_id], ...]
        """
        # Koristi se za praćenje koji su GT box-ovi već "upareni"
        detected_gt_indices = set()

        # Sortiranje predikcija po pouzdanosti (confidence) opadajuće
        predictions.sort(key=lambda x: x[4], reverse=True)

        for pred_box in predictions:
            pred_xyxy = pred_box[:4]
            pred_cls = int(pred_box[5])
            
            best_iou = 0
            best_gt_idx = -1

            for i, gt_box in enumerate(ground_truths):
                gt_cls = int(gt_box[4])
                if gt_cls == pred_cls and i not in detected_gt_indices:
                    iou = self._calculate_iou(pred_xyxy, gt_box[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_iou >= self.iou_threshold:
                if best_gt_idx not in detected_gt_indices:
                    self.stats[pred_cls]['TP'] += 1
                    detected_gt_indices.add(best_gt_idx)
                else:
                    # GT je već uparen sa predikcijom veće pouzdanosti
                    self.stats[pred_cls]['FP'] += 1
            else:
                self.stats[pred_cls]['FP'] += 1
        
        # Svi GT-ovi koji nisu detektovani (upareni) su False Negatives
        num_gt_per_class = defaultdict(int)
        for gt_box in ground_truths:
            num_gt_per_class[int(gt_box[4])] += 1
        
        for class_id in range(self.num_classes):
            self.stats[class_id]['FN'] = num_gt_per_class[class_id] - self.stats[class_id]['TP']


    def calculate_metrics(self) -> dict:
        """
        Izračunava finalne metrike (Precision, Recall, F1) na osnovu svih obrađenih frejmova.
        """
        results = {}
        total_tp, total_fp, total_fn = 0, 0, 0

        for class_id in range(self.num_classes):
            tp = self.stats[class_id]['TP']
            fp = self.stats[class_id]['FP']
            fn = self.stats[class_id]['FN']

            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results[f'class_{class_id}'] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1_score, 4),
                'TP': tp,
                'FP': fp,
                'FN': fn
            }
        
        # Macro-average metrike (ukupne performanse)
        macro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        macro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0

        results['overall_summary'] = {
            'precision': round(macro_precision, 4),
            'recall': round(macro_recall, 4),
            'f1_score': round(macro_f1, 4)
        }
        
        return results