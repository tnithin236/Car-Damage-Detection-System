"""Model loading and raw inference. No training code lives here."""
import os, time
from pathlib import Path
import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS = os.environ.get("DAMAGE_MODEL_PATH", "models/best.pt")


class Detector:
    def __init__(self, weights: str = DEFAULT_WEIGHTS):
        if not Path(weights).exists():
            raise FileNotFoundError(f"Model weights not found at '{weights}'. Put best.pt in models/.")
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights)
        self.names = self.model.names              # read from the model, never hardcoded
        self.task = self.model.task                # 'detect' or 'segment'
        self.n_params = sum(p.numel() for p in self.model.model.parameters())
        self.size_mb = Path(weights).stat().st_size / 1e6

    def predict(self, image_bgr, conf=0.25, iou=0.5, imgsz=640):
        t0 = time.perf_counter()
        r = self.model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, device=self.device,
                               retina_masks=(self.task == "segment"), verbose=False)[0]
        return r, (time.perf_counter() - t0) * 1000
