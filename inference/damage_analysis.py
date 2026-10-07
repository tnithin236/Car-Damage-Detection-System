"""Turns raw YOLO output into a structured analysis (area, severity, per-damage rows)."""
import numpy as np, cv2
from .severity import detection_severity, overall_severity


def analyze(result, names, inference_ms: float) -> dict:
    H, W = result.orig_shape
    img_area = float(H * W)
    union = np.zeros((H, W), bool)
    has_masks = result.masks is not None
    dets = []
    n = len(result.boxes)
    for i in range(n):
        x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy()
        if has_masks:
            m = result.masks.data[i].cpu().numpy() > 0.5
            if m.shape != (H, W): m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m = np.zeros((H, W), bool); m[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)] = True
        union |= m
        area_pct = m.sum() / img_area * 100
        name = names[int(result.boxes.cls[i])]
        dets.append(dict(damage=name, confidence=float(result.boxes.conf[i]), area_pct=float(area_pct),
                         box=[float(x1), float(y1), float(x2), float(y2)],
                         severity=detection_severity(name, area_pct)))
    total = float(union.sum() / img_area * 100)
    return dict(detections=dets, count=n, total_area_pct=total,
                severity=overall_severity(dets, total),
                mean_confidence=float(np.mean([d["confidence"] for d in dets])) if dets else 0.0,
                area_method="segmentation masks" if has_masks else "bounding boxes",
                area_reference="full image area (no vehicle segmentation available)",
                inference_ms=inference_ms)
