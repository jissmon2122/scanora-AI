"""
Scanora AI — Deep Learning Lesion Localization Engine (Faster R-CNN).

Detects:
1. Mass
2. Suspicious Calcification
3. Architectural Distortion
4. Asymmetry

Applies:
- Non-Maximum Suppression (NMS)
- Anatomical quadrant localization
- Safe fallback when weights are not present (returns empty list with clear indicator)
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms


LESION_CLASSES = [
    "__background__",
    "Mass",
    "Suspicious Calcification",
    "Architectural Distortion",
    "Asymmetry",
]


class MammographyLesionDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = len(LESION_CLASSES),
        pretrained_backbone: bool = True,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.35,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
        self.detector = fasterrcnn_resnet50_fpn(weights=weights)

        # Adapt first conv layer to 1-channel grayscale
        orig_conv = self.detector.backbone.body.conv1
        self.adapted_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        if pretrained_backbone:
            with torch.no_grad():
                self.adapted_conv1.weight.data = orig_conv.weight.data.mean(
                    dim=1, keepdim=True
                )

        self.detector.backbone.body.conv1 = self.adapted_conv1

        # Replace box predictor head for mammography classes
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.is_trained: bool = False
        self.checkpoint_id: Optional[str] = None

    @staticmethod
    def infer_quadrant(cx_norm: float, cy_norm: float, laterality: str = "right") -> str:
        """
        Maps normalized center coordinate (0.0 to 1.0) to anatomical breast quadrant.
        Takes laterality into account (e.g. outer is lateral).
        """
        is_upper = cy_norm < 0.5
        is_outer = cx_norm > 0.5 if laterality.lower() == "right" else cx_norm < 0.5

        # Check retroareolar (central circle within 15% radius)
        dist_from_center = np.sqrt((cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2)
        if dist_from_center < 0.15:
            return "Retroareolar"

        if is_upper and is_outer:
            return "Upper Outer Quadrant"
        elif is_upper and not is_outer:
            return "Upper Inner Quadrant"
        elif not is_upper and is_outer:
            return "Lower Outer Quadrant"
        else:
            return "Lower Inner Quadrant"

    @torch.no_grad()
    def detect(
        self,
        tensor_1x1xHxW: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        laterality: str = "right",
    ) -> Dict[str, Any]:
        """
        Executes real object detection.
        If no trained checkpoint is loaded, explicitly returns detector unavailable.
        """
        if not self.is_trained:
            return {
                "detector_status": "UNAVAILABLE",
                "message": (
                    "Lesion detection model is not loaded. Bounding boxes are disabled. "
                    "Absence of boxes does NOT indicate absence of lesions."
                ),
                "findings": [],
            }

        self.eval()
        image_input = tensor_1x1xHxW.squeeze(0).to(device)  # Shape [1, H, W]

        # Faster R-CNN expects a list of tensors [C, H, W] in [0, 1] range
        predictions = self.detector([image_input])
        pred = predictions[0]

        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        # Apply confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        if len(boxes) == 0:
            return {
                "detector_status": "ACTIVE",
                "findings": [],
                "message": "No focal lesions detected at current confidence threshold.",
            }

        # Apply Non-Maximum Suppression (NMS)
        keep = nms(boxes, scores, self.nms_iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        findings: List[Dict[str, Any]] = []
        target_size = tensor_1x1xHxW.shape[-1]

        for i in range(len(boxes)):
            box = boxes[i].numpy()
            score = float(scores[i].item())
            class_id = int(labels[i].item())
            class_name = LESION_CLASSES[class_id] if class_id < len(LESION_CLASSES) else "Lesion"

            # Normalized coordinates in [0, 1]
            xmin_norm = float(box[0] / target_size)
            ymin_norm = float(box[1] / target_size)
            xmax_norm = float(box[2] / target_size)
            ymax_norm = float(box[3] / target_size)

            cx = (xmin_norm + xmax_norm) / 2.0
            cy = (ymin_norm + ymax_norm) / 2.0
            location = self.infer_quadrant(cx, cy, laterality)

            risk_level = "Suspicious" if score > 0.6 else "Indeterminate"
            if "Calcification" in class_name and score > 0.75:
                risk_level = "Highly Suspicious"

            findings.append({
                "id": f"det-{i+1}",
                "type": class_name,
                "location": location,
                "confidencePercent": round(score * 100.0, 1),
                "riskLevel": risk_level,
                "raw_model_box": {
                    "xmin": xmin_norm,
                    "ymin": ymin_norm,
                    "xmax": xmax_norm,
                    "ymax": ymax_norm,
                },
            })

        return {
            "detector_status": "ACTIVE",
            "findings": findings,
            "count": len(findings),
        }

    def load_checkpoint(self, checkpoint_path: str, device: torch.device = torch.device("cpu")):
        """Loads trained Faster R-CNN checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Detector checkpoint '{checkpoint_path}' does not exist.")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
            self.checkpoint_id = checkpoint.get("checkpoint_id", os.path.basename(checkpoint_path))
        else:
            self.load_state_dict(checkpoint)
            self.checkpoint_id = os.path.basename(checkpoint_path)

        self.is_trained = True
        self.to(device)
        self.eval()
