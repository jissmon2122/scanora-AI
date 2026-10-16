"""
Scanora AI — Deep Neural Network for Mammographic Malignancy & Density Classification.

Features:
1. 1-Channel Medical Grayscale Input [B, 1, H, W].
2. Dual-Head Multi-Task Learning:
   - Malignancy Binary Head: P(Malignant | Image)
   - Breast Density Head: P(Density in {A, B, C, D} | Image)
3. Temperature-Scaled Calibrated Probabilities.
4. Prediction Entropy for Explicit Uncertainty / Indeterminate Output.
"""

import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from ai_service.models.calibration import TemperatureScaler


class MammographyClassifier(nn.Module):
    def __init__(
        self,
        pretrained_backbone: bool = True,
        dropout_rate: float = 0.3,
        temperature: float = 1.25,
    ):
        super().__init__()
        self.backbone_name = "efficientnet_b2"

        # Load backbone
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained_backbone else None
        base_model = efficientnet_b2(weights=weights)

        # Adapt first convolutional layer from 3-channel RGB to 1-channel Grayscale
        orig_conv = base_model.features[0][0]
        self.initial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        # Initialize grayscale weights by averaging RGB pretrained weights
        if pretrained_backbone:
            with torch.no_grad():
                self.initial_conv.weight.data = orig_conv.weight.data.mean(
                    dim=1, keepdim=True
                )

        # Replace first layer and retain feature extractor
        base_model.features[0][0] = self.initial_conv
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = base_model.classifier[1].in_features  # 1408 for EfficientNet-B2

        # Dual Task Heads
        # 1. Malignancy Binary Head (1 logit for BCEWithLogitsLoss)
        self.malignancy_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.Mish(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, 1),
        )

        # 2. Breast Density Head (4 classes: A, B, C, D)
        self.density_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.Mish(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 4),
        )

        # Temperature calibration layer
        self.temp_scaler = TemperatureScaler(initial_temperature=temperature)

        # Checkpoint provenance metadata
        self.checkpoint_id: Optional[str] = None
        self.is_trained: bool = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            malignancy_logits: Shape [B, 1]
            density_logits: Shape [B, 4]
        """
        feats = self.features(x)
        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)

        raw_malignancy_logits = self.malignancy_head(flattened)
        density_logits = self.density_head(flattened)

        # Scale malignancy logits by temperature
        calibrated_malignancy_logits = self.temp_scaler(raw_malignancy_logits)

        return calibrated_malignancy_logits, density_logits

    @torch.no_grad()
    def predict_single(
        self,
        tensor_1x1xHxW: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        uncertainty_entropy_threshold: float = 0.55,
    ) -> Dict[str, Any]:
        """
        Executes single-image inference and returns calibrated probabilities,
        density classification, and uncertainty assessment.
        """
        self.eval()
        tensor_input = tensor_1x1xHxW.to(device)

        cal_mal_logits, density_logits = self.forward(tensor_input)

        # Malignancy probability via Sigmoid
        mal_prob = float(torch.sigmoid(cal_mal_logits).squeeze().cpu().item())
        benign_prob = float(1.0 - mal_prob)

        # Binary Entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
        p = max(1e-7, min(1.0 - 1e-7, mal_prob))
        entropy = float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
        is_indeterminate = entropy > uncertainty_entropy_threshold

        # Density Probabilities via Softmax
        density_probs = F.softmax(density_logits, dim=-1).squeeze().cpu().numpy()
        density_classes = ["A", "B", "C", "D"]
        density_idx = int(np.argmax(density_probs))
        predicted_density = density_classes[density_idx]
        density_confidence = float(density_probs[density_idx])

        density_descriptions = {
            "A": "Entirely fatty tissue (ACR Density A)",
            "B": "Scattered fibroglandular densities (ACR Density B)",
            "C": "Heterogeneously dense tissue (ACR Density C)",
            "D": "Extremely dense tissue (ACR Density D)",
        }

        return {
            "model_architecture": self.backbone_name,
            "is_trained_checkpoint": self.is_trained,
            "checkpoint_id": self.checkpoint_id or "uncalibrated_baseline",
            "malignancy_score": round(mal_prob * 100.0, 2),
            "benign_score": round(benign_prob * 100.0, 2),
            "calibrated_malignancy_probability": round(mal_prob, 4),
            "prediction_entropy": round(entropy, 4),
            "is_indeterminate_case": is_indeterminate,
            "density": predicted_density,
            "density_description": density_descriptions[predicted_density],
            "density_confidence": round(density_confidence * 100.0, 2),
            "all_density_probabilities": {
                c: round(float(p), 4) for c, p in zip(density_classes, density_probs)
            },
        }

    def load_checkpoint(self, checkpoint_path: str, device: torch.device = torch.device("cpu")):
        """Loads verified model weights and updates model provenance."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist.")

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
