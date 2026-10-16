"""
Scanora AI — Probability Calibration & Reliability Metrics.

Provides:
1. Temperature Scaling for post-hoc probability calibration of deep neural networks.
2. Expected Calibration Error (ECE) calculation.
3. Brier Score computation.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss


class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to neural network logits to produce calibrated probabilities.
    Guo et al. 'On Calibration of Modern Neural Networks', ICML 2017.
    """
    def __init__(self, initial_temperature: float = 1.25):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by temperature parameter T > 0."""
        # Ensure temperature remains strictly positive
        temp = torch.clamp(self.temperature, min=0.01)
        return logits / temp

    @torch.no_grad()
    def calibrate_probability(self, logit: float) -> float:
        """Converts raw scalar binary logit into calibrated probability [0.0, 1.0]."""
        t_logit = logit / max(0.01, float(self.temperature.item()))
        prob = 1.0 / (1.0 + np.exp(-t_logit))
        return float(prob)


def compute_expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across probability bins.
    ECE measures the difference in expectation between confidence and accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probabilities)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Identify samples falling into this bin
        in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            bin_accuracy = np.mean(labels[in_bin])
            bin_confidence = np.mean(probabilities[in_bin])
            ece += np.abs(bin_accuracy - bin_confidence) * prop_in_bin

    return float(ece)


def compute_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Computes mean squared error of probabilistic predictions (Brier Score)."""
    return float(brier_score_loss(labels, probabilities))
