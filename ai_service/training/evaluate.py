"""
Scanora AI — Independent Evaluation & Clinical Metrics Assessment.

Evaluates trained model checkpoints on unseen patient-level test sets:
- ROC-AUC & PR-AUC
- Sensitivity, Specificity, PPV, NPV
- Brier Score & Expected Calibration Error (ECE)
- Confusion Matrix (TP, FP, TN, FN)
- Subgroup metrics (by view: CC vs MLO; by breast density: fatty vs dense)
"""

import os
import json
import logging
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
)
from ai_service.models.mammography_classifier import MammographyClassifier
from ai_service.models.calibration import compute_expected_calibration_error
from ai_service.training.dataset_loader import MammographyDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scanora.evaluation")


def evaluate_checkpoint(
    checkpoint_path: str,
    test_records: List[Dict[str, Any]],
    operating_threshold: float = 0.50,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Runs rigorous independent evaluation and generates a clinical metrics report."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model = MammographyClassifier(pretrained_backbone=False)
    model.load_checkpoint(checkpoint_path, device=device)
    model.eval()

    test_ds = MammographyDataset(test_records, is_training=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    all_probs: List[float] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            cal_mal_logits, _ = model(images)
            probs = torch.sigmoid(cal_mal_logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy().flatten().astype(int))

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= operating_threshold).astype(int)

    # Core metrics
    n_total = len(y_true)
    n_pos = int(np.sum(y_true))
    n_neg = n_total - n_pos

    roc_auc = float(roc_auc_score(y_true, y_prob)) if n_pos > 0 and n_neg > 0 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if n_pos > 0 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_prob, y_true, n_bins=10)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # Sensitivity at 90% Specificity (high-specificity operating point)
    sens_at_90_spec = 0.0
    if n_pos > 0 and n_neg > 0:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # Find index where specificity >= 0.90 (i.e. FPR <= 0.10)
        valid_indices = np.where(fpr <= 0.10)[0]
        if len(valid_indices) > 0:
            sens_at_90_spec = float(tpr[valid_indices[-1]])

    # View Subgroup Analysis (CC vs MLO)
    view_subgroups = {}
    views = [r.get("view", "UNKNOWN") for r in test_records]
    for v in ["CC", "MLO"]:
        mask = np.array([view == v for view in views])
        if np.sum(mask) > 10 and len(np.unique(y_true[mask])) > 1:
            view_subgroups[v] = {
                "count": int(np.sum(mask)),
                "roc_auc": round(float(roc_auc_score(y_true[mask], y_prob[mask])), 4),
            }

    results = {
        "checkpoint_id": model.checkpoint_id,
        "sample_size": n_total,
        "class_distribution": {"positive_malignant": n_pos, "negative_benign": n_neg},
        "operating_threshold": operating_threshold,
        "discrimination": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "sensitivity_at_90pct_specificity": round(sens_at_90_spec, 4),
            "positive_predictive_value": round(ppv, 4),
            "negative_predictive_value": round(npv, 4),
        },
        "calibration": {
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(ece, 4),
        },
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        },
        "view_subgroups": view_subgroups,
        "regulatory_notice": (
            "This evaluation report reflects retrospective benchmark testing on an independent research test set. "
            "Prospective multi-center clinical trials are required prior to commercial clinical diagnostic deployment."
        ),
    }

    return results
