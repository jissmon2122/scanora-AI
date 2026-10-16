"""
Scanora AI — Independent Evaluation & Clinical Metrics Assessment.

Evaluates trained model checkpoints on unseen patient-level test sets:
- ROC-AUC & PR-AUC
- Sensitivity, Specificity, PPV, NPV
- False Negative Rate (FNR)
- Brier Score & Expected Calibration Error (ECE)
- Confusion Matrix (TP, FP, TN, FN)
- 95% Percentile Bootstrap Confidence Interval
- Subgroup metrics (by view: CC vs MLO; by breast density: A, B, C, D)
- Patient-level aggregation (max-pooled probability per patient)
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple
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


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Computes clinical discrimination and calibration metrics from ground truth and predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
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
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0  # False Negative Rate
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    sens_at_90_spec = 0.0
    if n_pos > 0 and n_neg > 0:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        valid_indices = np.where(fpr <= 0.10)[0]
        if len(valid_indices) > 0:
            sens_at_90_spec = float(tpr[valid_indices[-1]])

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "false_negative_rate_fnr": round(fnr, 4),
        "sensitivity_at_90pct_specificity": round(sens_at_90_spec, 4),
        "positive_predictive_value": round(ppv, 4),
        "negative_predictive_value": round(npv, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error_ece": round(ece, 4),
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        },
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes a 95% percentile bootstrap confidence interval for ROC-AUC."""
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    bootstrap_aucs: List[float] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[boot_idx])) > 1:
            boot_auc = float(roc_auc_score(y_true[boot_idx], y_prob[boot_idx]))
            bootstrap_aucs.append(boot_auc)

    if bootstrap_aucs:
        ci_lower = round(float(np.percentile(bootstrap_aucs, 2.5)), 4)
        ci_upper = round(float(np.percentile(bootstrap_aucs, 97.5)), 4)
    else:
        fallback = round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5
        ci_lower, ci_upper = fallback, fallback

    return ci_lower, ci_upper


def aggregate_patient_predictions(
    predictions: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Max-pools malignancy probability across views per patient.
    Returns:
        (patient_true_labels, patient_max_probabilities)
    """
    patient_max_prob: Dict[str, float] = {}
    patient_true_label: Dict[str, int] = {}

    for item in predictions:
        pid = item["patient_id"]
        prob = float(item["calibrated_probability"])
        target = int(item["y_true"])
        patient_max_prob[pid] = max(patient_max_prob.get(pid, 0.0), prob)
        patient_true_label[pid] = max(patient_true_label.get(pid, 0), target)

    p_true = np.array(list(patient_true_label.values()))
    p_prob = np.array([patient_max_prob[pid] for pid in patient_true_label.keys()])
    return p_true, p_prob


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

    metrics = compute_metrics(y_true, y_prob, threshold=operating_threshold)
    ci_lower, ci_upper = compute_bootstrap_ci(y_true, y_prob)

    # Subgroups
    view_subgroups = {}
    views = [r.get("view", "UNKNOWN") for r in test_records]
    for v in ["CC", "MLO"]:
        mask = np.array([view == v for view in views])
        if np.sum(mask) >= 5 and len(np.unique(y_true[mask])) > 1:
            view_subgroups[v] = {
                "count": int(np.sum(mask)),
                "roc_auc": round(float(roc_auc_score(y_true[mask], y_prob[mask])), 4),
            }

    density_subgroups = {}
    densities = [r.get("density", "UNKNOWN") for r in test_records]
    for d in ["A", "B", "C", "D"]:
        mask = np.array([density == d for density in densities])
        if np.sum(mask) >= 5 and len(np.unique(y_true[mask])) > 1:
            density_subgroups[f"Density_{d}"] = {
                "count": int(np.sum(mask)),
                "roc_auc": round(float(roc_auc_score(y_true[mask], y_prob[mask])), 4),
            }

    # Patient-level aggregation
    prediction_items = [
        {
            "patient_id": r.get("patient_id", f"pt_{i}"),
            "y_true": target,
            "calibrated_probability": prob,
        }
        for i, (r, target, prob) in enumerate(zip(test_records, y_true, y_prob))
    ]
    p_true, p_prob = aggregate_patient_predictions(prediction_items)
    patient_roc_auc = (
        round(float(roc_auc_score(p_true, p_prob)), 4) if len(np.unique(p_true)) > 1 else metrics["roc_auc"]
    )

    results = {
        "checkpoint_id": model.checkpoint_id,
        "sample_size_images": len(y_true),
        "sample_size_patients": len(p_true),
        "class_distribution": {
            "positive_malignant": int(np.sum(y_true)),
            "negative_benign": int(len(y_true) - np.sum(y_true)),
        },
        "operating_threshold": operating_threshold,
        "empirical_calculated_metrics": {
            "image_level_roc_auc": metrics["roc_auc"],
            "image_level_roc_auc_95ci": [ci_lower, ci_upper],
            "patient_level_roc_auc": patient_roc_auc,
            "pr_auc": metrics["pr_auc"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "false_negative_rate": metrics["false_negative_rate_fnr"],
            "sensitivity_at_90pct_specificity": metrics["sensitivity_at_90pct_specificity"],
            "positive_predictive_value": metrics["positive_predictive_value"],
            "negative_predictive_value": metrics["negative_predictive_value"],
        },
        "calibration": {
            "brier_score": metrics["brier_score"],
            "expected_calibration_error": metrics["expected_calibration_error_ece"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "subgroup_analysis": {
            "views": view_subgroups,
            "density": density_subgroups,
        },
        "target_clinical_benchmarks": {
            "target_roc_auc": ">= 0.8500",
            "target_ece": "<= 0.0500",
            "target_sensitivity_at_90pct_spec": ">= 0.7000",
            "benchmark_status": "UNVALIDATED_TARGETS (Requires physical test set evaluation)",
        },
        "regulatory_notice": (
            "This evaluation report reflects calculated metrics on held-out retrospective test data. "
            "Internal benchmark evaluation does not constitute clinical certification or autonomous diagnostic clearance."
        ),
    }

    return results
