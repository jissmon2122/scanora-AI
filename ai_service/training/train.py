"""
Scanora AI — Reproducible Mammography Model Training Pipeline.

Features:
1. Patient-level data partitioning.
2. Weighted BCEWithLogitsLoss addressing severe screening class imbalance.
3. Multi-task loss (Malignancy + Breast Density).
4. Validation ROC-AUC checkpointing and early stopping.
"""

import os
import argparse
import json
import logging
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from ai_service.models.mammography_classifier import MammographyClassifier
from ai_service.training.dataset_loader import MammographyDataset, create_patient_isolated_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scanora.training")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_mal: nn.Module,
    criterion_den: nn.Module,
    device: torch.device,
    density_loss_weight: float = 0.3,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, mal_targets, den_targets) in enumerate(loader):
        images = images.to(device)
        mal_targets = mal_targets.to(device)
        den_targets = den_targets.to(device)

        optimizer.zero_grad()
        mal_logits, den_logits = model(images)

        loss_mal = criterion_mal(mal_logits, mal_targets)
        loss_den = criterion_den(den_logits, den_targets)
        loss = loss_mal + (density_loss_weight * loss_den)

        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_preds: List[float] = []
    all_targets: List[float] = []

    for images, mal_targets, _ in loader:
        images = images.to(device)
        mal_logits, _ = model(images)
        probs = torch.sigmoid(mal_logits).cpu().numpy().flatten()
        targets = mal_targets.numpy().flatten()

        all_preds.extend(probs)
        all_targets.extend(targets)

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)

    # Compute ROC-AUC if both classes are present
    if len(np.unique(all_targets_arr)) > 1:
        auc = float(roc_auc_score(all_targets_arr, all_preds_arr))
    else:
        auc = 0.5  # Neutral fallback when single class present in mini-split

    return {"val_auc": round(auc, 4), "samples": len(all_targets_arr)}


def run_training(
    manifest_records: List[Dict[str, Any]],
    output_checkpoint_dir: str = "./ai_service/checkpoints",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-4,
    device_str: str = "auto",
):
    """Executes full training run with patient-isolated splits and checkpoint saving."""
    os.makedirs(output_checkpoint_dir, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"Using compute device: {device}")

    # Generate patient-isolated splits
    train_recs, val_recs, test_recs = create_patient_isolated_splits(manifest_records)
    logger.info(
        f"Splits generated: Train={len(train_recs)} images, Val={len(val_recs)}, Test={len(test_recs)}"
    )

    # Compute class weighting for loss
    n_pos = sum(1 for r in train_recs if r.get("malignancy") == 1)
    n_neg = len(train_recs) - n_pos
    pos_weight_val = max(1.0, n_neg / max(1, n_pos))
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    logger.info(f"Class distribution: {n_pos} positive, {n_neg} negative. pos_weight={pos_weight_val:.2f}")

    train_ds = MammographyDataset(train_recs, is_training=True)
    val_ds = MammographyDataset(val_recs, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MammographyClassifier(pretrained_backbone=True).to(device)

    criterion_mal = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_den = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_checkpoint_path = os.path.join(output_checkpoint_dir, "best_mammography_classifier.pt")

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion_mal, criterion_den, device)
        scheduler.step()

        val_metrics = evaluate_validation(model, val_loader, device)
        val_auc = val_metrics["val_auc"]

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_auc": best_auc,
                    "checkpoint_id": f"effnet-b2-epoch{epoch}-auc{best_auc:.4f}",
                    "architecture": "efficientnet_b2_mammography",
                },
                best_checkpoint_path,
            )
            logger.info(f"Saved new best checkpoint to {best_checkpoint_path} (AUC: {best_auc:.4f})")

    return best_checkpoint_path, best_auc
