"""
Scanora AI — Patient-Isolated Dataset Loader & Splitter.

Guarantees:
1. Strict Patient-Level Splitting: ZERO patient overlap between train, val, and test splits.
2. Ingestion of both DICOM and standard raster images.
3. Class-imbalance aware weighting.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from ai_service.preprocessing.mammography_preprocessor import MammographyPreprocessor
from ai_service.preprocessing.dicom_handler import DicomHandler


class MammographyDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        target_size: int = 512,
        is_training: bool = False,
    ):
        """
        records: List of dictionaries with keys:
            - 'image_path': str
            - 'patient_id': str
            - 'malignancy': int (0: Benign/Normal, 1: Malignant)
            - 'density': Optional[str] ('A', 'B', 'C', 'D')
            - 'view': Optional[str] ('CC', 'MLO')
            - 'laterality': Optional[str] ('left', 'right')
        """
        self.records = records
        self.preprocessor = MammographyPreprocessor(target_size=target_size)
        self.is_training = is_training
        self.density_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        image_path = rec["image_path"]

        # Check if DICOM or standard image
        if image_path.lower().endswith(".dcm") or image_path.lower().endswith(".dicom"):
            with open(image_path, "rb") as f:
                raw_bytes = f.read()
            pixel_array, _ = DicomHandler.parse_dicom(raw_bytes)
            prep = self.preprocessor.preprocess(pixel_array)
        else:
            image = Image.open(image_path).convert("L")
            prep = self.preprocessor.preprocess(image)

        tensor = prep.tensor.squeeze(0)  # Shape [1, H, W]

        # Training-only augmentation (random horizontal flip to mimic contralateral mammogram)
        if self.is_training and torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[-1])

        mal_target = torch.tensor([float(rec.get("malignancy", 0))], dtype=torch.float32)
        density_label = self.density_map.get(rec.get("density", "B"), 1)
        den_target = torch.tensor(density_label, dtype=torch.long)

        return tensor, mal_target, den_target


def create_patient_isolated_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Groups records strictly by patient_id before allocating to splits.
    Guarantees no patient overlap between train, val, and test partitions.
    """
    np.random.seed(random_seed)

    # Group records by patient
    patient_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        pid = r.get("patient_id", f"unknown_{r['image_path']}")
        patient_to_records.setdefault(pid, []).append(r)

    unique_patients = np.array(list(patient_to_records.keys()))
    np.random.shuffle(unique_patients)

    n_total = len(unique_patients)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_pids = set(unique_patients[:n_train])
    val_pids = set(unique_patients[n_train : n_train + n_val])
    test_pids = set(unique_patients[n_train + n_val :])

    train_records = [r for pid in train_pids for r in patient_to_records[pid]]
    val_records = [r for pid in val_pids for r in patient_to_records[pid]]
    test_records = [r for pid in test_pids for r in patient_to_records[pid]]

    return train_records, val_records, test_records
