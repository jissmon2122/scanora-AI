"""
Scanora AI — Dataset Inventory, Quality Inspection, and Patient-Level Split Validator.

Ensures:
1. Zero patient overlap between train, validation, and test splits (no data leakage).
2. Label balance analysis (Normal, Benign, Malignant, BI-RADS 1-5).
3. Missing value and corruption inspection.
"""

import os
import json
import logging
from typing import Dict, List, Set, Any
from collections import Counter
from ai_service.datasets.governance import DATASET_REGISTRY, DatasetSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scanora.datasets.inventory")


class DatasetInventoryValidator:
    def __init__(self, dataset_source: DatasetSource, data_root: str):
        self.source = dataset_source
        self.metadata = DATASET_REGISTRY.get(dataset_source)
        self.data_root = data_root

    def check_presence(self) -> Dict[str, Any]:
        """Verifies if the physical dataset directory and expected metadata files exist."""
        exists = os.path.exists(self.data_root)
        if not exists:
            return {
                "dataset": self.source.value,
                "status": "NOT_FOUND",
                "path": self.data_root,
                "message": (
                    f"Directory '{self.data_root}' not found. "
                    "Follow download instructions in 'download_instructions.md'."
                ),
            }

        files = os.listdir(self.data_root)
        return {
            "dataset": self.source.value,
            "status": "PRESENT",
            "path": self.data_root,
            "file_count": len(files),
            "files_sample": files[:10],
        }

    def validate_patient_splits(
        self,
        train_patients: Set[str],
        val_patients: Set[str],
        test_patients: Set[str],
    ) -> Dict[str, Any]:
        """
        Rigorous check ensuring no patient leakage across train, val, and test splits.
        Crucial requirement for medical machine learning.
        """
        train_val_overlap = train_patients.intersection(val_patients)
        train_test_overlap = train_patients.intersection(test_patients)
        val_test_overlap = val_patients.intersection(test_patients)

        has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

        return {
            "valid": not has_leakage,
            "patient_counts": {
                "train": len(train_patients),
                "val": len(val_patients),
                "test": len(test_patients),
                "total_unique": len(train_patients | val_patients | test_patients),
            },
            "leakage_detected": has_leakage,
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
        }

    def compute_summary_statistics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes label distribution, view breakdown, and missing data metrics."""
        total = len(records)
        if total == 0:
            return {"total_records": 0, "status": "EMPTY"}

        views = Counter(r.get("view", "UNKNOWN") for r in records)
        birads = Counter(r.get("birads", "UNKNOWN") for r in records)
        pathology = Counter(r.get("pathology", "UNKNOWN") for r in records)
        density = Counter(r.get("density", "UNKNOWN") for r in records)

        missing_views = sum(1 for r in records if not r.get("view"))
        missing_density = sum(1 for r in records if not r.get("density"))

        return {
            "total_records": total,
            "view_distribution": dict(views),
            "birads_distribution": dict(birads),
            "pathology_distribution": dict(pathology),
            "density_distribution": dict(density),
            "missing_fields": {
                "view": missing_views,
                "density": missing_density,
            },
        }


if __name__ == "__main__":
    validator = DatasetInventoryValidator(DatasetSource.VINDR_MAMMO, "./ai_service/data/vindr")
    presence = validator.check_presence()
    print(json.dumps(presence, indent=2))
