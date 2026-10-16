"""
Scanora AI — Mammography Dataset Governance and Metadata Specification.

This module defines the schema, provenance, licensing, and label mappings
for standardized research mammography datasets: VinDr-Mammo and CBIS-DDSM.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DatasetSource(str, Enum):
    VINDR_MAMMO = "VinDr-Mammo"
    CBIS_DDSM = "CBIS-DDSM"
    INBREAST = "INbreast"
    CUSTOM_CLINICAL = "CustomClinical"


class ModalityType(str, Enum):
    FFDM = "Full-Field Digital Mammography"
    SFM_DIGITIZED = "Scanned Film Mammography"


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    source: DatasetSource
    modality: ModalityType
    official_url: str
    license_type: str
    access_requirements: str
    patient_count: int
    image_count: int
    views_supported: List[str]
    has_lesion_bounding_boxes: bool
    has_density_labels: bool
    has_birads_labels: bool
    has_pathology_ground_truth: bool
    limitations: List[str]


# Authoritative dataset profiles
DATASET_REGISTRY: Dict[DatasetSource, DatasetMetadata] = {
    DatasetSource.VINDR_MAMMO: DatasetMetadata(
        name="VinDr-Mammo: A Large-Scale Benchmark Dataset for Computer-Aided Diagnosis in Full-Field Digital Mammography",
        source=DatasetSource.VINDR_MAMMO,
        modality=ModalityType.FFDM,
        official_url="https://physionet.org/content/vindr-mammo/1.0.0/",
        license_type="PhysioNet Credentialed Data Use Agreement (v1.5.0)",
        access_requirements="PhysioNet credentialed access with CITI Human Subjects Research training certification.",
        patient_count=5000,
        image_count=20000,
        views_supported=["CC", "MLO"],
        has_lesion_bounding_boxes=True,
        has_density_labels=True,
        has_birads_labels=True,
        has_pathology_ground_truth=False,  # Radiology assessment consensus, not biopsy-proven for all cases
        limitations=[
            "Single-country cohort (Vietnamese population); potential demographic shift.",
            "Labels derived from expert radiologist consensus reading rather than universal histopathology.",
            "Raw DICOM files require proprietary vendor VOI LUT windowing application.",
        ],
    ),
    DatasetSource.CBIS_DDSM: DatasetMetadata(
        name="CBIS-DDSM: Curated Breast Imaging Subset of the Digital Database for Screening Mammography",
        source=DatasetSource.CBIS_DDSM,
        modality=ModalityType.SFM_DIGITIZED,
        official_url="https://www.cancerimagingarchive.net/collection/cbis-ddsm/",
        license_type="Creative Commons Attribution 3.0 Unported (CC BY 3.0)",
        access_requirements="Open access via The Cancer Imaging Archive (TCIA).",
        patient_count=1566,
        image_count=2620,
        views_supported=["CC", "MLO"],
        has_lesion_bounding_boxes=True,
        has_density_labels=True,
        has_birads_labels=True,
        has_pathology_ground_truth=True,  # Histopathology verified: BENIGN, BENIGN_WITHOUT_CALLBACK, MALIGNANT
        limitations=[
            "Digitized analog film mammography from the 1990s; noise, dust particles, and scanner artifacts.",
            "Different pixel distribution compared to modern high-resolution full-field digital mammography (FFDM).",
            "Scanner optical density variations.",
        ],
    ),
}


# Standardized Target Label Mappings
class BinaryMalignancyLabel(int, Enum):
    BENIGN_OR_NORMAL = 0
    MALIGNANT = 1


class BreastDensityACR(str, Enum):
    DENSITY_A = "A"  # Almost entirely fatty
    DENSITY_B = "B"  # Scattered areas of fibroglandular density
    DENSITY_C = "C"  # Heterogeneously dense
    DENSITY_D = "D"  # Extremely dense


class BiradsCategory(str, Enum):
    BIRADS_1 = "BI-RADS 1"  # Negative
    BIRADS_2 = "BI-RADS 2"  # Benign
    BIRADS_3 = "BI-RADS 3"  # Probably Benign (<2% risk)
    BIRADS_4 = "BI-RADS 4"  # Suspicious (2-95% risk)
    BIRADS_5 = "BI-RADS 5"  # Highly Suggestive (>95% risk)


class LesionType(str, Enum):
    MASS = "Mass"
    CALCIFICATION = "Suspicious Calcification"
    ASYMMETRY = "Asymmetry"
    ARCHITECTURAL_DISTORTION = "Architectural Distortion"
    OTHER = "Other Abnormality"


@dataclass
class NormalizedFindingAnnotation:
    lesion_type: LesionType
    xmin: float  # Normalized coordinates [0.0, 1.0]
    ymin: float
    xmax: float
    ymax: float
    confidence: Optional[float] = None
