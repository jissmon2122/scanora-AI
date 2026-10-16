"""
Scanora AI — Clinical DICOM Mammography Parser & Windowing Engine.

Handles:
1. PhotometricInterpretation: Inverts MONOCHROME1 to standard MONOCHROME2 display.
2. VOI LUT Windowing: Uses WindowCenter and WindowWidth tags if available.
3. Linear Rescale: Applies RescaleSlope and RescaleIntercept.
4. Privacy: Scrubs direct PHI while preserving technical acquisition parameters (Laterality, View, Modality, Pixel Spacing).
"""

import io
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


class DicomProcessingError(ValueError):
    """Raised when DICOM cannot be safely decoded or is clinically incompatible."""
    pass


class DicomHandler:
    @staticmethod
    def is_dicom(file_bytes: bytes) -> bool:
        """Checks for DICOM standard byte signature."""
        return len(file_bytes) >= 132 and file_bytes[128:132] == b"DICM"

    @classmethod
    def parse_dicom(cls, file_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Parses DICOM byte stream, extracts pixel data, applies VOI LUT / windowing,
        corrects photometric interpretation, and extracts non-PHI metadata.
        """
        try:
            dataset = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        except Exception as e:
            raise DicomProcessingError(f"Failed to parse DICOM header: {str(e)}")

        # Modality verification
        modality = getattr(dataset, "Modality", "UNKNOWN")
        is_mammography = modality in ["MG", "DX", "CR"]

        # Pixel array extraction
        try:
            pixel_array = dataset.pixel_array.astype(np.float32)
        except Exception as e:
            raise DicomProcessingError(f"Could not decompress DICOM pixel array: {str(e)}")

        # Rescale Slope & Intercept
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        if slope != 1.0 or intercept != 0.0:
            pixel_array = pixel_array * slope + intercept

        # Windowing (VOI LUT or Window Center / Width)
        try:
            # apply_voi_lut handles standard VOI LUT curves and linear windowing
            windowed = apply_voi_lut(pixel_array, dataset)
        except Exception:
            # Fallback to manual min-max windowing if VOI LUT fails
            windowed = pixel_array

        # Photometric Interpretation handling
        # MONOCHROME1: 0 = White, Max = Black (requires inversion for standard display and model inference)
        # MONOCHROME2: 0 = Black, Max = White (standard)
        photometric = getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")
        if photometric == "MONOCHROME1":
            windowed = np.amax(windowed) - windowed

        # Normalize pixel values to 8-bit [0, 255]
        min_val = np.amin(windowed)
        max_val = np.amax(windowed)
        if max_val > min_val:
            normalized_8bit = ((windowed - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        else:
            normalized_8bit = np.zeros_like(windowed, dtype=np.uint8)

        # Anonymized Acquisition Metadata
        metadata = {
            "modality": modality,
            "is_mammography": is_mammography,
            "photometric_interpretation": photometric,
            "rows": int(getattr(dataset, "Rows", normalized_8bit.shape[0])),
            "columns": int(getattr(dataset, "Columns", normalized_8bit.shape[1])),
            "laterality": getattr(dataset, "ImageLaterality", getattr(dataset, "Laterality", "UNKNOWN")),
            "view_position": getattr(dataset, "ViewPosition", "UNKNOWN"),
            "body_part_examined": getattr(dataset, "BodyPartExamined", "BREAST"),
            "pixel_spacing": [float(x) for x in getattr(dataset, "PixelSpacing", [1.0, 1.0])],
            "bits_allocated": int(getattr(dataset, "BitsAllocated", 16)),
            "bits_stored": int(getattr(dataset, "BitsStored", 12)),
        }

        return normalized_8bit, metadata
