"""
Scanora AI — Clinical Mammography Preprocessing Pipeline.

Provides:
1. Aspect-ratio preserving padding to square dimensions.
2. Standardized intensity normalization.
3. Coordinate transformation mapping (model tensor space <-> original image space).
4. Direct conversion to PyTorch tensors.
"""

from typing import Tuple, Dict, Any, Union
import numpy as np
from PIL import Image
import torch


class PreprocessingResult:
    def __init__(
        self,
        tensor: torch.Tensor,
        original_width: int,
        original_height: int,
        padded_size: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ):
        self.tensor = tensor  # Shape: [1, 1, target_size, target_size]
        self.original_width = original_width
        self.original_height = original_height
        self.padded_size = padded_size
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y

    def map_box_to_original(
        self, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> Dict[str, float]:
        """
        Maps a bounding box in normalized model space [0, 1] back to
        the original unpadded image coordinate space [0, 1] (or pixels).
        """
        # Convert from normalized model space to pixel space of padded image
        padded_xmin = xmin * self.padded_size
        padded_ymin = ymin * self.padded_size
        padded_xmax = xmax * self.padded_size
        padded_ymax = ymax * self.padded_size

        # Remove padding
        unpad_xmin = (padded_xmin - self.pad_x) / self.scale
        unpad_ymin = (padded_ymin - self.pad_y) / self.scale
        unpad_xmax = (padded_xmax - self.pad_x) / self.scale
        unpad_ymax = (padded_ymax - self.pad_y) / self.scale

        # Clamp to original image bounds
        orig_xmin = max(0.0, min(float(self.original_width), unpad_xmin))
        orig_ymin = max(0.0, min(float(self.original_height), unpad_ymin))
        orig_xmax = max(0.0, min(float(self.original_width), unpad_xmax))
        orig_ymax = max(0.0, min(float(self.original_height), unpad_ymax))

        # Normalized to original image percentages [0.0 - 100.0] for frontend UI
        norm_x = (orig_xmin / self.original_width) * 100.0
        norm_y = (orig_ymin / self.original_height) * 100.0
        norm_width = ((orig_xmax - orig_xmin) / self.original_width) * 100.0
        norm_height = ((orig_ymax - orig_ymin) / self.original_height) * 100.0

        return {
            "x": round(norm_x, 2),
            "y": round(norm_y, 2),
            "width": round(norm_width, 2),
            "height": round(norm_height, 2),
            "raw_pixels": {
                "xmin": round(orig_xmin, 1),
                "ymin": round(orig_ymin, 1),
                "xmax": round(orig_xmax, 1),
                "ymax": round(orig_ymax, 1),
            },
        }


class MammographyPreprocessor:
    def __init__(self, target_size: int = 512, mean: float = 0.28, std: float = 0.22):
        self.target_size = target_size
        # Mean and standard deviation computed over VinDr-Mammo FFDM training split
        self.mean = mean
        self.std = std

    def preprocess(
        self, input_image: Union[Image.Image, np.ndarray]
    ) -> PreprocessingResult:
        """
        Converts grayscale image into standardized PyTorch tensor with aspect-ratio preserving padding.
        """
        if isinstance(input_image, Image.Image):
            # Ensure single-channel grayscale
            if input_image.mode != "L":
                input_image = input_image.convert("L")
            img_array = np.array(input_image, dtype=np.float32)
        elif isinstance(input_image, np.ndarray):
            img_array = input_image.astype(np.float32)
            if img_array.ndim == 3 and img_array.shape[2] == 3:
                # RGB to Grayscale via standard luminance weights
                img_array = (
                    0.2989 * img_array[:, :, 0]
                    + 0.5870 * img_array[:, :, 1]
                    + 0.1140 * img_array[:, :, 2]
                )
        else:
            raise TypeError("Unsupported image type. Expected PIL.Image or np.ndarray.")

        orig_h, orig_w = img_array.shape[:2]

        # Calculate aspect-ratio scale
        scale = self.target_size / max(orig_h, orig_w)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # Resize using bilinear interpolation via PIL
        pil_resized = Image.fromarray(img_array.astype(np.uint8)).resize(
            (new_w, new_h), resample=Image.Resampling.BILINEAR
        )
        resized_array = np.array(pil_resized, dtype=np.float32)

        # Pad to square (target_size x target_size)
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2

        padded = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_array

        # Scale intensity to [0.0, 1.0]
        normalized = padded / 255.0

        # Standardize using mammography training statistics
        standardized = (normalized - self.mean) / self.std

        # Convert to PyTorch Tensor: [1, 1, H, W]
        tensor = torch.from_numpy(standardized).unsqueeze(0).unsqueeze(0)

        return PreprocessingResult(
            tensor=tensor,
            original_width=orig_w,
            original_height=orig_h,
            padded_size=self.target_size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
