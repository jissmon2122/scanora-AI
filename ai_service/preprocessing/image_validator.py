"""
Scanora AI — Secure Image Ingestion and File Signature Validation.

Validates:
1. Actual file magic bytes (signatures) to prevent extension spoofing.
2. Image size, dimension limits, and decompression bomb protection.
3. Safe rejection of empty, corrupted, or unsupported formats.
"""

import io
from enum import Enum
from typing import Tuple, Dict, Any, Optional
from PIL import Image


class SupportedImageFormat(str, Enum):
    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"
    DICOM = "DICOM"
    UNSUPPORTED = "UNSUPPORTED"


# File signatures (magic bytes)
MAGIC_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": SupportedImageFormat.PNG,
    b"\xff\xd8\xff": SupportedImageFormat.JPEG,
    b"RIFF": SupportedImageFormat.WEBP,  # Needs 'WEBP' at offset 8
}

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit
MAX_IMAGE_PIXELS = 100_000_000  # 100 Megapixel limit to prevent decompression bombs


class ImageValidationError(ValueError):
    """Raised when an uploaded file violates safety or clinical integrity requirements."""
    pass


class ImageValidator:
    @staticmethod
    def detect_format(file_bytes: bytes) -> SupportedImageFormat:
        """Determines the true image format from header magic bytes rather than file extensions."""
        if len(file_bytes) < 132:
            return SupportedImageFormat.UNSUPPORTED

        # Check standard 2D image signatures
        for magic, fmt in MAGIC_SIGNATURES.items():
            if file_bytes.startswith(magic):
                if fmt == SupportedImageFormat.WEBP:
                    if file_bytes[8:12] == b"WEBP":
                        return SupportedImageFormat.WEBP
                else:
                    return fmt

        # Check DICOM: standard DICOM has 'DICM' at byte offset 128
        if len(file_bytes) >= 132 and file_bytes[128:132] == b"DICM":
            return SupportedImageFormat.DICOM

        return SupportedImageFormat.UNSUPPORTED

    @classmethod
    def validate_and_load_raster(
        cls, file_bytes: bytes
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Validates buffer size, format, and loads standard raster image safely.
        Enforces decompression limits.
        """
        if not file_bytes:
            raise ImageValidationError("File is empty (0 bytes).")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise ImageValidationError(
                f"File size exceeds maximum allowable limit of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
            )

        fmt = cls.detect_format(file_bytes)
        if fmt == SupportedImageFormat.UNSUPPORTED:
            raise ImageValidationError(
                "Unsupported file format. Scanora accepts valid PNG, JPEG, WebP, or DICOM files."
            )

        if fmt == SupportedImageFormat.DICOM:
            # Delegate to specialized DICOM handler
            return None, {"format": "DICOM", "raw_bytes_len": len(file_bytes)}

        # Set Pillow decompression bomb protection
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()  # Verify structural integrity
            # Reopen because verify() closes the file descriptor
            image = Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            raise ImageValidationError(f"Corrupted or malformed image data: {str(e)}")

        width, height = image.size
        if width < 64 or height < 64:
            raise ImageValidationError(
                f"Image dimensions ({width}x{height}) are too small for clinical diagnostic analysis."
            )

        metadata = {
            "format": fmt.value,
            "width": width,
            "height": height,
            "mode": image.mode,
            "file_size_bytes": len(file_bytes),
        }

        return image, metadata
