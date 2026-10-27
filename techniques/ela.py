"""
ELA (Error Level Analysis) Module
----------------------------------
ELA reveals JPEG compression inconsistencies.
Re-saves the image at a known quality and computes
the difference — tampered regions show higher error levels.
"""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import os


def perform_ela(image_path: str, quality: int = 90, scale: int = 15) -> np.ndarray:
    """
    Perform Error Level Analysis on an image.

    Args:
        image_path: Path to the input image
        quality:    JPEG re-save quality (default 90)
        scale:      Amplification factor for difference visibility

    Returns:
        ela_array: ELA result as NumPy array (RGB)
    """
    # Open original image
    original = Image.open(image_path).convert("RGB")

    # Re-save at specified quality into memory buffer
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Load re-saved image
    resaved = Image.open(buffer).convert("RGB")

    # Compute absolute difference
    ela_image = ImageChops.difference(original, resaved)

    # Scale the difference for better visibility
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale * 255.0 / max_diff)

    ela_array = np.array(ela_image)
    return ela_array


def get_ela_heatmap(image_path: str, quality: int = 90) -> np.ndarray:
    """
    Generate a heatmap from ELA result.

    Returns:
        heatmap: BGR heatmap image as NumPy array
    """
    ela = perform_ela(image_path, quality=quality)

    # Convert to grayscale
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)

    # Apply colormap
    heatmap = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
    return heatmap


def compute_ela_score(image_path: str, quality: int = 90) -> float:
    """
    Compute a forgery probability score based on ELA.
    Higher score = more likely tampered.

    Returns:
        score: float between 0.0 and 1.0
    """
    ela = perform_ela(image_path, quality=quality)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)

    mean_val = np.mean(ela_gray)
    std_val  = np.std(ela_gray)

    # Normalize: higher mean error → more suspicious
    score = float(np.clip((mean_val + std_val) / 255.0, 0.0, 1.0))
    return round(score, 4)
