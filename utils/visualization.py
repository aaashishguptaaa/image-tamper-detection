"""
Visualization Utilities
------------------------
Generates heatmaps and overlay visualizations
for forgery localization results.
"""

import cv2
import numpy as np
import base64
import os
from io import BytesIO
from PIL import Image


def generate_forgery_heatmap(image_path: str,
                              ela_array: np.ndarray,
                              sift_keypoints: list = None,
                              alpha: float = 0.6) -> np.ndarray:
    """
    Overlay ELA heatmap and SIFT keypoints onto original image.

    Args:
        image_path:     Path to original image
        ela_array:      ELA result as RGB numpy array
        sift_keypoints: List of (x,y) tuples from SIFT
        alpha:          Blend factor for overlay

    Returns:
        overlay: BGR numpy array
    """
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = original.shape[:2]

    # Resize ELA to match original
    ela_bgr = cv2.cvtColor(ela_array, cv2.COLOR_RGB2BGR)
    ela_resized = cv2.resize(ela_bgr, (w, h))

    # Convert ELA to gray then heatmap
    ela_gray = cv2.cvtColor(ela_resized, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)

    # Blend original + heatmap
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

    # Draw SIFT keypoints as circles
    if sift_keypoints:
        for (x, y) in sift_keypoints:
            cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)
            cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)

    return overlay


def add_forgery_border(image: np.ndarray, is_fake: bool) -> np.ndarray:
    """Add colored border: RED if fake, GREEN if real."""
    color = (0, 0, 255) if is_fake else (0, 255, 0)
    bordered = cv2.copyMakeBorder(image, 8, 8, 8, 8,
                                   cv2.BORDER_CONSTANT, value=color)
    return bordered


def ndarray_to_base64(image_array: np.ndarray, fmt: str = "PNG") -> str:
    """Convert numpy array (BGR) to base64 string for HTML embedding."""
    rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def image_path_to_base64(image_path: str) -> str:
    """Convert image file to base64 for HTML embedding."""
    img = cv2.imread(image_path)
    return ndarray_to_base64(img)


def save_result_image(image_array: np.ndarray,
                       output_dir: str,
                       filename: str) -> str:
    """Save result image and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, image_array)
    return out_path


def create_side_by_side(original_path: str,
                         heatmap: np.ndarray,
                         width: int = 400) -> np.ndarray:
    """
    Create a side-by-side comparison: original | heatmap overlay.
    """
    original = cv2.imread(original_path)
    h, w = original.shape[:2]
    new_h = int(h * width / w)

    orig_resized = cv2.resize(original, (width, new_h))
    heat_resized = cv2.resize(heatmap, (width, new_h))

    combined = np.hstack([orig_resized, heat_resized])
    return combined
