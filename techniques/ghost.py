"""
JPEG Ghost Detection Module
-----------------------------
JPEG Ghost is a powerful splicing detection technique.
When a region from another JPEG image is pasted into a
target image and re-saved, it was compressed TWICE.

By re-saving at multiple quality levels and looking at
the difference energy, double-compressed (ghost) regions
stand out clearly against singly-compressed regions.
"""

import cv2
import numpy as np
from PIL import Image
import io


def detect_jpeg_ghost(image_path: str,
                       quality_range: range = range(50, 100, 10),
                       threshold_ratio: float = 0.15) -> dict:
    """
    Detect JPEG double compression (splicing from another JPEG).

    Args:
        image_path:      Path to input image
        quality_range:   JPEG quality levels to test
        threshold_ratio: Sensitivity threshold

    Returns:
        dict with:
            score         (float 0-1)
            ghost_map     (np.ndarray BGR) visualization
            best_quality  (int) quality level that shows most ghosts
            ghost_ratio   (float) fraction of pixels flagged as ghost
    """
    original = Image.open(image_path).convert("RGB")
    orig_array = np.array(original, dtype=np.float32)

    best_ghost_ratio = 0.0
    best_quality     = 70
    best_ghost_map   = None

    for quality in quality_range:
        # Re-save at this quality
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")
        resaved_array = np.array(resaved, dtype=np.float32)

        # Compute squared difference
        diff = np.mean((orig_array - resaved_array) ** 2, axis=2)

        # Normalize
        diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        # Ghost pixels = unusually LOW error (already compressed at this Q)
        ghost_mask  = diff_norm < threshold_ratio
        ghost_ratio = float(ghost_mask.mean())

        if ghost_ratio > best_ghost_ratio and ghost_ratio < 0.95:
            best_ghost_ratio = ghost_ratio
            best_quality     = quality
            best_ghost_map   = _create_ghost_visualization(
                orig_array, diff_norm, ghost_mask
            )

    # Score: moderate ghost ratio = likely spliced
    # Very low or very high ghost ratio = uniform image or poor detection
    if 0.05 < best_ghost_ratio < 0.80:
        score = float(np.clip(best_ghost_ratio * 1.5, 0.0, 1.0))
    else:
        score = 0.1

    if best_ghost_map is None:
        img_bgr = cv2.imread(image_path)
        best_ghost_map = img_bgr if img_bgr is not None else np.zeros((100,100,3), np.uint8)

    return {
        "score":        round(score, 4),
        "ghost_map":    best_ghost_map,
        "best_quality": best_quality,
        "ghost_ratio":  round(best_ghost_ratio, 4)
    }


def _create_ghost_visualization(original: np.ndarray,
                                  diff_norm: np.ndarray,
                                  ghost_mask: np.ndarray) -> np.ndarray:
    """
    Create a visualization of the JPEG ghost map.
    Ghost regions shown in cyan, normal in dark.
    """
    orig_bgr = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Apply hot colormap to difference
    diff_vis = (diff_norm * 255).astype(np.uint8)
    diff_colored = cv2.applyColorMap(diff_vis, cv2.COLORMAP_VIRIDIS)

    # Overlay ghost mask in bright cyan
    overlay = orig_bgr.copy()
    overlay[ghost_mask] = [255, 255, 0]  # Cyan for ghost regions

    # Blend
    result = cv2.addWeighted(diff_colored, 0.5, overlay, 0.5, 0)
    return result
