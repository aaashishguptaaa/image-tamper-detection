"""
Noise Pattern Analysis Module
-------------------------------
Every camera sensor produces a unique noise pattern (PRNU).
When an image is spliced from different sources, the noise
patterns in different regions won't match — revealing forgery.

Also detects:
  - Gaussian blur inconsistency (blurred regions = hidden edits)
  - Sharpness inconsistency across regions
"""

import cv2
import numpy as np


def analyze_noise_inconsistency(image_path: str,
                                 block_size: int = 64) -> dict:
    """
    Detect noise pattern inconsistency across image regions.
    Consistent noise = likely real. Inconsistent = tampered.

    Args:
        image_path: Path to input image
        block_size: Size of analysis blocks in pixels

    Returns:
        dict with:
            score          (float 0-1)
            noise_map      (np.ndarray) visual noise map
            inconsistency  (float) standard deviation of noise levels
            suspicious_blocks (list of (x,y,w,h))
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # ── Estimate noise using high-pass filter ─────────────────────
    # Subtract blurred version → residual = noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = np.abs(gray - blurred)

    # ── Compute noise level per block ─────────────────────────────
    noise_levels = []
    block_coords = []

    for y in range(0, h - block_size, block_size // 2):
        for x in range(0, w - block_size, block_size // 2):
            block = noise_residual[y:y+block_size, x:x+block_size]
            if block.shape == (block_size, block_size):
                noise_levels.append(float(np.std(block)))
                block_coords.append((x, y, block_size, block_size))

    if not noise_levels:
        return {"score": 0.0, "noise_map": image, "inconsistency": 0.0,
                "suspicious_blocks": []}

    noise_array = np.array(noise_levels)
    mean_noise  = np.mean(noise_array)
    std_noise   = np.std(noise_array)
    inconsistency = float(std_noise / (mean_noise + 1e-6))

    # ── Find suspicious blocks (outliers) ─────────────────────────
    threshold = mean_noise + 2.0 * std_noise
    suspicious = []
    for i, (nl, coords) in enumerate(zip(noise_levels, block_coords)):
        if nl > threshold or nl < mean_noise - 2.0 * std_noise:
            suspicious.append(coords)

    # ── Score: higher inconsistency = more likely tampered ────────
    score = float(np.clip(inconsistency * 0.5, 0.0, 1.0))

    # ── Build noise visualization map ─────────────────────────────
    noise_map = _create_noise_visualization(image, noise_residual,
                                             suspicious)

    return {
        "score":               round(score, 4),
        "noise_map":           noise_map,
        "inconsistency":       round(inconsistency, 4),
        "suspicious_blocks":   suspicious[:20],
        "mean_noise_level":    round(mean_noise, 4),
        "std_noise_level":     round(std_noise, 4)
    }


def _create_noise_visualization(original: np.ndarray,
                                  noise_residual: np.ndarray,
                                  suspicious_blocks: list) -> np.ndarray:
    """Draw noise map with suspicious regions highlighted."""
    # Normalize noise to 0-255
    noise_norm = cv2.normalize(noise_residual, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    noise_colored = cv2.applyColorMap(noise_norm, cv2.COLORMAP_HOT)

    # Blend with original
    result = cv2.addWeighted(original, 0.4, noise_colored, 0.6, 0)

    # Draw suspicious block borders
    for (x, y, bw, bh) in suspicious_blocks:
        cv2.rectangle(result, (x, y), (x+bw, y+bh), (0, 0, 255), 2)

    return result


def detect_blur_inconsistency(image_path: str,
                               num_regions: int = 9) -> dict:
    """
    Detect sharpness inconsistency across image regions.
    A real photo should have consistent focus/blur.
    Artificially blurred regions may hide edited content.

    Returns:
        dict with:
            score          (float 0-1)
            region_scores  (list of float) sharpness per region
            inconsistency  (float)
            blur_map       (np.ndarray)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Divide into grid
    grid = int(np.sqrt(num_regions))
    rh, rw = h // grid, w // grid

    region_scores = []
    region_coords = []

    for i in range(grid):
        for j in range(grid):
            y1, y2 = i * rh, (i+1) * rh
            x1, x2 = j * rw, (j+1) * rw
            region = gray[y1:y2, x1:x2]

            # Laplacian variance = sharpness score
            sharpness = float(cv2.Laplacian(region, cv2.CV_64F).var())
            region_scores.append(sharpness)
            region_coords.append((x1, y1, x2-x1, y2-y1))

    arr  = np.array(region_scores)
    mean = np.mean(arr)
    std  = np.std(arr)
    cv_  = float(std / (mean + 1e-6))  # Coefficient of variation

    score = float(np.clip(cv_ * 0.3, 0.0, 1.0))

    # Build blur visualization
    blur_map = _create_blur_map(image, region_scores, region_coords)

    return {
        "score":          round(score, 4),
        "region_scores":  [round(s, 2) for s in region_scores],
        "inconsistency":  round(cv_, 4),
        "blur_map":       blur_map
    }


def _create_blur_map(image: np.ndarray,
                      scores: list,
                      coords: list) -> np.ndarray:
    """Color-code regions by sharpness level."""
    result = image.copy()
    if not scores:
        return result

    max_s = max(scores) or 1
    for score, (x, y, bw, bh) in zip(scores, coords):
        # Normalize: green = sharp, red = blurry
        ratio = score / max_s
        color = (int((1-ratio)*255), int(ratio*255), 0)
        overlay = result.copy()
        cv2.rectangle(overlay, (x, y), (x+bw, y+bh), color, -1)
        result = cv2.addWeighted(result, 0.75, overlay, 0.25, 0)
        cv2.rectangle(result, (x, y), (x+bw, y+bh), color, 2)

    return result
