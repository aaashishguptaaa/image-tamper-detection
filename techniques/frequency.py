"""
Frequency Domain Analysis Module
----------------------------------
AI-generated and GAN-synthesized images leave artifacts
in the frequency domain that are invisible to the human eye
but detectable via Fourier Transform analysis.

Also detects:
  - DCT coefficient anomalies (JPEG manipulation)
  - Unnatural frequency distributions
"""

import cv2
import numpy as np


def analyze_frequency_domain(image_path: str) -> dict:
    """
    Perform FFT-based frequency analysis for AI/GAN artifact detection.

    Real camera images have a characteristic frequency falloff.
    AI-generated images often show grid artifacts or unusual peaks
    in the frequency spectrum.

    Returns:
        dict with:
            score       (float 0-1) probability of being AI/manipulated
            fft_image   (np.ndarray BGR) FFT spectrum visualization
            grid_score  (float) strength of grid artifacts
            flags       (list of str)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── Compute 2D FFT ────────────────────────────────────────────
    fft    = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)

    # ── Detect grid artifacts (GAN fingerprint) ────────────────────
    # GAN images often show periodic patterns at 8x8 or 16x16 frequencies
    h, w = gray.shape
    cx, cy = w // 2, h // 2

    grid_score = _detect_grid_artifacts(log_magnitude, cx, cy)

    # ── Analyze frequency falloff ──────────────────────────────────
    falloff_score = _analyze_frequency_falloff(magnitude, cx, cy)

    # ── DCT Block artifact analysis ────────────────────────────────
    dct_score = _analyze_dct_artifacts(gray)

    # ── Combine scores ─────────────────────────────────────────────
    combined = 0.4 * grid_score + 0.3 * falloff_score + 0.3 * dct_score
    score = float(np.clip(combined, 0.0, 1.0))

    flags = []
    if grid_score > 0.5:
        flags.append("Grid artifacts detected in frequency domain (possible AI/GAN generation)")
    if dct_score > 0.5:
        flags.append("DCT block artifacts detected (possible double JPEG compression)")
    if falloff_score > 0.5:
        flags.append("Unusual frequency distribution detected")

    # ── Create FFT visualization ───────────────────────────────────
    fft_vis = _create_fft_visualization(log_magnitude)

    return {
        "score":        round(score, 4),
        "fft_image":    fft_vis,
        "grid_score":   round(grid_score, 4),
        "dct_score":    round(dct_score, 4),
        "falloff_score": round(falloff_score, 4),
        "flags":        flags
    }


def _detect_grid_artifacts(log_magnitude: np.ndarray,
                             cx: int, cy: int) -> float:
    """
    Look for periodic peaks in FFT that indicate GAN/AI artifacts.
    These appear as bright spots in a grid pattern.
    """
    h, w = log_magnitude.shape

    # Sample along horizontal and vertical axes
    h_profile = log_magnitude[cy, :]
    v_profile = log_magnitude[:, cx]

    # Look for periodicity using autocorrelation
    h_ac = np.correlate(h_profile - h_profile.mean(),
                         h_profile - h_profile.mean(), mode='full')
    v_ac = np.correlate(v_profile - v_profile.mean(),
                         v_profile - v_profile.mean(), mode='full')

    # Peaks in autocorrelation (excluding center) indicate grid patterns
    h_mid = len(h_ac) // 2
    h_side = np.max(np.abs(h_ac[h_mid+8:h_mid+64])) / (np.abs(h_ac[h_mid]) + 1e-8)
    v_side = np.max(np.abs(v_ac[h_mid+8:h_mid+64])) / (np.abs(v_ac[h_mid]) + 1e-8)

    grid_strength = float(max(h_side, v_side))
    return float(np.clip(grid_strength * 0.3, 0.0, 1.0))


def _analyze_frequency_falloff(magnitude: np.ndarray,
                                 cx: int, cy: int) -> float:
    """
    Natural images have a 1/f frequency falloff.
    AI images often deviate from this expected pattern.
    """
    h, w = magnitude.shape
    max_radius = min(cx, cy)

    radial_means = []
    for r in range(1, max_radius, 2):
        y_coords, x_coords = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (x_coords**2 + y_coords**2 <= r**2) & \
               (x_coords**2 + y_coords**2 > (r-2)**2)
        if mask.sum() > 0:
            radial_means.append(float(magnitude[mask].mean()))

    if len(radial_means) < 5:
        return 0.1

    # Expected: exponential decay. Check deviation from expected falloff
    radial_arr = np.array(radial_means)
    log_radial  = np.log1p(radial_arr)

    # Fit a line to log(magnitude) vs log(radius)
    x = np.log(np.arange(1, len(radial_means)+1))
    coeffs = np.polyfit(x, log_radial, 1)
    fitted = np.polyval(coeffs, x)
    residuals = np.std(log_radial - fitted)

    return float(np.clip(residuals * 2.0, 0.0, 1.0))


def _analyze_dct_artifacts(gray: np.ndarray) -> float:
    """
    Detect DCT 8x8 block artifacts in image.
    Strong block artifacts indicate heavy compression or re-compression.
    """
    h, w = gray.shape
    h_aligned = (h // 8) * 8
    w_aligned = (w // 8) * 8
    gray_aligned = gray[:h_aligned, :w_aligned]

    # Compute horizontal and vertical differences at 8-pixel boundaries
    h_diffs = []
    for y in range(8, h_aligned, 8):
        diff = float(np.mean(np.abs(
            gray_aligned[y, :] - gray_aligned[y-1, :]
        )))
        h_diffs.append(diff)

    v_diffs = []
    for x in range(8, w_aligned, 8):
        diff = float(np.mean(np.abs(
            gray_aligned[:, x] - gray_aligned[:, x-1]
        )))
        v_diffs.append(diff)

    # Compare boundary differences to interior differences
    all_diffs = float(np.mean(np.abs(np.diff(gray_aligned.flatten()))))
    boundary_diffs = float(np.mean(h_diffs + v_diffs)) if h_diffs else 0

    ratio = boundary_diffs / (all_diffs + 1e-8)
    return float(np.clip((ratio - 1.0) * 0.5, 0.0, 1.0))


def _create_fft_visualization(log_magnitude: np.ndarray) -> np.ndarray:
    """Create a colorized FFT spectrum image."""
    norm = cv2.normalize(log_magnitude, None, 0, 255,
                          cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)

    # Draw center crosshair
    h, w = colored.shape[:2]
    cv2.line(colored, (w//2, 0), (w//2, h), (100, 100, 100), 1)
    cv2.line(colored, (0, h//2), (w, h//2), (100, 100, 100), 1)

    return colored
