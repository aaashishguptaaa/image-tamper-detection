"""
SIFT (Scale-Invariant Feature Transform) Module
------------------------------------------------
SIFT detects copy-move forgery — when a region of the image
is copied and pasted elsewhere within the same image.
It finds keypoints, matches similar descriptors, and
flags suspiciously matched regions.
"""

import cv2
import numpy as np


def detect_copy_move(image_path: str,
                     min_match_count: int = 10,
                     ratio_threshold: float = 0.75) -> dict:
    """
    Detect copy-move forgery using SIFT feature matching.

    Args:
        image_path:       Path to the input image
        min_match_count:  Minimum good matches to flag as forgery
        ratio_threshold:  Lowe's ratio test threshold

    Returns:
        result dict with keys:
            is_copy_move (bool)
            match_count  (int)
            score        (float 0-1)
            keypoints    (list of (x,y))
            match_image  (np.ndarray BGR annotated image)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 2:
        return {
            "is_copy_move": False,
            "match_count": 0,
            "score": 0.0,
            "keypoints": [],
            "match_image": image
        }

    # Use FLANN-based matcher for speed
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors with themselves (k=2 nearest)
    try:
        matches = flann.knnMatch(descriptors, descriptors, k=2)
    except Exception:
        matches = []

    # Apply Lowe's ratio test & filter self-matches
    good_matches = []
    for m_list in matches:
        if len(m_list) < 2:
            continue
        m, n = m_list[0], m_list[1]
        # Skip self-matches (same keypoint)
        if m.queryIdx == m.trainIdx:
            continue
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # Filter spatially distant matches (copy-move pairs should be far apart)
    spatial_matches = []
    for m in good_matches:
        pt1 = np.array(keypoints[m.queryIdx].pt)
        pt2 = np.array(keypoints[m.trainIdx].pt)
        dist = np.linalg.norm(pt1 - pt2)
        if dist > 30:  # Minimum spatial distance to avoid noise
            spatial_matches.append(m)

    is_copy_move = len(spatial_matches) >= min_match_count
    score = float(np.clip(len(spatial_matches) / (min_match_count * 3), 0.0, 1.0))

    # Draw matches on annotated image
    annotated = _draw_copy_move(image, keypoints, spatial_matches[:50])

    keypoint_coords = [(int(keypoints[m.queryIdx].pt[0]),
                        int(keypoints[m.queryIdx].pt[1]))
                       for m in spatial_matches[:50]]

    return {
        "is_copy_move": is_copy_move,
        "match_count": len(spatial_matches),
        "score": round(score, 4),
        "keypoints": keypoint_coords,
        "match_image": annotated
    }


def _draw_copy_move(image: np.ndarray,
                    keypoints: list,
                    matches: list) -> np.ndarray:
    """Draw matched keypoint pairs on image."""
    annotated = image.copy()
    for m in matches:
        pt1 = tuple(map(int, keypoints[m.queryIdx].pt))
        pt2 = tuple(map(int, keypoints[m.trainIdx].pt))
        cv2.circle(annotated, pt1, 5, (0, 255, 0), -1)
        cv2.circle(annotated, pt2, 5, (0, 0, 255), -1)
        cv2.line(annotated, pt1, pt2, (255, 255, 0), 1)
    return annotated


def get_sift_score(image_path: str) -> float:
    """
    Return a forgery probability score based on SIFT copy-move detection.

    Returns:
        score: float between 0.0 and 1.0
    """
    result = detect_copy_move(image_path)
    return result["score"]
