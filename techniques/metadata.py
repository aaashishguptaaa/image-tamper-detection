"""
EXIF Metadata Analysis Module
-------------------------------
Real cameras embed metadata (EXIF) in every photo.
Edited images often:
  - Have missing/stripped EXIF data
  - Show editing software (Photoshop, GIMP, etc.)
  - Have inconsistent timestamps
  - Have mismatched thumbnail vs main image
"""

import subprocess
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
import datetime


# Software signatures that indicate editing
EDITING_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "paint", "canva",
    "snapseed", "facetune", "meitu", "pixlr", "affinity",
    "capture one", "darktable", "rawtherapee", "inkscape",
    "illustrator", "corel", "skylum", "luminar", "fotor",
    "picsart", "adobe", "imagemagick", "opencv"
]


def extract_exif(image_path: str) -> dict:
    """
    Extract all EXIF metadata from an image.

    Returns:
        dict of tag_name -> value
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if not exif_data:
            return {}

        readable = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="ignore").strip()
                except Exception:
                    value = str(value)
            readable[str(tag)] = str(value)[:200]  # Truncate long values

        return readable

    except Exception:
        return {}


def analyze_metadata(image_path: str) -> dict:
    """
    Analyze metadata for forgery indicators.

    Returns:
        result dict with:
            has_exif      (bool)
            software      (str|None)  editing software detected
            is_edited     (bool)
            camera        (str|None)
            datetime      (str|None)
            gps           (bool)
            thumbnail_mismatch (bool)
            score         (float 0-1) forgery probability
            flags         (list of str) suspicious findings
            raw_exif      (dict)
    """
    exif = extract_exif(image_path)
    flags = []
    score = 0.0

    # ── Check for editing software ─────────────────────────────────
    software = exif.get("Software", "")
    is_edited = False
    detected_software = None

    if software:
        sw_lower = software.lower()
        for editor in EDITING_SOFTWARE:
            if editor in sw_lower:
                is_edited = True
                detected_software = software.strip()
                flags.append(f"Edited with: {detected_software}")
                score += 0.4
                break

    # ── No EXIF at all (suspicious for JPEG) ──────────────────────
    has_exif = bool(exif)
    if not has_exif:
        flags.append("No EXIF metadata found (possible stripping after editing)")
        score += 0.2

    # ── Check camera make/model ────────────────────────────────────
    make  = exif.get("Make", "")
    model = exif.get("Model", "")
    camera = f"{make} {model}".strip() if (make or model) else None

    if not camera:
        flags.append("No camera make/model in metadata")
        score += 0.1

    # ── Check datetime ─────────────────────────────────────────────
    dt_original  = exif.get("DateTimeOriginal", "")
    dt_digitized = exif.get("DateTimeDigitized", "")
    dt_modified  = exif.get("DateTime", "")
    datetime_val = dt_original or dt_digitized or dt_modified or None

    # If modified time differs from original — editing happened
    if dt_original and dt_modified and dt_original != dt_modified:
        flags.append("File modification time differs from capture time")
        score += 0.2

    # ── GPS data ───────────────────────────────────────────────────
    has_gps = "GPSInfo" in exif

    # ── Thumbnail mismatch check ───────────────────────────────────
    thumbnail_mismatch = _check_thumbnail_mismatch(image_path)
    if thumbnail_mismatch:
        flags.append("Thumbnail does not match main image (strong forgery indicator!)")
        score += 0.5

    # ── File format sanity ─────────────────────────────────────────
    try:
        img = Image.open(image_path)
        declared_format = img.format
        ext = os.path.splitext(image_path)[1].lower().strip(".")
        if declared_format and ext:
            fmt_map = {"jpg": "JPEG", "jpeg": "JPEG"}
            expected = fmt_map.get(ext, ext.upper())
            if declared_format != expected:
                flags.append(f"File extension mismatch: .{ext} but format is {declared_format}")
                score += 0.15
    except Exception:
        pass

    return {
        "has_exif":            has_exif,
        "software":            detected_software,
        "is_edited":           is_edited,
        "camera":              camera,
        "datetime":            datetime_val,
        "gps":                 has_gps,
        "thumbnail_mismatch":  thumbnail_mismatch,
        "score":               round(min(score, 1.0), 4),
        "flags":               flags,
        "raw_exif":            exif
    }


def _check_thumbnail_mismatch(image_path: str) -> bool:
    """
    Check if embedded thumbnail differs from main image.
    This is a classic sign of re-saving a tampered image.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return False

        # Tag 513 = JPEGInterchangeFormat (thumbnail offset)
        if 513 not in exif_data:
            return False

        # Get thumbnail
        img.seek(0)
        from PIL import ImageChops
        thumb = img.copy()
        thumb.thumbnail((64, 64))

        # Compare basic stats — if drastically different, flag it
        main_array = list(img.resize((64, 64)).getdata())
        thumb_array = list(thumb.getdata())

        if len(main_array) != len(thumb_array):
            return False

        diff = sum(abs(int(a[0]) - int(b[0])) for a, b in
                   zip(main_array[:100], thumb_array[:100])
                   if isinstance(a, tuple) and isinstance(b, tuple))
        return diff > 5000

    except Exception:
        return False
