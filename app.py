"""
Main Flask Application — Image Forgery Detection System
=========================================================
Premium Features:
    1. ELA   — Error Level Analysis
    2. SIFT  — Copy-Move Detection
    3. CNN   — Deep Learning Classification
    4. EXIF  — Metadata Forensics      ★ NEW
    5. Noise — Pattern Inconsistency   ★ NEW
    6. Ghost — JPEG Ghost Detection    ★ NEW
    7. FFT   — Frequency/AI Analysis   ★ NEW
"""

import os
import uuid
import traceback
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ── Core Techniques ────────────────────────────────────────────────
from techniques.ela       import perform_ela, compute_ela_score, get_ela_heatmap
from techniques.sift      import detect_copy_move
from techniques.metadata  import analyze_metadata
from techniques.noise     import analyze_noise_inconsistency, detect_blur_inconsistency
from techniques.ghost     import detect_jpeg_ghost
from techniques.frequency import analyze_frequency_domain

# ── Visualization ──────────────────────────────────────────────────
from utils.visualization import (generate_forgery_heatmap, ndarray_to_base64,
                                  image_path_to_base64, add_forgery_border,
                                  create_side_by_side)
from utils.ai_explainer import generate_ai_explanation

# ── Optional CNN ───────────────────────────────────────────────────
try:
    from models.cnn_model import predict_with_cnn
    CNN_AVAILABLE = True
except Exception:
    CNN_AVAILABLE = False

# ── App Setup ──────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "forgery-detection-ipcv-2025"

UPLOAD_FOLDER  = os.path.join("static", "uploads")
ALLOWED_EXT    = {"jpg", "jpeg", "png", "bmp", "tiff"}
MAX_MB         = 16

app.config["UPLOAD_FOLDER"]       = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"]  = MAX_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def compute_final_score(ela: float, sift: float, cnn: float | None,
                         meta: float, noise: float,
                         ghost: float, freq: float) -> float:
    """
    Weighted combination of all 7 analysis scores.
    Weights tuned for maximum accuracy.
    """
    if cnn is not None:
        return round(
            0.25 * cnn   +
            0.20 * ela   +
            0.15 * sift  +
            0.15 * meta  +
            0.10 * noise +
            0.10 * ghost +
            0.05 * freq,
            4
        )
    # Without CNN
    return round(
        0.30 * ela   +
        0.20 * sift  +
        0.20 * meta  +
        0.15 * noise +
        0.10 * ghost +
        0.05 * freq,
        4
    )


def get_risk_level(score: float) -> dict:
    """Convert numeric score to risk level."""
    if score < 0.25:
        return {"level": "LOW",    "label": "Likely Authentic",     "color": "#3fb950"}
    elif score < 0.45:
        return {"level": "MEDIUM", "label": "Possibly Manipulated", "color": "#d29922"}
    elif score < 0.65:
        return {"level": "HIGH",   "label": "Likely Forged",        "color": "#f78166"}
    else:
        return {"level": "CRITICAL","label": "Heavily Manipulated",  "color": "#f85149"}


# ── Routes ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG/PNG/BMP"}), 400

    try:
        # ── Save uploaded file ─────────────────────────────────────
        ext      = file.filename.rsplit(".", 1)[1].lower()
        uid      = uuid.uuid4().hex[:8]
        filename = secure_filename(f"{uid}_input.{ext}")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        results = {}

        # ════════════════════════════════════════════════════
        # 1. ELA Analysis
        # ════════════════════════════════════════════════════
        ela_array        = perform_ela(save_path)
        ela_score        = compute_ela_score(save_path)
        ela_heatmap_bgr  = get_ela_heatmap(save_path)
        results["ela"] = {
            "score":   ela_score,
            "percent": round(ela_score * 100, 1),
            "image":   ndarray_to_base64(ela_heatmap_bgr)
        }

        # ════════════════════════════════════════════════════
        # 2. SIFT Copy-Move Detection
        # ════════════════════════════════════════════════════
        sift_result  = detect_copy_move(save_path)
        sift_score   = sift_result["score"]
        sift_kps     = sift_result["keypoints"]
        results["sift"] = {
            "score":        sift_score,
            "percent":      round(sift_score * 100, 1),
            "is_copy_move": sift_result["is_copy_move"],
            "match_count":  sift_result["match_count"],
            "image":        ndarray_to_base64(sift_result["match_image"])
        }

        # ════════════════════════════════════════════════════
        # 3. CNN Prediction
        # ════════════════════════════════════════════════════
        cnn_score   = None
        cnn_trained = False
        if CNN_AVAILABLE:
            cnn_result  = predict_with_cnn(save_path)
            cnn_score   = cnn_result.get("cnn_score")
            cnn_trained = cnn_result.get("is_trained", False)
        results["cnn"] = {
            "score":      cnn_score,
            "percent":    round((cnn_score or 0) * 100, 1),
            "is_trained": cnn_trained,
            "available":  CNN_AVAILABLE
        }

        # ════════════════════════════════════════════════════
        # 4. ★ EXIF Metadata Analysis
        # ════════════════════════════════════════════════════
        meta_result = analyze_metadata(save_path)
        meta_score  = meta_result["score"]
        results["metadata"] = {
            "score":        meta_score,
            "percent":      round(meta_score * 100, 1),
            "has_exif":     meta_result["has_exif"],
            "software":     meta_result["software"],
            "is_edited":    meta_result["is_edited"],
            "camera":       meta_result["camera"],
            "datetime":     meta_result["datetime"],
            "gps":          meta_result["gps"],
            "flags":        meta_result["flags"],
            "thumbnail_mismatch": meta_result["thumbnail_mismatch"]
        }

        # ════════════════════════════════════════════════════
        # 5. ★ Noise Pattern Analysis
        # ════════════════════════════════════════════════════
        noise_result = analyze_noise_inconsistency(save_path)
        blur_result  = detect_blur_inconsistency(save_path)
        noise_score  = round((noise_result["score"] + blur_result["score"]) / 2, 4)
        results["noise"] = {
            "score":         noise_score,
            "percent":       round(noise_score * 100, 1),
            "inconsistency": noise_result["inconsistency"],
            "noise_map":     ndarray_to_base64(noise_result["noise_map"]),
            "blur_map":      ndarray_to_base64(blur_result["blur_map"]),
            "suspicious_blocks": len(noise_result["suspicious_blocks"])
        }

        # ════════════════════════════════════════════════════
        # 6. ★ JPEG Ghost Detection
        # ════════════════════════════════════════════════════
        ghost_result = detect_jpeg_ghost(save_path)
        ghost_score  = ghost_result["score"]
        results["ghost"] = {
            "score":        ghost_score,
            "percent":      round(ghost_score * 100, 1),
            "ghost_ratio":  ghost_result["ghost_ratio"],
            "best_quality": ghost_result["best_quality"],
            "image":        ndarray_to_base64(ghost_result["ghost_map"])
        }

        # ════════════════════════════════════════════════════
        # 7. ★ Frequency Domain / AI Detection
        # ════════════════════════════════════════════════════
        freq_result = analyze_frequency_domain(save_path)
        freq_score  = freq_result["score"]
        results["frequency"] = {
            "score":       freq_score,
            "percent":     round(freq_score * 100, 1),
            "grid_score":  freq_result["grid_score"],
            "dct_score":   freq_result["dct_score"],
            "image":       ndarray_to_base64(freq_result["fft_image"]),
            "flags":       freq_result["flags"]
        }

        # ════════════════════════════════════════════════════
        # FINAL: Combined Score & Verdict
        # ════════════════════════════════════════════════════
        final_score = compute_final_score(
            ela_score, sift_score, cnn_score,
            meta_score, noise_score, ghost_score, freq_score
        )
        threshold = 0.35
        is_fake   = final_score >= threshold
        risk      = get_risk_level(final_score)

        # ── Forgery Localization Heatmap ──────────────────────────
        overlay      = generate_forgery_heatmap(save_path, ela_array, sift_kps)
        bordered     = add_forgery_border(overlay, is_fake)
        side_by_side = create_side_by_side(save_path, bordered)

        # ── All flags combined ────────────────────────────────────
        all_flags = (meta_result["flags"] +
                     freq_result["flags"])

        # ── Pre-build dictionary for AI explanation ────────────────
        partial_data = {
            "verdict":     "FAKE" if is_fake else "REAL",
            "is_fake":     is_fake,
            "final_score": final_score,
            "confidence":  round(final_score * 100, 1),
            "all_flags":   all_flags,
            **results
        }
        ai_explanation = generate_ai_explanation(partial_data)

        # ── Build response ────────────────────────────────────────
        response = {
            "verdict":        "FAKE" if is_fake else "REAL",
            "is_fake":        is_fake,
            "final_score":    final_score,
            "confidence":     round(final_score * 100, 1),
            "risk":           risk,
            "all_flags":      all_flags,
            "ai_explanation": ai_explanation,

            **results,

            "images": {
                "original":    image_path_to_base64(save_path),
                "heatmap":     ndarray_to_base64(bordered),
                "side_by_side": ndarray_to_base64(side_by_side)
            }
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "cnn_available": CNN_AVAILABLE,
        "features":      7
    })


# ── Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Image Forgery Detection System  [PREMIUM]")
    print("  IPCV Final Year Project")
    print("  ELA + SIFT + CNN + EXIF + Noise + Ghost + FFT")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
