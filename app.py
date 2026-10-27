"""
Main Flask Application — Image Forgery Detection System
=========================================================
Routes:
    GET  /           → Upload page
    POST /analyze    → Run ELA + SIFT + CNN analysis
    GET  /about      → Project info page
"""

import os
import uuid
import json
import traceback
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# ── Technique imports ──────────────────────────────────────────────
from techniques.ela  import perform_ela, compute_ela_score, get_ela_heatmap
from techniques.sift import detect_copy_move
from utils.visualization import (generate_forgery_heatmap, ndarray_to_base64,
                                  image_path_to_base64, add_forgery_border,
                                  create_side_by_side)

# ── Optional CNN ───────────────────────────────────────────────────
try:
    from models.cnn_model import predict_with_cnn
    CNN_AVAILABLE = True
except Exception:
    CNN_AVAILABLE = False

# ── App Setup ──────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "forgery-detection-ipcv-2025"

UPLOAD_FOLDER    = os.path.join("static", "uploads")
ALLOWED_EXT      = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
MAX_CONTENT_MB   = 16

app.config["UPLOAD_FOLDER"]    = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def weighted_forgery_score(ela_score: float,
                            sift_score: float,
                            cnn_score: float | None) -> float:
    """
    Combine ELA, SIFT, CNN scores into one forgery probability.
    Weights: CNN 50% | ELA 30% | SIFT 20%  (if CNN available)
             ELA 60% | SIFT 40%             (if CNN unavailable)
    """
    if cnn_score is not None:
        return round(0.50 * cnn_score + 0.30 * ela_score + 0.20 * sift_score, 4)
    return round(0.60 * ela_score + 0.40 * sift_score, 4)


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

        # ── 1. ELA Analysis ────────────────────────────────────────
        ela_array = perform_ela(save_path)
        ela_score = compute_ela_score(save_path)
        ela_heatmap_bgr = get_ela_heatmap(save_path)

        # ── 2. SIFT Copy-Move Detection ────────────────────────────
        sift_result   = detect_copy_move(save_path)
        sift_score    = sift_result["score"]
        sift_kps      = sift_result["keypoints"]
        is_copy_move  = sift_result["is_copy_move"]
        match_count   = sift_result["match_count"]

        # ── 3. CNN Prediction ──────────────────────────────────────
        cnn_score   = None
        cnn_trained = False
        if CNN_AVAILABLE:
            cnn_result  = predict_with_cnn(save_path)
            cnn_score   = cnn_result.get("cnn_score")
            cnn_trained = cnn_result.get("is_trained", False)

        # ── 4. Combined Score & Verdict ────────────────────────────
        final_score = weighted_forgery_score(ela_score, sift_score, cnn_score)
        threshold   = 0.40
        is_fake     = final_score >= threshold

        # ── 5. Forgery Localization Heatmap ───────────────────────
        overlay  = generate_forgery_heatmap(save_path, ela_array, sift_kps)
        bordered = add_forgery_border(overlay, is_fake)
        side_by_side = create_side_by_side(save_path, bordered)

        # ── 6. Encode images for browser ──────────────────────────
        original_b64    = image_path_to_base64(save_path)
        ela_b64         = ndarray_to_base64(ela_heatmap_bgr)
        overlay_b64     = ndarray_to_base64(bordered)
        sift_b64        = ndarray_to_base64(sift_result["match_image"])
        side_by_side_b64 = ndarray_to_base64(side_by_side)

        # ── Build response ────────────────────────────────────────
        response = {
            "verdict":      "FAKE" if is_fake else "REAL",
            "is_fake":      is_fake,
            "final_score":  final_score,
            "confidence":   round(final_score * 100, 1),

            "ela": {
                "score":   ela_score,
                "percent": round(ela_score * 100, 1),
                "image":   ela_b64
            },
            "sift": {
                "score":         sift_score,
                "percent":       round(sift_score * 100, 1),
                "is_copy_move":  is_copy_move,
                "match_count":   match_count,
                "image":         sift_b64
            },
            "cnn": {
                "score":      cnn_score,
                "percent":    round((cnn_score or 0) * 100, 1),
                "is_trained": cnn_trained,
                "available":  CNN_AVAILABLE
            },

            "images": {
                "original":    original_b64,
                "heatmap":     overlay_b64,
                "side_by_side": side_by_side_b64
            }
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "cnn_available": CNN_AVAILABLE})


# ── Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Image Forgery Detection System")
    print("  IPCV Final Year Project")
    print("  ELA + SIFT + CNN")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
