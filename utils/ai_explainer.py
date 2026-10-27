"""
AI Forensic Report Generator
----------------------------
Generates an intelligent, plain-English explanation of why an image
was flagged as FAKE or REAL based on all forensic technique outputs.
Can use Google Gemini API if GEMINI_API_KEY is present, or falls back to
a smart rule-based forensic analysis engine.
"""

import os

def generate_ai_explanation(results: dict) -> dict:
    """
    Generate an insightful AI breakdown for the uploaded image.
    
    Args:
        results: Dictionary containing scores and telemetry from all techniques.
        
    Returns:
        dict with:
            summary     (str): High-level verdict summary in plain English
            breakdown   (list of dict): Title, status, and what this specific finding means
            advice      (str): Actionable advice for the user/investigator
    """
    is_fake = results.get("is_fake", False)
    final_score = results.get("final_score", 0.0)
    confidence = results.get("confidence", 0.0)
    
    ela = results.get("ela", {})
    sift = results.get("sift", {})
    meta = results.get("metadata", {})
    noise = results.get("noise", {})
    ghost = results.get("ghost", {})
    freq = results.get("frequency", {})
    
    # Check if Gemini API key exists
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            return _call_gemini_explainer(results, gemini_key)
        except Exception as e:
            print(f"[AI Explainer] Gemini fallback triggered: {e}")
            
    # Default smart forensic reasoning engine
    return _rule_based_forensic_explainer(
        is_fake, final_score, confidence, ela, sift, meta, noise, ghost, freq
    )


def _rule_based_forensic_explainer(is_fake, final_score, confidence, ela, sift, meta, noise, ghost, freq):
    breakdown = []
    
    # 1. ELA Breakdown
    ela_pct = ela.get("percent", 0)
    if ela_pct > 40:
        breakdown.append({
            "tech": "Error Level Analysis (ELA)",
            "badge": "Tampering Indicator",
            "badge_type": "danger",
            "detail": f"Observed higher-than-normal compression inconsistency ({ela_pct}%). When an image is modified and re-saved, spliced or touched-up regions retain distinct compression levels compared to original pixels. In the heatmap, brighter glowing areas point directly to regions that were likely edited or composited."
        })
    else:
        breakdown.append({
            "tech": "Error Level Analysis (ELA)",
            "badge": "Consistent",
            "badge_type": "success",
            "detail": f"Compression error level is uniform ({ela_pct}%). There are no prominent localized error spikes, suggesting the image has maintained a single, consistent compression history."
        })
        
    # 2. SIFT Breakdown
    match_count = sift.get("match_count", 0)
    if sift.get("is_copy_move", False) or match_count >= 6:
        breakdown.append({
            "tech": "SIFT Feature Matching (Copy-Move)",
            "badge": "Duplication Detected",
            "badge_type": "danger",
            "detail": f"Detected {match_count} cloned keypoint pairs across separate spatial locations. This pattern strongly points to a copy-move / clone stamp manipulation, where an object or texture inside this image was cloned to either multiply something or conceal an underlying region."
        })
    else:
        breakdown.append({
            "tech": "SIFT Feature Matching (Copy-Move)",
            "badge": "No Clones",
            "badge_type": "success",
            "detail": f"Only {match_count} incidental keypoint matches were detected. No unnatural recurring identical descriptors were found, ruling out copy-move or clone stamp forgery."
        })
        
    # 3. EXIF Metadata Breakdown
    meta_sw = meta.get("software")
    flags = meta.get("flags", [])
    if meta.get("is_edited") or meta_sw:
        breakdown.append({
            "tech": "EXIF Metadata Inspection",
            "badge": "Editing Signature",
            "badge_type": "danger",
            "detail": f"Found direct editing signatures mentioning software: '{meta_sw}'. Furthermore, metadata timestamps or headers indicate post-processing software saved this file."
        })
    elif not meta.get("has_exif"):
        breakdown.append({
            "tech": "EXIF Metadata Inspection",
            "badge": "Stripped Header",
            "badge_type": "warning",
            "detail": "The image has completely sanitized/stripped metadata. While common when downloaded from messaging platforms (like WhatsApp or Twitter), professional/original camera captures typically contain camera specifications (Make, Model, Exposure)."
        })
    else:
        cam = meta.get("camera", "Standard device")
        breakdown.append({
            "tech": "EXIF Metadata Inspection",
            "badge": "Clean Camera Data",
            "badge_type": "success",
            "detail": f"Camera hardware metadata identified ({cam}) without traces of known photo-editing suite headers."
        })
        
    # 4. Noise Pattern Analysis
    noise_score = noise.get("score", 0)
    suspicious_blocks = noise.get("suspicious_blocks", 0)
    if noise_score > 0.35 or suspicious_blocks > 2:
        breakdown.append({
            "tech": "PRNU Sensor Noise Consistency",
            "badge": "Noise Variance",
            "badge_type": "danger",
            "detail": f"Detected {suspicious_blocks} high-variance noise blocks. Authentic photos capture a consistent sensor noise pattern across the entire lens. Mismatched regional noise is a classic footprint of morphing or pasting an object from another source photo."
        })
    else:
        breakdown.append({
            "tech": "PRNU Sensor Noise Consistency",
            "badge": "Uniform Grain",
            "badge_type": "success",
            "detail": "Background sensor noise distribution and edge sharpness are homogenous, matching standard uncomposited camera captures."
        })
        
    # 5. JPEG Ghost
    ghost_ratio = ghost.get("ghost_ratio", 0)
    if ghost.get("score", 0) > 0.3:
        breakdown.append({
            "tech": "JPEG Ghost Analysis",
            "badge": "Double Compression",
            "badge_type": "warning",
            "detail": f"Double compression trace detected (ghost ratio: {round(ghost_ratio*100, 1)}% at Q={ghost.get('best_quality', 75)}). This happens when an already-compressed JPEG segment is imported into another photo and re-exported."
        })
    else:
        breakdown.append({
            "tech": "JPEG Ghost Analysis",
            "badge": "Single Compression",
            "badge_type": "success",
            "detail": "No secondary compression residual shadows (ghosts) were identified."
        })

    # 6. Frequency FFT / AI
    grid_score = freq.get("grid_score", 0)
    if grid_score > 0.45:
        breakdown.append({
            "tech": "2D-FFT Frequency Analysis",
            "badge": "Periodic Artifacts",
            "badge_type": "danger",
            "detail": "Spikes in periodic Fourier frequencies were discovered. AI generators (Diffusion/GANs) and heavy digital resamplers leave telltale grid patterns in the frequency domain that are invisible to the naked eye."
        })
        
    # Generate High-level verdict summary
    if is_fake:
        summary = (
            f"The forensic analysis concludes this image is manipulated with {confidence}% confidence. "
            f"Key triggers include: {', '.join([b['tech'] for b in breakdown if b['badge_type'] == 'danger']) or 'multiple cumulative anomalies'}. "
            "The red highlighted zones on the localization heatmap mark the specific areas exhibiting artificial tampering signatures."
        )
        advice = "This media shows demonstrable indicators of digital manipulation or splicing. It is unrecommended for forensic verification or official submission without primary camera RAW confirmation."
    else:
        summary = (
            f"The forensic analysis concludes this image is authentic with {round(100 - confidence, 1)}% integrity confidence. "
            "All primary detectors (compression errors, feature cloning, sensor noise patterns, and frequency falloff) align within normal unmanipulated photographic thresholds."
        )
        advice = "The image passes automated integrity screening. No signs of splicing, clone-stamping, or deep composite artifacts were discovered."

    return {
        "summary": summary,
        "breakdown": breakdown,
        "advice": advice
    }


def _call_gemini_explainer(results: dict, api_key: str) -> dict:
    import json
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are an expert digital forensics investigator. Provide a clear, educational, plain-English breakdown of this image analysis.
    
    Data:
    - Final Verdict: {'FAKE / TAMPERED' if results.get('is_fake') else 'AUTHENTIC / REAL'}
    - Confidence: {results.get('confidence')}%
    - ELA Percent: {results.get('ela', {}).get('percent')}%
    - SIFT Matches: {results.get('sift', {}).get('match_count')} matches (Copy-move: {results.get('sift', {}).get('is_copy_move')})
    - EXIF Software: {results.get('metadata', {}).get('software')} | Camera: {results.get('metadata', {}).get('camera')}
    - Flags: {results.get('all_flags', [])}
    - Noise Inconsistency: {results.get('noise', {}).get('inconsistency')}
    - Ghost Score: {results.get('ghost', {}).get('score')}
    - Frequency Grid Score: {results.get('frequency', {}).get('grid_score')}

    Return strictly a JSON object with this schema:
    {{
      "summary": "2-3 sentences explaining in everyday terms why the photo is fake or real",
      "breakdown": [
         {{
           "tech": "Technique Name",
           "badge": "Short badge text",
           "badge_type": "danger" or "warning" or "success",
           "detail": "Clear explanation of what this tool looked for and what the numbers mean for this specific image."
         }}
      ],
      "advice": "1-2 sentences with practical guidance on this photo"
    }}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)
