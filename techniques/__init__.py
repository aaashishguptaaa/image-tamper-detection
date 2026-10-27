# techniques/__init__.py
from .ela       import perform_ela, get_ela_heatmap, compute_ela_score
from .sift      import detect_copy_move, get_sift_score
from .metadata  import analyze_metadata
from .noise     import analyze_noise_inconsistency, detect_blur_inconsistency
from .ghost     import detect_jpeg_ghost
from .frequency import analyze_frequency_domain
