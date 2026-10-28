/* ═══════════════════════════════════════════════
   Image Forgery Detection — Frontend JavaScript
   Handles: 7-Technique Analysis, EXIF, Visuals
═══════════════════════════════════════════════ */

const dropZone        = document.getElementById('dropZone');
const fileInput       = document.getElementById('fileInput');
const previewContainer= document.getElementById('previewContainer');
const previewImg      = document.getElementById('previewImg');
const previewName     = document.getElementById('previewName');
const clearBtn        = document.getElementById('clearBtn');
const analyzeBtn      = document.getElementById('analyzeBtn');
const loadingOverlay  = document.getElementById('loadingOverlay');
const loadingText     = document.getElementById('loadingText');
const resultsSection  = document.getElementById('resultsSection');

let selectedFile = null;

// Drag & drop
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
  const name = file.name.toLowerCase();
  const validExt = name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.png') || name.endsWith('.bmp') || name.endsWith('.tiff');
  
  if (!validExt && !allowed.includes(file.type)) {
    alert('Please upload a valid image file (JPG, PNG, BMP, TIFF).');
    return;
  }
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewName.textContent = file.name;
  previewContainer.classList.remove('hidden');
  analyzeBtn.disabled = false;
}

clearBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewContainer.classList.add('hidden');
  analyzeBtn.disabled = true;
  resultsSection.classList.add('hidden');
});

document.getElementById('analyzeAnotherBtn')?.addEventListener('click', () => {
  resultsSection.classList.add('hidden');
  clearBtn.click();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Loading steps
const stepsData = [
  { id: 'step1', msg: '① Running Error Level Analysis (ELA)...' },
  { id: 'step2', msg: '② SIFT keypoint matching (Copy-Move)...' },
  { id: 'step3', msg: '③ Checking CNN classification model...' },
  { id: 'step4', msg: '④ Extracting & inspecting EXIF metadata...' },
  { id: 'step5', msg: '⑤ Sensor noise & blur variation analysis...' },
  { id: 'step6', msg: '⑥ Detecting JPEG ghost & double-compression...' },
  { id: 'step7', msg: '⑦ 2D-FFT frequency domain artifact analysis...' }
];
let stepTimers = [];

function startLoading() {
  stepTimers = [];
  stepsData.forEach(s => {
    const el = document.getElementById(s.id);
    if (el) el.className = 'step pending';
  });

  stepsData.forEach((s, idx) => {
    const t = setTimeout(() => {
      if (idx > 0) {
        const prev = document.getElementById(stepsData[idx - 1].id);
        if (prev) prev.className = 'step done';
      }
      const curr = document.getElementById(s.id);
      if (curr) curr.className = 'step active';
      loadingText.textContent = s.msg;
    }, idx * 600);
    stepTimers.push(t);
  });
}

function stopLoading() {
  stepTimers.forEach(clearTimeout);
  stepsData.forEach(s => {
    const el = document.getElementById(s.id);
    if (el) el.className = 'step done';
  });
}

// Analyze button
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  loadingOverlay.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  startLoading();

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(Server status : );
    }

    const data = await res.json();

    stopLoading();
    loadingOverlay.classList.add('hidden');

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    renderResults(data);
  } catch (err) {
    stopLoading();
    loadingOverlay.classList.add('hidden');
    alert('Analysis failed: ' + err.message);
  }
});

function renderResults(data) {
  // Verdict banner
  const banner = document.getElementById('verdictBanner');
  banner.className = 'verdict-banner ' + (data.is_fake ? 'fake' : 'real');

  document.getElementById('verdictIcon').textContent = data.is_fake ? '🔴' : '🟢';
  document.getElementById('verdictLabel').textContent = data.verdict;
  document.getElementById('confidenceVal').textContent = data.confidence + '%';

  // Risk badge
  const riskBadge = document.getElementById('riskBadge');
  if (data.risk) {
    riskBadge.textContent = `${data.risk.level} RISK — ${data.risk.label}`;
    riskBadge.style.backgroundColor = data.risk.color;
    riskBadge.style.color = '#fff';
  }

  // Score arc animation
  animateScoreArc(data.final_score * 100);
  document.getElementById('scoreText').textContent = data.confidence + '%';

  // ── Store active analysis data for interactive card clicks ──
  window.lastAnalysisData = data;
  setupClickableFeatureCards(data);

  // Findings list
  const flagsContainer = document.getElementById('flagsContainer');
  const flagsList = document.getElementById('flagsList');
  flagsList.innerHTML = '';
  if (data.all_flags && data.all_flags.length > 0) {
    flagsContainer.classList.remove('hidden');
    data.all_flags.forEach(f => {
      const li = document.createElement('li');
      li.textContent = f;
      flagsList.appendChild(li);
    });
  } else {
    flagsContainer.classList.add('hidden');
  }

  // 7 Scores
  animateBar('elaBar', data.ela.percent);
  document.getElementById('elaScore').textContent = data.ela.percent + '%';

  animateBar('siftBar', data.sift.percent);
  document.getElementById('siftScore').textContent = data.sift.percent + '%';
  document.getElementById('matchCount').textContent = data.sift.match_count;

  if (data.cnn && data.cnn.available && data.cnn.score !== null) {
    animateBar('cnnBar', data.cnn.percent);
    document.getElementById('cnnScore').textContent = data.cnn.percent + '%';
    document.getElementById('cnnDesc').textContent = data.cnn.is_trained ? 'Trained model' : 'Untrained baseline';
  } else {
    document.getElementById('cnnScore').textContent = 'N/A';
    document.getElementById('cnnDesc').textContent = 'TensorFlow optional';
  }

  animateBar('metaBar', data.metadata.percent);
  document.getElementById('metaScore').textContent = data.metadata.percent + '%';

  animateBar('noiseBar', data.noise.percent);
  document.getElementById('noiseScore').textContent = data.noise.percent + '%';

  animateBar('ghostBar', data.ghost.percent);
  document.getElementById('ghostScore').textContent = data.ghost.percent + '%';

  animateBar('freqBar', data.frequency.percent);
  document.getElementById('freqScore').textContent = data.frequency.percent + '%';

  // EXIF Metadata details
  const metaGrid = document.getElementById('metaGrid');
  metaGrid.innerHTML = `
    <div class="meta-item"><div class="meta-label">Camera</div><div class="meta-val">${data.metadata.camera || 'None detected'}</div></div>
    <div class="meta-item"><div class="meta-label">Software</div><div class="meta-val">${data.metadata.software || 'None / Cleared'}</div></div>
    <div class="meta-item"><div class="meta-label">Timestamp</div><div class="meta-val">${data.metadata.datetime || 'Unavailable'}</div></div>
    <div class="meta-item"><div class="meta-label">GPS Geotag</div><div class="meta-val">${data.metadata.gps ? 'Present' : 'Not attached'}</div></div>
  `;

  // Images
  document.getElementById('originalImg').src = data.images.original;
  document.getElementById('elaImg').src      = data.ela.image;
  document.getElementById('siftImg').src     = data.sift.image;
  document.getElementById('noiseImg').src    = data.noise.noise_map;
  document.getElementById('ghostImg').src    = data.ghost.image;
  document.getElementById('freqImg').src     = data.frequency.image;
  document.getElementById('heatmapImg').src  = data.images.heatmap;

  // Show section
  resultsSection.classList.remove('hidden');

  // Download
  document.getElementById('downloadBtn').onclick = () => {
    const a = document.createElement('a');
    a.href = data.images.heatmap;
    a.download = 'forensic_report_heatmap.png';
    a.click();
  };

  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

function setupClickableFeatureCards(data) {
  const cards = document.querySelectorAll('.clickable-card');
  const panel = document.getElementById('techDetailPanel');
  const panelTitle = document.getElementById('panelTitle');
  const panelIcon = document.getElementById('panelIcon');
  const panelBadge = document.getElementById('panelBadge');
  const panelScoreVal = document.getElementById('panelScoreVal');
  const panelExplanation = document.getElementById('panelExplanation');
  const closeBtn = document.getElementById('closeDetailPanel');

  closeBtn.onclick = () => {
    panel.classList.add('hidden');
    cards.forEach(c => c.classList.remove('active-card'));
  };

  const explanationsMap = {
    ela: () => {
      const pct = data.ela.percent;
      const isHigh = pct > 40;
      return {
        title: "Error Level Analysis (ELA)",
        icon: "fa-layer-group",
        badge: isHigh ? "Suspicious Compression Spike" : "Uniform Compression",
        badgeType: isHigh ? "danger" : "success",
        score: pct + "%",
        text: `When an image is edited and re-saved as a JPEG, newly altered or spliced regions save at a distinct compression level compared to untouched pixels. Your image has an error level discrepancy of ${pct}%. ${isHigh ? 'The glowing patches on the ELA heatmap highlight areas that were likely manipulated or pasted in.' : 'The error distribution is even across the entire surface, showing no localized editing hotspots.'}`
      };
    },
    sift: () => {
      const count = data.sift.match_count;
      const isCopy = data.sift.is_copy_move || count >= 6;
      return {
        title: "SIFT Keypoint Matching (Copy-Move)",
        icon: "fa-vector-square",
        badge: isCopy ? "Duplication Found" : "No Repeated Regions",
        badgeType: isCopy ? "danger" : "success",
        score: data.sift.percent + "%",
        text: `SIFT searches for duplicated visual descriptors inside the same image. We detected ${count} matching feature points. ${isCopy ? 'This indicates a copy-move or clone stamp forgery, where content was copied from one place to another to duplicate an object or hide a region.' : 'No unnatural recurring patterns were discovered, ruling out clone-stamping.'}`
      };
    },
    cnn: () => {
      const avail = data.cnn && data.cnn.available && data.cnn.score !== null;
      return {
        title: "Convolutional Neural Network (CNN)",
        icon: "fa-brain",
        badge: avail ? (data.cnn.is_trained ? "Trained Model" : "Baseline") : "Optional",
        badgeType: avail ? "warning" : "warning",
        score: avail ? data.cnn.percent + "%" : "N/A",
        text: avail
          ? `The CNN analyzes deep visual textures and spatial patterns learned from training on authentic vs manipulated datasets.`
          : `Deep learning CNN classifier is optional. It works alongside classical computer vision algorithms once trained on the CASIA dataset.`
      };
    },
    meta: () => {
      const sw = data.metadata.software;
      const edited = data.metadata.is_edited;
      const hasExif = data.metadata.has_exif;
      return {
        title: "EXIF Metadata Inspection",
        icon: "fa-tags",
        badge: edited ? "Editor Tagged" : (hasExif ? "Original Header" : "Stripped Header"),
        badgeType: edited ? "danger" : (hasExif ? "success" : "warning"),
        score: data.metadata.percent + "%",
        text: edited
          ? `Direct software footprint detected: '${sw}'. The file header confirms it was processed in an image editing application.`
          : (!hasExif
              ? `No camera metadata found. While typical for messaging apps (WhatsApp, Instagram), original camera RAW/JPEG photos normally store device hardware specifications.`
              : `Camera model identified (${data.metadata.camera || 'Detected'}) with unmanipulated capture headers.`)
      };
    },
    noise: () => {
      const score = data.noise.percent;
      const isHigh = data.noise.score > 0.35;
      return {
        title: "Sensor Noise Consistency (PRNU)",
        icon: "fa-wave-square",
        badge: isHigh ? "Inconsistent Noise Grain" : "Homogeneous Noise Grain",
        badgeType: isHigh ? "danger" : "success",
        score: score + "%",
        text: `Every camera sensor generates a characteristic, subtle noise pattern. In spliced images, noise levels differ across regions. Your image recorded a noise inconsistency of ${data.noise.inconsistency}. ${isHigh ? 'Multiple regional variance outliers were detected, indicating elements may have been composited from different camera sources.' : 'The noise grain is consistent throughout the frame, typical of a single original shot.'}`
      };
    },
    ghost: () => {
      const ratio = Math.round(data.ghost.ghost_ratio * 100);
      const isGhost = data.ghost.score > 0.3;
      return {
        title: "JPEG Ghost Analysis",
        icon: "fa-ghost",
        badge: isGhost ? "Double-Compression Shadow" : "Single JPEG History",
        badgeType: isGhost ? "warning" : "success",
        score: data.ghost.percent + "%",
        text: `JPEG Ghost detects when a region imported from another JPEG was re-compressed at a different quality level (tested at Q=${data.ghost.best_quality}). Ghost ratio is ${ratio}%. ${isGhost ? 'Discovered secondary compression residuals, suggesting composite insertion.' : 'No secondary compression artifacts found.'}`
      };
    },
    freq: () => {
      const grid = data.frequency.grid_score;
      const isHigh = grid > 0.45;
      return {
        title: "2D-FFT Frequency Analysis (AI / Resampling)",
        icon: "fa-chart-line",
        badge: isHigh ? "Periodic Lattice Detected" : "Natural Falloff",
        badgeType: isHigh ? "danger" : "success",
        score: data.frequency.percent + "%",
        text: `Analyzes the Fourier spectrum for mathematical periodicities. Generative AI tools (GANs, Diffusion) and resizing operations introduce periodic frequency spikes. Grid artifact index is ${grid}. ${isHigh ? 'High periodic energy detected, suggesting synthetic AI generation or strong resampling.' : 'Spectrum follows natural 1/f photographic frequency decay.'}`
      };
    }
  };

  cards.forEach(card => {
    card.onclick = () => {
      cards.forEach(c => c.classList.remove('active-card'));
      card.classList.add('active-card');

      const tech = card.dataset.tech;
      if (explanationsMap[tech]) {
        const info = explanationsMap[tech]();
        panelTitle.textContent = info.title;
        panelIcon.className = `fa ${info.icon}`;
        panelBadge.textContent = info.badge;
        panelBadge.className = `ai-badge ai-badge-${info.badgeType}`;
        panelScoreVal.textContent = info.score;
        panelExplanation.textContent = info.text;

        panel.classList.remove('hidden');
        panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    };
  });
}

function animateBar(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.width = '0%';
  setTimeout(() => { el.style.width = Math.min(val, 100) + '%'; }, 80);
}

function animateScoreArc(percent) {
  const arc = document.getElementById('scoreArc');
  if (!arc) return;
  const circum = 2 * Math.PI * 40;
  const offset = circum * (Math.min(percent, 100) / 100);
  arc.style.strokeDasharray = `0 ${circum}`;
  setTimeout(() => {
    arc.style.strokeDasharray = `${offset} ${circum}`;
  }, 100);
}
