/* ═══════════════════════════════════════════════
   Image Forgery Detection — Frontend JavaScript
   Handles: Upload, Drag & Drop, Analysis, Results
═══════════════════════════════════════════════ */

// ── DOM refs ─────────────────────────────────────
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

// ── Drag & Drop ──────────────────────────────────
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

// ── File Selection ───────────────────────────────
function setFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please upload a valid image file.');
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

// ── Animated Loading Steps ───────────────────────
const loadingMessages = [
  'Running ELA Analysis...',
  'Detecting Copy-Move with SIFT...',
  'Running CNN Classification...',
  'Generating Forgery Heatmap...'
];
const stepIds = ['step1','step2','step3','step4'];
let stepTimers = [];

function startLoadingAnimation() {
  stepTimers = [];
  stepIds.forEach(id => {
    const el = document.getElementById(id);
    el.className = 'step pending';
  });
  loadingText.textContent = loadingMessages[0];

  loadingMessages.forEach((msg, i) => {
    const t = setTimeout(() => {
      if (i > 0) document.getElementById(stepIds[i-1]).className = 'step done';
      document.getElementById(stepIds[i]).className = 'step active';
      loadingText.textContent = msg;
    }, i * 1400);
    stepTimers.push(t);
  });
}

function stopLoadingAnimation() {
  stepTimers.forEach(clearTimeout);
  stepIds.forEach(id => {
    document.getElementById(id).className = 'step done';
  });
}

// ── Analyze ──────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Show loader
  loadingOverlay.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  startLoadingAnimation();

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();

    stopLoadingAnimation();
    loadingOverlay.classList.add('hidden');

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    renderResults(data);

  } catch (err) {
    stopLoadingAnimation();
    loadingOverlay.classList.add('hidden');
    alert('Network error: ' + err.message);
  }
});

// ── Render Results ───────────────────────────────
function renderResults(data) {
  // ── Verdict Banner ──
  const banner = document.getElementById('verdictBanner');
  banner.className = 'verdict-banner ' + (data.is_fake ? 'fake' : 'real');

  document.getElementById('verdictIcon').textContent = data.is_fake ? '🔴' : '🟢';
  document.getElementById('verdictLabel').textContent = data.verdict;
  document.getElementById('verdictSub').innerHTML =
    `Confidence: <strong>${data.confidence}%</strong>`;

  // Score arc animation
  animateScoreArc(data.final_score * 100);
  document.getElementById('scoreText').textContent = data.confidence + '%';

  // ── ELA ──
  document.getElementById('elaScore').textContent = data.ela.percent + '%';
  animateBar('elaBar', data.ela.percent);
  document.getElementById('elaImg').src = data.ela.image;

  // ── SIFT ──
  document.getElementById('siftScore').textContent = data.sift.percent + '%';
  animateBar('siftBar', data.sift.percent);
  document.getElementById('matchCount').textContent = data.sift.match_count;
  document.getElementById('siftImg').src = data.sift.image;

  // ── CNN ──
  if (data.cnn.available && data.cnn.score !== null) {
    document.getElementById('cnnScore').textContent = data.cnn.percent + '%';
    animateBar('cnnBar', data.cnn.percent);
    document.getElementById('cnnDesc').textContent =
      data.cnn.is_trained
        ? 'Trained CNN classification'
        : '⚠️ CNN not yet trained — score may be inaccurate';
  } else {
    document.getElementById('cnnScore').textContent = 'N/A';
    document.getElementById('cnnDesc').textContent = 'TensorFlow not installed — install to enable CNN';
  }

  // ── Images ──
  document.getElementById('originalImg').src  = data.images.original;
  document.getElementById('heatmapImg').src   = data.images.heatmap;

  // ── Show results ──
  resultsSection.classList.remove('hidden');

  // ── Download button ──
  document.getElementById('downloadBtn').onclick = () => {
    const link = document.createElement('a');
    link.href = data.images.heatmap;
    link.download = 'forgery_result.png';
    link.click();
  };

  // Scroll to results
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

// ── Animations ───────────────────────────────────
function animateBar(elementId, percent) {
  const el = document.getElementById(elementId);
  el.style.width = '0%';
  setTimeout(() => { el.style.width = percent + '%'; }, 50);
}

function animateScoreArc(percent) {
  const arc = document.getElementById('scoreArc');
  const circumference = 2 * Math.PI * 40; // r=40
  const offset = circumference * (percent / 100);
  arc.style.strokeDasharray = `0 ${circumference}`;
  setTimeout(() => {
    arc.style.strokeDasharray = `${offset} ${circumference}`;
  }, 100);
}
