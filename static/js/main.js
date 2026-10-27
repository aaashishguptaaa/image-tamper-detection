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

  // ── AI Explanation Rendering ──
  if (data.ai_explanation) {
    document.getElementById('aiSummaryText').textContent = data.ai_explanation.summary || 'Summary unavailable.';
    document.getElementById('aiAdviceText').textContent = data.ai_explanation.advice || '';

    const list = document.getElementById('aiBreakdownList');
    list.innerHTML = '';
    if (data.ai_explanation.breakdown && data.ai_explanation.breakdown.length > 0) {
      data.ai_explanation.breakdown.forEach(item => {
        const row = document.createElement('div');
        row.className = 'ai-item-row';
        row.innerHTML = `
          <div class="ai-item-head">
            <span class="ai-item-title"><i class="fa fa-chevron-right" style="font-size:0.7rem;color:var(--accent);"></i> ${item.tech}</span>
            <span class="ai-badge ai-badge-${item.badge_type}">${item.badge}</span>
          </div>
          <p class="ai-item-desc">${item.detail}</p>
        `;
        list.appendChild(row);
      });
    }
  }

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
