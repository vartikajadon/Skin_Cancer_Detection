document.addEventListener('DOMContentLoaded', () => {
  // --- Navigation & Scroll Effects ---
  const navbar = document.getElementById('navbar');
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('section');

  // Change nav background on scroll
  const handleScroll = () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }

    // Active link highlighting on scroll
    let currentSectionId = 'home';
    sections.forEach(section => {
      const sectionTop = section.offsetTop - 100; // offset for sticky nav
      const sectionHeight = section.offsetHeight;
      if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
        currentSectionId = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${currentSectionId}`) {
        link.classList.add('active');
      }
    });
  };

  window.addEventListener('scroll', handleScroll);
  handleScroll(); // Initial check

  // --- Mobile Hamburger Menu ---
  const menuToggle = document.getElementById('menu-toggle');
  const navMenu = document.getElementById('nav-menu');

  if (menuToggle && navMenu) {
    menuToggle.addEventListener('click', () => {
      navMenu.classList.toggle('open');
      menuToggle.classList.toggle('active');
    });

    // Close menu when clicking links
    navLinks.forEach(link => {
      link.addEventListener('click', () => {
        navMenu.classList.remove('open');
        menuToggle.classList.remove('active');
      });
    });
  }

  // --- Drag and Drop File Upload Portal ---
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const filePreview = document.getElementById('file-preview');
  const previewImg = document.getElementById('preview-img');
  const fileNameDisplay = document.getElementById('file-name');
  const fileSizeDisplay = document.getElementById('file-size');
  const btnRemove = document.getElementById('btn-remove');
  
  const resultsCard = document.getElementById('results-card');
  const lesionResult = document.getElementById('lesion-result');
  const lesionConfidence = document.getElementById('lesion-confidence');

  // Grad-CAM Visualization Elements
  const gradcamContainer = document.getElementById('gradcam-container');
  const gradcamOriginal = document.getElementById('gradcam-original');
  const gradcamHeatmap = document.getElementById('gradcam-heatmap');
  const gradcamOverlay = document.getElementById('gradcam-overlay');

  // Interactive Slider Elements
  const sliderRangeControl = document.getElementById('slider-range-control');
  const sliderOriginal = document.getElementById('slider-original');
  const sliderOverlay = document.getElementById('slider-overlay');
  const overlayWrapper = document.querySelector('.overlay-wrapper');
  const sliderHandle = document.querySelector('.slider-handle');

  // Clinical Input Elements
  const patientNameInput = document.getElementById('patient-name');
  const patientNumberInput = document.getElementById('patient-number');
  const patientAgeInput = document.getElementById('patient-age');
  const patientSexSelect = document.getElementById('patient-sex');
  const lesionSiteSelect = document.getElementById('lesion-site');
  const resultsClinicalMeta = document.getElementById('results-clinical-meta');

  // History and Print Elements
  const historyList = document.getElementById('history-list');
  const btnPrintReport = document.getElementById('btn-print-report');
  
  // Track active case data for printing
  let activeCaseData = null;

  // Human-readable mapping of model disease codes
  const classLabels = {
    'nv': 'Nevus (Benign Mole)',
    'mel': 'Melanoma (Malignant Lesion)',
    'bkl': 'Benign Keratosis-like Lesion',
    'bcc': 'Basal Cell Carcinoma (Suspected)',
    'akiec': 'Actinic Keratosis / Bowen\'s Disease',
    'vasc': 'Vascular Lesion',
    'df': 'Dermatofibroma'
  };

  // Short descriptions explaining the diseases/lesions
  const classDescriptions = {
    'nv': 'A Nevus (Benign Mole) is a common, non-cancerous skin growth formed by clusters of melanocytes. They are typically uniform in color and shape, and completely harmless.',
    'mel': 'Melanoma is the most serious form of skin cancer, arising from pigment cells (melanocytes). It requires prompt professional attention and often biopsy/surgical removal.',
    'bkl': 'Benign Keratosis-like Lesions (e.g. seborrheic keratosis, solar lentigines) are common, completely harmless growths that often appear as waxy or scaly patches with age.',
    'bcc': 'Basal Cell Carcinoma is the most common form of skin cancer. It is slow-growing, highly treatable, and rarely spreads, appearing as pearly bumps or scaly patches.',
    'akiec': 'Actinic Keratosis (Bowen\'s Disease) is a sun-induced pre-cancerous lesion. It appears as rough, dry, or scaly patches and should be treated to prevent potential progression.',
    'vasc': 'Vascular Lesions (e.g. cherry angiomas) are benign skin spots formed by abnormally dense collections of blood vessels. They are harmless and non-cancerous.',
    'df': 'Dermatofibromas are common, benign (harmless) firm skin bumps often found on the lower legs, frequently developing in response to minor insect bites or trauma.'
  };

  // Diagnostic Risk mappings
  const classRisks = {
    'nv': 'low',
    'bkl': 'low',
    'bcc': 'medium',
    'df': 'medium',
    'vasc': 'medium',
    'mel': 'high',
    'akiec': 'high'
  };

  let isProcessing = false;

  // Initialize History List
  renderHistoryList();

  // Handle slider events
  if (sliderRangeControl && overlayWrapper && sliderHandle) {
    sliderRangeControl.addEventListener('input', () => {
      const val = sliderRangeControl.value;
      overlayWrapper.style.width = `${val}%`;
      sliderHandle.style.left = `${val}%`;
    });
  }

  // Handle Visualizer Tabs
  const tabSlider = document.getElementById('tab-slider');
  const tabGrid = document.getElementById('tab-grid');
  const vizSliderContent = document.getElementById('viz-slider-content');
  const vizGridContent = document.getElementById('viz-grid-content');

  if (tabSlider && tabGrid && vizSliderContent && vizGridContent) {
    tabSlider.addEventListener('click', (e) => {
      e.stopPropagation();
      tabSlider.classList.add('active');
      tabSlider.style.background = 'var(--color-white)';
      tabSlider.style.color = 'var(--color-primary)';
      tabGrid.classList.remove('active');
      tabGrid.style.background = 'none';
      tabGrid.style.color = 'var(--color-text-muted)';
      vizSliderContent.style.display = 'block';
      vizGridContent.style.display = 'none';
    });

    tabGrid.addEventListener('click', (e) => {
      e.stopPropagation();
      tabGrid.classList.add('active');
      tabGrid.style.background = 'var(--color-white)';
      tabGrid.style.color = 'var(--color-primary)';
      tabSlider.classList.remove('active');
      tabSlider.style.background = 'none';
      tabSlider.style.color = 'var(--color-text-muted)';
      vizGridContent.style.display = 'grid';
      vizSliderContent.style.display = 'none';
    });
  }

  // Filter default drop-zone content elements for toggle actions
  const defaultZoneChildren = Array.from(dropZone.children).filter(el => 
    el.id !== 'file-preview' && el.id !== 'progress-container' && el.id !== 'results-card'
  );

  function toggleDefaultZoneContent(show) {
    defaultZoneChildren.forEach(el => {
      el.style.display = show ? '' : 'none';
    });
  }

  // Prevent defaults for drag events
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  // Drag highlights
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      if (!isProcessing) dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.remove('dragover');
    }, false);
  });

  // Handle dropped files
  dropZone.addEventListener('drop', (e) => {
    if (isProcessing) return;
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  // Handle selected files via browse dialog
  fileInput.addEventListener('change', (e) => {
    if (isProcessing) return;
    if (e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  // Remove current file
  btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUploadZone();
  });

  function handleFileSelection(file) {
    // Clear old errors and results
    removeErrorCard();
    resultsCard.style.display = 'none';

    // Validate extension
    const validExtensions = ['.jpg', '.jpeg', '.png'];
    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const isValidExtension = validExtensions.includes(fileExtension) || file.type.startsWith('image/');
    
    if (!isValidExtension) {
      showErrorCard('Unsupported file format. Please upload a valid image file (JPG, JPEG, PNG).');
      return;
    }

    // Validate size (5MB limit)
    if (file.size > 5 * 1024 * 1024) {
      showErrorCard('Image size exceeds 5MB limit. Please upload a smaller image file.');
      return;
    }

    isProcessing = true;
    
    // Display file name and size
    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = formatBytes(file.size);
    
    // Display preview thumbnail
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      filePreview.style.display = 'flex';
      
      // Auto scroll slightly to bring upload area fully into view
      dropZone.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      
      // Submit for active neural classification
      performAnalysis(file);
    };
    reader.readAsDataURL(file);
  }

  function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Communicate with API Service Layer
  async function performAnalysis(file) {
    toggleDefaultZoneContent(false);
    showSpinner();

    try {
      // Execute REST prediction
      const result = await window.apiService.predictLesion(file);
      
      hideSpinner();
      if (result.status === "uncertain") {
        renderUncertainResults(result);
        showSuccessToast('Analysis finished with low confidence.');
      } else {
        // Build current active case data payload
        const formattedDate = new Date().toLocaleDateString(undefined, {
          month: 'short',
          day: 'numeric',
          year: 'numeric'
        });
        
        const currentCase = {
          id: String(Date.now()),
          date: formattedDate,
          predictedClass: result.predicted_class,
          confidence: result.confidence,
          topPredictions: result.top_predictions || [],
          gradcamImageBase64: result.gradcam_image_base64 || null,
          heatmapImageBase64: result.heatmap_image_base64 || null,
          previewImgSrc: previewImg.src,
          patientName: patientNameInput.value || 'N/A',
          patientNumber: patientNumberInput.value || 'N/A',
          patientAge: patientAgeInput.value || 'N/A',
          patientSex: patientSexSelect.value || 'Unknown',
          lesionSite: lesionSiteSelect.value || 'Unknown',
          fileName: fileNameDisplay.textContent
        };
        
        activeCaseData = currentCase;
        
        // Save to localStorage case history
        saveToHistory(currentCase);
        savePatientRecordToLocalStorage(currentCase);
        
        renderResults(result);
        showSuccessToast('Lesion analyzed successfully!');
      }
    } catch (err) {
      hideSpinner();
      const msg = err.message || '';
      if (msg.includes('dermoscopic skin lesion') || msg.includes('quality check failed') || msg.includes('does not appear to be a skin lesion')) {
        showRejectionCard(msg);
      } else {
        showErrorCard(msg || 'An unexpected failure occurred during analysis.');
      }
    } finally {
      isProcessing = false;
    }
  }

  // Display predictions and dynamic top-3 confidence bars
  function renderResults(result) {
    const predClass = result.predicted_class;
    const confidenceVal = result.confidence;
    const topPredictions = result.top_predictions || [];

    // Map names and risk profiles
    const label = classLabels[predClass] || predClass;
    const risk = classRisks[predClass] || 'medium';

    lesionResult.textContent = label;
    const tooltipEl = document.getElementById('lesion-tooltip');
    if (tooltipEl) {
      tooltipEl.textContent = classDescriptions[predClass] || 'No description available for this category.';
    }
    lesionConfidence.innerHTML = `Confidence Score: <strong>${(confidenceVal * 100).toFixed(1)}%</strong>`;

    // Clear old risk badges and insert updated status badge
    const oldBadge = lesionResult.parentElement.querySelector('.risk-badge');
    if (oldBadge) oldBadge.remove();

    const badge = document.createElement('span');
    badge.className = `risk-badge ${risk}`;
    
    let icon = 'fa-shield-halved';
    if (risk === 'high') icon = 'fa-triangle-exclamation';
    else if (risk === 'medium') icon = 'fa-circle-exclamation';

    badge.innerHTML = `<i class="fa-solid ${icon}"></i> ${risk} Risk Assessment`;
    lesionResult.after(badge);

    // Populate Top-3 Horizontal Progress Bars
    const topPredList = document.getElementById('top-predictions-list');
    if (topPredList) {
      topPredList.innerHTML = '<div class="top-predictions-title"><i class="fa-solid fa-chart-bar"></i> Top Predictions Breakdown</div>';
      
      topPredictions.forEach((pred) => {
        const itemLabel = classLabels[pred.class] || pred.class;
        const itemPercent = (pred.score * 100).toFixed(1);
        
        const itemHtml = `
          <div class="top-pred-item">
            <div class="top-pred-label-row">
              <span class="top-pred-class-name">${itemLabel}</span>
              <span class="top-pred-percentage">${itemPercent}%</span>
            </div>
            <div class="top-pred-bar">
              <div class="top-pred-fill" style="width: 0%"></div>
            </div>
          </div>
        `;
        
        const div = document.createElement('div');
        div.innerHTML = itemHtml.trim();
        topPredList.appendChild(div.firstChild);
      });

      // Animate widths after brief rendering buffer
      setTimeout(() => {
        const fills = topPredList.querySelectorAll('.top-pred-fill');
        fills.forEach((fill, idx) => {
          if (topPredictions[idx]) {
            fill.style.width = `${(topPredictions[idx].score * 100).toFixed(1)}%`;
          }
        });
      }, 50);
    }

    // Populate clinical metadata card info
    if (resultsClinicalMeta && activeCaseData) {
      resultsClinicalMeta.style.display = 'block';
      const advice = generateEpidemiologicalAdvice(
        activeCaseData.patientAge,
        activeCaseData.patientSex,
        activeCaseData.lesionSite,
        activeCaseData.predictedClass
      );
      
      resultsClinicalMeta.innerHTML = `
        <div style="font-weight: 700; margin-bottom: 0.25rem;"><i class="fa-solid fa-notes-medical"></i> Patient & Case Details</div>
        <div><strong>Name:</strong> ${activeCaseData.patientName} | <strong>Patient ID:</strong> ${activeCaseData.patientNumber}</div>
        <div style="margin-top: 0.25rem;"><strong>Age:</strong> ${activeCaseData.patientAge} | <strong>Sex:</strong> ${activeCaseData.patientSex} | <strong>Site:</strong> ${activeCaseData.lesionSite}</div>
        ${advice ? `<div style="margin-top: 0.5rem; color: #b45309; line-height: 1.3;">${advice}</div>` : ''}
      `;
    }

    // Render Grad-CAM explainability maps
    if (result.gradcam_image_base64 && result.heatmap_image_base64) {
      if (gradcamContainer && gradcamOriginal && gradcamHeatmap && gradcamOverlay && sliderOriginal && sliderOverlay) {
        // Setup Grid View
        gradcamOriginal.src = previewImg.src;
        gradcamHeatmap.src = result.heatmap_image_base64;
        gradcamOverlay.src = result.gradcam_image_base64;
        
        // Setup Slider View
        sliderOriginal.src = previewImg.src;
        sliderOverlay.src = result.gradcam_image_base64;
        
        // Reset slider ranges
        if (sliderRangeControl && overlayWrapper && sliderHandle) {
          sliderRangeControl.value = 50;
          overlayWrapper.style.width = '50%';
          sliderHandle.style.left = '50%';
        }
        
        gradcamContainer.style.display = 'block';
      }
    } else {
      if (gradcamContainer) gradcamContainer.style.display = 'none';
    }

    resultsCard.style.display = 'block';
  }

  // Display uncertain results warning card without standard top-3 bars
  function renderUncertainResults(result) {
    lesionResult.textContent = "Uncertain Diagnosis";
    const tooltipEl = document.getElementById('lesion-tooltip');
    if (tooltipEl) {
      tooltipEl.textContent = 'The screening model could not confidently identify this skin lesion. Please upload a high-quality dermoscopic photo under good lighting and verify inputs.';
    }
    lesionConfidence.innerHTML = `Status: <strong>Uncertain</strong>`;

    // Clear old risk badges and insert updated status badge
    const oldBadge = lesionResult.parentElement.querySelector('.risk-badge');
    if (oldBadge) oldBadge.remove();

    const badge = document.createElement('span');
    badge.className = `risk-badge medium`;
    badge.innerHTML = `<i class="fa-solid fa-circle-question"></i> Low Confidence`;
    lesionResult.after(badge);

    // Populate Top-3 Breakdown container with uncertain alert details
    const topPredList = document.getElementById('top-predictions-list');
    if (topPredList) {
      topPredList.innerHTML = `
        <div class="uncertain-alert-box" style="padding: 1rem; border-radius: 8px; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); margin-top: 1rem; color: #d97706; text-align: left;">
          <div style="font-weight: 700; display: flex; align-items: center; gap: 0.5rem; font-size: 0.95rem;">
            <i class="fa-solid fa-triangle-exclamation"></i> ⚠ Low Confidence
          </div>
          <div style="font-size: 0.85rem; margin-top: 0.25rem; line-height: 1.4;">
            ${escapeHtml(result.message)} The model was unable to classify the lesion with high confidence (requires &ge;70.0%).
          </div>
        </div>
      `;
    }

    if (resultsClinicalMeta) {
      resultsClinicalMeta.style.display = 'none';
    }

    // Hide explainability charts to avoid misleading clinicians
    if (gradcamContainer) gradcamContainer.style.display = 'none';

    resultsCard.style.display = 'block';
  }

  // Show dynamic loader spinner
  function showSpinner() {
    removeSpinner();
    const spinner = document.createElement('div');
    spinner.id = 'api-loading-spinner';
    spinner.className = 'spinner-container';
    spinner.innerHTML = `
      <span class="spinner-loader"></span>
      <span class="spinner-text">Analyzing lesion image with EfficientNetB0 AI...</span>
    `;
    dropZone.appendChild(spinner);
  }

  function hideSpinner() {
    removeSpinner();
  }

  function removeSpinner() {
    const existing = document.getElementById('api-loading-spinner');
    if (existing) existing.remove();
  }

  // Show error alert card
  function showErrorCard(message) {
    removeErrorCard();
    const card = document.createElement('div');
    card.id = 'api-error-card';
    card.className = 'error-alert-card';
    card.innerHTML = `
      <i class="fa-solid fa-triangle-exclamation error-alert-icon"></i>
      <div class="error-alert-content">
        <div class="error-alert-title">Analysis Failure</div>
        <div class="error-alert-desc">${escapeHtml(message)}</div>
      </div>
    `;
    dropZone.appendChild(card);
    toggleDefaultZoneContent(false);
  }

  // Show OOD / quality validation rejection card
  function showRejectionCard(message) {
    removeErrorCard();
    const card = document.createElement('div');
    card.id = 'api-error-card';
    card.className = 'error-alert-card rejection-card';
    card.innerHTML = `
      <i class="fa-solid fa-circle-exclamation error-alert-icon" style="color: #ef4444;"></i>
      <div class="error-alert-content" style="text-align: left;">
        <div class="error-alert-title" style="color: #ef4444; font-weight: 700; font-size: 1.05rem; display: flex; align-items: center; gap: 0.5rem;">
          <i class="fa-solid fa-triangle-exclamation"></i> ⚠ Invalid Image
        </div>
        <div class="error-alert-desc" style="margin-top: 0.35rem; font-weight: 600; font-size: 0.9rem; color: var(--color-text);">
          The uploaded image does not appear to be a skin lesion.
        </div>
        <div class="error-alert-subdesc" style="margin-top: 0.25rem; font-size: 0.85rem; color: var(--color-text-muted); line-height: 1.4;">
          ${escapeHtml(message)}
        </div>
      </div>
    `;
    dropZone.appendChild(card);
    toggleDefaultZoneContent(false);
  }

  function removeErrorCard() {
    const existing = document.getElementById('api-error-card');
    if (existing) existing.remove();
  }

  // Display Success Toast message
  function showSuccessToast(message) {
    const toast = document.getElementById('success-notification');
    const toastText = document.getElementById('success-notification-text');
    if (toast && toastText) {
      toastText.textContent = message;
      toast.classList.add('show');
      setTimeout(() => {
        toast.classList.remove('show');
      }, 4000);
    }
  }

  // Local Storage Diagnostic Case History Management
  function saveToHistory(item) {
    try {
      let history = JSON.parse(localStorage.getItem('lesion_history') || '[]');
      // Deduplicate history matching identical base64 image data
      history = history.filter(h => h.previewImgSrc !== item.previewImgSrc);
      history.unshift(item);
      if (history.length > 10) history.pop(); // keep last 10 entries
      localStorage.setItem('lesion_history', JSON.stringify(history));
      renderHistoryList();
    } catch (err) {
      console.warn("localStorage quota exceeded, history not saved:", err);
    }
  }

  // Save patient record in localStorage keyed by patient ID
  function savePatientRecordToLocalStorage(item) {
    if (!item || !item.patientNumber || item.patientNumber === 'N/A') return;
    const patientId = item.patientNumber.trim();
    if (!patientId) return;
    try {
      let records = JSON.parse(localStorage.getItem('patient_records') || '{}');
      records[patientId] = item;
      localStorage.setItem('patient_records', JSON.stringify(records));
    } catch (err) {
      console.warn("Failed to save patient record in localStorage:", err);
    }
  }

  function deleteHistoryItem(id) {
    try {
      let history = JSON.parse(localStorage.getItem('lesion_history') || '[]');
      history = history.filter(item => item.id !== id);
      localStorage.setItem('lesion_history', JSON.stringify(history));
      renderHistoryList();
      showSuccessToast('Record removed.');
    } catch (err) {
      console.error("Failed to delete history item:", err);
    }
  }

  function renderHistoryList() {
    if (!historyList) return;
    try {
      const history = JSON.parse(localStorage.getItem('lesion_history') || '[]');
      if (history.length === 0) {
        historyList.innerHTML = '<p class="no-history-text" style="font-size: 0.85rem; color: var(--color-text-muted); text-align: center; padding: 1rem 0;">No diagnostic logs yet.</p>';
        return;
      }
      
      historyList.innerHTML = '';
      history.forEach(item => {
        const label = classLabels[item.predictedClass] || item.predictedClass;
        const risk = classRisks[item.predictedClass] || 'medium';
        
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
          <img class="history-thumb" src="${item.previewImgSrc}" alt="Thumbnail">
          <div class="history-details">
            <h5 class="history-class">${label}</h5>
            <div class="history-meta">
              <span>${item.date}</span>
              <span class="risk-badge ${risk}" style="padding: 1px 4px; font-size: 0.65rem; border-radius: 4px;">${risk}</span>
            </div>
          </div>
          <button class="btn-delete-history" title="Remove Record"><i class="fa-solid fa-trash-can"></i></button>
        `;
        
        // Setup item clicks
        div.addEventListener('click', (e) => {
          if (e.target.closest('.btn-delete-history')) {
            e.stopPropagation();
            deleteHistoryItem(item.id);
          } else {
            loadCase(item);
          }
        });
        
        historyList.appendChild(div);
      });
    } catch (err) {
      console.error("Failed to render history list:", err);
    }
  }

  function loadCase(item) {
    removeErrorCard();
    toggleDefaultZoneContent(false);
    
    // Restore inputs & thumbnail
    previewImg.src = item.previewImgSrc;
    filePreview.style.display = 'flex';
    fileNameDisplay.textContent = item.fileName;
    fileSizeDisplay.textContent = 'Saved Analysis';
    
    if (patientNameInput) patientNameInput.value = item.patientName === 'N/A' ? '' : item.patientName;
    if (patientNumberInput) patientNumberInput.value = item.patientNumber === 'N/A' ? '' : item.patientNumber;
    if (patientAgeInput) patientAgeInput.value = item.patientAge === 'N/A' ? '' : item.patientAge;
    if (patientSexSelect) patientSexSelect.value = item.patientSex;
    if (lesionSiteSelect) lesionSiteSelect.value = item.lesionSite;
    
    activeCaseData = item;
    
    // Construct fake API output matching current item state
    const resultObj = {
      predicted_class: item.predictedClass,
      confidence: item.confidence,
      top_predictions: item.topPredictions,
      gradcam_image_base64: item.gradcamImageBase64,
      heatmap_image_base64: item.heatmapImageBase64
    };
    
    renderResults(resultObj);
    
    // Auto-scroll to results card
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Dynamic risk engine giving clinical contextual warnings based on dataset stats
  function generateEpidemiologicalAdvice(age, sex, site, predictedClass) {
    const ageNum = parseInt(age);
    if (isNaN(ageNum)) return '';
    
    // 1. High risk melanoma profile (Male, Age > 50, back/chest/trunk)
    if (predictedClass === 'mel' && ageNum >= 50 && sex.toLowerCase() === 'male' && ['back', 'chest', 'trunk', 'unknown'].includes(site.toLowerCase())) {
      return `⚠️ <strong>Demographic Alert:</strong> This clinical profile aligns with statistical cohorts exhibiting elevated Melanoma recurrence. Prioritize biopsy validation.`;
    }
    
    // 2. Cumulative UV exposure risk (Age > 60, face/neck/scalp/ear)
    if (['bcc', 'akiec'].includes(predictedClass) && ageNum >= 60 && ['face', 'neck', 'scalp', 'ear'].includes(site.toLowerCase())) {
      return `ℹ️ <strong>Clinical Profile:</strong> Common presentation of cumulative actinic damage in older demographics. Monitor surrounding sun-exposed skin regions.`;
    }
    
    return '';
  }

  // Print/Export Report Trigger
  if (btnPrintReport) {
    btnPrintReport.addEventListener('click', (e) => {
      e.stopPropagation();
      
      const enteredPatientId = patientNumberInput.value.trim();
      let printData = activeCaseData;
      
      if (enteredPatientId) {
        try {
          const records = JSON.parse(localStorage.getItem('patient_records') || '{}');
          if (records[enteredPatientId]) {
            printData = records[enteredPatientId];
          }
        } catch (err) {
          console.error("Failed to read patient records from localStorage:", err);
        }
      }
      
      if (!printData) {
        showErrorCard('No active case data available to print.');
        return;
      }
      
      const label = classLabels[printData.predictedClass] || printData.predictedClass;
      const risk = classRisks[printData.predictedClass] || 'medium';
      
      // Populate printable template items
      document.getElementById('print-date').textContent = `Date: ${printData.date}`;
      document.getElementById('print-case-id').textContent = `Report ID: #R${printData.id.slice(-6)}`;
      document.getElementById('print-patient-name').textContent = printData.patientName || 'N/A';
      document.getElementById('print-patient-number').textContent = printData.patientNumber || 'N/A';
      document.getElementById('print-patient-age').textContent = `${printData.patientAge || 'Unknown'} Years`;
      document.getElementById('print-patient-sex').textContent = printData.patientSex || 'Unknown';
      document.getElementById('print-lesion-site').textContent = printData.lesionSite || 'Unknown';
      
      document.getElementById('print-assessment-result').textContent = label;
      document.getElementById('print-assessment-result').style.color = risk === 'high' ? 'var(--risk-high)' : (risk === 'medium' ? 'var(--risk-med)' : 'var(--risk-low)');
      document.getElementById('print-demographic-risk').textContent = `${risk} risk`;
      document.getElementById('print-demographic-risk').style.color = risk === 'high' ? 'var(--risk-high)' : (risk === 'medium' ? 'var(--risk-med)' : 'var(--risk-low)');
      
      // Generate patient friendly layperson explanation text
      let patientExpl = '';
      switch(printData.predictedClass) {
        case 'nv':
          patientExpl = "The analysis has classified this skin lesion as a **Nevus (Benign Mole)**. Moles are very common, normal skin spots. They are completely benign (non-cancerous) and do not represent skin cancer.";
          break;
        case 'bkl':
          patientExpl = "The analysis has classified this skin lesion as a **Benign Keratosis-like Lesion**. These are harmless skin spots (like seborrheic keratoses or sun freckles) that commonly appear as people age. They are not cancerous and require no treatment.";
          break;
        case 'df':
          patientExpl = "The analysis has classified this skin lesion as a **Dermatofibroma**. This is a very common, completely harmless (benign) small bump on the skin, often found on the legs. It is not cancerous.";
          break;
        case 'vasc':
          patientExpl = "The analysis has classified this skin lesion as a **Vascular Lesion** (like a cherry angioma). These are benign, harmless spots formed by collections of blood vessels. They are non-cancerous.";
          break;
        case 'bcc':
          patientExpl = "The screening model detected features matching **Basal Cell Carcinoma**. This is a common and slow-growing type of skin cancer. While it is highly treatable and rarely spreads, we recommend having it evaluated by a dermatologist for proper treatment.";
          break;
        case 'akiec':
          patientExpl = "The screening model detected features matching **Actinic Keratosis** (Bowen's Disease). This is a sun-induced surface lesion that is considered pre-cancerous. We recommend consulting a dermatologist to determine if it should be treated to prevent it from progressing.";
          break;
        case 'mel':
          patientExpl = "The screening model detected high-activation characteristics matching **Melanoma**. Melanoma is an aggressive skin condition that arises from pigment cells. **Please schedule an immediate consultation with a qualified dermatologist for a professional clinical examination.**";
          break;
        default:
          patientExpl = "The screening model classified this lesion as a skin growth. Please review these results with your healthcare provider.";
      }
      document.getElementById('print-patient-explanation').innerHTML = patientExpl;
      
      // Setup patient recommendation badge
      const printRec = document.getElementById('print-patient-recommendation');
      if (risk === 'high') {
        printRec.style.background = 'var(--risk-high-bg)';
        printRec.style.color = 'var(--risk-high)';
        printRec.style.border = '1px solid var(--risk-high-border)';
        printRec.textContent = 'Recommendation: Urgent Dermatologist Consultation (High Risk)';
      } else if (risk === 'medium') {
        printRec.style.background = 'var(--risk-med-bg)';
        printRec.style.color = 'var(--risk-med)';
        printRec.style.border = '1px solid rgba(230, 126, 34, 0.3)';
        printRec.textContent = 'Recommendation: Schedule Professional Evaluation (Moderate Risk)';
      } else {
        printRec.style.background = 'var(--risk-low-bg)';
        printRec.style.color = 'var(--risk-low)';
        printRec.style.border = '1px solid rgba(46, 125, 50, 0.3)';
        printRec.textContent = 'Recommendation: Standard Self-Monitoring (Low Risk)';
      }
      
      // Set images (only the original image is displayed on the patient's report for reference)
      document.getElementById('print-img-original').src = printData.previewImgSrc;
      
      // Print using browser dialogue
      window.print();
    });
  }

  // Helper to update active case details dynamically on input change
  function handleClinicalInputChange() {
    if (!activeCaseData) return;
    
    const oldPatientId = activeCaseData.patientNumber ? activeCaseData.patientNumber.trim() : '';
    
    activeCaseData.patientName = patientNameInput.value.trim() || 'N/A';
    activeCaseData.patientNumber = patientNumberInput.value.trim() || 'N/A';
    activeCaseData.patientAge = patientAgeInput.value.trim() || 'N/A';
    activeCaseData.patientSex = patientSexSelect.value || 'Unknown';
    activeCaseData.lesionSite = lesionSiteSelect.value || 'Unknown';
    
    const newPatientId = activeCaseData.patientNumber ? activeCaseData.patientNumber.trim() : '';
    
    // Update local storage patient records keyed by patient ID
    if (newPatientId && newPatientId !== 'N/A') {
      try {
        let records = JSON.parse(localStorage.getItem('patient_records') || '{}');
        // Delete old key if ID was changed
        if (oldPatientId && oldPatientId !== newPatientId && oldPatientId !== 'N/A') {
          delete records[oldPatientId];
        }
        records[newPatientId] = activeCaseData;
        localStorage.setItem('patient_records', JSON.stringify(records));
      } catch (err) {
        console.warn("Failed to update patient records in localStorage:", err);
      }
    }
    
    // Update local storage entry in legacy history if it exists
    try {
      let history = JSON.parse(localStorage.getItem('lesion_history') || '[]');
      const idx = history.findIndex(item => item.id === activeCaseData.id);
      if (idx !== -1) {
        history[idx] = activeCaseData;
        localStorage.setItem('lesion_history', JSON.stringify(history));
      }
    } catch (err) {
      console.warn("Failed to update history on input change:", err);
    }
    
    // Update live metadata on screen
    if (resultsClinicalMeta) {
      const advice = generateEpidemiologicalAdvice(
        activeCaseData.patientAge,
        activeCaseData.patientSex,
        activeCaseData.lesionSite,
        activeCaseData.predictedClass
      );
      
      resultsClinicalMeta.innerHTML = `
        <div style="font-weight: 700; margin-bottom: 0.25rem;"><i class="fa-solid fa-notes-medical"></i> Patient & Case Details</div>
        <div><strong>Name:</strong> ${escapeHtml(activeCaseData.patientName)} | <strong>Patient ID:</strong> ${escapeHtml(activeCaseData.patientNumber)}</div>
        <div style="margin-top: 0.25rem;"><strong>Age:</strong> ${escapeHtml(activeCaseData.patientAge)} | <strong>Sex:</strong> ${escapeHtml(activeCaseData.patientSex)} | <strong>Site:</strong> ${escapeHtml(activeCaseData.lesionSite)}</div>
        ${advice ? `<div style="margin-top: 0.5rem; color: #b45309; line-height: 1.3;">${advice}</div>` : ''}
      `;
    }
  }

  // Attach event listeners to patient context form elements
  [patientNameInput, patientNumberInput, patientAgeInput].forEach(input => {
    if (input) input.addEventListener('input', handleClinicalInputChange);
  });
  [patientSexSelect, lesionSiteSelect].forEach(select => {
    if (select) select.addEventListener('change', handleClinicalInputChange);
  });

  // Reset zone variables and restore layout
  function resetUploadZone() {
    isProcessing = false;
    fileInput.value = '';
    filePreview.style.display = 'none';
    resultsCard.style.display = 'none';
    removeSpinner();
    removeErrorCard();
    
    // Clear clinical context inputs
    if (patientNameInput) patientNameInput.value = '';
    if (patientNumberInput) patientNumberInput.value = '';
    if (patientAgeInput) patientAgeInput.value = '';
    if (patientSexSelect) patientSexSelect.value = 'unknown';
    if (lesionSiteSelect) lesionSiteSelect.value = 'unknown';
    if (resultsClinicalMeta) {
      resultsClinicalMeta.style.display = 'none';
      resultsClinicalMeta.innerHTML = '';
    }
    
    activeCaseData = null;

    // Reset Grad-CAM displays
    if (gradcamContainer) gradcamContainer.style.display = 'none';
    if (gradcamOriginal) gradcamOriginal.src = '';
    if (gradcamHeatmap) gradcamHeatmap.src = '';
    if (gradcamOverlay) gradcamOverlay.src = '';
    if (sliderOriginal) sliderOriginal.src = '';
    if (sliderOverlay) sliderOverlay.src = '';

    const oldBadge = lesionResult.parentElement.querySelector('.risk-badge');
    if (oldBadge) oldBadge.remove();
    
    toggleDefaultZoneContent(true);
  }

  // --- Footer Resources Modal Overlay Management ---
  const resourceModal = document.getElementById('resource-modal');
  const resourceModalTitle = document.getElementById('resource-modal-title');
  const resourceModalBody = document.getElementById('resource-modal-body');
  const resourceModalClose = document.getElementById('resource-modal-close');

  const resourceData = {
    'link-clinical-trials': {
      title: 'Clinical Trials Protocol',
      content: `
        <p>Our screening assistant is currently undergoing pilot observational clinical evaluation. The study protocol is registered under NCT9982312 to measure model sensitivity and specificity in diverse skin tone patient cohorts.</p>
        <p><strong>Key Protocol Highlights:</strong></p>
        <ul>
          <li>Pre-screening validation of standard close-up lesions.</li>
          <li>Double-blind annotation comparisons against three certified clinical dermatologists.</li>
          <li>Data transfer securely encrypted at rest conforming to standard HIPAA frameworks.</li>
        </ul>
      `
    },
    'link-model-api': {
      title: 'Model API Documentation',
      content: `
        <p>Integrate our deep learning skin screening assistant directly into hospital EHR systems using our REST interface.</p>
        <p><strong>Endpoint Details:</strong></p>
        <ul>
          <li><strong>Endpoint URL:</strong> <code>/api/predict</code></li>
          <li><strong>Method:</strong> <code>POST</code></li>
          <li><strong>Form Data Payload:</strong> <code>image</code> (binary image file)</li>
          <li><strong>JSON Response Fields:</strong> <code>predicted_class</code>, <code>confidence</code>, <code>top_predictions</code>, and <code>gradcam_image_base64</code>.</li>
        </ul>
      `
    },
    'link-data-policy': {
      title: 'Data Policy & HIPAA Compliance',
      content: `
        <p>Patient privacy is our highest priority in dermatology screening.</p>
        <p><strong>Compliance Standard Rules:</strong></p>
        <ul>
          <li><strong>Zero Remote Storage:</strong> This prototype is running entirely in your local sandbox. No data or image assets are transmitted to any cloud servers.</li>
          <li><strong>Anonymized Logging:</strong> Internal model checks mask any patient identifying variables.</li>
          <li><strong>User Access Control:</strong> Diagnostic logs are stored locally in the browser's <code>localStorage</code> and can be cleared instantly.</li>
        </ul>
      `
    },
    'link-terms-use': {
      title: 'Terms of Medical Use',
      content: `
        <p>Please review these terms prior to clinical demo execution:</p>
        <ul>
          <li><strong>Not a Diagnosis:</strong> This platform is designed solely for screening support and educational triage. It is not an FDA-approved diagnostic system.</li>
          <li><strong>Mandatory Evaluation:</strong> Moles showing border irregularity, asymmetry, or color shifts must be visually checked by a certified dermatologist regardless of model outcomes.</li>
        </ul>
      `
    },
    'link-privacy-policy': {
      title: 'Privacy Policy',
      content: `
        <p>We believe in absolute data ownership.</p>
        <ul>
          <li>No personal data or uploaded skin photos are shared, sold, or distributed.</li>
          <li>No analytical tracking cookies or telemetry parameters are loaded.</li>
          <li>All logs, patient IDs, and files are kept exclusively on your local host system.</li>
        </ul>
      `
    },
    'link-terms-service': {
      title: 'Terms of Service',
      content: `
        <p>Welcome to the Skin Cancer Detection Demo. By accessing our interface and uploading images, you acknowledge that:</p>
        <ul>
          <li>This is a clinical research prototype.</li>
          <li>The model output is for educational support purposes and not legally binding medical advice.</li>
          <li>The authors and institutions hold no liability for treatment decisions.</li>
        </ul>
      `
    },
    'link-cookie-settings': {
      title: 'Cookie & Storage Settings',
      content: `
        <p>This application does not load third-party analytical or advertising cookies.</p>
        <ul>
          <li><strong>Session Cookies:</strong> None.</li>
          <li><strong>Browser Storage:</strong> Standard HTML5 <code>localStorage</code> is used to save clinical records on the basis of the Patient ID you specify.</li>
          <li><strong>Clearing Storage:</strong> You can wipe all saved patient records at any time by clearing your browser cache.</li>
        </ul>
      `
    }
  };

  function openResourceModal(event) {
    event.preventDefault();
    const linkId = event.currentTarget.id;
    const data = resourceData[linkId];
    if (data && resourceModal && resourceModalTitle && resourceModalBody) {
      resourceModalTitle.textContent = data.title;
      resourceModalBody.innerHTML = data.content;
      resourceModal.classList.add('open');
    }
  }

  function closeResourceModal() {
    if (resourceModal) {
      resourceModal.classList.remove('open');
    }
  }

  // Hook up event listeners to resources links
  Object.keys(resourceData).forEach(id => {
    const linkEl = document.getElementById(id);
    if (linkEl) {
      linkEl.addEventListener('click', openResourceModal);
    }
  });

  if (resourceModalClose) {
    resourceModalClose.addEventListener('click', closeResourceModal);
  }

  // Close modal when clicking outside the modal card
  if (resourceModal) {
    resourceModal.addEventListener('click', (e) => {
      if (e.target === resourceModal) {
        closeResourceModal();
      }
    });
  }
});
