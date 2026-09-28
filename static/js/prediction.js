/**
 * Star Prediction script:
 * - Load and handle Demo Mode presets
 * - Execute real-time prediction via /api/model/predict
 * - Render predicted star type, confidence bars, and astrophysical reasoning
 */

const starClassStyles = {
  'Brown Dwarf': { badge: 'brown-dwarf', color: '#b45309' },
  'Red Dwarf': { badge: 'red-dwarf', color: '#ef4444' },
  'White Dwarf': { badge: 'white-dwarf', color: '#38bdf8' },
  'Main Sequence': { badge: 'main-sequence', color: '#f59e0b' },
  'Supergiant': { badge: 'supergiant', color: '#ec4899' },
  'Hypergiant': { badge: 'hypergiant', color: '#a855f7' }
};

document.addEventListener('DOMContentLoaded', () => {
  loadDemoPresets();

  const form = document.getElementById('predictionForm');
  if (form) {
    form.addEventListener('submit', handlePrediction);
  }

  const btnReset = document.getElementById('btnReset');
  if (btnReset) {
    btnReset.addEventListener('click', resetForm);
  }
});

async function loadDemoPresets() {
  const container = document.getElementById('demoPresetsContainer');
  try {
    const res = await fetch('/api/demo/examples');
    const json = await res.json();
    if (json.status !== 'success') throw new Error(json.message);

    container.innerHTML = '';
    json.examples.forEach(ex => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn-preset';
      const style = starClassStyles[ex.target_class] || { color: '#38bdf8' };
      
      btn.innerHTML = `
        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${style.color};"></span>
        ${ex.target_class}
      `;
      btn.title = ex.note;
      btn.addEventListener('click', () => {
        populateForm(ex);
        // Automatically predict for instant demo flow!
        executePrediction(ex);
      });
      container.appendChild(btn);
    });
  } catch (err) {
    console.error('Error loading presets:', err);
    container.innerHTML = `<span style="color: var(--text-dim);">Presets unavailable</span>`;
  }
}

function populateForm(data) {
  document.getElementById('inputTemperature').value = data.temperature;
  document.getElementById('inputLuminosity').value = data.luminosity;
  document.getElementById('inputRadius').value = data.radius;
  document.getElementById('inputMagnitude').value = data.absolute_magnitude;
  
  const selSpectral = document.getElementById('inputSpectralClass');
  if (selSpectral) selSpectral.value = data.spectral_class;

  const selColor = document.getElementById('inputColor');
  if (selColor) {
    // Attempt exact or normalized match
    let matched = false;
    for (let opt of selColor.options) {
      if (opt.value.toLowerCase() === data.star_color.toLowerCase().trim()) {
        selColor.value = opt.value;
        matched = true;
        break;
      }
    }
    if (!matched) selColor.value = 'Red';
  }

  showToast(`Loaded ${data.target_class || 'preset'} example values.`, 'info');
}

function resetForm() {
  document.getElementById('predictionForm').reset();
  document.getElementById('resultCard').style.display = 'none';
  document.getElementById('explanationCard').style.display = 'none';
  document.getElementById('emptyResultPlaceholder').style.display = 'block';
  showToast('Form reset.', 'info');
}

async function handlePrediction(e) {
  e.preventDefault();
  
  const payload = {
    temperature: parseFloat(document.getElementById('inputTemperature').value),
    luminosity: parseFloat(document.getElementById('inputLuminosity').value),
    radius: parseFloat(document.getElementById('inputRadius').value),
    absolute_magnitude: parseFloat(document.getElementById('inputMagnitude').value),
    spectral_class: document.getElementById('inputSpectralClass').value,
    star_color: document.getElementById('inputColor').value
  };

  await executePrediction(payload);
}

async function executePrediction(payload) {
  const btn = document.getElementById('btnPredict');
  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Classifying...`;

  try {
    const res = await fetch('/api/model/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const json = await res.json();
    if (json.status !== 'success') throw new Error(json.message);

    displayPredictionResult(json.data);
    showToast(`Prediction: ${json.data.predicted_type_name} (${json.data.confidence_percentage}%)`, 'success');
  } catch (err) {
    console.error('Prediction failed:', err);
    showToast('Prediction failed: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

function displayPredictionResult(data) {
  // Hide empty state, show results
  document.getElementById('emptyResultPlaceholder').style.display = 'none';
  document.getElementById('resultCard').style.display = 'block';
  document.getElementById('explanationCard').style.display = 'block';

  const typeName = data.predicted_type_name;
  const style = starClassStyles[typeName] || { badge: 'main-sequence', color: '#38bdf8' };

  // Set predicted type name & badge
  const typeElem = document.getElementById('resultTypeName');
  typeElem.textContent = typeName;
  typeElem.style.color = style.color;

  const badgeElem = document.getElementById('resultBadge');
  badgeElem.className = `badge-class ${style.badge}`;
  badgeElem.textContent = `Type ${data.predicted_type_id} &bull; ${typeName}`;

  // Description
  const stellarInfo = data.stellar_info || {};
  document.getElementById('resultClassSummary').textContent = stellarInfo.summary || '';

  // Top confidence badge
  document.getElementById('topConfidenceBadge').textContent = `${data.confidence_percentage}% Confidence`;

  // Render probability distribution bars
  const probContainer = document.getElementById('probListContainer');
  probContainer.innerHTML = '';

  data.probabilities.forEach(item => {
    const isWinner = item.class_name === typeName;
    const itemElem = document.createElement('div');
    itemElem.className = 'prob-item';

    itemElem.innerHTML = `
      <div class="prob-header">
        <span style="color: ${isWinner ? '#ffffff' : 'var(--text-muted)'}; font-weight: ${isWinner ? '700' : '500'};">
          ${item.class_name}
        </span>
        <span style="font-family: var(--font-mono); color: ${isWinner ? style.color : 'var(--text-muted)'}; font-weight: 700;">
          ${item.percentage}%
        </span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${item.percentage}%; background: ${item.color || style.color};"></div>
      </div>
    `;
    probContainer.appendChild(itemElem);
  });

  // Submitted Features Table
  const feat = data.input_features;
  const tbody = document.getElementById('submittedFeaturesTbody');
  tbody.innerHTML = `
    <tr><td><strong>Temperature:</strong> <code>${feat.temperature.toLocaleString()} K</code></td><td><strong>Luminosity:</strong> <code>${feat.luminosity.toLocaleString()} L☉</code></td></tr>
    <tr><td><strong>Radius:</strong> <code>${feat.radius.toLocaleString()} R☉</code></td><td><strong>Absolute Magnitude:</strong> <code>${feat.absolute_magnitude} Mv</code></td></tr>
    <tr><td><strong>Spectral Class:</strong> <span style="font-weight:700; color:var(--accent-cyan);">${feat.spectral_class}</span></td><td><strong>Star Colour:</strong> ${feat.star_color}</td></tr>
  `;

  // Explanation fields
  const expl = data.explanation || {};
  document.getElementById('explainHrPosition').textContent = expl.hr_position || '';
  document.getElementById('explainMlReasoning').textContent = expl.ml_reasoning || '';
  document.getElementById('explainScientificNote').textContent = expl.scientific_note || '';
}
