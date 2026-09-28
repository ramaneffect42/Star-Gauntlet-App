/**
 * Model Training and Evaluation script:
 * - Trigger training via /api/model/train
 * - Display real metrics (Accuracy, Precision, Recall, F1)
 * - Render 6x6 confusion matrix
 * - Render feature importance chart
 * - Render model comparison benchmark table
 */

let featChart = null;

document.addEventListener('DOMContentLoaded', () => {
  loadModelStatus();

  const btnTrain = document.getElementById('btnTrainModel');
  if (btnTrain) {
    btnTrain.addEventListener('click', handleTrainModel);
  }
});

async function loadModelStatus() {
  try {
    const res = await fetch('/api/model/status');
    const json = await res.json();
    if (json.status !== 'ready') {
      showToast('Model not trained yet. Click "Train Model".', 'info');
      return;
    }

    updateTrainingUI(json);
  } catch (err) {
    console.error('Error fetching model status:', err);
    showToast('Could not fetch model diagnostics: ' + err.message, 'error');
  }
}

async function handleTrainModel() {
  const btn = document.getElementById('btnTrainModel');
  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Training Pipeline...`;

  const badge = document.getElementById('trainStatusBadge');
  badge.textContent = 'Training in progress...';
  badge.style.background = 'rgba(245, 158, 11, 0.2)';
  badge.style.color = '#fbbf24';

  try {
    const res = await fetch('/api/model/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const json = await res.json();

    if (json.status !== 'success') throw new Error(json.message);

    showToast('Model trained and evaluated successfully!', 'success');
    updateTrainingUI(json);

    badge.textContent = 'Trained & Saved';
    badge.style.background = 'rgba(16, 185, 129, 0.2)';
    badge.style.color = '#34d399';
  } catch (err) {
    console.error('Training failed:', err);
    showToast('Training failed: ' + err.message, 'error');
    badge.textContent = 'Training Error';
    badge.style.background = 'rgba(239, 68, 68, 0.2)';
    badge.style.color = '#f87171';
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

function updateTrainingUI(data) {
  const metrics = data.metrics;

  // Metric cards
  document.getElementById('metric-accuracy').textContent = `${metrics.accuracy_percentage}%`;
  document.getElementById('metric-precision').textContent = `${metrics.precision_percentage}%`;
  document.getElementById('metric-recall').textContent = `${metrics.recall_percentage}%`;
  document.getElementById('metric-f1').textContent = `${metrics.f1_percentage}%`;

  // Render Confusion Matrix
  if (data.confusion_matrix && data.class_labels) {
    renderConfusionMatrix(data.confusion_matrix, data.class_labels);
  }

  // Render Feature Importance Chart
  if (data.feature_importance_high_level) {
    renderFeatureImportanceChart(data.feature_importance_high_level);
  }

  // Render Model Comparison
  if (data.comparison) {
    renderComparisonTable(data.comparison);
  }
}

function renderConfusionMatrix(matrix, labels) {
  const container = document.getElementById('matrixContainer');
  container.innerHTML = '';

  const table = document.createElement('table');
  table.className = 'matrix-table';

  // Header row (Predicted classes)
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  headerRow.innerHTML = `<th style="text-align: right; color: var(--accent-cyan); font-size: 0.72rem;">Actual \\ Pred</th>`;
  
  labels.forEach(lbl => {
    const th = document.createElement('th');
    th.textContent = lbl;
    th.title = `Predicted: ${lbl}`;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Body rows (Actual classes)
  const tbody = document.createElement('tbody');
  matrix.forEach((row, rowIdx) => {
    const tr = document.createElement('tr');
    
    // Row label (Actual class)
    const labelCell = document.createElement('th');
    labelCell.style.textAlign = 'right';
    labelCell.textContent = labels[rowIdx];
    labelCell.title = `Actual: ${labels[rowIdx]}`;
    tr.appendChild(labelCell);

    // Cells
    row.forEach((val, colIdx) => {
      const td = document.createElement('td');
      td.className = 'matrix-cell';
      td.textContent = val;

      if (rowIdx === colIdx) {
        if (val > 0) td.classList.add('correct');
        else td.classList.add('zero');
      } else {
        if (val > 0) td.classList.add('error');
        else td.classList.add('zero');
      }

      td.title = `Actual: ${labels[rowIdx]} | Predicted: ${labels[colIdx]} | Count: ${val}`;
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  container.appendChild(table);
}

function renderFeatureImportanceChart(features) {
  const ctx = document.getElementById('featureImportanceChart').getContext('2d');
  
  const labels = features.map(f => f.feature);
  const values = features.map(f => f.percentage);

  if (featChart) featChart.destroy();

  featChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Contribution (%)',
        data: values,
        backgroundColor: [
          'rgba(56, 189, 248, 0.85)',
          'rgba(168, 85, 247, 0.85)',
          'rgba(245, 158, 11, 0.85)',
          'rgba(236, 72, 153, 0.85)',
          'rgba(16, 185, 129, 0.85)',
          'rgba(59, 130, 246, 0.85)'
        ],
        borderColor: [
          '#38bdf8',
          '#a855f7',
          '#f59e0b',
          '#ec4899',
          '#10b981',
          '#3b82f6'
        ],
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Importance: ${ctx.raw}%`
          }
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          suggestedMax: 40,
          ticks: {
            color: '#94a3b8',
            callback: (v) => v + '%'
          },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          ticks: { color: '#f8fafc', font: { weight: '600' } },
          grid: { display: false }
        }
      }
    }
  });
}

function renderComparisonTable(comparison) {
  const tbody = document.getElementById('comparisonTbody');
  tbody.innerHTML = '';

  for (const [modelName, data] of Object.entries(comparison)) {
    const tr = document.createElement('tr');
    const isRF = modelName === 'Random Forest';
    
    if (isRF) {
      tr.style.background = 'rgba(56, 189, 248, 0.06)';
    }

    tr.innerHTML = `
      <td>
        <strong>${modelName}</strong>
        ${isRF ? '<span style="margin-left: 0.5rem; font-size: 0.72rem; padding: 0.15rem 0.45rem; background: rgba(56, 189, 248, 0.2); color: #38bdf8; border-radius: 4px;">Primary</span>' : ''}
      </td>
      <td><code>${data.type}</code></td>
      <td><strong style="color: var(--accent-emerald);">${data.accuracy}%</strong></td>
      <td>${data.precision}%</td>
      <td>${data.recall}%</td>
      <td>${data.f1_score}%</td>
      <td>
        ${isRF 
          ? '<span class="badge-class" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4);"><i class="fa-solid fa-check"></i> Selected & Deployed</span>' 
          : '<span class="badge-class" style="background: rgba(255, 255, 255, 0.05); color: var(--text-dim); border: 1px solid rgba(255, 255, 255, 0.1);">Benchmark Baseline</span>'}
      </td>
    `;
    tbody.appendChild(tr);
  }
}
