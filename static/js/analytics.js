/**
 * Analytics Dashboard script:
 * - Fetches grouped astronomical dataset points from /api/analytics/charts
 * - Initializes 5 distinct interactive Chart.js visualizations
 */

const classColors = {
  'Brown Dwarf': '#b45309',
  'Red Dwarf': '#ef4444',
  'White Dwarf': '#38bdf8',
  'Main Sequence': '#f59e0b',
  'Supergiant': '#ec4899',
  'Hypergiant': '#a855f7'
};

document.addEventListener('DOMContentLoaded', () => {
  loadAnalyticsData();
});

async function loadAnalyticsData() {
  try {
    const res = await fetch('/api/analytics/charts');
    const json = await res.json();
    if (json.status !== 'success') throw new Error(json.message);

    renderChart1(json.distribution);
    renderChart5(json.feature_importance);
    renderChart4HrDiagram(json.temp_vs_mag, json.classes);
    renderChart2TempLum(json.temp_vs_lum, json.classes);
    renderChart3TempRad(json.temp_vs_rad, json.classes);
  } catch (err) {
    console.error('Error loading analytics:', err);
    showToast('Failed to load analytics: ' + err.message, 'error');
  }
}

// Chart 1: Distribution
function renderChart1(distribution) {
  const ctx = document.getElementById('chartDistribution').getContext('2d');
  const labels = distribution.map(d => d.class);
  const data = distribution.map(d => d.count);
  const colors = labels.map(l => classColors[l] || '#38bdf8');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: colors.map(c => c + 'cc'),
        borderColor: colors,
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.raw} stars`
          }
        }
      },
      scales: {
        x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
        y: { beginAtZero: true, suggestedMax: 50, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
      }
    }
  });
}

// Chart 5: Feature Importance
function renderChart5(featureImportance) {
  const ctx = document.getElementById('chartFeatureImportance').getContext('2d');
  const labels = featureImportance.map(f => f.feature);
  const values = featureImportance.map(f => f.percentage);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: 'rgba(168, 85, 247, 0.75)',
        borderColor: '#a855f7',
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
        x: { beginAtZero: true, ticks: { color: '#94a3b8', callback: v => v + '%' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
        y: { ticks: { color: '#f8fafc', font: { size: 10 } }, grid: { display: false } }
      }
    }
  });
}

// Helper to build datasets for grouped scatter plots
function buildScatterDatasets(groupedData, classes) {
  return classes.map(cls => {
    const color = classColors[cls] || '#38bdf8';
    return {
      label: cls,
      data: groupedData[cls] || [],
      backgroundColor: color + 'cc',
      borderColor: color,
      borderWidth: 1,
      pointRadius: 4.5,
      pointHoverRadius: 7
    };
  });
}

// Chart 4: HR Diagram (Absolute Magnitude vs Temperature)
function renderChart4HrDiagram(tempVsMag, classes) {
  const ctx = document.getElementById('chartHrDiagram').getContext('2d');
  const datasets = buildScatterDatasets(tempVsMag, classes);

  new Chart(ctx, {
    type: 'scatter',
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#94a3b8', boxWidth: 12, padding: 15, font: { size: 11 } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: Temp = ${ctx.raw.x.toLocaleString()} K, Abs Mag = ${ctx.raw.y} Mv`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          title: { display: true, text: 'Effective Surface Temperature (K)', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          reverse: true, // Inverted: lower/negative magnitudes at top as in astronomy!
          title: { display: true, text: 'Absolute Magnitude (Mv) [Inverted: Brighter ↑]', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });
}

// Chart 2: Temperature vs Luminosity (Log scale)
function renderChart2TempLum(tempVsLum, classes) {
  const ctx = document.getElementById('chartTempVsLum').getContext('2d');
  const datasets = buildScatterDatasets(tempVsLum, classes);

  new Chart(ctx, {
    type: 'scatter',
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { color: '#94a3b8', boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: Temp = ${ctx.raw.x.toLocaleString()} K, Lum = ${ctx.raw.y.toLocaleString()} L☉`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Temperature (K)', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'Luminosity (L/L☉) [Log Scale]', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });
}

// Chart 3: Temperature vs Radius (Log scale)
function renderChart3TempRad(tempVsRad, classes) {
  const ctx = document.getElementById('chartTempVsRad').getContext('2d');
  const datasets = buildScatterDatasets(tempVsRad, classes);

  new Chart(ctx, {
    type: 'scatter',
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { color: '#94a3b8', boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: Temp = ${ctx.raw.x.toLocaleString()} K, Radius = ${ctx.raw.y.toLocaleString()} R☉`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Temperature (K)', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'Radius (R/R☉) [Log Scale]', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });
}
