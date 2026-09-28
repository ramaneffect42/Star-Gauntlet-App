/**
 * Dataset Explorer JavaScript logic:
 * - Load summary statistics
 * - Render class distribution breakdown bars
 * - Render and paginate/search 240 stellar records
 */

let allRecords = [];
let filteredRecords = [];
let currentPage = 1;
const pageSize = 20;

const units = {
  'Temperature (K)': 'Kelvin [ K ]',
  'Luminosity (L/Lo)': 'Solar Lum [ L☉ ]',
  'Radius (R/Ro)': 'Solar Rad [ R☉ ]',
  'Absolute magnitude (Mv)': 'Magnitude [ Mv ]'
};

document.addEventListener('DOMContentLoaded', () => {
  loadDatasetSummary();
  loadDatasetRecords();

  document.getElementById('tableSearch').addEventListener('input', applyFilters);
  document.getElementById('classFilter').addEventListener('change', applyFilters);
  document.getElementById('btnPrevPage').addEventListener('click', () => changePage(-1));
  document.getElementById('btnNextPage').addEventListener('click', () => changePage(1));
});

async function loadDatasetSummary() {
  try {
    const res = await fetch('/api/dataset/summary');
    const json = await res.json();
    if (json.status !== 'success') throw new Error(json.message);

    const data = json.data;

    // Numerical statistics table
    const tbody = document.getElementById('stats-tbody');
    tbody.innerHTML = '';

    for (const [colName, stats] of Object.entries(data.numerical_stats)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="color: var(--text-white); font-weight: 700;">${colName.toUpperCase()}</td>
        <td style="color: var(--text-gray-2);">${units[colName] || '--'}</td>
        <td>${stats.mean.toLocaleString()}</td>
        <td>${stats.std.toLocaleString()}</td>
        <td>${stats.min.toLocaleString()}</td>
        <td>${stats.max.toLocaleString()}</td>
      `;
      tbody.appendChild(tr);
    }

    // Class Distribution Breakdown
    renderClassDistBars(data.class_distribution);
  } catch (err) {
    console.error('Error loading summary:', err);
    showToast('Failed to load dataset summary: ' + err.message, 'error');
  }
}

function renderClassDistBars(distribution) {
  const container = document.getElementById('classDistBars');
  container.innerHTML = '';

  for (const [cls, count] of Object.entries(distribution)) {
    const item = document.createElement('div');
    item.style.background = '#000000';
    item.style.border = '1px solid var(--border-subtle)';
    item.style.padding = '0.75rem';

    item.innerHTML = `
      <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
        <span style="color: var(--text-white); font-weight: 700;">${cls.toUpperCase()}</span>
        <span style="color: var(--accent-green);">${count} OBS (16.67%)</span>
      </div>
      <div style="width: 100%; height: 4px; background: #111;">
        <div style="width: ${(count / 40) * 100}%; height: 100%; background: var(--text-white);"></div>
      </div>
    `;
    container.appendChild(item);
  }
}

async function loadDatasetRecords() {
  try {
    const res = await fetch('/api/dataset/records');
    const json = await res.json();
    if (json.status !== 'success') throw new Error(json.message);

    allRecords = json.records;
    filteredRecords = [...allRecords];
    currentPage = 1;
    renderTable();
  } catch (err) {
    console.error('Error loading records:', err);
    document.getElementById('records-tbody').innerHTML = `
      <tr><td colspan="8" style="text-align: center; color: var(--accent-red);">Failed to load records.</td></tr>
    `;
  }
}

function applyFilters() {
  const query = document.getElementById('tableSearch').value.toLowerCase().trim();
  const classFilter = document.getElementById('classFilter').value;

  filteredRecords = allRecords.filter(row => {
    const matchesClass = (classFilter === 'all') || (row['Star category'] === classFilter);
    if (!matchesClass) return false;

    if (!query) return true;
    const searchString = `${row['Temperature (K)']} ${row['Luminosity (L/Lo)']} ${row['Radius (R/Ro)']} ${row['Absolute magnitude (Mv)']} ${row['Spectral Class']} ${row['Star color']} ${row['Star category']}`.toLowerCase();
    return searchString.includes(query);
  });

  currentPage = 1;
  renderTable();
}

function renderTable() {
  const tbody = document.getElementById('records-tbody');
  tbody.innerHTML = '';

  const total = filteredRecords.length;
  const start = (currentPage - 1) * pageSize;
  const end = Math.min(start + pageSize, total);
  const pageItems = filteredRecords.slice(start, end);

  if (pageItems.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-gray-3); padding: 2rem;">No matching observations found.</td></tr>`;
  } else {
    pageItems.forEach((row, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="color: var(--text-gray-3);">${start + i + 1}</td>
        <td>${Number(row['Temperature (K)']).toLocaleString()}</td>
        <td>${Number(row['Luminosity (L/Lo)']).toLocaleString()}</td>
        <td>${Number(row['Radius (R/Ro)']).toLocaleString()}</td>
        <td>${row['Absolute magnitude (Mv)']}</td>
        <td style="color: var(--text-white); font-weight: 700;">${row['Spectral Class']}</td>
        <td style="color: var(--text-gray-1);">${row['Star color']}</td>
        <td style="color: var(--text-white); font-weight: 600;">[ ${row['Star category'].toUpperCase()} ]</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Update pagination info
  document.getElementById('recordCountInfo').textContent = `Showing ${total > 0 ? start + 1 : 0} to ${end} of ${total} observations`;
  const totalPages = Math.ceil(total / pageSize) || 1;
  document.getElementById('pageIndicator').textContent = `PAGE ${currentPage} OF ${totalPages}`;
  document.getElementById('btnPrevPage').disabled = currentPage <= 1;
  document.getElementById('btnNextPage').disabled = currentPage >= totalPages;
}

function changePage(delta) {
  const totalPages = Math.ceil(filteredRecords.length / pageSize) || 1;
  currentPage += delta;
  if (currentPage < 1) currentPage = 1;
  if (currentPage > totalPages) currentPage = totalPages;
  renderTable();
}
