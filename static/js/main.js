/**
 * Scientific Terminal background stars and shared utilities.
 * Design: Sparse, crisp, sharp 1px points, concentrated toward the edges.
 * No blur, no glowing bloom. Minimal and quiet.
 */

document.addEventListener('DOMContentLoaded', () => {
  initSparseStarfield();
});

function initSparseStarfield() {
  const canvas = document.getElementById('star-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    createSparseStars();
  });

  // Significantly reduced count: ~45 stars total
  const numStars = 48;
  let stars = [];

  function createSparseStars() {
    stars = [];
    const centerX = width / 2;
    const centerY = height / 2;
    const maxDist = Math.sqrt(centerX * centerX + centerY * centerY);

    while (stars.length < numStars) {
      const x = Math.random() * width;
      const y = Math.random() * height;

      // Distance from center: prefer outer regions
      const dx = x - centerX;
      const dy = y - centerY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const edgeWeight = dist / maxDist;

      // Reject some points near the center to keep center clean
      if (Math.random() > edgeWeight * 1.4 && dist < maxDist * 0.45) {
        continue;
      }

      stars.push({
        x: Math.floor(x),
        y: Math.floor(y),
        size: 1, // Crisp 1px point
        alpha: Math.random() * 0.45 + 0.15,
        twinkleRate: (Math.random() * 0.005 + 0.002) * (Math.random() > 0.5 ? 1 : -1)
      });
    }
  }

  createSparseStars();

  function render() {
    ctx.clearRect(0, 0, width, height);

    // Render crisp 1px points without blur/shadows
    ctx.fillStyle = '#ffffff';
    for (let i = 0; i < stars.length; i++) {
      const s = stars[i];
      s.alpha += s.twinkleRate;
      if (s.alpha > 0.65 || s.alpha < 0.12) {
        s.twinkleRate = -s.twinkleRate;
      }

      ctx.fillStyle = `rgba(255, 255, 255, ${Math.max(0.08, Math.min(0.7, s.alpha))})`;
      ctx.fillRect(s.x, s.y, 1, 1);
    }

    requestAnimationFrame(render);
  }

  render();
}

/**
 * Technical terminal notification helper
 */
function showToast(message, type = 'info') {
  let toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    toastContainer.style.position = 'fixed';
    toastContainer.style.bottom = '20px';
    toastContainer.style.right = '20px';
    toastContainer.style.zIndex = '9999';
    toastContainer.style.display = 'flex';
    toastContainer.style.flexDirection = 'column';
    toastContainer.style.gap = '8px';
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement('div');
  const borderCol = type === 'success' ? '#22c55e' : type === 'error' ? '#ef4444' : '#666666';
  const prefix = type === 'success' ? '[ OK ]' : type === 'error' ? '[ ERR ]' : '[ SYS ]';

  toast.style.background = '#0a0a0a';
  toast.style.color = '#ffffff';
  toast.style.border = `1px solid ${borderCol}`;
  toast.style.borderRadius = '0';
  toast.style.padding = '8px 14px';
  toast.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
  toast.style.fontSize = '0.8rem';
  toast.style.boxShadow = 'none';
  toast.style.display = 'flex';
  toast.style.alignItems = 'center';
  toast.style.gap = '10px';
  toast.innerHTML = `<span style="color: ${borderCol}; font-weight: bold;">${prefix}</span> <span>${message}</span>`;

  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 250);
  }, 3500);
}
