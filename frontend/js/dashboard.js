/**
 * Dashboard JavaScript — Privacy-Preserving News Recommender
 *
 * Sections: Overview, Recommend, Browse News, Privacy
 * API base: http://localhost:5000/api
 */

const API = 'http://localhost:5000/api';

/* ── State ──────────────────────────────────────────────────────────── */
const state = {
  newsPage: 1,
  newsPerPage: 20,
  newsCategory: '',
  newsQuery: '',
  newsTotal: 0,
};

/* ── Helpers ────────────────────────────────────────────────────────── */
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

async function apiFetch(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function fmt(n, decimals = 4) {
  return typeof n === 'number' ? n.toFixed(decimals) : '—';
}

/* ── Navigation ─────────────────────────────────────────────────────── */
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    const target = item.dataset.section;
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    item.classList.add('active');
    document.getElementById('section-' + target).classList.add('active');
    if (target === 'overview') loadOverview();
    if (target === 'news') loadNews();
  });
});

/* ── Health check ───────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const data = await apiFetch('/health');
    if (data.status === 'ok') {
      document.getElementById('api-status').textContent = 'API Online';
      document.querySelector('.status-dot').style.background = 'var(--accent-emerald)';
    }
  } catch {
    document.getElementById('api-status').textContent = 'API Offline';
    document.querySelector('.status-dot').style.background = 'var(--accent-rose)';
    document.querySelector('.status-dot').style.boxShadow = '0 0 8px var(--accent-rose)';
  }
}

/* ══════════════════════════════════════════════════════════════════════
   SECTION 1 — OVERVIEW
   ══════════════════════════════════════════════════════════════════════ */
let charts = {};

async function loadOverview() {
  try {
    const [summary, history] = await Promise.all([
      apiFetch('/metrics/summary'),
      apiFetch('/metrics/'),
    ]);
    renderSummaryCards(summary);
    renderCharts(history);
  } catch (err) {
    toast('Could not load metrics — is the API running?', 'error');
    console.error(err);
    // Render with embedded fallback data
    renderSummaryCards(FALLBACK_SUMMARY);
    renderCharts(FALLBACK_HISTORY);
  }
}

function renderSummaryCards(summary) {
  const c = summary.centralized || {};
  const f = summary.federated   || {};

  document.getElementById('metric-cen-auc').textContent    = fmt(c.auc);
  document.getElementById('metric-cen-ndcg10').textContent = fmt(c.ndcg10);
  document.getElementById('metric-fed-auc').textContent    = fmt(f.auc);
  document.getElementById('metric-fed-ndcg10').textContent = fmt(f.ndcg10);
  document.getElementById('metric-cen-mrr').textContent    = fmt(c.mrr);
  document.getElementById('metric-fed-mrr').textContent    = fmt(f.mrr);
}

function renderCharts(history) {
  const cen = history.centralized || [];
  const fed = history.federated   || [];

  const cenLabels = cen.map(r => `Ep ${r.epoch}`);
  const fedLabels = fed.map(r => `Rd ${r.round}`);

  const chartDefs = [
    {
      id: 'chart-loss',
      title: 'Training Loss',
      datasets: [
        { label: 'Centralized', data: cen.map(r => r.train_loss), color: '#6366f1', labels: cenLabels },
        { label: 'Federated',   data: fed.map(r => r.train_loss), color: '#10b981', labels: fedLabels },
      ],
    },
    {
      id: 'chart-auc',
      title: 'AUC',
      datasets: [
        { label: 'Centralized', data: cen.map(r => r.AUC), color: '#6366f1', labels: cenLabels },
        { label: 'Federated',   data: fed.map(r => r.AUC), color: '#10b981', labels: fedLabels },
      ],
    },
    {
      id: 'chart-ndcg5',
      title: 'nDCG@5',
      datasets: [
        { label: 'Centralized', data: cen.map(r => r['nDCG@5']), color: '#6366f1', labels: cenLabels },
        { label: 'Federated',   data: fed.map(r => r['nDCG@5']), color: '#10b981', labels: fedLabels },
      ],
    },
    {
      id: 'chart-mrr',
      title: 'MRR',
      datasets: [
        { label: 'Centralized', data: cen.map(r => r.MRR), color: '#6366f1', labels: cenLabels },
        { label: 'Federated',   data: fed.map(r => r.MRR), color: '#10b981', labels: fedLabels },
      ],
    },
  ];

  // Use max-length labels (federated usually same count)
  chartDefs.forEach(def => {
    if (charts[def.id]) charts[def.id].destroy();
    const ctx = document.getElementById(def.id).getContext('2d');
    const maxLen = Math.max(...def.datasets.map(d => d.labels.length));
    const sharedLabels = def.datasets.reduce((acc, d) => d.labels.length === maxLen ? d.labels : acc, def.datasets[0].labels);
    charts[def.id] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: sharedLabels,
        datasets: def.datasets.map(d => ({
          label: d.label,
          data: d.data,
          borderColor: d.color,
          backgroundColor: d.color + '22',
          borderWidth: 2.5,
          pointRadius: 5,
          pointHoverRadius: 7,
          fill: true,
          tension: 0.4,
        })),
      },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { size: 12 } } },
          tooltip: { backgroundColor: 'rgba(17,24,39,0.95)', titleColor: '#f1f5f9', bodyColor: '#94a3b8' },
        },
        scales: {
          x: { ticks: { color: '#475569', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#475569', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
        },
      },
    });
  });
}

/* ══════════════════════════════════════════════════════════════════════
   SECTION 2 — RECOMMEND
   ══════════════════════════════════════════════════════════════════════ */
document.getElementById('btn-recommend').addEventListener('click', async () => {
  const rawHistory  = document.getElementById('history-input').value.trim();
  const topK        = parseInt(document.getElementById('topk-input').value) || 10;
  const history     = rawHistory.split(/[\n,\s]+/).filter(Boolean);

  const resultBox = document.getElementById('rec-results');
  resultBox.innerHTML = '<div class="loading-state"><span class="spinner"></span> Getting recommendations…</div>';

  try {
    const data = await fetch(API + '/recommend/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history, top_k: topK }),
    }).then(r => r.json());

    renderRecommendations(data.recommendations || []);
  } catch (err) {
    toast('Recommendation request failed — check API.', 'error');
    resultBox.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><p>Could not fetch recommendations.</p></div>';
  }
});

document.getElementById('btn-demo').addEventListener('click', async () => {
  const resultBox = document.getElementById('rec-results');
  resultBox.innerHTML = '<div class="loading-state"><span class="spinner"></span> Loading demo recommendations…</div>';
  try {
    const data = await apiFetch('/recommend/demo');
    if (data.history_seed) {
      document.getElementById('history-input').value = data.history_seed.join('\n');
    }
    renderRecommendations(data.recommendations || []);
    toast('Demo loaded ✓', 'success');
  } catch {
    toast('Demo failed — is the API running?', 'error');
    resultBox.innerHTML = '<div class="empty-state"><div class="empty-icon">🎬</div><p>Start the backend API to see live demo.</p></div>';
  }
});

function renderRecommendations(recs) {
  const resultBox = document.getElementById('rec-results');
  if (!recs.length) {
    resultBox.innerHTML = '<div class="empty-state"><div class="empty-icon">🔍</div><p>No recommendations found.</p></div>';
    return;
  }
  resultBox.innerHTML = recs.map((r, i) => `
    <div class="rec-card" style="animation-delay:${i * 0.04}s">
      <div class="rec-rank">#${r.rank}</div>
      <div class="rec-body">
        <div class="rec-title">${escHtml(r.title || r.news_id)}</div>
        <div class="rec-meta">
          <span class="rec-badge">${escHtml(r.category)}</span>
          ${r.subcategory ? `<span style="color:var(--text-muted);font-size:12px">${escHtml(r.subcategory)}</span>` : ''}
          <span style="color:var(--text-muted);font-size:12px">${escHtml(r.news_id)}</span>
        </div>
        ${r.abstract ? `<p style="font-size:12px;color:var(--text-secondary);margin-top:6px;line-height:1.5">${escHtml(r.abstract)}</p>` : ''}
      </div>
      <div class="rec-score">${r.score.toFixed(4)}</div>
    </div>
  `).join('');
}

/* ══════════════════════════════════════════════════════════════════════
   SECTION 3 — BROWSE NEWS
   ══════════════════════════════════════════════════════════════════════ */
let newsDebounceTimer;

async function loadNews() {
  const grid = document.getElementById('news-grid');
  grid.innerHTML = '<div class="loading-state"><span class="spinner"></span> Loading articles…</div>';

  try {
    await loadCategoryFilter();
    await fetchAndRenderNews();
  } catch (err) {
    toast('Could not load news catalogue.', 'error');
    grid.innerHTML = '<div class="empty-state"><div class="empty-icon">📰</div><p>Could not load articles.</p></div>';
  }
}

async function loadCategoryFilter() {
  const sel = document.getElementById('category-filter');
  if (sel.options.length > 1) return; // already loaded
  try {
    const data = await apiFetch('/news/categories');
    (data.categories || []).forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.category;
      opt.textContent = `${c.category} (${c.count})`;
      sel.appendChild(opt);
    });
  } catch { /* ignore */ }
}

async function fetchAndRenderNews() {
  const q    = state.newsQuery ? `&q=${encodeURIComponent(state.newsQuery)}` : '';
  const cat  = state.newsCategory ? `&category=${encodeURIComponent(state.newsCategory)}` : '';
  const url  = `/news/?page=${state.newsPage}&per_page=${state.newsPerPage}${cat}${q}`;
  const data = await apiFetch(url);

  state.newsTotal = data.total || 0;
  renderNewsGrid(data.articles || []);
  renderPagination();
  document.getElementById('news-count').textContent = `${state.newsTotal.toLocaleString()} articles`;
}

function renderNewsGrid(articles) {
  const grid = document.getElementById('news-grid');
  if (!articles.length) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><div class="empty-icon">🔍</div><p>No articles found.</p></div>';
    return;
  }
  grid.innerHTML = articles.map((a, i) => `
    <div class="news-article-card" style="animation-delay:${i * 0.03}s"
         onclick="addToHistory('${escHtml(a.news_id)}')">
      <div class="article-category">${escHtml(a.category)}</div>
      <div class="article-title">${escHtml(a.title)}</div>
      <div class="article-abstract">${escHtml(a.abstract)}</div>
      <div class="article-id">${escHtml(a.news_id)} · click to add to history</div>
    </div>
  `).join('');
}

function renderPagination() {
  const totalPages = Math.ceil(state.newsTotal / state.newsPerPage);
  const container  = document.getElementById('news-pagination');
  const pages = [];

  const btn = (label, page, active = false, disabled = false) =>
    `<button class="page-btn${active ? ' active' : ''}" ${disabled ? 'disabled' : ''}
      onclick="gotoPage(${page})">${label}</button>`;

  pages.push(btn('←', state.newsPage - 1, false, state.newsPage <= 1));
  const start = Math.max(1, state.newsPage - 2);
  const end   = Math.min(totalPages, start + 4);
  for (let p = start; p <= end; p++) pages.push(btn(p, p, p === state.newsPage));
  pages.push(btn('→', state.newsPage + 1, false, state.newsPage >= totalPages));
  pages.push(`<span class="page-info">Page ${state.newsPage} / ${totalPages}</span>`);

  container.innerHTML = pages.join('');
}

window.gotoPage = async (page) => {
  const totalPages = Math.ceil(state.newsTotal / state.newsPerPage);
  if (page < 1 || page > totalPages) return;
  state.newsPage = page;
  await fetchAndRenderNews();
};

window.addToHistory = (newsId) => {
  const ta = document.getElementById('history-input');
  const existing = ta.value.trim().split(/[\n,\s]+/).filter(Boolean);
  if (!existing.includes(newsId)) {
    ta.value = [...existing, newsId].join('\n');
    toast(`Added ${newsId} to history ✓`, 'success');
  }
};

document.getElementById('news-search').addEventListener('input', e => {
  clearTimeout(newsDebounceTimer);
  newsDebounceTimer = setTimeout(async () => {
    state.newsQuery = e.target.value.trim();
    state.newsPage  = 1;
    await fetchAndRenderNews();
  }, 400);
});

document.getElementById('category-filter').addEventListener('change', async e => {
  state.newsCategory = e.target.value;
  state.newsPage     = 1;
  await fetchAndRenderNews();
});

document.getElementById('btn-news-refresh').addEventListener('click', () => {
  state.newsPage = 1; fetchAndRenderNews();
});

/* ══════════════════════════════════════════════════════════════════════
   FALLBACK DATA (when API is offline)
   ══════════════════════════════════════════════════════════════════════ */
const FALLBACK_SUMMARY = {
  centralized: { auc: 0.9748, mrr: 0.9683, ndcg5: 0.9700, ndcg10: 0.9785, loss: 0.334 },
  federated:   { auc: 0.9795, mrr: 0.9717, ndcg5: 0.9762, ndcg10: 0.9802, loss: 0.203 },
};
const FALLBACK_HISTORY = {
  centralized: [
    { epoch:1, train_loss:16.09, AUC:0.9668, MRR:0.9567, 'nDCG@5':0.9575, 'nDCG@10':0.9704 },
    { epoch:2, train_loss:1.340, AUC:0.9748, MRR:0.9707, 'nDCG@5':0.9709, 'nDCG@10':0.9791 },
    { epoch:3, train_loss:0.763, AUC:0.9729, MRR:0.9695, 'nDCG@5':0.9709, 'nDCG@10':0.9788 },
    { epoch:4, train_loss:0.491, AUC:0.9748, MRR:0.9683, 'nDCG@5':0.9700, 'nDCG@10':0.9785 },
    { epoch:5, train_loss:0.334, AUC:0.9748, MRR:0.9683, 'nDCG@5':0.9700, 'nDCG@10':0.9785 },
  ],
  federated: [
    { round:1, train_loss:34.37, AUC:0.9634, MRR:0.9467, 'nDCG@5':0.9502, 'nDCG@10':0.9630 },
    { round:2, train_loss:2.312, AUC:0.9731, MRR:0.9700, 'nDCG@5':0.9661, 'nDCG@10':0.9787 },
    { round:3, train_loss:0.291, AUC:0.9764, MRR:0.9707, 'nDCG@5':0.9709, 'nDCG@10':0.9793 },
    { round:4, train_loss:0.267, AUC:0.9812, MRR:0.9717, 'nDCG@5':0.9762, 'nDCG@10':0.9805 },
    { round:5, train_loss:0.203, AUC:0.9795, MRR:0.9717, 'nDCG@5':0.9762, 'nDCG@10':0.9802 },
  ],
};

/* ── Utils ──────────────────────────────────────────────────────────── */
function escHtml(str = '') {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ── Init ───────────────────────────────────────────────────────────── */
(async () => {
  await checkHealth();
  loadOverview();   // Load overview section by default
})();
