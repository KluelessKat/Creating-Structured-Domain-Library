// Snug Forest — gene & domain diff app
// Vanilla JS. No build step.

(() => {
  'use strict';

  // ---------- State ----------
  const sides = {
    before: { file: null, rows: null, error: null },
    after:  { file: null, rows: null, error: null },
  };

  let analysis = null; // populated on Analyze

  // Per-column UI state for Q1 & Q2a
  const q1State = {
    sort: 'asc',
    cols: {
      onlyBefore: { search: '', filter: new Set() },
      both:       { search: '', filter: new Set() },
      onlyAfter:  { search: '', filter: new Set() },
    },
  };
  const q2aState = {
    sort: 'asc',
    cols: {
      onlyBefore: { search: '' },
      both:       { search: '' },
      onlyAfter:  { search: '' },
    },
  };
  const q2bState = { sort: 'desc' };

  // ---------- TSV / CSV parsing ----------
  function parseTabular(text, ext) {
    let delim = '\t';
    if (ext === 'csv') delim = ',';
    else if (ext !== 'tsv') {
      const firstLine = text.split(/\r?\n/, 1)[0] || '';
      const tabs = (firstLine.match(/\t/g) || []).length;
      const commas = (firstLine.match(/,/g) || []).length;
      delim = tabs >= commas ? '\t' : ',';
    }
    const rows = parseDelimited(text, delim);
    if (rows.length === 0) return { header: [], data: [] };
    const header = rows[0].map((h) => h.trim());
    const data = [];
    for (let i = 1; i < rows.length; i++) {
      const r = rows[i];
      if (r.length === 1 && r[0] === '') continue; // skip blank
      const obj = {};
      for (let j = 0; j < header.length; j++) obj[header[j]] = r[j] !== undefined ? r[j] : '';
      data.push(obj);
    }
    return { header, data };
  }

  // RFC4180-ish CSV parser; works for tab too.
  function parseDelimited(text, delim) {
    const rows = [];
    let row = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inQuotes) {
        if (c === '"') {
          if (text[i + 1] === '"') { cur += '"'; i++; }
          else { inQuotes = false; }
        } else {
          cur += c;
        }
      } else {
        if (c === '"') { inQuotes = true; }
        else if (c === delim) { row.push(cur); cur = ''; }
        else if (c === '\n') { row.push(cur); rows.push(row); row = []; cur = ''; }
        else if (c === '\r') { /* ignore, handled at \n */ }
        else { cur += c; }
      }
    }
    if (cur !== '' || row.length > 0) { row.push(cur); rows.push(row); }
    return rows;
  }

  const REQUIRED_COLS = ['Gene Name', 'Domain', 'Start', 'End'];

  function validateRows(parsed) {
    const missing = REQUIRED_COLS.filter((c) => !parsed.header.includes(c));
    if (missing.length) return `Missing column(s): ${missing.join(', ')}`;
    if (parsed.data.length === 0) return 'No data rows found';
    return null;
  }

  // ---------- Dropzone wiring ----------
  function setupDropzone(zoneEl) {
    const side = zoneEl.dataset.side;
    const fileInput = zoneEl.querySelector('[data-role="file-input"]');
    const statusEl = zoneEl.querySelector('[data-role="status"]');
    const errorEl = zoneEl.querySelector('[data-role="error"]');

    zoneEl.addEventListener('click', (e) => {
      // Don't trigger when clicking the (invisible) input itself.
      if (e.target === fileInput) return;
      fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) handleFile(side, f, statusEl, errorEl, zoneEl);
    });

    ['dragenter', 'dragover'].forEach((evt) => {
      zoneEl.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        zoneEl.classList.add('drag-over');
      });
    });
    ['dragleave', 'drop'].forEach((evt) => {
      zoneEl.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        zoneEl.classList.remove('drag-over');
      });
    });
    zoneEl.addEventListener('drop', (e) => {
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) handleFile(side, f, statusEl, errorEl, zoneEl);
    });
  }

  function handleFile(side, file, statusEl, errorEl, zoneEl) {
    sides[side].file = file;
    sides[side].rows = null;
    sides[side].error = null;
    zoneEl.classList.remove('error', 'loaded');
    errorEl.textContent = '';
    statusEl.innerHTML = `<span class="filename">${escapeHtml(file.name)}</span> <span class="meta">· reading…</span>`;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = reader.result;
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        const parsed = parseTabular(text, ext);
        const err = validateRows(parsed);
        if (err) {
          sides[side].error = err;
          errorEl.textContent = err;
          zoneEl.classList.add('error');
          statusEl.innerHTML = `<span class="filename">${escapeHtml(file.name)}</span>`;
        } else {
          sides[side].rows = parsed.data;
          zoneEl.classList.add('loaded');
          const nRows = parsed.data.length;
          const nGenes = new Set(parsed.data.map((r) => r['Gene Name'])).size;
          statusEl.innerHTML = `<span class="filename">${escapeHtml(file.name)}</span><br /><span class="meta">${nRows.toLocaleString()} rows · ${nGenes.toLocaleString()} unique genes</span>`;
        }
      } catch (ex) {
        const msg = `Could not parse file: ${ex.message || ex}`;
        sides[side].error = msg;
        errorEl.textContent = msg;
        zoneEl.classList.add('error');
        statusEl.innerHTML = `<span class="filename">${escapeHtml(file.name)}</span>`;
      }
      updateAnalyzeButton();
    };
    reader.onerror = () => {
      const msg = 'Failed to read file';
      sides[side].error = msg;
      errorEl.textContent = msg;
      zoneEl.classList.add('error');
      statusEl.innerHTML = `<span class="filename">${escapeHtml(file.name)}</span>`;
      updateAnalyzeButton();
    };
    reader.readAsText(file);
  }

  function updateAnalyzeButton() {
    const btn = document.getElementById('analyze-btn');
    btn.disabled = !(sides.before.rows && sides.after.rows);
  }

  // ---------- Diff helpers ----------
  function buildSide(rows) {
    const genes = new Set();
    const geneDomainNames = new Map();   // gene -> Set<domain>
    const geneDomainEntries = new Map(); // gene -> Array<{domain, start, end}>
    const geneAllDomainTypes = new Map();// gene -> Set<domain>
    for (const r of rows) {
      const g = (r['Gene Name'] || '').trim();
      if (!g) continue;
      const d = (r['Domain'] || '').trim();
      const start = parseInt(r['Start'], 10);
      const end = parseInt(r['End'], 10);
      genes.add(g);
      if (!geneDomainNames.has(g)) geneDomainNames.set(g, new Set());
      if (!geneDomainEntries.has(g)) geneDomainEntries.set(g, []);
      if (!geneAllDomainTypes.has(g)) geneAllDomainTypes.set(g, new Set());
      if (d) {
        geneDomainNames.get(g).add(d);
        geneDomainEntries.get(g).push({ domain: d, start, end });
        geneAllDomainTypes.get(g).add(d);
      }
    }
    return { genes, geneDomainNames, geneDomainEntries, geneAllDomainTypes };
  }

  function setDiff(a, b) {
    const onlyA = [], onlyB = [], both = [];
    for (const x of a) (b.has(x) ? both : onlyA).push(x);
    for (const x of b) if (!a.has(x)) onlyB.push(x);
    return { onlyA, both, onlyB };
  }

  function runAnalysis() {
    const before = buildSide(sides.before.rows);
    const after = buildSide(sides.after.rows);
    const { onlyA, both, onlyB } = setDiff(before.genes, after.genes);

    // Q2a: 4-tuple (domain, gene, start, end) set diff across ALL rows
    const tupleKey = (t) => `${t.domain}${t.gene}${t.start}${t.end}`;
    const beforeTuples = new Map(); // key -> {domain, gene, start, end}
    const afterTuples  = new Map();
    for (const [g, entries] of before.geneDomainEntries) {
      for (const e of entries) {
        const t = { domain: e.domain, gene: g, start: e.start, end: e.end };
        beforeTuples.set(tupleKey(t), t);
      }
    }
    for (const [g, entries] of after.geneDomainEntries) {
      for (const e of entries) {
        const t = { domain: e.domain, gene: g, start: e.start, end: e.end };
        afterTuples.set(tupleKey(t), t);
      }
    }
    const q2aRows = { onlyBefore: [], both: [], onlyAfter: [] };
    for (const [k, t] of beforeTuples) {
      if (afterTuples.has(k)) q2aRows.both.push(t);
      else q2aRows.onlyBefore.push(t);
    }
    for (const [k, t] of afterTuples) {
      if (!beforeTuples.has(k)) q2aRows.onlyAfter.push(t);
    }

    // Q2b: per-domain-name occurrence counts across ALL rows in each file
    const counts = new Map(); // domain -> { before, after }
    for (const entries of before.geneDomainEntries.values()) {
      for (const e of entries) {
        if (!counts.has(e.domain)) counts.set(e.domain, { before: 0, after: 0 });
        counts.get(e.domain).before++;
      }
    }
    for (const entries of after.geneDomainEntries.values()) {
      for (const e of entries) {
        if (!counts.has(e.domain)) counts.set(e.domain, { before: 0, after: 0 });
        counts.get(e.domain).after++;
      }
    }

    analysis = {
      before, after,
      q1: { onlyBefore: onlyA, both, onlyAfter: onlyB },
      q2a: q2aRows,
      q2b: counts,
    };
  }

  // ---------- Fuzzy match ----------
  function fuzzyMatch(query, str) {
    if (!query) return true;
    const q = query.toLowerCase();
    const s = str.toLowerCase();
    if (s.includes(q)) return true;
    let qi = 0;
    for (let i = 0; i < s.length && qi < q.length; i++) {
      if (s[i] === q[qi]) qi++;
    }
    return qi === q.length;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ---------- Render: Q1 ----------
  function renderQ1() {
    const grid = document.getElementById('q1-grid');
    const buckets = {
      onlyBefore: { items: analysis.q1.onlyBefore, sideKey: 'before' },
      both:       { items: analysis.q1.both,       sideKey: 'both' },
      onlyAfter:  { items: analysis.q1.onlyAfter,  sideKey: 'after' },
    };
    grid.querySelectorAll('.tri-col').forEach((col) => {
      const bucket = col.dataset.bucket;
      const data = buckets[bucket];
      const sideKey = data.sideKey;
      const allDomainTypes = collectDomainTypesForBucket(bucket, data.items);

      // Populate filter options once per bucket (or refresh if missing)
      buildFilterOptions(col, bucket, allDomainTypes);

      const search = q1State.cols[bucket].search;
      const filter = q1State.cols[bucket].filter;
      const sort = q1State.sort;

      const filtered = data.items.filter((g) => {
        if (search && !fuzzyMatch(search, g)) return false;
        if (filter.size > 0) {
          const hasAny = geneHasAnyDomain(g, sideKey, filter);
          if (!hasAny) return false;
        }
        return true;
      });
      filtered.sort((a, b) => sort === 'asc' ? a.localeCompare(b) : b.localeCompare(a));

      const list = col.querySelector('[data-role="list"]');
      list.innerHTML = '';
      if (filtered.length === 0) {
        const li = document.createElement('li');
        li.className = 'empty';
        li.textContent = 'no genes';
        list.appendChild(li);
      } else {
        const frag = document.createDocumentFragment();
        for (const g of filtered) {
          const li = document.createElement('li');
          li.textContent = g;
          frag.appendChild(li);
        }
        list.appendChild(frag);
      }
      col.querySelector('[data-role="count"]').textContent = filtered.length.toLocaleString();
    });
  }

  function geneHasAnyDomain(gene, sideKey, filterSet) {
    let domainSet;
    if (sideKey === 'before') {
      domainSet = analysis.before.geneAllDomainTypes.get(gene);
    } else if (sideKey === 'after') {
      domainSet = analysis.after.geneAllDomainTypes.get(gene);
    } else {
      // 'both' bucket: gene exists on both sides; consider union so user can filter on either side
      const b = analysis.before.geneAllDomainTypes.get(gene) || new Set();
      const a = analysis.after.geneAllDomainTypes.get(gene)  || new Set();
      for (const d of filterSet) if (b.has(d) || a.has(d)) return true;
      return false;
    }
    if (!domainSet) return false;
    for (const d of filterSet) if (domainSet.has(d)) return true;
    return false;
  }

  function collectDomainTypesForBucket(bucket, geneList) {
    const set = new Set();
    for (const g of geneList) {
      if (bucket === 'onlyBefore' || bucket === 'both') {
        const s = analysis.before.geneAllDomainTypes.get(g);
        if (s) for (const d of s) set.add(d);
      }
      if (bucket === 'onlyAfter' || bucket === 'both') {
        const s = analysis.after.geneAllDomainTypes.get(g);
        if (s) for (const d of s) set.add(d);
      }
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }

  function buildFilterOptions(col, bucket, allDomainTypes) {
    const optsEl = col.querySelector('[data-role="filter-options"]');
    if (!optsEl || optsEl.dataset.populated === '1') return;
    optsEl.innerHTML = '';
    const selectedSet = q1State.cols[bucket].filter;
    for (const d of allDomainTypes) {
      const id = `flt-${bucket}-${cssSafe(d)}`;
      const label = document.createElement('label');
      label.dataset.value = d.toLowerCase();
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = d;
      cb.id = id;
      cb.checked = selectedSet.has(d);
      cb.addEventListener('change', () => {
        if (cb.checked) selectedSet.add(d);
        else selectedSet.delete(d);
        const toggle = col.querySelector('[data-role="filter-toggle"]');
        toggle.classList.toggle('active', selectedSet.size > 0);
        toggle.textContent = selectedSet.size > 0
          ? `Domain filter (${selectedSet.size})`
          : 'Filter by domain type';
        renderQ1();
      });
      const span = document.createElement('span');
      span.textContent = d;
      label.appendChild(cb);
      label.appendChild(span);
      optsEl.appendChild(label);
    }
    optsEl.dataset.populated = '1';
  }

  function cssSafe(s) { return String(s).replace(/[^a-zA-Z0-9_-]/g, '_'); }

  // ---------- Render: Q2a ----------
  function renderQ2a() {
    const grid = document.getElementById('q2a-grid');
    const buckets = {
      onlyBefore: analysis.q2a.onlyBefore,
      both:       analysis.q2a.both,
      onlyAfter:  analysis.q2a.onlyAfter,
    };
    grid.querySelectorAll('.tri-col').forEach((col) => {
      const bucket = col.dataset.bucket;
      const items = buckets[bucket];
      const search = q2aState.cols[bucket].search;
      const sort = q2aState.sort;

      const filtered = items.filter((it) => {
        if (!search) return true;
        const hay = `${it.gene} ${it.domain}`;
        return fuzzyMatch(search, hay);
      });
      filtered.sort((a, b) => {
        const cmp = a.domain.localeCompare(b.domain) || a.gene.localeCompare(b.gene);
        return sort === 'asc' ? cmp : -cmp;
      });

      const list = col.querySelector('[data-role="list"]');
      list.innerHTML = '';
      if (filtered.length === 0) {
        const li = document.createElement('li');
        li.className = 'empty';
        li.textContent = 'no domains';
        list.appendChild(li);
      } else {
        const frag = document.createDocumentFragment();
        for (const it of filtered) {
          const li = document.createElement('li');
          const pos = (it.start != null && it.end != null && !Number.isNaN(it.start) && !Number.isNaN(it.end)) ? `${it.start}–${it.end}` : '—';
          li.innerHTML = `<strong>${escapeHtml(it.domain)}</strong> · ${escapeHtml(it.gene)} · <span class="row-meta">${escapeHtml(pos)}</span>`;
          frag.appendChild(li);
        }
        list.appendChild(frag);
      }
      col.querySelector('[data-role="count"]').textContent = filtered.length.toLocaleString();
    });
  }

  // ---------- Render: Q2b ----------
  function renderQ2b() {
    const svg = document.getElementById('q2b-bars');
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const entries = Array.from(analysis.q2b.entries()).map(([domain, c]) => ({
      domain, before: c.before, after: c.after, mean: (c.before + c.after) / 2,
    }));
    entries.sort((a, b) => q2bState.sort === 'desc' ? b.mean - a.mean : a.mean - b.mean);

    const margin = { top: 14, right: 70, bottom: 28, left: 180 };
    const rowH = 30; // per domain (two stacked bars within)
    const barH = 11;
    const gap = 2;
    const innerH = entries.length * rowH;
    const container = svg.parentElement;
    const containerW = Math.max(560, container.clientWidth || 800);
    const innerW = Math.max(160, containerW - margin.left - margin.right);
    const totalH = innerH + margin.top + margin.bottom;

    svg.setAttribute('viewBox', `0 0 ${containerW} ${totalH}`);
    svg.setAttribute('width', containerW);
    svg.setAttribute('height', totalH);
    svg.setAttribute('preserveAspectRatio', 'xMinYMin meet');

    if (entries.length === 0) {
      const t = svgEl('text', { x: containerW / 2, y: 80, 'text-anchor': 'middle', class: 'axis-label' });
      t.textContent = 'No domains to count.';
      svg.appendChild(t);
      return;
    }

    const maxVal = Math.max(1, ...entries.map((e) => Math.max(e.before, e.after)));
    const xScale = (v) => (v / maxVal) * innerW;

    const ticks = niceTicks(0, maxVal, 5);
    for (const t of ticks) {
      const x = margin.left + xScale(t);
      const line = svgEl('line', {
        x1: x, x2: x,
        y1: margin.top, y2: margin.top + innerH,
        class: 'grid-line',
      });
      svg.appendChild(line);
      const lbl = svgEl('text', {
        x, y: margin.top + innerH + 16,
        'text-anchor': 'middle', class: 'axis-label',
      });
      lbl.textContent = t;
      svg.appendChild(lbl);
    }
    const xAxisLabel = svgEl('text', {
      x: margin.left + innerW / 2,
      y: totalH - 4,
      'text-anchor': 'middle', class: 'axis-label',
    });
    xAxisLabel.textContent = 'occurrence count (rows) per file';
    svg.appendChild(xAxisLabel);

    entries.forEach((e, i) => {
      const yTop = margin.top + i * rowH;
      // Row stripe
      if (i % 2 === 0) {
        const bg = svgEl('rect', {
          x: margin.left, y: yTop,
          width: innerW, height: rowH, class: 'row-bg',
        });
        svg.appendChild(bg);
      }
      // Domain label (truncate if too long)
      const label = svgEl('text', {
        x: margin.left - 10, y: yTop + rowH / 2 + 3,
        'text-anchor': 'end', class: 'domain-label',
      });
      const trunc = e.domain.length > 26 ? e.domain.slice(0, 25) + '…' : e.domain;
      label.textContent = trunc;
      const titleNode = svgEl('title', {});
      titleNode.textContent = `${e.domain}\nbefore: ${e.before}, after: ${e.after}`;
      label.appendChild(titleNode);
      svg.appendChild(label);

      // Bar before (top)
      const yB = yTop + (rowH - 2 * barH - gap) / 2;
      const bw = xScale(e.before);
      const rectB = svgEl('rect', {
        x: margin.left, y: yB,
        width: Math.max(0.5, bw), height: barH, rx: 3, ry: 3,
        class: 'bar-before',
      });
      const titleB = svgEl('title', {});
      titleB.textContent = `${e.domain} · before: ${e.before}`;
      rectB.appendChild(titleB);
      svg.appendChild(rectB);
      const tB = svgEl('text', {
        x: margin.left + bw + 5, y: yB + barH - 1,
        class: 'value-label',
      });
      tB.textContent = e.before;
      svg.appendChild(tB);

      // Bar after (bottom)
      const yA = yB + barH + gap;
      const aw = xScale(e.after);
      const rectA = svgEl('rect', {
        x: margin.left, y: yA,
        width: Math.max(0.5, aw), height: barH, rx: 3, ry: 3,
        class: 'bar-after',
      });
      const titleA = svgEl('title', {});
      titleA.textContent = `${e.domain} · after: ${e.after}`;
      rectA.appendChild(titleA);
      svg.appendChild(rectA);
      const tA = svgEl('text', {
        x: margin.left + aw + 5, y: yA + barH - 1,
        class: 'value-label',
      });
      tA.textContent = e.after;
      svg.appendChild(tA);
    });
  }

  function niceTicks(min, max, count) {
    const range = max - min;
    if (range <= 0) return [0, 1];
    const rawStep = range / count;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    let step;
    if (norm < 1.5) step = 1 * mag;
    else if (norm < 3) step = 2 * mag;
    else if (norm < 7) step = 5 * mag;
    else step = 10 * mag;
    const ticks = [];
    let t = 0;
    while (t <= max + 1e-9) { ticks.push(Math.round(t)); t += step; }
    return ticks;
  }

  function svgEl(name, attrs) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  // ---------- Wire-up ----------
  function init() {
    document.querySelectorAll('.dropzone').forEach(setupDropzone);

    document.getElementById('analyze-btn').addEventListener('click', () => {
      runAnalysis();
      document.getElementById('results').classList.remove('hidden');
      // Reset per-column UI state
      q1State.sort = 'asc';
      q2aState.sort = 'asc';
      q2bState.sort = 'desc';
      for (const k of Object.keys(q1State.cols)) {
        q1State.cols[k].search = '';
        q1State.cols[k].filter.clear();
      }
      for (const k of Object.keys(q2aState.cols)) q2aState.cols[k].search = '';
      // Clear inputs in DOM
      document.querySelectorAll('#q1-grid [data-role="search"], #q2a-grid [data-role="search"]').forEach((i) => i.value = '');
      document.querySelectorAll('#q1-grid [data-role="filter-options"]').forEach((o) => o.dataset.populated = '');
      document.querySelectorAll('#q1-grid [data-role="filter-toggle"]').forEach((b) => {
        b.classList.remove('active');
        b.textContent = 'Filter by domain type';
      });

      renderQ1();
      renderQ2a();
      renderQ2b();
      document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    // Q1 grid: per-column search, filter toggles
    document.querySelectorAll('#q1-grid .tri-col').forEach((col) => {
      const bucket = col.dataset.bucket;
      const search = col.querySelector('[data-role="search"]');
      search.addEventListener('input', () => { q1State.cols[bucket].search = search.value; renderQ1(); });

      const toggle = col.querySelector('[data-role="filter-toggle"]');
      const panel = col.querySelector('[data-role="filter-panel"]');
      const filterSearch = col.querySelector('[data-role="filter-search"]');
      const clearBtn = col.querySelector('[data-role="filter-clear"]');
      const closeBtn = col.querySelector('[data-role="filter-close"]');

      toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        // Close other open panels
        document.querySelectorAll('#q1-grid [data-role="filter-panel"]').forEach((p) => {
          if (p !== panel) p.classList.add('hidden');
        });
        panel.classList.toggle('hidden');
      });
      filterSearch.addEventListener('input', () => {
        const q = filterSearch.value.toLowerCase();
        const labels = col.querySelectorAll('[data-role="filter-options"] label');
        labels.forEach((lab) => {
          const v = lab.dataset.value || '';
          lab.style.display = (!q || v.includes(q)) ? '' : 'none';
        });
      });
      clearBtn.addEventListener('click', () => {
        q1State.cols[bucket].filter.clear();
        const cbs = col.querySelectorAll('[data-role="filter-options"] input[type="checkbox"]');
        cbs.forEach((cb) => { cb.checked = false; });
        toggle.classList.remove('active');
        toggle.textContent = 'Filter by domain type';
        renderQ1();
      });
      closeBtn.addEventListener('click', () => panel.classList.add('hidden'));
    });

    // Close any open Q1 filter panel when clicking outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.tri-filter')) {
        document.querySelectorAll('#q1-grid [data-role="filter-panel"]').forEach((p) => p.classList.add('hidden'));
      }
    });

    document.getElementById('q1-sort').addEventListener('change', (e) => {
      q1State.sort = e.target.value; renderQ1();
    });

    // Q2a grid: per-column search
    document.querySelectorAll('#q2a-grid .tri-col').forEach((col) => {
      const bucket = col.dataset.bucket;
      const search = col.querySelector('[data-role="search"]');
      search.addEventListener('input', () => { q2aState.cols[bucket].search = search.value; renderQ2a(); });
    });
    document.getElementById('q2a-sort').addEventListener('change', (e) => {
      q2aState.sort = e.target.value; renderQ2a();
    });

    // Q2b sort
    document.getElementById('q2b-sort').addEventListener('change', (e) => {
      q2bState.sort = e.target.value; renderQ2b();
    });

    // Re-render bar plot on resize (debounced)
    let rt;
    window.addEventListener('resize', () => {
      if (!analysis) return;
      clearTimeout(rt);
      rt = setTimeout(renderQ2b, 120);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
