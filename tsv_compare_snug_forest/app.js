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

  // ---------- Report export ----------
  function buildReportHtml() {
    const beforeName = sides.before.file ? sides.before.file.name : 'before';
    const afterName  = sides.after.file  ? sides.after.file.name  : 'after';
    const now = new Date().toLocaleString();

    // Summary stats
    const nGenesBefore = analysis.before.genes.size;
    const nGenesAfter  = analysis.after.genes.size;
    const genesAdded   = analysis.q1.onlyAfter.length;
    const genesRemoved = analysis.q1.onlyBefore.length;

    let instancesBefore = 0;
    for (const entries of analysis.before.geneDomainEntries.values()) instancesBefore += entries.length;
    let instancesAfter = 0;
    for (const entries of analysis.after.geneDomainEntries.values()) instancesAfter += entries.length;
    const instancesDelta = instancesAfter - instancesBefore;

    // Top 10 domains by absolute delta
    const top10 = Array.from(analysis.q2b.entries())
      .map(([domain, c]) => ({ domain, before: c.before, after: c.after, delta: c.after - c.before }))
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
      .slice(0, 10);

    // Full domain table sorted by |delta|
    const q2bFull = Array.from(analysis.q2b.entries())
      .map(([domain, c]) => ({ domain, before: c.before, after: c.after, delta: c.after - c.before }))
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

    // Sorted gene/tuple lists
    const q1Before = [...analysis.q1.onlyBefore].sort((a, b) => a.localeCompare(b));
    const q1Both   = [...analysis.q1.both].sort((a, b) => a.localeCompare(b));
    const q1After  = [...analysis.q1.onlyAfter].sort((a, b) => a.localeCompare(b));
    const tupleSort = (a, b) => a.domain.localeCompare(b.domain) || a.gene.localeCompare(b.gene);
    const q2aBefore = [...analysis.q2a.onlyBefore].sort(tupleSort);
    const q2aBoth   = [...analysis.q2a.both].sort(tupleSort);
    const q2aAfter  = [...analysis.q2a.onlyAfter].sort(tupleSort);

    function dSign(d) {
      if (d > 0) return `<span style="color:#4a8a4a;font-weight:700">+${d}</span>`;
      if (d < 0) return `<span style="color:#b1564a;font-weight:700">${d}</span>`;
      return `<span style="color:#6b6b6b">0</span>`;
    }
    function geneItems(arr) {
      if (!arr.length) return '<li style="color:#6b6b6b;font-style:italic;padding:16px 12px;text-align:center">none</li>';
      return arr.map((g) => `<li>${escapeHtml(g)}</li>`).join('');
    }
    function tupleItems(arr) {
      if (!arr.length) return '<li style="color:#6b6b6b;font-style:italic;padding:16px 12px;text-align:center">none</li>';
      return arr.map((it) => {
        const pos = (it.start != null && it.end != null && !Number.isNaN(it.start) && !Number.isNaN(it.end))
          ? `${it.start}–${it.end}` : '—';
        return `<li><strong>${escapeHtml(it.domain)}</strong> · ${escapeHtml(it.gene)} · <span style="color:#6b6b6b;font-size:11px">${escapeHtml(pos)}</span></li>`;
      }).join('');
    }
    function domainRows(arr) {
      return arr.map((d) => `<tr><td>${escapeHtml(d.domain)}</td><td>${d.before}</td><td>${d.after}</td><td>${dSign(d.delta)}</td></tr>`).join('');
    }

    // Serialize the live SVG chart
    const svgEl = document.getElementById('q2b-bars');
    const svgHtml = svgEl ? new XMLSerializer().serializeToString(svgEl) : '';

    const triColStyle = 'background:#fcfaf0;border-radius:8px;border:1px solid rgba(94,122,94,0.18);overflow:hidden;';
    const headStyle = 'padding:10px 12px 8px;border-bottom:1px solid rgba(94,122,94,0.18);background:#fffdf5;';
    const h3Style = 'margin:0 0 0;font-size:14px;font-weight:700;color:#5e7a5e;';
    const countStyle = 'font-size:11px;color:#6b6b6b;background:#f5ecd6;border-radius:999px;padding:2px 8px;font-weight:600;margin-left:6px;';
    const listStyle = 'list-style:none;margin:0;padding:4px 0;max-height:420px;overflow:auto;';
    const listItemStyle = 'padding:5px 12px;font-size:12px;border-bottom:1px dashed rgba(94,122,94,0.10);word-break:break-word;';
    const thStyle = 'text-align:left;color:#6b6b6b;font-weight:600;border-bottom:1px solid rgba(94,122,94,0.18);padding:6px 10px;';
    const tdStyle = 'padding:6px 10px;border-bottom:1px solid rgba(94,122,94,0.08);';

    function triCol(stripeColor, title, count, listInner) {
      return `<div style="${triColStyle}">
        <div style="height:4px;background:${stripeColor}"></div>
        <div style="${headStyle}"><h3 style="${h3Style}">${title} <span style="${countStyle}">${count.toLocaleString()}</span></h3></div>
        <ul style="${listStyle}">${listInner}</ul>
      </div>`;
    }

    const geneDelta = nGenesAfter - nGenesBefore;

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Snug Forest Report · ${escapeHtml(beforeName)} vs ${escapeHtml(afterName)}</title>
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:#fbf6e7;color:#2a2a2a;margin:0;padding:0;font-size:14px;font-weight:500}
.wrap{max-width:1100px;margin:0 auto;padding:40px 28px 60px}
h1{color:#5e7a5e;font-size:26px;margin:0 0 4px}
h2{color:#5e7a5e;font-size:18px;margin:0 0 14px}
.meta-line{color:#6b6b6b;font-size:13px;margin:0 0 32px}
.card{background:#fff;border-radius:12px;padding:22px;box-shadow:0 4px 18px rgba(60,80,50,0.08);border:1px solid rgba(94,122,94,0.18);margin-bottom:32px}
.stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px}
.stat-cell{background:#fcfaf0;border-radius:8px;padding:14px 16px;border:1px solid rgba(94,122,94,0.18)}
.stat-label{font-size:12px;color:#6b6b6b;margin-bottom:4px}
.stat-value{font-size:22px;font-weight:700;color:#5e7a5e}
.stat-sub{font-size:12px;color:#6b6b6b;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13px}
.tri-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.bar-svg{display:block;width:100%;overflow:visible}
.bar-svg .axis-label{fill:#6b6b6b;font-size:11px;font-family:-apple-system,sans-serif}
.bar-svg .domain-label{fill:#2a2a2a;font-size:12px;font-family:-apple-system,sans-serif;font-weight:600}
.bar-svg .value-label{fill:#2a2a2a;font-size:11px;font-family:-apple-system,sans-serif;font-weight:600}
.bar-svg .row-bg{fill:rgba(245,236,214,0.45)}
.bar-svg .grid-line{stroke:rgba(94,122,94,0.15);stroke-width:1}
.bar-svg .bar-before{fill:#8faf85}
.bar-svg .bar-after{fill:#a8c5d8}
@media(max-width:700px){.tri-grid{grid-template-columns:1fr}.stat-grid{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Snug Forest · Comparison Report</h1>
  <p class="meta-line">Generated ${escapeHtml(now)} &nbsp;·&nbsp; Before: <strong>${escapeHtml(beforeName)}</strong> &nbsp;·&nbsp; After: <strong>${escapeHtml(afterName)}</strong></p>

  <div class="card">
    <h2>Summary</h2>
    <div class="stat-grid">
      <div class="stat-cell"><div class="stat-label">Unique genes · Before</div><div class="stat-value">${nGenesBefore.toLocaleString()}</div></div>
      <div class="stat-cell"><div class="stat-label">Unique genes · After</div><div class="stat-value">${nGenesAfter.toLocaleString()}</div></div>
      <div class="stat-cell">
        <div class="stat-label">Gene delta</div>
        <div class="stat-value">${geneDelta >= 0 ? '+' : ''}${geneDelta.toLocaleString()}</div>
        <div class="stat-sub">+${genesAdded} added &nbsp;·&nbsp; −${genesRemoved} removed</div>
      </div>
      <div class="stat-cell"><div class="stat-label">Domain instances · Before</div><div class="stat-value">${instancesBefore.toLocaleString()}</div></div>
      <div class="stat-cell"><div class="stat-label">Domain instances · After</div><div class="stat-value">${instancesAfter.toLocaleString()}</div></div>
      <div class="stat-cell"><div class="stat-label">Instance delta</div><div class="stat-value">${instancesDelta >= 0 ? '+' : ''}${instancesDelta.toLocaleString()}</div></div>
    </div>
    <h2 style="font-size:15px;margin-bottom:10px">Top 10 domains by absolute change</h2>
    <table>
      <thead><tr><th style="${thStyle}">Domain</th><th style="${thStyle}">Before</th><th style="${thStyle}">After</th><th style="${thStyle}">Delta</th></tr></thead>
      <tbody>${domainRows(top10)}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>1. Genes added &amp; removed</h2>
    <div class="tri-grid">
      ${triCol('#d8c9a8', 'Only in Before', q1Before.length, geneItems(q1Before))}
      ${triCol('#8faf85', 'In Both', q1Both.length, geneItems(q1Both))}
      ${triCol('#a8c5d8', 'Only in After', q1After.length, geneItems(q1After))}
    </div>
  </div>

  <div class="card">
    <h2>2a. Domain instances added &amp; removed</h2>
    <div class="tri-grid">
      ${triCol('#d8c9a8', 'Only in Before', q2aBefore.length, tupleItems(q2aBefore))}
      ${triCol('#8faf85', 'In Both', q2aBoth.length, tupleItems(q2aBoth))}
      ${triCol('#a8c5d8', 'Only in After', q2aAfter.length, tupleItems(q2aAfter))}
    </div>
  </div>

  <div class="card">
    <h2>2b. Domain count comparison</h2>
    ${svgHtml}
    <h2 style="font-size:15px;margin:24px 0 10px">All domain counts (sorted by |delta|)</h2>
    <table>
      <thead><tr><th style="${thStyle}">Domain</th><th style="${thStyle}">Before</th><th style="${thStyle}">After</th><th style="${thStyle}">Delta</th></tr></thead>
      <tbody>${domainRows(q2bFull)}</tbody>
    </table>
  </div>
</div>
</body>
</html>`;
  }

  function downloadReport() {
    const html = buildReportHtml();
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `snug-forest-report-${new Date().toISOString().slice(0, 10)}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ---------- Wire-up ----------
  function init() {
    document.querySelectorAll('.dropzone').forEach(setupDropzone);

    document.getElementById('export-btn').disabled = true;
    document.getElementById('export-btn').addEventListener('click', downloadReport);

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
      document.getElementById('export-btn').disabled = false;
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
