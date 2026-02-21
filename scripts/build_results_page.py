"""
Reads all JSON result files from results/ and generates docs/index.html
with the data embedded for GitHub Pages.

Usage:
    python scripts/build_results_page.py
"""

from __future__ import annotations

import json
import os
import html
from pathlib import Path


RESULTS_DIR = Path("results")
OUTPUT_PATH = Path("docs/index.html")


def collect_results() -> list[dict]:
    results = []
    for path in sorted(RESULTS_DIR.rglob("*.json")):
        with open(path) as f:
            data = json.load(f)
        rel = path.relative_to(RESULTS_DIR)
        category = str(rel.parent) if str(rel.parent) != "." else "uncategorized"

        entry = {
            "model": data.get("model"),
            "dataset": data.get("dataset"),
            "dataset_kwargs": data.get("dataset_kwargs", {}),
            "metrics": data.get("metrics", {}),
            "num_samples": data.get("num_samples"),
            "_filename": path.name,
            "_category": category,
            "_path": str(rel),
        }
        results.append(entry)
    return results


def file_label(name: str) -> str:
    base = name.replace(".json", "")
    parts = base.split("_")
    if len(parts) >= 4:
        tgt = parts.pop()
        src = parts.pop()
        dataset = parts.pop()
        model = "_".join(parts)
        return f"{model} · {dataset} · {src}→{tgt}"
    return base


def build_html(results: list[dict]) -> str:
    results_json = json.dumps(results, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Translation Benchmark Results</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --orange: #d29922;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; font-size: 0.95rem; }}
  .tabs {{
    display: flex;
    gap: 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
  }}
  .tab-btn {{
    background: none;
    border: none;
    color: var(--text-muted);
    padding: 0.75rem 1.25rem;
    cursor: pointer;
    font-size: 0.9rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
  }}
  .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
  }}
  .metric-card .label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
  }}
  .metric-card .value {{ font-size: 1.5rem; font-weight: 600; color: var(--accent); }}
  .info-bar {{
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
  }}
  .info-bar .lbl {{ color: var(--text-muted); margin-right: 0.5rem; }}
  .section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    margin: 2rem 0 1rem 0;
    color: var(--text);
  }}
  .category-title {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin: 2.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 2rem;
    font-size: 0.9rem;
  }}
  th {{
    text-align: left;
    padding: 0.75rem 1rem;
    background: var(--surface);
    border-bottom: 2px solid var(--border);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    position: sticky;
    top: 0;
    z-index: 1;
  }}
  td {{
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  tr:hover td {{ background: rgba(88, 166, 255, 0.04); }}
  .best {{ color: var(--green); font-weight: 600; }}
</style>
</head>
<body>

<h1>Translation Benchmark Results</h1>
<p class="subtitle">Benchmark results for translation models on African language datasets.</p>

<div class="tabs">
  <button class="tab-btn active" data-tab="overview">Overview</button>
  <button class="tab-btn" data-tab="comparison">Comparison</button>
</div>

<div class="tab-panel active" id="tab-overview"></div>
<div class="tab-panel" id="tab-comparison"></div>

<script>
const DATA = {results_json};

function esc(s) {{ return s ? s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : ''; }}

function fileLabel(name) {{
  const base = name.replace(/\\.json$/, '');
  const parts = base.split('_');
  if (parts.length >= 4) {{
    const tgt = parts.pop();
    const src = parts.pop();
    const dataset = parts.pop();
    const model = parts.join('_');
    return model + ' \\u00b7 ' + dataset + ' \\u00b7 ' + src + '\\u2192' + tgt;
  }}
  return base;
}}

function getSrcLang(d) {{
  return d.dataset_kwargs?.source_language || (d.dataset_kwargs?.source_languages||[]).join(',') || '?';
}}
function getTgtLang(d) {{ return d.dataset_kwargs?.target_language || '?'; }}

// Tabs
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  }});
}});

// Overview
function renderOverview() {{
  const el = document.getElementById('tab-overview');
  const cats = {{}};
  DATA.forEach(d => {{
    const c = d._category || 'uncategorized';
    if (!cats[c]) cats[c] = [];
    cats[c].push(d);
  }});

  let h = '';
  for (const [cat, items] of Object.entries(cats).sort()) {{
    h += '<div class="category-title">' + esc(cat) + '</div>';
    for (const d of items) {{
      h += '<div class="section-title">' + esc(fileLabel(d._filename)) + '</div>';
      h += '<div class="info-bar">';
      h += '<div><span class="lbl">Model:</span>' + esc(d.model) + '</div>';
      h += '<div><span class="lbl">Dataset:</span>' + esc(d.dataset) + '</div>';
      h += '<div><span class="lbl">Samples:</span>' + (d.num_samples ?? '—') + '</div>';
      h += '<div><span class="lbl">Direction:</span>' + esc(getSrcLang(d)) + '→' + esc(getTgtLang(d)) + '</div>';
      h += '<div><span class="lbl">Time:</span>' + (d.elapsed_seconds ? d.elapsed_seconds + 's' : '—') + '</div>';
      h += '</div>';
      if (d.metrics) {{
        h += '<div class="metrics-grid">';
        for (const [k,v] of Object.entries(d.metrics)) {{
          h += '<div class="metric-card"><div class="label">' + esc(k) + '</div><div class="value">' + (typeof v==='number'?v.toFixed(2):v) + '</div></div>';
        }}
        h += '</div>';
      }}
    }}
  }}
  el.innerHTML = h;
}}

// Comparison – grouped by translation direction
function renderComparison() {{
  const el = document.getElementById('tab-comparison');

  // Group results by translation direction (e.g. "yo→en")
  const groups = {{}};
  DATA.forEach(d => {{
    const dir = getSrcLang(d) + '→' + getTgtLang(d);
    if (!groups[dir]) groups[dir] = [];
    groups[dir].push(d);
  }});

  let h = '';
  for (const [dir, items] of Object.entries(groups).sort()) {{
    // Collect all metrics in this group
    const allMetrics = new Set();
    items.forEach(d => Object.keys(d.metrics||{{}}).forEach(k => allMetrics.add(k)));
    const mList = [...allMetrics].sort();

    // Find best value per metric within this group
    const best = {{}};
    for (const m of mList) {{
      const lower = m.toLowerCase().includes('ter');
      let bv = lower ? Infinity : -Infinity, bi = -1;
      items.forEach((d, i) => {{
        const v = d.metrics?.[m];
        if (v == null) return;
        if (lower ? v < bv : v > bv) {{ bv = v; bi = i; }}
      }});
      best[m] = bi;
    }}

    h += '<div class="category-title">' + esc(dir) + '</div>';
    h += '<table><thead><tr><th>Category</th><th>Model</th><th>Samples</th>';
    mList.forEach(m => h += '<th>' + esc(m.toUpperCase()) + '</th>');
    h += '</tr></thead><tbody>';
    items.forEach((d, i) => {{
      h += '<tr><td>' + esc(d._category || 'uncategorized') + '</td>';
      h += '<td>' + esc(d.model) + '</td>';
      h += '<td>' + (d.num_samples ?? '—') + '</td>';
      for (const m of mList) {{
        const v = d.metrics?.[m];
        const cls = best[m] === i && items.length > 1 ? ' class="best"' : '';
        h += '<td' + cls + '>' + (v != null ? v.toFixed(2) : '—') + '</td>';
      }}
      h += '</tr>';
    }});
    h += '</tbody></table>';
  }}

  if (!h) h = '<div style="text-align:center;padding:3rem;color:var(--text-muted);">No results to compare.</div>';
  el.innerHTML = h;
}}

renderOverview();
renderComparison();
</script>
</body>
</html>"""


def main():
    results = collect_results()
    if not results:
        print("No JSON result files found in results/")
        return

    print(f"Found {len(results)} result file(s):")
    for r in results:
        print(f"  {r['_path']}: {r['model']} ({r.get('num_samples', '?')} samples)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_html(results)
    OUTPUT_PATH.write_text(html_content, encoding="utf-8")
    print(f"\nGenerated {OUTPUT_PATH} ({len(html_content):,} bytes)")


if __name__ == "__main__":
    main()
