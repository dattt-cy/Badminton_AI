"""Render cached ShuttleSet hard examples into an HTML visual audit."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def render_strip(cache_path: Path) -> np.ndarray:
    tensor = np.load(cache_path)  # T,C,H,W uint8
    indices = np.linspace(0, len(tensor) - 1, 5).round().astype(int)
    frames = []
    for index in indices:
        rgb = np.transpose(tensor[index], (1, 2, 0))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames.append(cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_CUBIC))
    return np.concatenate(frames, axis=1)


def main() -> None:
    args = parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    missing = []
    for index, item in enumerate(queue, 1):
        cache_path = args.cache_dir / f"{item['sample_id']}.npy"
        if not cache_path.is_file():
            missing.append(item["sample_id"])
            continue
        strip = render_strip(cache_path)
        name = f"{index:03d}_{item['sample_id']}.jpg"
        cv2.imwrite(str(image_dir / name), strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rendered.append((item, name))

    rows = []
    for index, (item, name) in enumerate(rendered, 1):
        truth = html.escape(f"{item['stroke_side']} {item['expected_stroke']}")
        prediction = html.escape(f"{item['predicted_side']} {item['predicted_stroke']}")
        rows.append(f"""
        <article class="card">
          <h3>{index:03d} · {html.escape(item['sample_id'])}</h3>
          <img src="images/{html.escape(name)}" loading="lazy">
          <div class="meta">
            <span class="truth">GT: {truth}</span>
            <span class="prediction">Pred: {prediction} ({item['stroke_probability']:.3f})</span>
            <span>raw: {html.escape(item['raw_label'])}</span>
            <span>court: {html.escape(item['player_side'])}</span>
          </div>
          <div class="audit" data-sample="{html.escape(item['sample_id'])}">
            <label><input type="radio" name="s{index}" value="keep"> Keep label</label>
            <label><input type="radio" name="s{index}" value="model"> Model looks right</label>
            <label><input type="radio" name="s{index}" value="crop"> Bad crop</label>
            <label><input type="radio" name="s{index}" value="unclear"> Unclear</label>
          </div>
        </article>""")
    document = f"""<!doctype html><meta charset="utf-8">
    <title>ShuttleSet hard-example audit</title>
    <style>
    body{{font:14px system-ui;background:#111;color:#eee;margin:20px}}
    h1{{margin-bottom:4px}} .summary{{color:#bbb;margin-bottom:20px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(580px,1fr));gap:14px}}
    .card{{background:#1d1d1d;border:1px solid #444;border-radius:8px;padding:10px}}
    img{{width:100%;background:#000}} .meta,.audit{{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}}
    .truth{{color:#7ee787}} .prediction{{color:#ff7b72}}
    </style>
    <h1>ShuttleSet hard-example audit</h1>
    <div class="summary">Rendered {len(rendered)} samples · missing {len(missing)} · frames: early → contact → late</div>
    <p><button id="export">Export audit JSON</button> <button id="clear">Clear saved choices</button>
    <span id="progress">0/{len(rendered)} reviewed</span></p>
    <main class="grid">{''.join(rows)}</main>
    <script>
    const key='shuttleset-hard-example-audit-v1';
    let state=JSON.parse(localStorage.getItem(key)||'{{}}');
    const audits=[...document.querySelectorAll('.audit')];
    function refresh(){{
      audits.forEach(div=>{{const value=state[div.dataset.sample]; if(value){{
        const input=div.querySelector(`input[value="${{value}}"]`); if(input) input.checked=true;
      }}}});
      document.getElementById('progress').textContent=`${{Object.keys(state).length}}/{len(rendered)} reviewed`;
    }}
    audits.forEach(div=>div.addEventListener('change',event=>{{
      state[div.dataset.sample]=event.target.value; localStorage.setItem(key,JSON.stringify(state)); refresh();
    }}));
    document.getElementById('export').onclick=()=>{{
      const blob=new Blob([JSON.stringify(state,null,2)],{{type:'application/json'}});
      const link=document.createElement('a'); link.href=URL.createObjectURL(blob); link.download='audit_decisions.json'; link.click();
      URL.revokeObjectURL(link.href);
    }};
    document.getElementById('clear').onclick=()=>{{if(confirm('Clear all saved choices?')){{state={{}};localStorage.removeItem(key);location.reload();}}}};
    refresh();
    </script>"""
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    (args.output_dir / "missing.json").write_text(
        json.dumps(missing, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rendered": len(rendered), "missing": len(missing),
                      "html": str((args.output_dir / 'index.html').resolve())}, indent=2))


if __name__ == "__main__":
    main()
