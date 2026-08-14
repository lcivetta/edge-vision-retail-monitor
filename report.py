"""Generate a zero-dependency local HTML index of stored video runs."""

from __future__ import annotations

import html
import json
from pathlib import Path

import config


def main() -> None:
    config.ensure_directories()
    cards = []
    for manifest_path in sorted(config.RUNS_DIR.glob("*/manifest.json")):
        run = json.loads(manifest_path.read_text())
        run_dir = manifest_path.parent
        video = Path(run["output_video"])
        relative_video = video.relative_to(config.OUTPUT_DIR)
        alerts = sorted((run_dir / "alerts").glob("*.jpg"))
        alert_html = "".join(
            f'<img src="{html.escape(str(p.relative_to(config.OUTPUT_DIR)))}" alt="alert" loading="lazy">'
            for p in alerts
        ) or "<p>No review screenshot generated.</p>"
        cards.append(f"""
        <article>
          <h2>{html.escape(run['run_name'])}</h2>
          <dl>
            <dt>Expected label</dt><dd>{html.escape(run['expected_label'])}</dd>
            <dt>System outcome</dt><dd>{html.escape(run['system_outcome'])}</dd>
            <dt>Evaluation</dt><dd>{html.escape(run['evaluation'])}</dd>
            <dt>Maximum review score</dt><dd>{run['max_risk']}</dd>
            <dt>Detected classes</dt><dd><code>{html.escape(run['detected_classes'])}</code></dd>
          </dl>
          <video controls preload="metadata" src="{html.escape(str(relative_video))}"></video>
          <div class="alerts">{alert_html}</div>
        </article>""")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge Vision Run Results</title><style>
body{{font:16px system-ui;margin:0;background:#101418;color:#edf2f7}}main{{max-width:1100px;margin:auto;padding:24px}}
article{{background:#1b2229;padding:20px;margin:20px 0;border-radius:12px}}dl{{display:grid;grid-template-columns:180px 1fr;gap:8px}}
dt{{color:#9fb3c8}}dd{{margin:0}}video{{width:100%;max-height:560px;background:#000;margin-top:16px}}img{{max-width:48%;margin:8px}}
code{{white-space:pre-wrap}}a{{color:#8bc5ff}}
</style></head><body><main><h1>Edge Vision Run Results</h1>
<p>Observable-event prototype results. PASS/FAIL compares a review/no-review outcome with the staged dataset label; it does not declare intent.</p>
{''.join(cards) or '<p>No named runs yet.</p>'}</main></body></html>"""
    destination = config.OUTPUT_DIR / "report.html"
    destination.write_text(document)
    print(destination)


if __name__ == "__main__":
    main()
