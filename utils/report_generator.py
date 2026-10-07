import base64, io
from datetime import datetime
from PIL import Image


def _b64(rgb):
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_html_report(analysis: dict, annotated_rgb, model_info: str) -> str:
    rows = "".join(f"<tr><td>{d['damage']}</td><td>{d['confidence']*100:.1f}%</td>"
                   f"<td>{d['area_pct']:.2f}%</td><td>{d['severity']}</td></tr>" for d in analysis["detections"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Damage Inspection Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:860px;margin:32px auto;color:#1a1f2b}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d5d9e0;padding:8px;text-align:left}}
th{{background:#f2f4f7}}img{{max-width:100%}}.note{{color:#5b6472;font-size:13px}}</style></head><body>
<h1>Vehicle Damage Inspection Report</h1>
<p>Inspection date/time: {datetime.now():%Y-%m-%d %H:%M:%S}</p>
<h2>Summary</h2><ul>
<li>Detected damages: <b>{analysis['count']}</b></li>
<li>Overall severity (AI estimate): <b>{analysis['severity']}</b></li>
<li>Estimated affected area: <b>{analysis['total_area_pct']:.1f}%</b> of image ({analysis['area_method']}; reference: {analysis['area_reference']})</li>
<li>Mean model confidence: <b>{analysis['mean_confidence']*100:.1f}%</b></li>
<li>Repair cost: not estimated (no labeled cost data available)</li></ul>
<img src="data:image/jpeg;base64,{_b64(annotated_rgb)}">
<h2>Detections</h2><table><tr><th>Damage</th><th>Confidence</th><th>Area</th><th>Severity</th></tr>{rows}</table>
<p class="note">Model: {model_info}. Severity and area are AI estimates, not a professional insurance or mechanical assessment.</p>
</body></html>"""
