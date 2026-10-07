import cv2, numpy as np, streamlit as st
from datetime import datetime
from inference.detector import Detector, DEFAULT_WEIGHTS
from inference.damage_analysis import analyze
from utils.visualization import annotate
from utils.report_generator import build_html_report

st.set_page_config(page_title="AI Car Damage Inspection", page_icon="🚘", layout="wide")

st.markdown("""<style>
:root{--bg:#0a1020;--panel:#0f1a30;--line:#1c2a47;--txt:#e8eefc;--mut:#8b9bbd;--blue:#2f80ed}
.stApp,[data-testid="stHeader"]{background:var(--bg)!important;color:var(--txt)}
.stApp p,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp [data-testid="stWidgetLabel"] p{color:var(--txt)}
.block-container{padding:1.6rem 2.2rem;max-width:1400px}
[data-testid="stSidebar"]{background:#0c1426!important;border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{padding:1.2rem}
/* brand + header */
.brand{display:flex;gap:12px;align-items:center;margin-bottom:18px}
.brand .bi{font-size:34px}.brand .bt{font-weight:700;font-size:17px;line-height:1.2}
.brand .bs{font-size:11px;color:var(--mut)}
.nav{background:#16264a;border-radius:10px;padding:10px 14px;font-weight:600;margin-bottom:18px}
.sec{color:#7fb0ff;font-weight:600;margin:6px 0 8px}
.info{background:#0f1c38;border:1px solid var(--line);border-radius:12px;padding:14px;font-size:13px;color:var(--mut);line-height:1.7}
.info b{color:var(--txt)}
.hero{display:flex;align-items:center;gap:18px;margin-bottom:18px}
.hero .hi{width:76px;height:76px;border-radius:16px;background:#15305f;display:grid;place-items:center;font-size:36px;border:1px solid #24427a}
.hero h1{margin:0;font-size:34px;font-weight:700}.hero p{margin:2px 0 0;color:var(--mut)!important;font-size:16px}
.online{margin-left:auto;background:#10213f;border:1px solid var(--line);padding:6px 14px;border-radius:20px;font-size:12px;color:var(--txt)}
.online:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:8px}
/* uploader */
[data-testid="stFileUploader"] section{background:#0c162d;border:1.5px dashed #2f5fb3;border-radius:14px;padding:18px}
[data-testid="stFileUploader"] *{color:var(--txt)!important}
[data-testid="stFileUploader"] small{color:var(--mut)!important}
[data-testid="stFileUploader"] button{background:var(--blue)!important;border:0;border-radius:8px;color:#fff!important}
/* buttons + inputs */
.stButton>button,.stDownloadButton>button{background:var(--blue);color:#fff;border:0;border-radius:10px;font-weight:600;padding:.6rem 1.6rem}
.stButton>button:hover,.stDownloadButton>button:hover{background:#1f6fe0;color:#fff}
[data-baseweb="input"],[data-baseweb="select"]>div{background:#0c162d!important;border-color:var(--line)!important}
[data-baseweb="input"] input{color:var(--txt)!important}
/* metric cards */
.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:20px 0}
.card{display:flex;gap:12px;align-items:center;padding:16px 18px;border-radius:14px;border:1px solid}
.ic{width:44px;height:44px;min-width:44px;border-radius:50%;display:grid;place-items:center;font-size:20px}
.k{font-size:11px;letter-spacing:.07em;text-transform:uppercase;font-weight:700}
.v{font-size:30px;font-weight:700;line-height:1.15;color:var(--txt)}
.red{background:#2a1119;border-color:#6b1f2c}.red .ic{background:#e5384f}.red .k{color:#ff8b9b}
.blue{background:#0f1f3d;border-color:#1f4a8f}.blue .ic{background:#2f80ed}.blue .k{color:#7fb0ff}
.orange{background:#2a1c0d;border-color:#7a4a12}.orange .ic{background:#f59e0b}.orange .k{color:#fbbf6a}
.purple{background:#1c1640;border-color:#4a3a9a}.purple .ic{background:#8b5cf6}.purple .k{color:#b9a3ff}
.green{background:#0b2a26;border-color:#17695c}.green .ic{background:#10b981}.green .k{color:#5eead4}
/* panels */
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);border:1px solid var(--line)!important;border-radius:16px}
.ph{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.pi{width:42px;height:42px;border-radius:10px;background:#16264a;display:grid;place-items:center;font-size:19px}
.pt{font-weight:700;font-size:18px}.ps{font-size:13px;color:var(--mut)}
.pill{margin-left:auto;background:#10213f;border:1px solid var(--line);padding:5px 12px;border-radius:18px;font-size:12px}
[data-testid="stImage"] img{width:100%;border-radius:10px}
/* table */
.tbl{width:100%;border-collapse:collapse;font-size:14px}
.tbl th{color:var(--mut);text-align:left;font-weight:600;padding:10px 12px;border-bottom:1px solid var(--line)}
.tbl td{padding:12px;border-bottom:1px solid #15223d}
.chip{padding:3px 12px;border-radius:14px;font-size:12px;font-weight:700}
.c-High{background:#3a1520;color:#ff8b9b}.c-Medium{background:#3a2810;color:#fbbf6a}.c-Low{background:#0f3328;color:#5eead4}
.note{color:var(--mut)!important;font-size:12.5px;margin-top:10px}
@media(max-width:1000px){.cards{grid-template-columns:repeat(2,1fr)}}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load(path):
    return Detector(path)


def card(color, icon, label, value):
    return (f'<div class="card {color}"><div class="ic">{icon}</div>'
            f'<div><div class="k">{label}</div><div class="v">{value}</div></div></div>')


def panel_head(icon, title, sub, pill=""):
    p = f'<div class="pill">{pill}</div>' if pill else ""
    return (f'<div class="ph"><div class="pi">{icon}</div><div><div class="pt">{title}</div>'
            f'<div class="ps">{sub}</div></div>{p}</div>')


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="bi">🚘</div><div><div class="bt">AI Car Damage Inspection</div>'
                '<div class="bs">Automated vehicle damage detection &amp; assessment</div></div></div>'
                '<div class="nav">🏠&nbsp; Home</div><div class="sec">⚙️ Settings</div>', unsafe_allow_html=True)
    weights = st.text_input("Model path", DEFAULT_WEIGHTS)
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    imgsz = st.select_slider("Inference size", [320, 480, 640, 800, 960], value=640)

try:
    det = load(weights)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

with st.sidebar:
    show_masks = st.checkbox("Show segmentation masks", True, disabled=det.task != "segment",
                             help="Only available when the model is a segmentation model.")
    st.markdown(f'<div class="info"><b>Task:</b> {det.task}<br><b>Classes:</b> {", ".join(det.names.values())}<br>'
                f'<b>Params:</b> {det.n_params/1e6:.1f}M &nbsp;•&nbsp; <b>Size:</b> {det.size_mb:.1f} MB<br>'
                f'<b>Device:</b> {"GPU" if det.device != "cpu" else "CPU"}</div>', unsafe_allow_html=True)

# ---------------- Header + upload ----------------
st.markdown('<div class="hero"><div class="hi">🛡️</div><div><h1>AI Car Damage Inspection</h1>'
            '<p>Upload a vehicle image to detect and assess visible damage.</p></div>'
            '<div class="online">System Online</div></div>', unsafe_allow_html=True)

up = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png", "webp"])
if up is None:
    st.session_state.pop("out", None)
    st.info("Upload an image to begin.")
    st.stop()

raw = up.getvalue()
img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read this image. Try a different file.")
    st.stop()

key = (up.name, len(raw), conf, imgsz)
if st.session_state.get("key") != key:          # new image or settings -> clear stale result
    st.session_state.pop("out", None)
    st.session_state["key"] = key

if st.button("Analyze Vehicle"):
    with st.spinner("Analyzing..."):
        try:
            res, ms = det.predict(img, conf=conf, imgsz=imgsz)
            st.session_state["out"] = (analyze(res, det.names, ms), annotate(res, show_masks and det.task == "segment"))
        except Exception as e:
            st.error(f"Inference failed: {e}")

orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out = st.session_state.get("out")

# ---------------- Results ----------------
if out:
    a, ann = out
    sev = a["severity"]
    sev_color = {"High": "orange", "Medium": "orange", "Low": "green"}.get(sev, "blue")
    st.markdown('<div class="cards">' +
                card("red", "⚠️", "Damage detected", "Yes" if a["count"] else "No") +
                card("blue", "🔲", "Damaged areas", a["count"]) +
                card(sev_color, "✳️", "Severity", sev) +
                card("purple", "📐", "Damage area (est.)", f'{a["total_area_pct"]:.1f}%') +
                card("green", "🎯", "Confidence", f'{a["mean_confidence"]*100:.1f}%') + '</div>',
                unsafe_allow_html=True)
else:
    st.caption("Click **Analyze Vehicle** to run the inspection.")

left, right = st.columns(2)
with left.container(border=True):
    st.markdown(panel_head("🖼️", "Original", "Input image"), unsafe_allow_html=True)
    st.image(orig)
if out:
    n = a["count"]
    how = "segmentation masks" if det.task == "segment" else "bounding boxes"
    with right.container(border=True):
        st.markdown(panel_head("🧠", "AI Analysis", f"Detected damage with {how}",
                               f'🔴 {n} object{"s" if n != 1 else ""} detected'), unsafe_allow_html=True)
        st.image(ann)

    if a["detections"]:
        with st.container(border=True):
            st.markdown(panel_head("📋", "Detailed damage table", f'Inference time: {a["inference_ms"]:.0f} ms'),
                        unsafe_allow_html=True)
            rows = "".join(f'<tr><td>{d["damage"].title()}</td><td>{d["confidence"]*100:.1f}%</td>'
                           f'<td>{d["area_pct"]:.2f}%</td><td><span class="chip c-{d["severity"]}">{d["severity"]}</span></td></tr>'
                           for d in a["detections"])
            st.markdown(f'<table class="tbl"><tr><th>Damage</th><th>Confidence</th><th>Area</th><th>Severity</th></tr>{rows}</table>',
                        unsafe_allow_html=True)
    else:
        st.success("No damage detected at this confidence threshold. Try lowering it in the sidebar.")

    st.markdown(f'<p class="note">Area is computed from {a["area_method"]} relative to the {a["area_reference"]}'
                f'{" and includes background inside each box, so it overestimates true damage area" if det.task != "segment" else ""}. '
                'Severity is an AI estimate, not a professional insurance or mechanical assessment. '
                'Repair cost is not estimated because no labeled cost data exists.</p>', unsafe_allow_html=True)
    html = build_html_report(a, ann, f"{det.task} model, {det.n_params/1e6:.1f}M params")
    st.download_button("⬇️ Download report (HTML)", html,
                       f"damage_report_{datetime.now():%Y%m%d_%H%M%S}.html", "text/html")
