"""
SAR Oil Spill Detection - Streamlit GUI
========================================
Run with:  streamlit run app.py
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "gui_results")

sys.path.insert(0, BASE_DIR)

from train_models import UNet, AttentionUNet, DeepLabV3Plus
from inference_postprocess import (
    apply_postprocessing,
    create_overlay,
    NUM_CLASSES,
    IMG_SIZE,
    DEVICE,
)

# ── Config ─────────────────────────────────────────────────────

MODEL_CONFIG = {
    "U-Net": {
        "file": "unet.pth",
        "class": UNet,
        "params": "~7.8M",
        "tag": "Baseline",
        "tag_color": "#64748b",
    },
    "Attention U-Net": {
        "file": "attention_unet.pth",
        "class": AttentionUNet,
        "params": "~7.9M",
        "tag": "Best Accuracy",
        "tag_color": "#3b82f6",
    },
    "DeepLabV3+": {
        "file": "deeplabv3.pth",
        "class": DeepLabV3Plus,
        "params": "~4.1M",
        "tag": "Lightweight",
        "tag_color": "#8b5cf6",
    },
}

MODEL_DESCRIPTIONS = {
    "U-Net": "Classic encoder-decoder with skip connections. 4 resolution levels, ~7.8M parameters.",
    "Attention U-Net": "U-Net enhanced with attention gates at skip connections. Best overall accuracy. ~7.9M parameters.",
    "DeepLabV3+": "ASPP module with multi-scale context via dilated convolutions. ~4.1M parameters.",
}

# ── Page config ─────────────────────────────────────────────────

st.set_page_config(
    page_title="SAR Oil Spill Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS + animations ────────────────────────────────────────────

st.markdown("""
<style>
/* ─── Reset & base ───────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* ─── Animated background ────────────────────────────────────── */
.stApp {
    background: #080d1a;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 20%, rgba(59,130,246,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,92,246,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(15,23,42,0.9) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 8s ease-in-out infinite alternate;
}
@keyframes bgPulse {
    0%   { opacity: 0.7; }
    100% { opacity: 1; }
}

/* ─── Floating grid overlay ──────────────────────────────────── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ─── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0c1530 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.12) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: #94a3b8 !important;
    font-size: 0.87rem;
    line-height: 1.65;
}

/* ─── Sidebar brand ──────────────────────────────────────────── */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 0.4rem 0 1rem 0;
}
.sb-brand-icon {
    font-size: 2rem;
    animation: floatIcon 3s ease-in-out infinite;
}
@keyframes floatIcon {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-4px); }
}
.sb-brand-name {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.01em;
}
.sb-brand-sub {
    font-size: 0.7rem;
    color: #475569;
}

/* ─── Sidebar section label ──────────────────────────────────── */
.sb-label {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 1.3rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(59,130,246,0.18);
}

/* ─── Model card ─────────────────────────────────────────────── */
.sb-model-card {
    background: rgba(59,130,246,0.06);
    border: 1px solid rgba(59,130,246,0.16);
    border-radius: 10px;
    padding: 11px 13px;
    margin-top: 6px;
    transition: border-color 0.2s;
}
.sb-model-card:hover { border-color: rgba(59,130,246,0.35); }
.sb-model-card p {
    color: #94a3b8 !important;
    font-size: 0.81rem !important;
    margin: 0 !important;
    line-height: 1.55 !important;
}

/* ─── Steps ──────────────────────────────────────────────────── */
.step { display:flex; align-items:flex-start; gap:10px; margin-bottom:8px; }
.step-n {
    background: linear-gradient(135deg,#3b82f6,#8b5cf6);
    color:#fff; font-size:0.65rem; font-weight:700;
    width:18px; height:18px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; margin-top:1px;
}
.step-t { font-size:0.81rem; color:#94a3b8; line-height:1.5; }

/* ─── Legend ─────────────────────────────────────────────────── */
.leg {
    display:flex; align-items:center; gap:10px;
    padding:7px 10px; border-radius:8px; margin-bottom:5px;
    background:rgba(255,255,255,0.02);
    border:1px solid rgba(255,255,255,0.05);
    transition: background 0.2s;
}
.leg:hover { background:rgba(255,255,255,0.05); }
.leg-dot { width:11px; height:11px; border-radius:50%; flex-shrink:0; }
.leg-name { font-size:0.81rem; color:#cbd5e1; font-weight:500; }
.leg-desc { font-size:0.71rem; color:#475569; }

/* ─── Divider ────────────────────────────────────────────────── */
.divider {
    border:none;
    border-top:1px solid rgba(255,255,255,0.05);
    margin:1.2rem 0;
}

/* ─── Section label ──────────────────────────────────────────── */
.sec-label {
    font-size:0.66rem; font-weight:700;
    letter-spacing:0.13em; text-transform:uppercase;
    color:#334155; margin-bottom:0.75rem;
}

/* ─── Header ─────────────────────────────────────────────────── */
.hdr {
    text-align:center;
    padding:3rem 1rem 2rem 1rem;
    position:relative;
}
.hdr-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(59,130,246,0.1);
    border:1px solid rgba(59,130,246,0.25);
    color:#60a5fa;
    font-size:0.68rem; font-weight:600; letter-spacing:0.1em;
    text-transform:uppercase;
    padding:4px 14px; border-radius:100px;
    margin-bottom:1.1rem;
    animation: pillGlow 2.5s ease-in-out infinite alternate;
}
@keyframes pillGlow {
    0%   { box-shadow:0 0 0 rgba(59,130,246,0); }
    100% { box-shadow:0 0 14px rgba(59,130,246,0.35); }
}
.hdr-title {
    font-size:clamp(2rem,4vw,3rem);
    font-weight:800;
    background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 50%,#f472b6 100%);
    background-size:200% 200%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    line-height:1.1;
    margin-bottom:0.9rem;
    letter-spacing:-0.025em;
    animation: gradientShift 5s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position:0% 50%; }
    50%  { background-position:100% 50%; }
    100% { background-position:0% 50%; }
}
.hdr-sub {
    font-size:0.97rem; color:#4b5563;
    max-width:580px; margin:0 auto;
    line-height:1.65;
}

/* ─── Upload zone ────────────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    background:rgba(59,130,246,0.03) !important;
    border:2px dashed rgba(59,130,246,0.3) !important;
    border-radius:16px !important;
    transition:border-color 0.25s, background 0.25s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color:rgba(59,130,246,0.6) !important;
    background:rgba(59,130,246,0.07) !important;
}
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
    color:#4b5563 !important;
}

/* ─── File badge ─────────────────────────────────────────────── */
.file-badge {
    display:inline-flex; align-items:center; gap:7px;
    background:rgba(16,185,129,0.08);
    border:1px solid rgba(16,185,129,0.22);
    border-radius:100px; padding:5px 13px;
    margin-top:8px; font-size:0.78rem;
    color:#34d399; font-weight:500;
    animation: fadeUp 0.35s ease;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(6px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ─── Run button ─────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#3b82f6 0%,#8b5cf6 100%) !important;
    color:#fff !important; border:none !important;
    border-radius:14px !important;
    padding:0.8rem 2rem !important;
    font-size:0.98rem !important; font-weight:600 !important;
    letter-spacing:0.01em !important;
    transition:opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow:0 4px 24px rgba(59,130,246,0.4) !important;
    width:100% !important;
    position:relative; overflow:hidden;
}
div[data-testid="stButton"] > button::after {
    content:'';
    position:absolute; inset:0;
    background:linear-gradient(135deg,rgba(255,255,255,0.08),transparent);
    pointer-events:none;
}
div[data-testid="stButton"] > button:hover {
    opacity:0.92 !important;
    transform:translateY(-2px) !important;
    box-shadow:0 8px 32px rgba(59,130,246,0.55) !important;
}
div[data-testid="stButton"] > button:active {
    transform:translateY(0) !important;
}

/* ─── Preview card ───────────────────────────────────────────── */
.preview-card {
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:18px; padding:1.2rem;
    animation: fadeUp 0.4s ease;
    position:relative; overflow:hidden;
}
.preview-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(59,130,246,0.4),transparent);
}

/* ─── Image cards ────────────────────────────────────────────── */
.img-card {
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:16px; padding:1rem;
    transition:transform 0.25s, box-shadow 0.25s;
    animation: fadeUp 0.5s ease;
    position:relative; overflow:hidden;
}
.img-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(139,92,246,0.4),transparent);
}
.img-card:hover {
    transform:translateY(-3px);
    box-shadow:0 12px 40px rgba(0,0,0,0.45);
}
.img-card-title {
    font-size:0.68rem; font-weight:700;
    letter-spacing:0.1em; text-transform:uppercase;
    color:#334155; margin-bottom:0.75rem;
    padding-bottom:0.55rem;
    border-bottom:1px solid rgba(255,255,255,0.05);
}
.img-wrap img {
    border-radius:10px;
    transition:transform 0.3s ease, filter 0.3s ease;
    width:100%;
}
.img-wrap img:hover {
    transform:scale(1.04);
    filter:brightness(1.06);
}

/* ─── Scan-line animation overlay on result images ───────────── */
.img-scan {
    position:relative; overflow:hidden; border-radius:10px;
}
.img-scan::after {
    content:'';
    position:absolute; left:0; right:0; top:-100%; height:30%;
    background:linear-gradient(to bottom,
        transparent 0%,
        rgba(59,130,246,0.06) 50%,
        transparent 100%);
    animation: scanLine 3s linear infinite;
    pointer-events:none;
}
@keyframes scanLine {
    0%   { top:-30%; }
    100% { top:130%; }
}

/* ─── Metric cards ───────────────────────────────────────────── */
.m-card {
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:1.1rem 0.9rem;
    text-align:center;
    transition:transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    animation:fadeUp 0.5s ease;
    position:relative; overflow:hidden;
}
.m-card::before {
    content:'';
    position:absolute; bottom:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#3b82f6,#8b5cf6);
    transform:scaleX(0);
    transition:transform 0.3s ease;
    transform-origin:left;
}
.m-card:hover { transform:translateY(-2px); box-shadow:0 8px 28px rgba(0,0,0,0.3); }
.m-card:hover::before { transform:scaleX(1); }
.m-val {
    font-size:1.6rem; font-weight:800; line-height:1.1;
    background:linear-gradient(135deg,#60a5fa,#a78bfa);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
}
.m-lbl {
    font-size:0.68rem; font-weight:700;
    letter-spacing:0.09em; text-transform:uppercase;
    color:#334155; margin-top:5px;
}

/* ─── Confidence gauge bar ───────────────────────────────────── */
.gauge-wrap {
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:1.1rem 1.2rem;
    animation:fadeUp 0.5s ease;
}
.gauge-label { font-size:0.68rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase; color:#334155; margin-bottom:8px; }
.gauge-track {
    background:rgba(255,255,255,0.06);
    border-radius:100px; height:10px; overflow:hidden;
}
.gauge-fill {
    height:100%; border-radius:100px;
    background:linear-gradient(90deg,#3b82f6,#8b5cf6,#f472b6);
    background-size:200% 100%;
    animation:gaugeGrow 1s cubic-bezier(0.34,1.56,0.64,1) both,
              gaugeShimmer 2s linear infinite;
    transition:width 0.8s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes gaugeGrow {
    from { transform:scaleX(0); transform-origin:left; }
    to   { transform:scaleX(1); transform-origin:left; }
}
@keyframes gaugeShimmer {
    0%   { background-position:0% 0%; }
    100% { background-position:200% 0%; }
}
.gauge-val { font-size:0.85rem; font-weight:700; color:#60a5fa; margin-top:6px; }

/* ─── Class cards ────────────────────────────────────────────── */
.cls-card {
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:1.1rem 1.2rem;
    transition:transform 0.2s, box-shadow 0.2s;
    animation:fadeUp 0.5s ease;
}
.cls-card:hover { transform:translateY(-2px); box-shadow:0 10px 32px rgba(0,0,0,0.35); }
.cls-bar-track {
    background:rgba(255,255,255,0.05);
    border-radius:100px; height:5px;
    margin-top:10px; overflow:hidden;
}
.cls-bar-fill {
    height:100%; border-radius:100px;
    animation:gaugeGrow 0.9s cubic-bezier(0.34,1.56,0.64,1) both;
}

/* ─── Status saved ───────────────────────────────────────────── */
.saved-badge {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(16,185,129,0.08);
    border:1px solid rgba(16,185,129,0.22);
    border-radius:100px; padding:6px 16px;
    font-size:0.79rem; color:#34d399; font-weight:500;
    animation:fadeUp 0.4s ease;
}

/* ─── Info box ───────────────────────────────────────────────── */
.info-box {
    background:rgba(59,130,246,0.06);
    border:1px solid rgba(59,130,246,0.18);
    border-left:3px solid #3b82f6;
    border-radius:12px; padding:14px 18px;
    font-size:0.85rem; color:#94a3b8; line-height:1.65;
    animation:fadeUp 0.4s ease;
}
.info-box strong { color:#60a5fa; }

/* ─── Selectbox ──────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.09) !important;
    border-radius:10px !important; color:#e2e8f0 !important;
    transition:border-color 0.2s !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color:rgba(59,130,246,0.38) !important;
}

/* ─── Spinner ────────────────────────────────────────────────── */
[data-testid="stSpinner"] { color:#60a5fa !important; }

/* ─── Error box ──────────────────────────────────────────────── */
.err-box {
    background:rgba(239,68,68,0.08);
    border:1px solid rgba(239,68,68,0.25);
    border-radius:12px; padding:13px 17px;
    color:#f87171; font-size:0.88rem;
    animation:fadeUp 0.3s ease;
}

/* ─── Hide chrome ────────────────────────────────────────────── */
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ───────────────────────────────────────────────

@st.cache_resource
def load_model(model_name):
    config = MODEL_CONFIG[model_name]
    model_path = os.path.join(MODEL_DIR, config["file"])
    if not os.path.exists(model_path):
        return None
    model = config["class"](in_channels=1, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ── Inference ───────────────────────────────────────────────────

def run_inference(model, image_gray):
    resized = cv2.resize(image_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    prob_map  = probs[0, 1].cpu().numpy()
    pred_mask = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred_mask, prob_map

# ── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-icon">🛰️</span>
        <div>
            <div class="sb-brand-name">OilSense AI</div>
            <div class="sb-brand-sub">SAR Detection Platform</div>
        </div>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">⬡ &nbsp;Model Selection</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("model", list(MODEL_CONFIG.keys()), index=1, label_visibility="collapsed")

    cfg = MODEL_CONFIG[selected_model]
    st.markdown(f"""
    <div class="sb-model-card">
        <p>{MODEL_DESCRIPTIONS[selected_model]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider"><div class="sb-label">◎ &nbsp;How to Use</div>', unsafe_allow_html=True)
    for i, s in enumerate([
        "Upload a preprocessed SAR image (.png)",
        "Select a segmentation model above",
        "Click <strong>Run Prediction</strong>",
        "Explore results and metrics below",
    ]):
        st.markdown(f'<div class="step"><div class="step-n">{i+1}</div><div class="step-t">{s}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider"><div class="sb-label">◉ &nbsp;Color Legend</div>', unsafe_allow_html=True)
    for dot, name, desc in [
        ("#ef4444", "Oil Spill",       "High confidence (&gt;55%) + elongated"),
        ("#eab308", "Look-Alike",      "Medium confidence or round shape"),
        ("#1e293b", "Sea / Background","No detection", ),
    ]:
        border = "border:1px solid rgba(255,255,255,0.1);" if dot == "#1e293b" else ""
        st.markdown(f"""
        <div class="leg">
            <div class="leg-dot" style="background:{dot};{border}"></div>
            <div>
                <div class="leg-name">{name}</div>
                <div class="leg-desc">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <hr class="divider">
    <div style="font-size:0.68rem;color:#1e293b;text-align:center;">
        Deep Learning · Semantic Segmentation<br>PyTorch · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────

st.markdown("""
<div class="hdr">
    <div class="hdr-pill">🛰️ &nbsp; AI-Powered Remote Sensing</div>
    <div class="hdr-title">SAR Oil Spill Detection</div>
    <div class="hdr-sub">
        Semantic segmentation of Synthetic Aperture Radar imagery using
        state-of-the-art deep learning — U-Net, Attention U-Net &amp; DeepLabV3+.
    </div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Upload ───────────────────────────────────────────────────────

st.markdown('<div class="sec-label">📡 &nbsp; SAR Image Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "SAR Image",
    type=["png"],
    help="Upload a grayscale PNG image from the preprocessed dataset (256×256).",
    label_visibility="collapsed",
)
if uploaded_file:
    st.markdown(f'<div class="file-badge">✓ &nbsp; {uploaded_file.name}</div>', unsafe_allow_html=True)

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

# ── Prediction ───────────────────────────────────────────────────

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image_gray is None:
        st.markdown('<div class="err-box">✗ &nbsp; Failed to read image. Please upload a valid PNG file.</div>', unsafe_allow_html=True)
    else:
        # Preview
        st.markdown('<div class="sec-label">🖼️ &nbsp; Uploaded Image Preview</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns([1, 2, 1])
        with p2:
            st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.markdown('<div class="img-card-title">SAR Input · Grayscale</div>', unsafe_allow_html=True)
            st.markdown('<div class="img-scan">', unsafe_allow_html=True)
            st.image(image_gray, use_container_width=True, clamp=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        # Run button
        b1, b2, b3 = st.columns([1, 2, 1])
        with b2:
            run_predict = st.button(f"⚡  Run Prediction  ·  {selected_model}", use_container_width=True)

        if run_predict:
            with st.spinner(f"Loading {selected_model} weights…"):
                model = load_model(selected_model)

            if model is None:
                st.markdown(f'<div class="err-box" style="margin-top:1rem;">✗ &nbsp; Model file not found: <code>{MODEL_CONFIG[selected_model]["file"]}</code>. Place trained models in <code>models/</code>.</div>', unsafe_allow_html=True)
            else:
                prog = st.progress(0, text="Preparing image…")
                prog.progress(20, text="Running inference…")
                pred_mask, prob_map = run_inference(model, image_gray)
                prog.progress(65, text="Applying post-processing…")
                final_mask, num_regions, region_stats, region_centroids = apply_postprocessing(pred_mask, prob_map)
                prog.progress(100, text="Done ✓")
                prog.empty()

                st.session_state["results"] = {
                    "image_gray":    image_gray,
                    "final_mask":    final_mask,
                    "prob_map":      prob_map,
                    "num_regions":   num_regions,
                    "selected_model": selected_model,
                    "filename":      uploaded_file.name,
                }

        # ── Results display ──────────────────────────────────────

        if "results" in st.session_state:
            res        = st.session_state["results"]
            image_gray_r = res["image_gray"]
            final_mask   = res["final_mask"]
            prob_map     = res["prob_map"]
            num_regions  = res["num_regions"]

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">📊 &nbsp; Prediction Results</div>', unsafe_allow_html=True)

            display_img      = cv2.resize(image_gray_r, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            pred_overlay     = create_overlay(display_img, final_mask)
            pred_overlay_rgb = cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB)

            # Three image panels
            c1, c2, c3 = st.columns(3, gap="medium")

            with c1:
                st.markdown('<div class="img-card"><div class="img-card-title">Original SAR Image</div><div class="img-scan img-wrap">', unsafe_allow_html=True)
                st.image(display_img, use_container_width=True, clamp=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

            with c2:
                st.markdown(f'<div class="img-card"><div class="img-card-title">Prediction · {res["selected_model"]}</div><div class="img-wrap">', unsafe_allow_html=True)
                st.image(pred_overlay_rgb, use_container_width=True, clamp=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

            with c3:
                st.markdown('<div class="img-card"><div class="img-card-title">Confidence Heatmap</div><div class="img-wrap">', unsafe_allow_html=True)
                fig_hm, ax_hm = plt.subplots(figsize=(4, 4))
                fig_hm.patch.set_facecolor("none")
                ax_hm.set_facecolor("none")
                hm = ax_hm.imshow(prob_map, cmap="inferno", vmin=0, vmax=1)
                ax_hm.axis("off")
                cbar = plt.colorbar(hm, ax=ax_hm, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(colors="#64748b", labelsize=7)
                plt.tight_layout(pad=0)
                st.pyplot(fig_hm, use_container_width=True)
                plt.close(fig_hm)
                st.markdown('</div></div>', unsafe_allow_html=True)

            # ── Metrics ──────────────────────────────────────────

            detected_pixels  = (final_mask >= 1)
            mean_conf        = float(np.mean(prob_map[detected_pixels]) * 100) if np.any(detected_pixels) else 0.0
            oil_pixels       = int(np.sum(final_mask == 1))
            lookalike_pixels = int(np.sum(final_mask == 2))
            total_detected   = oil_pixels + lookalike_pixels
            total_pixels     = final_mask.size
            spill_area_pct   = (total_detected / total_pixels) * 100

            st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">📈 &nbsp; Key Metrics</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4, gap="small")
            for col, val, lbl in [
                (m1, f"{mean_conf:.1f}%",        "Mean Confidence"),
                (m2, f"{spill_area_pct:.2f}%",   "Spill Area"),
                (m3, str(num_regions),            "Regions Detected"),
                (m4, f"{oil_pixels:,}",           "Oil Pixels"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="m-card">
                        <div class="m-val">{val}</div>
                        <div class="m-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            # Confidence gauge
            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
            gauge_w = int(mean_conf)
            st.markdown(f"""
            <div class="gauge-wrap">
                <div class="gauge-label">Detection Confidence</div>
                <div class="gauge-track">
                    <div class="gauge-fill" style="width:{gauge_w}%;"></div>
                </div>
                <div class="gauge-val">{mean_conf:.1f}% average probability over detected pixels</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Class breakdown ───────────────────────────────────

            st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">🗂️ &nbsp; Predicted Class Breakdown</div>', unsafe_allow_html=True)

            sea_pixels = int(np.sum(final_mask == 0))
            cls_data = [
                ("#ef4444", "Oil Spill",       oil_pixels,       "linear-gradient(90deg,#ef4444,#f97316)"),
                ("#eab308", "Look-Alike",      lookalike_pixels, "linear-gradient(90deg,#eab308,#84cc16)"),
                ("#334155", "Sea / Background",sea_pixels,       "linear-gradient(90deg,#334155,#475569)"),
            ]
            cl1, cl2, cl3 = st.columns(3, gap="medium")
            for col, (color, name, px, grad) in zip([cl1, cl2, cl3], cls_data):
                pct = (px / total_pixels) * 100
                with col:
                    st.markdown(f"""
                    <div class="cls-card">
                        <div style="display:flex;align-items:center;gap:9px;margin-bottom:8px;">
                            <div style="width:11px;height:11px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                            <div style="font-size:0.82rem;font-weight:600;color:#cbd5e1;">{name}</div>
                        </div>
                        <div style="font-size:1.55rem;font-weight:800;color:#e2e8f0;">{pct:.2f}%</div>
                        <div style="font-size:0.73rem;color:#334155;margin-top:3px;">{px:,} pixels</div>
                        <div class="cls-bar-track">
                            <div class="cls-bar-fill" style="width:{pct:.1f}%;background:{grad};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # ── Save ──────────────────────────────────────────────

            st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            save_name = os.path.splitext(res["filename"])[0]
            save_path = os.path.join(OUTPUT_DIR, f"{save_name}_{res['selected_model']}.png")

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(display_img, cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("Original SAR Image", fontsize=13, fontweight="bold"); axes[0].axis("off")
            axes[1].imshow(pred_overlay_rgb)
            axes[1].set_title(f"Prediction ({res['selected_model']})", fontsize=13, fontweight="bold"); axes[1].axis("off")
            hm_p = axes[2].imshow(prob_map, cmap="inferno", vmin=0, vmax=1)
            axes[2].set_title("Confidence Heatmap", fontsize=13, fontweight="bold"); axes[2].axis("off")
            fig.colorbar(hm_p, ax=axes[2], fraction=0.046, pad=0.04)
            fig.text(0.5, 0.01, "■ Oil Spill (Red)  |  ■ Look-Alike (Yellow)  |  ■ Sea (Black)",
                     ha="center", fontsize=10, style="italic",
                     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
            fig.suptitle(res["filename"], fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0.04, 1, 0.95])
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            st.markdown(f'<div class="saved-badge">✓ &nbsp; Result saved → <code style="color:#6ee7b7;">{save_path}</code></div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-box">
        <strong>Get started:</strong> Upload a preprocessed SAR image (.png) using the upload area above.
        Test images can be found in <code>data/processed/train/images/</code>, or preprocess new
        SAR data using <code>preprocess.py</code>.
    </div>
    """, unsafe_allow_html=True)
