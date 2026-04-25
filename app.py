"""
SAR Oil Spill Detection - Streamlit GUI
========================================
Professional demo interface for SAR oil spill segmentation.
Upload a preprocessed SAR image (PNG), select a model, and
visualize the segmentation output with metrics.

Color Coding:
  RED    = Oil Spill (high-confidence detection)
  YELLOW = Look-Alike (medium-confidence or rounder shape)
  BLACK  = Sea / Background

Run with:  streamlit run app.py
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "gui_results")

# Add project root to path for imports
sys.path.insert(0, BASE_DIR)

from train_models import UNet, AttentionUNet, DeepLabV3Plus
from inference_postprocess import (
    apply_postprocessing,
    create_overlay,
    compute_image_metrics,
    NUM_CLASSES,
    IMG_SIZE,
    DEVICE,
    OVERLAY_COLORS_BGR,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Model file mapping
MODEL_CONFIG = {
    "U-Net": {
        "file": "unet.pth",
        "class": UNet,
    },
    "Attention U-Net": {
        "file": "attention_unet.pth",
        "class": AttentionUNet,
    },
    "DeepLabV3+": {
        "file": "deeplabv3.pth",
        "class": DeepLabV3Plus,
    },
}


# ============================================================
# MODEL LOADING (cached to avoid reloading)
# ============================================================

@st.cache_resource
def load_model(model_name):
    """Load a trained model from disk. Cached for performance."""
    config = MODEL_CONFIG[model_name]
    model_path = os.path.join(MODEL_DIR, config["file"])

    if not os.path.exists(model_path):
        return None

    # Instantiate model architecture
    model = config["class"](in_channels=1, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# INFERENCE
# ============================================================

def run_inference(model, image_gray):
    """
    Run model inference on a preprocessed grayscale image.

    Args:
        model: Loaded PyTorch model
        image_gray: Grayscale image (H, W), values 0-255

    Returns:
        pred_mask: Binary prediction mask (H, W)
        prob_map: Oil spill probability map (H, W)
    """
    # Resize to model input size
    resized = cv2.resize(image_gray, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] and convert to tensor
    img_tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)

    # Softmax probabilities
    probs = F.softmax(output, dim=1)                     # (1, 2, H, W)
    prob_map = probs[0, 1].cpu().numpy()                 # Oil spill probability
    pred_mask = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)

    return pred_mask, prob_map


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SAR Oil Spill Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS FOR CLEAN STYLING
# ============================================================

st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #1a1a2e;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 4px;
    }

    /* Color legend */
    .legend-box {
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 6px;
        vertical-align: middle;
        border-radius: 3px;
    }

    /* Hide default streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/oil-industry.png", width=60)
    st.markdown("## SAR Oil Spill Detector")
    st.markdown("---")

    # Model selection
    st.markdown("### Select Model")
    selected_model = st.selectbox(
        "Choose a segmentation model:",
        list(MODEL_CONFIG.keys()),
        index=1,  # Default: Attention U-Net (best performer)
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Instructions
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload a preprocessed SAR image (.png)
    2. Select a model from above
    3. Click **Run Prediction**
    4. View results in the multi-panel display
    """)

    st.markdown("---")

    # Model info
    st.markdown("### Model Info")
    model_descriptions = {
        "U-Net": "Classic encoder-decoder with skip connections. 4 resolution levels, ~7.8M parameters.",
        "Attention U-Net": "U-Net enhanced with attention gates at skip connections. Best overall accuracy. ~7.9M parameters.",
        "DeepLabV3+": "ASPP module with multi-scale context via dilated convolutions. ~4.1M parameters.",
    }
    st.info(model_descriptions[selected_model])

    st.markdown("---")

    # Color legend — updated to match RED=Oil, YELLOW=Lookalike
    st.markdown("### Color Legend")
    st.markdown("""
    <span class="legend-box" style="background:#ff0000;"></span> <b>Oil Spill</b> (Red)<br>
    <span class="legend-box" style="background:#ffff00; border:1px solid #ccc;"></span> <b>Look-Alike</b> (Yellow)<br>
    <span class="legend-box" style="background:#000000; border:1px solid #ccc;"></span> Sea / Background
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Detection legend explanation
    st.markdown("### Detection Logic")
    st.markdown("""
    - **Red (Oil Spill)**: High confidence (>55%) detections with elongated shape
    - **Yellow (Look-Alike)**: Medium confidence or round-shaped detections
    """)


# ============================================================
# MAIN CONTENT
# ============================================================

# Header
st.markdown('<div class="main-title">SAR Oil Spill Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Semantic Segmentation of SAR Imagery using Deep Learning</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a preprocessed SAR image (.png)",
    type=["png"],
    help="Upload a grayscale PNG image from the preprocessed dataset (256x256).",
)

# Optional ground truth upload
gt_file = st.file_uploader(
    "Upload ground truth mask (optional, .png)",
    type=["png"],
    help="Upload the corresponding ground truth mask to compute accuracy metrics.",
)

# ============================================================
# PREDICTION LOGIC
# ============================================================

if uploaded_file is not None:
    # Load and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image_gray is None:
        st.error("Failed to read the uploaded image. Please upload a valid PNG file.")
    else:
        # Load ground truth if provided
        gt_mask = None
        if gt_file is not None:
            gt_bytes = np.asarray(bytearray(gt_file.read()), dtype=np.uint8)
            gt_raw = cv2.imdecode(gt_bytes, cv2.IMREAD_GRAYSCALE)
            if gt_raw is not None:
                # Convert mask: 255 -> 1 (oil spill), 0 -> 0 (sea)
                gt_mask = np.where(gt_raw == 255, 1, 0).astype(np.uint8)
                # Resize to match model output
                gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE),
                                     interpolation=cv2.INTER_NEAREST)

        # Predict button
        if st.button("Run Prediction", type="primary", use_container_width=True):
            # Load model
            with st.spinner(f"Loading {selected_model}..."):
                model = load_model(selected_model)

            if model is None:
                st.error(f"Model file not found: {MODEL_CONFIG[selected_model]['file']}. "
                         f"Please ensure trained models are in the `models/` folder.")
            else:
                # Run inference
                with st.spinner("Running inference and postprocessing..."):
                    pred_mask, prob_map = run_inference(model, image_gray)

                    # Apply postprocessing pipeline (includes oil vs look-alike discrimination)
                    final_mask, num_regions, region_stats, region_centroids = \
                        apply_postprocessing(pred_mask, prob_map)

                # Store results in session state for persistent display
                st.session_state["results"] = {
                    "image_gray": image_gray,
                    "gt_mask": gt_mask,
                    "final_mask": final_mask,
                    "prob_map": prob_map,
                    "num_regions": num_regions,
                    "selected_model": selected_model,
                    "filename": uploaded_file.name,
                }

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    if "results" in st.session_state:
        res = st.session_state["results"]
        image_gray = res["image_gray"]
        gt_mask = res["gt_mask"]
        final_mask = res["final_mask"]
        prob_map = res["prob_map"]
        num_regions = res["num_regions"]

        st.markdown("---")
        st.markdown("### Prediction Results")

        # --- Multi-Panel Display ---
        # Resize image for overlay
        display_img = cv2.resize(image_gray, (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_LINEAR)

        # Create prediction overlay
        pred_overlay = create_overlay(display_img, final_mask)
        pred_overlay_rgb = cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB)

        # Panel columns
        if gt_mask is not None:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col3, col4 = st.columns(3)

        # Panel 1: Original SAR Image
        with col1:
            st.markdown("**Original SAR Image**")
            st.image(display_img, use_container_width=True, clamp=True)

        # Panel 2: Ground Truth (if available)
        if gt_mask is not None:
            with col2:
                st.markdown("**Ground Truth Mask**")
                gt_overlay = create_overlay(display_img, gt_mask)
                gt_overlay_rgb = cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB)
                st.image(gt_overlay_rgb, use_container_width=True, clamp=True)

        # Panel 3: Prediction Overlay
        with col3:
            st.markdown(f"**Prediction ({res['selected_model']})**")
            st.image(pred_overlay_rgb, use_container_width=True, clamp=True)

        # Panel 4: Confidence Heatmap
        with col4:
            st.markdown("**Confidence Heatmap**")
            # Create heatmap image using matplotlib
            fig_hm, ax_hm = plt.subplots(1, 1, figsize=(4, 4))
            hm = ax_hm.imshow(prob_map, cmap='inferno', vmin=0, vmax=1)
            ax_hm.axis('off')
            plt.colorbar(hm, ax=ax_hm, fraction=0.046, pad=0.04, label='Oil Probability')
            plt.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)

        # --- Metrics Display ---
        st.markdown("---")
        st.markdown("### Metrics Summary")

        # Compute confidence (for all detected pixels)
        detected_pixels = (final_mask >= 1)
        if np.any(detected_pixels):
            mean_conf = np.mean(prob_map[detected_pixels]) * 100
        else:
            mean_conf = 0.0

        # Pixel counts
        oil_pixels = int(np.sum(final_mask == 1))
        lookalike_pixels = int(np.sum(final_mask == 2))
        total_detected = oil_pixels + lookalike_pixels
        total_pixels = final_mask.size
        spill_area_pct = (total_detected / total_pixels) * 100

        # Metrics row
        if gt_mask is not None:
            # Full metrics with ground truth
            metrics = compute_image_metrics(final_mask, gt_mask)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Confidence", f"{mean_conf:.1f}%")
            with c2:
                st.metric("Spill Area", f"{spill_area_pct:.2f}%")
            with c3:
                st.metric("Regions", f"{num_regions}")
            with c4:
                st.metric("Dice Score", f"{metrics['dice']:.4f}")
            with c5:
                st.metric("IoU", f"{metrics['iou']:.4f}")
            with c6:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")

            # Additional metrics
            st.markdown("#### Detailed Metrics")
            det_c1, det_c2, det_c3, det_c4 = st.columns(4)
            with det_c1:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with det_c2:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with det_c3:
                st.metric("Detected (Pred)", f"{metrics.get('total_detected', oil_pixels + lookalike_pixels):,}")
            with det_c4:
                st.metric("Oil Pixels (GT)", f"{metrics['oil_pixels_gt']:,}")
        else:
            # Metrics without ground truth
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Confidence", f"{mean_conf:.1f}%")
            with c2:
                st.metric("Spill Area", f"{spill_area_pct:.2f}%")
            with c3:
                st.metric("Regions Detected", f"{num_regions}")

        # --- Class Summary ---
        st.markdown("---")
        st.markdown("### Predicted Class Summary")
        sea_pixels = int(np.sum(final_mask == 0))
        summary_data = {
            "Class": ["Sea / Background", "Oil Spill", "Look-Alike"],
            "Color": ["⬛ Black", "🟥 Red", "🟨 Yellow"],
            "Pixels": [f"{sea_pixels:,}", f"{oil_pixels:,}", f"{lookalike_pixels:,}"],
            "Percentage": [
                f"{(sea_pixels / total_pixels) * 100:.2f}%",
                f"{(oil_pixels / total_pixels) * 100:.2f}%",
                f"{(lookalike_pixels / total_pixels) * 100:.2f}%",
            ],
        }
        st.table(summary_data)

        # --- Save Results ---
        st.markdown("---")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save prediction overlay
        save_name = os.path.splitext(res["filename"])[0]
        save_path = os.path.join(OUTPUT_DIR, f"{save_name}_{res['selected_model']}.png")

        # Create and save the multi-panel figure
        num_panels = 4 if gt_mask is not None else 3
        fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 6))

        panel_idx = 0

        # Panel 1: Original
        axes[panel_idx].imshow(display_img, cmap='gray', vmin=0, vmax=255)
        axes[panel_idx].set_title('Original SAR Image', fontsize=13, fontweight='bold')
        axes[panel_idx].axis('off')
        panel_idx += 1

        # Panel 2: Ground Truth (if available)
        if gt_mask is not None:
            axes[panel_idx].imshow(gt_overlay_rgb)
            axes[panel_idx].set_title('Ground Truth Mask', fontsize=13, fontweight='bold')
            axes[panel_idx].axis('off')
            panel_idx += 1

        # Panel 3: Prediction
        axes[panel_idx].imshow(pred_overlay_rgb)
        axes[panel_idx].set_title(f'Prediction ({res["selected_model"]})',
                              fontsize=13, fontweight='bold')
        axes[panel_idx].axis('off')
        panel_idx += 1

        # Panel 4: Heatmap
        hm_plot = axes[panel_idx].imshow(prob_map, cmap='inferno', vmin=0, vmax=1)
        axes[panel_idx].set_title('Confidence Heatmap', fontsize=13, fontweight='bold')
        axes[panel_idx].axis('off')
        fig.colorbar(hm_plot, ax=axes[panel_idx], fraction=0.046, pad=0.04)

        # Legend
        fig.text(0.5, 0.01,
                 'Legend:  \u25A0 Oil Spill (Red)  |  \u25A0 Look-Alike (Yellow)  |  \u25A0 Sea (Black)',
                 ha='center', fontsize=10, style='italic',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        fig.suptitle(f'{res["filename"]}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        st.success(f"Result saved to: `{save_path}`")

else:
    # Placeholder when no image is uploaded
    st.markdown("---")
    st.info("Upload a preprocessed SAR image (.png) to get started. "
            "You can find test images in `data/processed/train/images/` or "
            "preprocess new SAR data using `preprocess.py`.")
