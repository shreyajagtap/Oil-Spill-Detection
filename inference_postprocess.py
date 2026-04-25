"""
SAR Oil Spill Detection - Inference & Postprocessing Pipeline
==============================================================
Runs trained models on unseen test SAR images, applies machine
vision postprocessing, and generates thesis-quality visualizations.

Steps:
  1. Load raw test TIFF images and preprocess (same pipeline as training)
  2. Run inference with all 3 models (U-Net, Attention U-Net, DeepLabV3+)
  3. Apply postprocessing: morphological ops, blob removal, CCA,
     contour smoothing, confidence thresholding, shape/area filtering
  4. Discriminate Oil Spill vs Look-Alike based on confidence + shape
  5. Generate 4-panel comparison PNGs (Original | GT | Prediction | Heatmap)
  6. Save per-image metrics CSV, overall summary, best/worst montage,
     and false positive analysis images

Color Coding:
  RED    = Oil Spill (high-confidence detections)
  YELLOW = Look-Alike (medium-confidence or rounder shape)
  BLACK  = Sea / Background
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Import preprocessing functions from preprocess.py
from preprocess import (
    preprocess_image,
    preprocess_mask,
    handle_invalid_pixels,
    sigma_clip,
    adaptive_db_normalize,
    gamma_correction,
    lee_filter,
    apply_clahe,
    morphological_enhance,
    unsharp_mask,
    contrast_stretch,
    fuse_bands,
    resize_image,
    resize_mask,
)

# Import model architectures from train_models.py
from train_models import UNet, AttentionUNet, DeepLabV3Plus

import tifffile


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TEST_DIR    = os.path.join(BASE_DIR, "data", "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "final_predictions")

# Model settings (must match training config)
NUM_CLASSES = 2
IMG_SIZE    = 256
CLASS_NAMES = ['Sea', 'Oil Spill']

# Postprocessing parameters (tuned for better sensitivity)
CONFIDENCE_THRESHOLD   = 0.35   # Minimum softmax probability to keep a prediction
LOOKALIKE_THRESHOLD    = 0.55   # Detections above this = Oil, below = Look-Alike
MIN_BLOB_AREA          = 25     # Remove connected components smaller than this (pixels)
MAX_ASPECT_RATIO       = 12.0   # Remove blobs with extreme aspect ratios (likely noise)
MIN_REGION_AREA        = 20     # Minimum area for shape/area filtering
MORPH_KERNEL_SIZE      = 3      # Kernel size for morphological opening/closing
CONTOUR_EPSILON_FACTOR = 0.015  # Smoothing factor for contour approximation

# Overlay colors (BGR format for OpenCV) — strong, vivid colors
# Oil Spill = BRIGHT RED, Look-Alike = BRIGHT YELLOW
OVERLAY_COLORS_BGR = {
    0: (0, 0, 0),          # Sea / Background = Black
    1: (0, 0, 255),        # Oil Spill = Red (BGR)
    2: (0, 255, 255),      # Look-Alike = Yellow (BGR)
}

# Overlay colors for legend (RGB for matplotlib)
LEGEND_COLORS_RGB = {
    'Oil Spill': (1.0, 0.0, 0.0),      # Red
    'Look-Alike': (1.0, 1.0, 0.0),     # Yellow
    'Sea': (0.0, 0.0, 0.0),            # Black
}

# Contour colors for borders (BGR)
CONTOUR_COLOR_OIL      = (0, 0, 255)    # Red border for oil spill
CONTOUR_COLOR_LOOKALIKE = (0, 200, 255)  # Yellow border for look-alike

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plot style for thesis quality
PLOT_PARAMS = {
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
}
plt.rcParams.update(PLOT_PARAMS)


# ============================================================
# FIND TEST IMAGE-MASK PAIRS
# ============================================================

def find_test_pairs(test_dir):
    """
    Find test image and mask pairs from raw TIFF data.

    Directory structure:
      test/test_images/{Oil, Lookalike, No oil}/00000.tif
      test/test_masks/{Oil, Lookalike, No oil}/00000_segmentation.tif

    Returns list of (image_path, mask_path, category) tuples.
    """
    img_base  = os.path.join(test_dir, "test_images")
    mask_base = os.path.join(test_dir, "test_masks")
    pairs = []

    if not os.path.exists(img_base):
        print(f"  ERROR: Test image folder not found: {img_base}")
        return pairs

    for subfolder in sorted(os.listdir(img_base)):
        img_sub  = os.path.join(img_base, subfolder)
        mask_sub = os.path.join(mask_base, subfolder)

        if not os.path.isdir(img_sub):
            continue

        # Find all TIFF files in this subfolder
        tiff_files = sorted(
            glob.glob(os.path.join(img_sub, "*.tif")) +
            glob.glob(os.path.join(img_sub, "*.tiff"))
        )

        for img_path in tiff_files:
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]

            # Mask has _segmentation suffix
            mask_name = f"{name_no_ext}_segmentation.tif"
            mask_path = os.path.join(mask_sub, mask_name)

            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path, subfolder))
            else:
                # Try same filename as fallback
                fallback = os.path.join(mask_sub, filename)
                if os.path.exists(fallback):
                    pairs.append((fallback, fallback, subfolder))

    return pairs


# ============================================================
# POSTPROCESSING STEP 1: Morphological Opening / Closing
# ============================================================

def postprocess_morphological(mask, kernel_size=MORPH_KERNEL_SIZE):
    """
    Apply morphological opening then closing to clean predictions.

    Opening (erosion + dilation):
      Removes small false positive blobs (salt noise)

    Closing (dilation + erosion):
      Fills small holes inside predicted oil spill regions

    This smooths jagged boundaries and removes isolated noise pixels.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # Opening: remove small false positive noise
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Closing: fill small holes in predictions
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


# ============================================================
# POSTPROCESSING STEP 2: Remove Small Blobs
# ============================================================

def postprocess_remove_small_blobs(mask, min_area=MIN_BLOB_AREA):
    """
    Remove connected components smaller than min_area pixels.

    Small isolated predictions are usually false positives (noise).
    Real oil spills cover a minimum area on SAR imagery.
    Uses 8-connectivity for connected component analysis.
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Create clean mask (keep only large enough components)
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # Skip background (label 0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 1

    return cleaned


# ============================================================
# POSTPROCESSING STEP 3: Connected Component Analysis
# ============================================================

def postprocess_connected_components(mask):
    """
    Perform connected component analysis to label distinct regions.

    Each separate oil spill region gets a unique label.
    Returns the labeled mask and component statistics.

    Stats per component: [x, y, width, height, area]
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    return num_labels, labels, stats, centroids


# ============================================================
# POSTPROCESSING STEP 4: Contour Smoothing
# ============================================================

def postprocess_smooth_contours(mask, epsilon_factor=CONTOUR_EPSILON_FACTOR):
    """
    Smooth jagged prediction boundaries using contour approximation.

    Uses Douglas-Peucker algorithm to simplify contour polygons.
    epsilon_factor controls smoothing: higher = smoother but less accurate.

    This produces cleaner, more natural-looking oil spill boundaries
    suitable for reports and publications.
    """
    # Find contours of predicted regions
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Create smooth mask by drawing approximated contours
    smoothed = np.zeros_like(mask)
    for contour in contours:
        # Approximate contour with fewer points
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        cv2.drawContours(smoothed, [approx], -1, 1, cv2.FILLED)

    return smoothed


# ============================================================
# POSTPROCESSING STEP 5: Confidence Thresholding
# ============================================================

def postprocess_confidence_threshold(prob_map, threshold=CONFIDENCE_THRESHOLD):
    """
    Apply confidence threshold to softmax probability map.

    Only pixels with Oil Spill probability >= threshold are kept.
    This reduces false positives by requiring high model confidence.

    Args:
        prob_map: Softmax probability for oil spill class (H, W)
        threshold: Minimum probability to classify as oil spill

    Returns:
        Binary mask after thresholding
    """
    return (prob_map >= threshold).astype(np.uint8)


# ============================================================
# POSTPROCESSING STEP 6: Shape / Area Filtering
# ============================================================

def postprocess_shape_area_filter(mask, min_area=MIN_REGION_AREA,
                                   max_aspect_ratio=MAX_ASPECT_RATIO):
    """
    Filter predicted regions by shape and area constraints.

    Removes regions that are:
      - Too small (below min_area pixels)
      - Too elongated (aspect ratio > max_aspect_ratio)

    Oil spills have characteristic shapes - extremely thin or tiny
    detections are likely noise or artifacts.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    filtered = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        area   = stats[label_id, cv2.CC_STAT_AREA]
        width  = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]

        # Skip regions that are too small
        if area < min_area:
            continue

        # Compute aspect ratio (always >= 1)
        aspect = max(width, height) / max(min(width, height), 1)

        # Skip extremely elongated regions (likely artifacts)
        if aspect > max_aspect_ratio:
            continue

        filtered[labels == label_id] = 1

    return filtered


# ============================================================
# FULL POSTPROCESSING PIPELINE
# ============================================================

def apply_postprocessing(raw_pred, prob_map):
    """
    Apply complete postprocessing pipeline to model predictions.

    Pipeline order:
      1. Confidence thresholding (filter by probability)
      2. Morphological opening/closing (clean boundaries)
      3. Remove small blobs (eliminate noise)
      4. Shape/area filtering (remove unrealistic shapes)
      5. Contour smoothing (smooth boundaries)
      6. Connected component analysis (label final regions)
      7. Oil vs Look-Alike discrimination (confidence + shape)

    Args:
        raw_pred: Raw argmax prediction mask (H, W), values 0 or 1
        prob_map: Softmax probability for oil spill class (H, W)

    Returns:
        final_mask: Multi-class mask (0=Sea, 1=Oil Spill, 2=Look-Alike)
        num_regions: Number of detected regions (oil + look-alike)
        region_stats: Stats for each region [x, y, w, h, area]
        region_centroids: Centroid coordinates for each region
    """
    # Step 1: Confidence thresholding
    mask = postprocess_confidence_threshold(prob_map, CONFIDENCE_THRESHOLD)

    # Step 2: Morphological opening/closing
    mask = postprocess_morphological(mask, MORPH_KERNEL_SIZE)

    # Step 3: Remove small blobs
    mask = postprocess_remove_small_blobs(mask, MIN_BLOB_AREA)

    # Step 4: Shape / area filtering
    mask = postprocess_shape_area_filter(mask, MIN_REGION_AREA, MAX_ASPECT_RATIO)

    # Step 5: Contour smoothing
    mask = postprocess_smooth_contours(mask, CONTOUR_EPSILON_FACTOR)

    # Step 6: Connected component analysis on final result
    num_regions, labels, stats, centroids = postprocess_connected_components(mask)

    # num_regions includes background (label 0), subtract 1
    num_spill_regions = num_regions - 1

    # Step 7: Oil vs Look-Alike discrimination per region
    # Classify each detected region based on mean confidence and shape
    classified_mask = discriminate_oil_lookalike(mask, labels, num_regions, stats, prob_map)

    return classified_mask, num_spill_regions, stats, centroids


# ============================================================
# OIL vs LOOK-ALIKE DISCRIMINATION
# ============================================================

def discriminate_oil_lookalike(binary_mask, labels, num_labels, stats, prob_map):
    """
    Classify each detected region as Oil Spill (1) or Look-Alike (2).

    Discrimination criteria:
      - Mean confidence >= LOOKALIKE_THRESHOLD → Oil Spill (1)
      - Mean confidence < LOOKALIKE_THRESHOLD  → Look-Alike (2)
      - Additionally, very round regions (circularity > 0.7) with
        moderate confidence are more likely Look-Alikes

    Args:
        binary_mask: Binary detection mask (H, W)
        labels: Connected component labels
        num_labels: Number of labels (including background)
        stats: CC stats [x, y, w, h, area] per label
        prob_map: Oil spill probability map (H, W)

    Returns:
        classified_mask: (H, W) with 0=Sea, 1=Oil, 2=Look-Alike
    """
    classified = np.zeros_like(binary_mask, dtype=np.uint8)

    for label_id in range(1, num_labels):  # Skip background (0)
        region_pixels = (labels == label_id)
        if not np.any(region_pixels):
            continue

        # Mean confidence for this region
        mean_conf = np.mean(prob_map[region_pixels])

        # Compute circularity (how round the region is)
        region_uint8 = region_pixels.astype(np.uint8)
        contours, _ = cv2.findContours(region_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circularity = 0.0
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)

        # Classification logic:
        # High confidence + not too round → Oil Spill
        # Lower confidence or very round → Look-Alike
        if mean_conf >= LOOKALIKE_THRESHOLD and circularity < 0.7:
            classified[region_pixels] = 1  # Oil Spill
        else:
            classified[region_pixels] = 2  # Look-Alike

    return classified


# ============================================================
# CREATE PREDICTION OVERLAY (Strong, Visible Colors)
# ============================================================

def create_overlay(image_gray, pred_mask, alpha=0.65):
    """
    Create a vivid color overlay of predictions on the original SAR image.

    Color coding:
      Oil Spill (1)  → Strong RED overlay with red contour border
      Look-Alike (2) → Strong YELLOW overlay with yellow contour border
      Sea (0)        → Original grayscale (slightly brightened)

    The overlay uses additive blending on dark regions to ensure
    colors remain visible even on very dark SAR backgrounds.

    Args:
        image_gray: Grayscale image (H, W), values 0-255
        pred_mask: Multi-class mask (H, W), 0=Sea, 1=Oil, 2=Look-Alike
        alpha: Overlay strength (higher = more color visible)
    """
    # Convert grayscale to 3-channel BGR
    if image_gray.ndim == 2:
        base = cv2.cvtColor(image_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        base = image_gray.copy()

    # Slightly brighten the base image so dark SAR regions show overlay better
    base_bright = cv2.convertScaleAbs(base, alpha=1.2, beta=15)

    # Create overlay layer
    overlay = base_bright.copy()

    # Oil Spill regions → RED
    oil_mask = (pred_mask == 1)
    if np.any(oil_mask):
        overlay[oil_mask] = OVERLAY_COLORS_BGR[1]  # Pure Red (BGR)

    # Look-Alike regions → YELLOW
    lookalike_mask = (pred_mask == 2)
    if np.any(lookalike_mask):
        overlay[lookalike_mask] = OVERLAY_COLORS_BGR[2]  # Pure Yellow (BGR)

    # Blend: use stronger alpha for marked regions
    result = base_bright.copy()
    detection_mask = oil_mask | lookalike_mask
    if np.any(detection_mask):
        # Strong blending only on detection regions
        result[detection_mask] = cv2.addWeighted(
            base_bright, 1 - alpha, overlay, alpha, 0
        )[detection_mask]

        # Additive brightness boost on very dark pixels to ensure visibility
        dark_detected = detection_mask & (np.mean(base, axis=2) < 60)
        if np.any(dark_detected):
            boost = np.zeros_like(result)
            boost[oil_mask & dark_detected] = [0, 0, 120]       # Red boost
            boost[lookalike_mask & dark_detected] = [0, 120, 120]  # Yellow boost
            result = cv2.add(result, boost)

    # Draw thick contours for oil spill regions
    if np.any(oil_mask):
        oil_contour_mask = oil_mask.astype(np.uint8)
        contours, _ = cv2.findContours(oil_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, CONTOUR_COLOR_OIL, 2)

    # Draw thick contours for look-alike regions
    if np.any(lookalike_mask):
        la_contour_mask = lookalike_mask.astype(np.uint8)
        contours, _ = cv2.findContours(la_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, CONTOUR_COLOR_LOOKALIKE, 2)

    return result


# ============================================================
# COMPUTE PER-IMAGE METRICS
# ============================================================

def compute_image_metrics(pred_mask, gt_mask):
    """
    Compute segmentation metrics for a single image.

    Handles multi-class predictions: both Oil (1) and Look-Alike (2)
    are treated as positive detections when computing metrics against GT.

    Returns dict with: dice, iou, precision, recall, accuracy,
    oil_pixels_pred, lookalike_pixels_pred, oil_pixels_gt, spill_area_percent.
    """
    # Convert multi-class prediction to binary for metric computation
    # Both Oil (1) and Look-Alike (2) count as "detected"
    pred_binary = ((pred_mask == 1) | (pred_mask == 2)).astype(np.int32).flatten()
    gt = gt_mask.flatten().astype(np.int32)
    smooth = 1e-6

    # True positives, false positives, false negatives
    tp = np.sum((pred_binary == 1) & (gt == 1))
    fp = np.sum((pred_binary == 1) & (gt == 0))
    fn = np.sum((pred_binary == 0) & (gt == 1))
    tn = np.sum((pred_binary == 0) & (gt == 0))

    # Metrics
    dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + smooth)

    # Spill area as percentage of total image
    total_pixels = pred_binary.size
    oil_pred = int(np.sum(pred_mask == 1))
    lookalike_pred = int(np.sum(pred_mask == 2))
    total_detected = oil_pred + lookalike_pred
    oil_gt = np.sum(gt == 1)
    spill_area_pct = (total_detected / total_pixels) * 100

    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'oil_pixels_pred': oil_pred,
        'lookalike_pixels_pred': lookalike_pred,
        'total_detected': total_detected,
        'oil_pixels_gt': int(oil_gt),
        'spill_area_percent': spill_area_pct,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


# ============================================================
# CREATE 4-PANEL VISUALIZATION
# ============================================================

def create_three_panel(image_gray, gt_mask, pred_mask, prob_map,
                       metrics, num_regions, filename, model_name):
    """
    Create a single PNG with 4 panels:
      Panel 1: Original SAR Image
      Panel 2: Ground Truth Mask (Red overlay)
      Panel 3: Predicted Segmentation Overlay (Red=Oil, Yellow=Look-alike)
      Panel 4: Confidence Heatmap

    Also displays confidence score, class summary, and spill area.
    Color legend: Red = Oil Spill, Yellow = Look-Alike, Black = Sea
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # --- Panel 1: Original SAR Image ---
    axes[0].imshow(image_gray, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original SAR Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # --- Panel 2: Ground Truth Mask (overlaid on original image) ---
    # GT mask is always binary (0=sea, 1=oil), use class 1 for red overlay
    gt_overlay = create_overlay(image_gray, gt_mask)
    gt_overlay_rgb = cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB)
    axes[1].imshow(gt_overlay_rgb)
    axes[1].set_title('Ground Truth Mask', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # Add GT info
    gt_oil_pct = np.sum(gt_mask == 1) / gt_mask.size * 100
    gt_oil_px  = np.sum(gt_mask == 1)
    gt_info = f'Oil: {gt_oil_pct:.1f}% ({gt_oil_px} px)'
    axes[1].text(0.02, 0.02, gt_info,
                 transform=axes[1].transAxes, fontsize=9,
                 color='white', bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

    # --- Panel 3: Predicted Segmentation Overlay ---
    overlay = create_overlay(image_gray, pred_mask)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f'Prediction ({model_name})', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # Compute mean confidence for detected regions
    detected_pixels = (pred_mask >= 1)
    if np.any(detected_pixels):
        mean_conf = np.mean(prob_map[detected_pixels]) * 100
    else:
        mean_conf = 0.0

    # Count oil vs look-alike
    oil_px = int(np.sum(pred_mask == 1))
    la_px = int(np.sum(pred_mask == 2))

    # Info text on prediction panel
    info_lines = [
        f"Confidence: {mean_conf:.1f}%",
        f"Oil Spill: {oil_px} px",
        f"Look-Alike: {la_px} px",
        f"Regions: {num_regions}",
        f"Dice: {metrics['dice']:.3f}",
        f"IoU: {metrics['iou']:.3f}",
    ]
    info_text = '\n'.join(info_lines)
    axes[2].text(0.02, 0.98, info_text, transform=axes[2].transAxes,
                 fontsize=8, verticalalignment='top', color='white',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    # Class summary below prediction panel
    pred_summary = f"Sea: {np.sum(pred_mask == 0)} px"
    axes[2].text(0.02, 0.02, pred_summary, transform=axes[2].transAxes,
                 fontsize=9, color='white',
                 bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

    # --- Panel 4: Confidence Heatmap ---
    heatmap = axes[3].imshow(prob_map, cmap='inferno', vmin=0, vmax=1)
    axes[3].set_title('Confidence Heatmap', fontsize=13, fontweight='bold')
    axes[3].axis('off')
    cbar = fig.colorbar(heatmap, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label('Oil Spill Probability', fontsize=10)

    # Add color-coded legend at bottom
    legend_text = (
        'Legend:  '
        '\u25A0 Oil Spill (Red)  |  '
        '\u25A0 Look-Alike (Yellow)  |  '
        '\u25A0 Sea (Black)'
    )
    fig.text(0.5, 0.01, legend_text,
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Title
    fig.suptitle(f'{filename}', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


# ============================================================
# CREATE BEST / WORST MONTAGE
# ============================================================

def create_best_worst_montage(results_list, output_dir, model_name, n=5):
    """
    Create a montage showing the N best and N worst predictions
    ranked by Dice score.

    Best = highest Dice (model does well)
    Worst = lowest Dice (model struggles)
    """
    # Sort by Dice score
    sorted_results = sorted(results_list, key=lambda x: x['dice'])

    worst = sorted_results[:n]
    best  = sorted_results[-n:][::-1]

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(f'{model_name} - Best & Worst Predictions (by Dice Score)',
                 fontsize=15, fontweight='bold')

    # Top row: Best predictions
    for i, res in enumerate(best):
        overlay = create_overlay(res['image'], res['pred_mask'])
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(overlay_rgb)
        axes[0, i].set_title(f"Dice: {res['dice']:.3f}", fontsize=10,
                             color='green', fontweight='bold')
        axes[0, i].axis('off')

    axes[0, 0].set_ylabel('BEST', fontsize=13, fontweight='bold',
                          color='green', rotation=0, labelpad=50)

    # Bottom row: Worst predictions
    for i, res in enumerate(worst):
        overlay = create_overlay(res['image'], res['pred_mask'])
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(overlay_rgb)
        axes[1, i].set_title(f"Dice: {res['dice']:.3f}", fontsize=10,
                             color='red', fontweight='bold')
        axes[1, i].axis('off')

    axes[1, 0].set_ylabel('WORST', fontsize=13, fontweight='bold',
                          color='red', rotation=0, labelpad=50)

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    save_path = os.path.join(output_dir, f'{model_name}_best_worst_montage.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ============================================================
# FALSE POSITIVE ANALYSIS
# ============================================================

def create_false_positive_analysis(results_list, output_dir, model_name, n=5):
    """
    Generate false positive analysis images.

    Shows images where the model predicted Oil Spill but ground truth
    had none (highest false positive count). Each image shows:
      - Original image
      - False positive regions highlighted in yellow
      - Ground truth for comparison
    """
    # Filter to images with false positives
    fp_results = [r for r in results_list if r['fp'] > 0]
    fp_results = sorted(fp_results, key=lambda x: x['fp'], reverse=True)

    if len(fp_results) == 0:
        print(f"  No false positives found for {model_name} - skipping FP analysis.")
        return

    # Take top N worst FP cases
    fp_show = fp_results[:min(n, len(fp_results))]

    fig, axes = plt.subplots(len(fp_show), 3, figsize=(15, 4 * len(fp_show)))
    if len(fp_show) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'{model_name} - False Positive Analysis (Top {len(fp_show)} cases)',
                 fontsize=15, fontweight='bold')

    for i, res in enumerate(fp_show):
        pred = res['pred_mask']
        gt   = res['gt_mask']
        img  = res['image']

        # Compute false positive mask
        fp_mask = ((pred == 1) & (gt == 0)).astype(np.uint8)

        # Panel 1: Original image
        axes[i, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title(f'{res["filename"]}', fontsize=10)
        axes[i, 0].axis('off')

        # Panel 2: False positive regions (Yellow overlay)
        base_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        fp_overlay = base_rgb.copy()
        fp_overlay[fp_mask == 1] = [255, 255, 0]  # Yellow for FP
        blended = cv2.addWeighted(base_rgb, 0.5, fp_overlay, 0.5, 0)
        axes[i, 1].imshow(blended)
        axes[i, 1].set_title(f'False Positives: {res["fp"]} px', fontsize=10,
                             color='orange', fontweight='bold')
        axes[i, 1].axis('off')

        # Panel 3: Ground truth
        gt_display = np.zeros((*gt.shape, 3), dtype=np.uint8)
        gt_display[gt == 1] = [255, 0, 0]
        axes[i, 2].imshow(gt_display)
        gt_oil = np.sum(gt == 1)
        axes[i, 2].set_title(f'Ground Truth (Oil: {gt_oil} px)', fontsize=10)
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, f'{model_name}_false_positive_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model(model_class, model_path, model_name):
    """
    Load a trained model from saved weights.

    Args:
        model_class: Model class (UNet, AttentionUNet, DeepLabV3Plus)
        model_path: Path to .pth weights file
        model_name: Display name for logging
    """
    model = model_class(in_channels=1, num_classes=NUM_CLASSES)
    if not os.path.exists(model_path):
        print(f"  ERROR: Model weights not found: {model_path}")
        return None

    # Load weights
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model_name}: {param_count:,} parameters")
    return model


# ============================================================
# RUN INFERENCE ON ONE IMAGE
# ============================================================

@torch.no_grad()
def run_inference(model, image_tensor):
    """
    Run model inference on a single preprocessed image.

    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor (1, 1, H, W)

    Returns:
        pred_mask: Argmax prediction (H, W), values 0 or 1
        prob_map: Softmax probability for oil spill class (H, W)
    """
    image_tensor = image_tensor.to(DEVICE)
    output = model(image_tensor)  # (1, NUM_CLASSES, H, W)

    # Softmax probabilities
    probs = F.softmax(output, dim=1)  # (1, 2, H, W)

    # Oil spill probability map (class 1)
    prob_map = probs[0, 1].cpu().numpy()  # (H, W)

    # Argmax prediction
    pred_mask = output.argmax(dim=1)[0].cpu().numpy()  # (H, W)

    return pred_mask.astype(np.uint8), prob_map


# ============================================================
# MAIN INFERENCE PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("  SAR Oil Spill Detection - Inference & Postprocessing")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Test data: {TEST_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Step 1: Find test image-mask pairs
    # --------------------------------------------------------
    print("\n[1/5] Finding test image-mask pairs...")
    test_pairs = find_test_pairs(TEST_DIR)
    print(f"  Found {len(test_pairs)} test pairs")

    if len(test_pairs) == 0:
        print("  ERROR: No test data found! Check data/test/ folder.")
        return

    # Count per category
    categories = {}
    for _, _, cat in test_pairs:
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in categories.items():
        print(f"    {cat}: {count} images")

    # --------------------------------------------------------
    # Step 2: Load all trained models
    # --------------------------------------------------------
    print("\n[2/5] Loading trained models...")
    models_config = {
        'U-Net': {
            'class': UNet,
            'path': os.path.join(MODEL_DIR, 'unet.pth'),
        },
        'Attention U-Net': {
            'class': AttentionUNet,
            'path': os.path.join(MODEL_DIR, 'attention_unet.pth'),
        },
        'DeepLabV3+': {
            'class': DeepLabV3Plus,
            'path': os.path.join(MODEL_DIR, 'deeplabv3.pth'),
        },
    }

    loaded_models = {}
    for name, config in models_config.items():
        model = load_model(config['class'], config['path'], name)
        if model is not None:
            loaded_models[name] = model

    if len(loaded_models) == 0:
        print("  ERROR: No models loaded! Train models first.")
        return

    # --------------------------------------------------------
    # Step 3: Run inference on all test images
    # --------------------------------------------------------
    print(f"\n[3/5] Running inference and postprocessing...")

    # Store results per model
    all_model_results = {}

    for model_name, model in loaded_models.items():
        print(f"\n  --- {model_name} ---")

        # Create model-specific output directory
        model_output_dir = os.path.join(OUTPUT_DIR, model_name.lower().replace(' ', '_').replace('+', 'plus'))
        os.makedirs(model_output_dir, exist_ok=True)

        model_results = []

        for img_path, mask_path, category in tqdm(test_pairs, desc=f"  {model_name}"):
            # --- Read raw TIFF data ---
            raw_image = tifffile.imread(img_path)
            raw_mask  = tifffile.imread(mask_path)

            # --- Preprocess image (same pipeline as training) ---
            processed_img = preprocess_image(raw_image)
            image_uint8 = (np.clip(processed_img, 0, 1) * 255).astype(np.uint8)

            # --- Preprocess mask ---
            if raw_mask.ndim == 3:
                raw_mask = raw_mask[:, :, 0]
            gt_mask = resize_mask(raw_mask, size=IMG_SIZE)
            gt_mask = np.where(gt_mask > 0, 1, 0).astype(np.uint8)

            # --- Prepare tensor for model ---
            img_float = image_uint8.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # --- Run inference ---
            raw_pred, prob_map = run_inference(model, img_tensor)

            # --- Apply postprocessing ---
            post_pred, num_regions, stats, centroids = apply_postprocessing(
                raw_pred, prob_map
            )

            # --- Compute metrics ---
            metrics = compute_image_metrics(post_pred, gt_mask)

            # --- Generate filename ---
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_filename = f"{category}_{base_name}"

            # --- Save 3-panel visualization ---
            fig = create_three_panel(
                image_uint8, gt_mask, post_pred, prob_map,
                metrics, num_regions, out_filename, model_name
            )
            panel_path = os.path.join(model_output_dir, f"{out_filename}.png")
            fig.savefig(panel_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            # --- Store results for analysis ---
            result_entry = {
                'filename': out_filename,
                'category': category,
                'image': image_uint8,
                'gt_mask': gt_mask,
                'pred_mask': post_pred,
                'prob_map': prob_map,
                'num_regions': num_regions,
                **metrics,
            }
            model_results.append(result_entry)

        all_model_results[model_name] = model_results

        # Move model to CPU to free GPU memory
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Step 4: Save metrics and analysis outputs
    # --------------------------------------------------------
    print(f"\n[4/5] Saving metrics and analysis outputs...")

    overall_summary_rows = []

    for model_name, results in all_model_results.items():
        model_key = model_name.lower().replace(' ', '_').replace('+', 'plus')
        model_output_dir = os.path.join(OUTPUT_DIR, model_key)

        # --- Per-image metrics CSV ---
        csv_rows = []
        for r in results:
            csv_rows.append({
                'Image': r['filename'],
                'Category': r['category'],
                'Dice': round(r['dice'], 4),
                'IoU': round(r['iou'], 4),
                'Precision': round(r['precision'], 4),
                'Recall': round(r['recall'], 4),
                'Accuracy': round(r['accuracy'], 4),
                'Oil_Pixels_Pred': r['oil_pixels_pred'],
                'Lookalike_Pixels_Pred': r.get('lookalike_pixels_pred', 0),
                'Oil_Pixels_GT': r['oil_pixels_gt'],
                'Spill_Area_Pct': round(r['spill_area_percent'], 4),
                'Num_Regions': r['num_regions'],
                'TP': r['tp'], 'FP': r['fp'], 'FN': r['fn'], 'TN': r['tn'],
            })

        df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(model_output_dir, 'per_image_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {model_key}/per_image_metrics.csv ({len(csv_rows)} images)")

        # --- Compute overall metrics ---
        mean_dice      = np.mean([r['dice'] for r in results])
        mean_iou       = np.mean([r['iou'] for r in results])
        mean_precision = np.mean([r['precision'] for r in results])
        mean_recall    = np.mean([r['recall'] for r in results])
        mean_accuracy  = np.mean([r['accuracy'] for r in results])
        total_fp       = sum(r['fp'] for r in results)
        total_fn       = sum(r['fn'] for r in results)

        # Oil-only images metrics (where GT has oil)
        oil_results = [r for r in results if r['oil_pixels_gt'] > 0]
        if len(oil_results) > 0:
            oil_dice   = np.mean([r['dice'] for r in oil_results])
            oil_iou    = np.mean([r['iou'] for r in oil_results])
            oil_recall = np.mean([r['recall'] for r in oil_results])
        else:
            oil_dice = oil_iou = oil_recall = 0.0

        overall_summary_rows.append({
            'Model': model_name,
            'Mean_Dice': round(mean_dice, 4),
            'Mean_IoU': round(mean_iou, 4),
            'Mean_Precision': round(mean_precision, 4),
            'Mean_Recall': round(mean_recall, 4),
            'Pixel_Accuracy': round(mean_accuracy, 4),
            'Oil_Only_Dice': round(oil_dice, 4),
            'Oil_Only_IoU': round(oil_iou, 4),
            'Oil_Only_Recall': round(oil_recall, 4),
            'Total_FP_Pixels': total_fp,
            'Total_FN_Pixels': total_fn,
            'Test_Images': len(results),
        })

        # --- Best/Worst montage ---
        create_best_worst_montage(results, model_output_dir, model_name, n=5)

        # --- False positive analysis ---
        create_false_positive_analysis(results, model_output_dir, model_name, n=5)

    # --- Overall test metrics summary CSV ---
    summary_df = pd.DataFrame(overall_summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'overall_test_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved: overall_test_summary.csv")
    print(f"\n{summary_df.to_string(index=False)}")

    # --------------------------------------------------------
    # Step 5: Generate combined comparison chart
    # --------------------------------------------------------
    print(f"\n[5/5] Generating comparison chart...")

    metric_labels = ['Dice', 'IoU', 'Precision', 'Recall', 'Accuracy']
    metric_keys = ['Mean_Dice', 'Mean_IoU', 'Mean_Precision', 'Mean_Recall', 'Pixel_Accuracy']

    model_colors = {'U-Net': '#2196F3', 'Attention U-Net': '#FF5722', 'DeepLabV3+': '#4CAF50'}

    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, row in enumerate(overall_summary_rows):
        name = row['Model']
        values = [row[k] for k in metric_keys]
        color = model_colors.get(name, '#999999')
        bars = ax.bar(x + i * width, values, width, label=name, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Test Set - Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.15])

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'test_model_comparison.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: test_model_comparison.png")

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  INFERENCE & POSTPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  3-Panel predictions saved to: {OUTPUT_DIR}/")
    print(f"  Per-model folders: {', '.join(all_model_results.keys())}")
    print(f"  Overall summary:  {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
