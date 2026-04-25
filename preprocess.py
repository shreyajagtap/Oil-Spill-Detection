"""
SAR Oil Spill Detection - Preprocessing Pipeline
==================================================
This script preprocesses SAR (Synthetic Aperture Radar) TIFF images
and their corresponding masks for oil spill detection.

Key insight: SAR images are 2-band (VV, VH polarization) in dB scale
with values from ~ -80 to +12. Zeros represent no-data regions.

Preprocessing steps for images:
  1. Handle NaN / Inf / No-data (zeros)
  2. Sigma clipping to remove extreme outliers
  3. Adaptive dB normalization (per-band percentile-based)
  4. Gamma correction for brightness recovery
  5. Lee filter (speckle noise removal)
  6. CLAHE (adaptive histogram equalization)
  7. Morphological enhancement (open/close to clean small artifacts)
  8. Unsharp masking (edge sharpening)
  9. Contrast stretching (2%-98%)
  10. Multi-band to single-band fusion (band averaging)
  11. Resize to 256x256

Preprocessing steps for masks:
  1. Preserve multiclass labels
  2. Resize with nearest-neighbor interpolation
"""

import os
import glob
import numpy as np
import cv2
import tifffile
from tqdm import tqdm


# ============================================================
# STEP 1: Handle NaN, Inf, and No-data values
# ============================================================
def handle_invalid_pixels(image):
    """
    Replace NaN, Inf, and zero (no-data) values.
    In SAR dB images, 0 dB is valid but large patches of exact 0.0
    are no-data regions. We replace them with the band's median
    so they don't distort normalization.
    """
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Create no-data mask (exact zeros in SAR dB data = no-data)
    nodata_mask = (image == 0.0)

    # Replace no-data pixels with median of valid pixels
    if image.ndim == 2:
        valid = image[~nodata_mask]
        if len(valid) > 0:
            image[nodata_mask] = np.median(valid)
    else:
        # Multi-band: handle each band separately
        for b in range(image.shape[2]):
            band = image[:, :, b]
            band_nodata = nodata_mask[:, :, b]
            valid = band[~band_nodata]
            if len(valid) > 0:
                band[band_nodata] = np.median(valid)
            image[:, :, b] = band

    return image


# ============================================================
# STEP 2: Sigma clipping to remove extreme outliers
# ============================================================
def sigma_clip(image, sigma=3):
    """
    Clip extreme outlier pixel values using sigma-based thresholds.
    Values beyond mean +/- sigma*std are clipped.
    This prevents a few extreme pixels from ruining normalization.
    """
    mean = np.mean(image)
    std = np.std(image)
    lower = mean - sigma * std
    upper = mean + sigma * std
    return np.clip(image, lower, upper)


# ============================================================
# STEP 3: Adaptive dB normalization (percentile-based)
# ============================================================
def adaptive_db_normalize(image, low_pct=2, high_pct=98):
    """
    Normalize SAR dB values using percentile-based scaling.

    Instead of using min/max (which is ruined by extreme outliers),
    we use the 2nd and 98th percentile as the effective range.
    This is the KEY fix for the black image problem.

    For example, if dB values range from -80 to 0:
      - p2 = -40 (2nd percentile)
      - p98 = -8 (98th percentile)
      - Map [-40, -8] to [0, 1]
      - Clip anything outside
    """
    p_low = np.percentile(image, low_pct)
    p_high = np.percentile(image, high_pct)

    if p_high - p_low == 0:
        return np.zeros_like(image, dtype=np.float64)

    # Clip to percentile range, then scale to [0, 1]
    image = np.clip(image, p_low, p_high)
    image = (image - p_low) / (p_high - p_low)
    return image


# ============================================================
# STEP 4: Gamma correction for brightness recovery
# ============================================================
def gamma_correction(image, gamma=0.7):
    """
    Apply gamma correction to brighten dark SAR images.

    gamma < 1.0 = brighten image (we use 0.7)
    gamma > 1.0 = darken image
    gamma = 1.0 = no change

    Formula: output = input^gamma
    Since SAR images tend to be dark, gamma < 1 lifts the midtones.
    """
    # Image must be in [0, 1] range
    return np.power(np.clip(image, 0, 1), gamma)


# ============================================================
# STEP 5: Lee filter (speckle noise removal) - Optimized
# ============================================================
def lee_filter(image, window_size=7):
    """
    Optimized Lee filter for SAR speckle removal using box filtering.

    The Lee filter reduces speckle noise while preserving edges:
      - In smooth regions (low variance): replaces pixel with local mean
      - In edge regions (high variance): keeps original pixel value

    Formula: output = mean + k * (pixel - mean)
    where k = max(0, (local_var - noise_var) / local_var)

    This version uses cv2.blur() for fast local mean/variance computation
    instead of slow pixel-by-pixel loops.
    """
    image = image.astype(np.float64)

    # Compute local mean using box filter (fast convolution)
    local_mean = cv2.blur(image, (window_size, window_size))

    # Compute local variance: E[X^2] - (E[X])^2
    local_sq_mean = cv2.blur(image ** 2, (window_size, window_size))
    local_variance = local_sq_mean - local_mean ** 2
    local_variance = np.maximum(local_variance, 0)  # Avoid negative due to float precision

    # Estimate noise variance from the overall image
    noise_variance = np.mean(local_variance)

    # Compute weighting factor k
    # k near 0 = smooth (use mean), k near 1 = preserve detail
    # Add tiny epsilon to avoid divide-by-zero warning
    k = np.where(
        local_variance > 0,
        np.maximum(0, (local_variance - noise_variance) / (local_variance + 1e-10)),
        0
    )

    # Apply Lee filter formula
    output = local_mean + k * (image - local_mean)
    return output


# ============================================================
# STEP 6: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ============================================================
def apply_clahe(image, clip_limit=3.0, tile_size=8):
    """
    CLAHE divides the image into small tiles and equalizes the
    histogram in each tile independently, with a clip limit to
    prevent noise amplification.

    This dramatically improves local contrast, making oil spill
    boundaries and sea features visible.
    """
    # Convert to uint8 for CLAHE (OpenCV requirement)
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    result = clahe.apply(image_uint8)

    return result.astype(np.float64) / 255.0


# ============================================================
# STEP 7: Morphological enhancement
# ============================================================
def morphological_enhance(image, kernel_size=3):
    """
    Apply morphological opening followed by closing.

    Opening (erosion + dilation): removes small bright noise spots
    Closing (dilation + erosion): fills small dark holes

    This cleans up small artifacts without affecting large structures
    like oil spill boundaries.
    """
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Opening: remove small bright spots (salt noise)
    opened = cv2.morphologyEx(image_uint8, cv2.MORPH_OPEN, kernel)

    # Closing: fill small dark holes (pepper noise)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed.astype(np.float64) / 255.0


# ============================================================
# STEP 8: Unsharp masking (edge sharpening)
# ============================================================
def unsharp_mask(image, sigma=1.0, strength=0.5):
    """
    Sharpen edges using unsharp masking.

    Method: subtract a blurred version from the original
    to extract edges, then add those edges back.

    Formula: sharpened = original + strength * (original - blurred)

    This makes oil spill boundaries and texture details crisper.
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Sharpen: original + strength * (original - blurred)
    sharpened = image + strength * (image - blurred)

    return np.clip(sharpened, 0, 1)


# ============================================================
# STEP 9: Contrast stretching (2%-98%)
# ============================================================
def contrast_stretch(image, low_pct=2, high_pct=98):
    """
    Final contrast stretching using percentile clipping.
    Ensures the output uses the full [0, 1] dynamic range.
    """
    p_low = np.percentile(image, low_pct)
    p_high = np.percentile(image, high_pct)

    if p_high - p_low == 0:
        return np.zeros_like(image)

    image = np.clip(image, p_low, p_high)
    return (image - p_low) / (p_high - p_low)


# ============================================================
# STEP 10: Multi-band fusion
# ============================================================
def fuse_bands(image):
    """
    Fuse VV and VH polarization bands into a single-channel image.

    SAR images typically have 2 bands:
      - Band 0: VV polarization (co-polarization)
      - Band 1: VH polarization (cross-polarization)

    We take the mean of both bands. This combines information
    from both polarizations for better feature representation.
    """
    if image.ndim == 3:
        return np.mean(image, axis=2)
    return image


# ============================================================
# STEP 11: Resize
# ============================================================
def resize_image(image, size=256):
    """Resize image to (size x size) using bilinear interpolation."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask, size=256):
    """Resize mask using nearest-neighbor to preserve class labels."""
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


# ============================================================
# FULL IMAGE PREPROCESSING PIPELINE
# ============================================================
def preprocess_image(image):
    """
    Apply the complete SAR preprocessing pipeline to one image.

    Pipeline order is carefully chosen:
      1. Fix invalid pixels first (NaN, Inf, no-data)
      2. Process each band in dB domain
      3. Fuse bands
      4. Enhance and sharpen
      5. Final contrast stretch and resize
    """
    image = image.astype(np.float64)

    # --- Stage 1: Clean raw data ---
    # Handle NaN, Inf, and no-data (zero) pixels
    image = handle_invalid_pixels(image)

    # --- Stage 2: Per-band dB normalization ---
    # Process each band separately for optimal normalization
    if image.ndim == 3:
        for b in range(image.shape[2]):
            band = image[:, :, b]
            band = sigma_clip(band, sigma=3)
            band = adaptive_db_normalize(band, low_pct=2, high_pct=98)
            band = gamma_correction(band, gamma=0.7)
            image[:, :, b] = band
    else:
        image = sigma_clip(image, sigma=3)
        image = adaptive_db_normalize(image, low_pct=2, high_pct=98)
        image = gamma_correction(image, gamma=0.7)

    # --- Stage 3: Fuse VV and VH bands into single channel ---
    image = fuse_bands(image)

    # --- Stage 4: Speckle removal ---
    image = lee_filter(image, window_size=7)

    # --- Stage 5: Local contrast enhancement ---
    image = apply_clahe(image, clip_limit=3.0, tile_size=8)

    # --- Stage 6: Morphological cleanup ---
    image = morphological_enhance(image, kernel_size=3)

    # --- Stage 7: Edge sharpening ---
    image = unsharp_mask(image, sigma=1.0, strength=0.5)

    # --- Stage 8: Final contrast stretch ---
    image = contrast_stretch(image, low_pct=2, high_pct=98)

    # --- Stage 9: Resize to model input size ---
    image = resize_image(image, size=256)

    return image


# ============================================================
# FULL MASK PREPROCESSING
# ============================================================
def preprocess_mask(mask):
    """
    Preprocess a segmentation mask.
    - Resize with nearest-neighbor to keep class labels intact
    - Scale labels to visible range: 0 stays 0, 1 becomes 255
      (so masks are visible when opened as images)

    Original mask values: 0 = background, 1 = oil spill / feature
    Saved mask values:    0 = background, 255 = oil spill / feature
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = resize_mask(mask, size=256)

    # Scale mask so labels are visible in PNG
    # 0 -> 0 (black = background), 1 -> 255 (white = oil region)
    mask = mask.astype(np.uint8) * 255

    return mask


# ============================================================
# FILE DISCOVERY: Find all TIFF files recursively
# ============================================================
def find_tiff_files(directory):
    """Recursively find all .tif and .tiff files in a directory."""
    patterns = ["**/*.tif", "**/*.tiff", "**/*.TIF", "**/*.TIFF"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    return sorted(set(files))


# ============================================================
# PAIR IMAGES WITH MASKS (for training data)
# ============================================================
def get_train_pairs(train_dir):
    """
    Pair training images with their masks.

    Folder structure expected:
      train/oil_spill_images/  <-> train/oil_spill_masks/
      train/lookalike_images/  <-> train/lookalike_masks/
      train/no_oil_images/     <-> train/no_oil_masks/

    Returns list of (image_path, mask_path) tuples.
    """
    categories = ["oil_spill", "lookalike", "no_oil"]
    pairs = []

    for category in categories:
        img_dir = os.path.join(train_dir, f"{category}_images")
        mask_dir = os.path.join(train_dir, f"{category}_masks")

        if not os.path.exists(img_dir):
            print(f"  [WARNING] Image folder not found: {img_dir}")
            continue
        if not os.path.exists(mask_dir):
            print(f"  [WARNING] Mask folder not found: {mask_dir}")
            continue

        image_files = find_tiff_files(img_dir)

        for img_path in image_files:
            filename = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, filename)

            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
            else:
                print(f"  [WARNING] No mask found for: {filename}")

    return pairs


def get_test_pairs(test_dir):
    """
    Pair test images with their masks.

    Folder structure expected:
      test/test_images/Lookalike/  <-> test/test_masks/Lookalike/
      test/test_images/No oil/     <-> test/test_masks/No oil/
      test/test_images/Oil/        <-> test/test_masks/Oil/

    Returns list of (image_path, mask_path) tuples.
    """
    img_base = os.path.join(test_dir, "test_images")
    mask_base = os.path.join(test_dir, "test_masks")
    pairs = []

    if not os.path.exists(img_base):
        print(f"  [WARNING] Test images folder not found: {img_base}")
        return pairs

    for subfolder in sorted(os.listdir(img_base)):
        img_sub = os.path.join(img_base, subfolder)
        mask_sub = os.path.join(mask_base, subfolder)

        if not os.path.isdir(img_sub):
            continue

        if not os.path.exists(mask_sub):
            print(f"  [WARNING] Test mask folder not found: {mask_sub}")
            continue

        image_files = find_tiff_files(img_sub)

        for img_path in image_files:
            filename = os.path.basename(img_path)
            mask_path = os.path.join(mask_sub, filename)

            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
            else:
                print(f"  [WARNING] No test mask for: {filename}")

    return pairs


# ============================================================
# PROCESS AND SAVE
# ============================================================
def process_and_save(pairs, output_img_dir, output_mask_dir, label):
    """
    Process a list of (image, mask) pairs and save as PNG.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    print(f"\n  Processing {len(pairs)} {label} image-mask pairs...")

    for img_path, mask_path in tqdm(pairs, desc=f"  {label}"):
        # Read TIFF image and mask
        image = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        # Preprocess
        processed_image = preprocess_image(image)
        processed_mask = preprocess_mask(mask)

        # Convert image to uint8 for saving as PNG
        image_uint8 = (np.clip(processed_image, 0, 1) * 255).astype(np.uint8)
        mask_uint8 = processed_mask.astype(np.uint8)

        # Generate output filename
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img_save_path = os.path.join(output_img_dir, f"{base_name}.png")
        mask_save_path = os.path.join(output_mask_dir, f"{base_name}.png")

        # Save as PNG
        cv2.imwrite(img_save_path, image_uint8)
        cv2.imwrite(mask_save_path, mask_uint8)

    print(f"  {label} processing complete!")


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    """Main function to run the entire preprocessing pipeline."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "data", "train")
    test_dir = os.path.join(base_dir, "data", "test")

    # Output directories
    processed_train_img = os.path.join(base_dir, "data", "processed", "train", "images")
    processed_train_mask = os.path.join(base_dir, "data", "processed", "train", "masks")
    processed_test_img = os.path.join(base_dir, "data", "processed", "test", "images")
    processed_test_mask = os.path.join(base_dir, "data", "processed", "test", "masks")

    print("=" * 60)
    print("  SAR Oil Spill Detection - Preprocessing Pipeline")
    print("=" * 60)

    # --- Process Training Data ---
    print("\n[1/2] Finding training image-mask pairs...")
    train_pairs = get_train_pairs(train_dir)
    print(f"  Found {len(train_pairs)} training pairs.")

    if train_pairs:
        process_and_save(train_pairs, processed_train_img, processed_train_mask, "Train")

    # --- Process Test Data ---
    print("\n[2/2] Finding test image-mask pairs...")
    test_pairs = get_test_pairs(test_dir)
    print(f"  Found {len(test_pairs)} test pairs.")

    if test_pairs:
        process_and_save(test_pairs, processed_test_img, processed_test_mask, "Test")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Train images saved to: {processed_train_img}")
    print(f"  Train masks saved to:  {processed_train_mask}")
    print(f"  Test images saved to:  {processed_test_img}")
    print(f"  Test masks saved to:   {processed_test_mask}")
    print("=" * 60)


if __name__ == "__main__":
    main()
