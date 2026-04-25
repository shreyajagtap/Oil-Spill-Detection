# SAR Oil Spill Detection using Deep Learning

MTech Project - Semantic segmentation of SAR imagery to detect oil spills using U-Net, Attention U-Net, and DeepLabV3+.

## Project Description

This project detects oil spills in ocean SAR (Synthetic Aperture Radar) images using deep learning-based semantic segmentation. Three architectures are trained and compared on preprocessed SAR data, then evaluated on unseen test imagery with machine vision postprocessing.

### Multi-Class Labels

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Sea / Background | Clean ocean surface |
| 1 | Oil Spill | Confirmed oil spill region |
| 2 | Look-Alike | Natural phenomena mimicking oil spills |
| 3 | Ship | Vessel detected in SAR image |
| 4 | Land | Land mass in SAR image |

## Model Descriptions

### 1. U-Net
Classic encoder-decoder architecture with skip connections. The encoder extracts features at 4 resolution levels, and the decoder upsamples with skip connections that preserve spatial detail. Well-suited for medical and remote sensing segmentation tasks.

### 2. Attention U-Net
Enhanced U-Net with attention gates at each skip connection. The attention mechanism learns to focus on relevant spatial regions and suppress irrelevant features. This improves segmentation accuracy, especially for small or irregularly shaped oil spill regions.

### 3. DeepLabV3+
State-of-the-art architecture using Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context. Uses dilated convolutions at rates 6, 12, and 18 to increase receptive field without losing resolution. The decoder fuses low-level and high-level features for sharp boundary prediction.

## Folder Structure

```
Oil_Spill/
├── preprocess.py             # SAR preprocessing pipeline
├── train_models.py           # Model training and comparison
├── inference_postprocess.py  # Inference + postprocessing on test data
├── app.py                    # Streamlit GUI for demo
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/
│   ├── train/                # Raw training data (TIFF)
│   ├── test/                 # Raw test data (TIFF)
│   └── processed/            # Preprocessed data (PNG)
│       ├── train/
│       │   ├── images/
│       │   └── masks/
│       └── test/
│           ├── images/
│           └── masks/
│
├── models/                   # Saved model weights
│   ├── unet.pth
│   ├── attention_unet.pth
│   └── deeplabv3.pth
│
└── outputs/
    ├── gui_results/              # GUI prediction outputs
    ├── training/             # Training evaluation outputs
    │   ├── loss_curves.png
    │   ├── dice_curves.png
    │   ├── iou_curves.png
    │   ├── mAP_curves.png
    │   ├── accuracy_curves.png
    │   ├── confusion_matrices.png
    │   ├── model_comparison.png
    │   └── model_comparison.csv
    │
    └── final_predictions/    # Inference + postprocessing outputs
        ├── overall_test_summary.csv
        ├── test_model_comparison.png
        ├── u-net/
        │   ├── *.png                     # 3-panel predictions per image
        │   ├── per_image_metrics.csv
        │   ├── U-Net_best_worst_montage.png
        │   └── U-Net_false_positive_analysis.png
        ├── attention_u-net/
        │   ├── *.png
        │   ├── per_image_metrics.csv
        │   ├── Attention U-Net_best_worst_montage.png
        │   └── Attention U-Net_false_positive_analysis.png
        └── deeplabv3plus/
            ├── *.png
            ├── per_image_metrics.csv
            ├── DeepLabV3+_best_worst_montage.png
            └── DeepLabV3+_false_positive_analysis.png
```

## Training Methodology

1. **Data Split**: 80% training, 20% validation (stratified random split)
2. **Augmentation**: Random horizontal/vertical flips, 90-degree rotations
3. **Loss Function**: Combined Dice Loss + Cross-Entropy Loss
4. **Optimizer**: Adam (learning rate = 1e-4)
5. **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
6. **Early Stopping**: Patience of 10 epochs on validation loss
7. **Best Model**: Saved based on lowest validation loss

## Postprocessing Pipeline

After model inference, six machine vision postprocessing techniques are applied to clean and refine predictions:

### 1. Confidence Thresholding
Applies a minimum probability threshold (default 0.5) to the model's softmax output. Only pixels where the Oil Spill class probability exceeds the threshold are classified as oil spill. This removes low-confidence predictions that are likely noise.

### 2. Morphological Opening / Closing
- **Opening** (erosion followed by dilation): Removes small isolated false positive blobs (salt noise) without affecting larger oil spill regions.
- **Closing** (dilation followed by erosion): Fills small holes inside predicted oil spill regions to create smoother, more complete detections.
- Uses an elliptical structuring element (5x5 kernel).

### 3. Small Blob Removal
Connected component analysis identifies individual predicted regions. Components with area below 50 pixels are removed. Real oil spills cover a minimum area on SAR imagery, so tiny isolated detections are almost always false positives.

### 4. Shape / Area Filtering
Each connected component is evaluated for geometric plausibility:
- Regions below minimum area threshold (30 pixels) are removed.
- Regions with extreme aspect ratios (> 10:1) are removed as they likely represent noise or sensor artifacts rather than actual oil spills.

### 5. Contour Smoothing
The Douglas-Peucker algorithm simplifies contour polygons by reducing the number of vertices while preserving overall shape. This produces cleaner, more natural-looking oil spill boundaries suitable for reports and publications.

### 6. Connected Component Analysis (Final)
After all filtering, a final connected component analysis labels each distinct oil spill region with a unique ID and computes statistics (area, bounding box, centroid) for each region.

## Metrics Explanation

| Metric | Formula | Description |
|--------|---------|-------------|
| **Dice Score** | 2TP / (2TP + FP + FN) | Measures overlap between prediction and ground truth. 1.0 = perfect overlap |
| **IoU (Jaccard)** | TP / (TP + FP + FN) | Intersection over Union. Stricter than Dice |
| **Accuracy** | Correct / Total pixels | Overall pixel classification accuracy |
| **mAP** | Mean of per-class precision | Mean Average Precision across all classes |
| **Precision** | TP / (TP + FP) | How many predicted positives are correct |
| **Recall** | TP / (TP + FN) | How many actual positives are detected |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess SAR data (if not already done)

```bash
python preprocess.py
```

### 3. Train all models

```bash
python train_models.py
```

This will:
- Train U-Net, Attention U-Net, and DeepLabV3+ sequentially
- Save best model weights to `models/`
- Generate all evaluation plots and comparison table to `outputs/training/`

### 4. Run inference on test data

```bash
python inference_postprocess.py
```

This will:
- Load raw test TIFF images from `data/test/`
- Apply the same SAR preprocessing pipeline used during training
- Run inference with all 3 trained models
- Apply 6-step postprocessing pipeline (morphological ops, blob removal, CCA, contour smoothing, confidence thresholding, shape/area filtering)
- Save outputs to `outputs/final_predictions/`

## Output Interpretation

### 3-Panel Prediction Images
Each test image generates a 3-panel PNG showing:
1. **Original SAR Image**: The preprocessed grayscale SAR input
2. **Ground Truth Mask**: The actual oil spill annotations (Red = Oil Spill)
3. **Prediction Overlay**: Model prediction overlaid on the original image with:
   - Red regions = Predicted oil spill
   - Red contour lines = Oil spill boundaries
   - Info box showing confidence, spill area %, region count, Dice, and IoU

### Overlay Color Legend
| Color | Class |
|-------|-------|
| Red | Oil Spill |
| Yellow | Look-Alike |
| Cyan | Ship |
| Green | Land |
| Black | Sea / Background |

### CSV Outputs
- **per_image_metrics.csv**: Per-image Dice, IoU, Precision, Recall, Accuracy, pixel counts, and spill area percentage for each test image
- **overall_test_summary.csv**: Aggregated metrics across all test images for each model, including oil-only subset analysis and false positive/negative counts

### Analysis Images
- **best_worst_montage.png**: Side-by-side comparison of the 5 best and 5 worst predictions (ranked by Dice score)
- **false_positive_analysis.png**: Detailed view of images with highest false positive rates, showing FP regions highlighted in yellow

## GUI Demo

A clean Streamlit-based GUI is provided for interactive demonstration.

### Features
- Upload preprocessed SAR images (.png)
- Select from 3 trained models (U-Net, Attention U-Net, DeepLabV3+)
- 3-panel display: Original | Ground Truth | Prediction Overlay
- Color-coded segmentation overlay (Red = Oil Spill)
- Real-time metrics: Confidence, Spill Area, Dice, IoU, Precision, Recall
- Optional ground truth upload for accuracy evaluation
- Auto-saves results to `outputs/gui_results/`

### Run the GUI

```bash
streamlit run app.py
```

This opens a browser interface at `http://localhost:8501`.

### Using the GUI
1. Upload a preprocessed SAR image from `data/processed/train/images/`
2. (Optional) Upload the corresponding mask from `data/processed/train/masks/`
3. Select a model from the sidebar
4. Click **Run Prediction**
5. View the segmentation overlay and metrics

## Dependencies

- Python 3.8+
- PyTorch (with CUDA support recommended)
- torchvision
- Streamlit
- NumPy, OpenCV, matplotlib
- scikit-learn, pandas
- tqdm, tifffile
# Oil-Spill-Detection
