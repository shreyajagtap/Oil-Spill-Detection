"""
SAR Oil Spill Detection - Model Training & Comparison
======================================================
Trains and compares 3 segmentation models:
  1. U-Net
  2. Attention U-Net
  3. DeepLabV3+

Multi-class segmentation with 5 classes:
  0 = Sea / Background
  1 = Oil Spill
  2 = Look-Alike
  3 = Ship
  4 = Land

NOTE: Metrics are computed ONLY for classes that actually exist
in the dataset. Empty classes (no ground truth pixels) are excluded
to avoid inflated scores.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "train")
MODEL_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "training")

# Training hyperparameters
NUM_CLASSES    = 2          # Only classes present: Sea (0), Oil Spill (1)
IMG_SIZE       = 256        # Input image size
BATCH_SIZE     = 4
NUM_EPOCHS     = 50
LEARNING_RATE  = 1e-4
VAL_SPLIT      = 0.2        # 20% for validation
RANDOM_SEED    = 42

# Class names for the classes that ACTUALLY exist in data
CLASS_NAMES    = ['Sea', 'Oil Spill']

# Class weights to handle imbalance
# Sea ~98.9% of pixels, Oil Spill ~1.1% -> ratio ~88:1
# Using inverse frequency weight: oil spill gets ~50x weight
# (not full 88x to avoid over-correcting)
CLASS_WEIGHTS  = [1.0, 50.0]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET CLASS
# ============================================================
class OilSpillDataset(Dataset):
    """
    Custom dataset for SAR oil spill segmentation.

    Reads preprocessed PNG images and masks.
    Masks are converted from [0, 255] to class labels [0, 1].
    """

    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Read image (grayscale) ---
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # --- Read mask (grayscale) ---
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Map mask values to class labels
        # Preprocessing saved: 0 -> 0 (background), 1 -> 255 (oil spill)
        # Map 255 back to 1
        mask = np.where(mask == 255, 1, 0).astype(np.int64)

        # --- Data augmentation (training only) ---
        if self.augment:
            image, mask = self._augment(image, mask)

        # --- Convert to tensors ---
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask  = torch.from_numpy(mask)                 # (H, W)

        return image, mask

    def _augment(self, image, mask):
        """Apply random augmentations for training."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask  = np.fliplr(mask).copy()

        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask  = np.flipud(mask).copy()

        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        image = np.rot90(image, k).copy()
        mask  = np.rot90(mask, k).copy()

        return image, mask


# ============================================================
# MODEL 1: U-Net
# ============================================================

class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for semantic segmentation.

    Structure:
      Encoder (downsampling path) with 4 levels
      Bottleneck
      Decoder (upsampling path) with 4 levels + skip connections
      Final 1x1 conv for classification
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        # Encoder (downsampling) - 32-base channels for GPU memory
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder (upsampling)
        self.up4   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4  = DoubleConv(512, 256)
        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3  = DoubleConv(256, 128)
        self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2  = DoubleConv(128, 64)
        self.up1   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1  = DoubleConv(64, 32)

        # Final classification layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# ============================================================
# MODEL 2: Attention U-Net
# ============================================================

class AttentionGate(nn.Module):
    """
    Attention gate for Attention U-Net.

    Learns to focus on relevant spatial regions in skip connections.
    The gating signal (from decoder) guides which parts of the
    skip connection (from encoder) are important.
    """
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode='bilinear', align_corners=True)
        attention = self.relu(g + s)
        attention = self.psi(attention)
        return skip * attention


class AttentionUNet(nn.Module):
    """
    Attention U-Net: U-Net with attention gates at skip connections.

    The attention gates learn to suppress irrelevant regions and
    highlight salient features, improving segmentation accuracy.
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        # Encoder - 32-base channels
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder with attention gates
        self.up4   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4  = AttentionGate(gate_ch=256, skip_ch=256, inter_ch=128)
        self.dec4  = DoubleConv(512, 256)

        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3  = AttentionGate(gate_ch=128, skip_ch=128, inter_ch=64)
        self.dec3  = DoubleConv(256, 128)

        self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2  = AttentionGate(gate_ch=64, skip_ch=64, inter_ch=32)
        self.dec2  = DoubleConv(128, 64)

        self.up1   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1  = AttentionGate(gate_ch=32, skip_ch=32, inter_ch=16)
        self.dec1  = DoubleConv(64, 32)

        # Final classification layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention gates
        u4 = self.up4(b)
        a4 = self.att4(gate=u4, skip=e4)
        d4 = self.dec4(torch.cat([u4, a4], dim=1))

        u3 = self.up3(d4)
        a3 = self.att3(gate=u3, skip=e3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))

        u2 = self.up2(d3)
        a2 = self.att2(gate=u2, skip=e2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))

        u1 = self.up1(d2)
        a1 = self.att1(gate=u1, skip=e1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))

        return self.final(d1)


# ============================================================
# MODEL 3: DeepLabV3+
# ============================================================

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module for DeepLabV3+.

    Uses multiple parallel atrous (dilated) convolutions with different
    dilation rates to capture multi-scale context.
    """
    def __init__(self, in_ch, out_ch=128):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.atrous6 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.atrous12 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.atrous18 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Global average pooling branch - GroupNorm to handle 1x1 spatial
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )
        # Projection after concatenation (5 branches * out_ch)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1x1(x)
        feat2 = self.atrous6(x)
        feat3 = self.atrous12(x)
        feat4 = self.atrous18(x)
        feat5 = F.interpolate(self.global_pool(x), size=size,
                              mode='bilinear', align_corners=True)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture for semantic segmentation.

    Uses a custom ResNet-like encoder with ASPP module and a decoder
    with low-level feature fusion.
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        # --- Encoder ---
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # --- ASPP module ---
        self.aspp = ASPP(in_ch=256, out_ch=128)

        # --- Decoder ---
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(32, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128 + 24, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ))
        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder
        x = self.stem(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = F.interpolate(x, size=low_level.shape[2:],
                          mode='bilinear', align_corners=True)
        low_level = self.low_level_conv(low_level)
        x = torch.cat([x, low_level], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)

        # Upsample to original size
        x = F.interpolate(x, size=input_size,
                          mode='bilinear', align_corners=True)
        return x


# ============================================================
# LOSS FUNCTION: Weighted Dice + Weighted Cross-Entropy
# ============================================================

class DiceCELoss(nn.Module):
    """
    Combined Dice Loss + Weighted Cross-Entropy Loss.

    - Class weights in CE penalize misclassifying the minority class
      (Oil Spill) more heavily than the majority (Sea).
    - Dice Loss directly optimizes overlap, which helps with imbalanced data.
    """
    def __init__(self, num_classes=2, class_weights=None, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

        # Weighted cross-entropy to handle class imbalance
        if class_weights is not None:
            weight = torch.FloatTensor(class_weights)
        else:
            weight = None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        # Move CE weight to same device as pred
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(pred.device)

        # Weighted Cross-Entropy loss
        ce_loss = self.ce(pred, target)

        # Dice loss (per-class, then averaged over PRESENT classes only)
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        dice_per_class = []
        for cls in range(self.num_classes):
            # Check if this class exists in the batch
            if target_onehot[:, cls].sum() == 0 and pred_soft[:, cls].sum() == 0:
                continue  # Skip empty classes

            intersection = (pred_soft[:, cls] * target_onehot[:, cls]).sum()
            union = pred_soft[:, cls].sum() + target_onehot[:, cls].sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)

        if len(dice_per_class) > 0:
            dice_loss = 1.0 - torch.stack(dice_per_class).mean()
        else:
            dice_loss = torch.tensor(0.0, device=pred.device)

        return ce_loss + dice_loss


# ============================================================
# METRICS COMPUTATION (Only for classes that EXIST in data)
# ============================================================

def compute_metrics(pred, target, num_classes=2):
    """
    Compute segmentation metrics ONLY for classes present in the data.

    This avoids inflated scores from empty classes.
    Reports both overall and per-class metrics.

    Returns dict with:
      - Per-class: dice_cls0, dice_cls1, iou_cls0, iou_cls1, etc.
      - Mean (over present classes): dice, iou, precision, recall
      - Oil-spill specific: oil_dice, oil_iou, oil_precision, oil_recall
      - Pixel accuracy (excludes class imbalance effect in averaging)
    """
    pred = pred.argmax(dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    smooth = 1e-6
    results = {}

    # Per-class metrics (only for classes with ground truth pixels)
    present_dice = []
    present_iou  = []
    present_prec = []
    present_rec  = []

    for cls in range(num_classes):
        pred_cls   = (pred == cls)
        target_cls = (target == cls)

        tp = np.sum(pred_cls & target_cls)
        fp = np.sum(pred_cls & ~target_cls)
        fn = np.sum(~pred_cls & target_cls)

        # Only compute if this class exists in ground truth OR predictions
        if np.sum(target_cls) == 0 and np.sum(pred_cls) == 0:
            continue

        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        iou  = (tp + smooth) / (tp + fp + fn + smooth)
        prec = (tp + smooth) / (tp + fp + smooth)
        rec  = (tp + smooth) / (tp + fn + smooth)

        results[f'dice_cls{cls}'] = dice
        results[f'iou_cls{cls}']  = iou
        results[f'prec_cls{cls}'] = prec
        results[f'rec_cls{cls}']  = rec

        present_dice.append(dice)
        present_iou.append(iou)
        present_prec.append(prec)
        present_rec.append(rec)

    # Mean metrics (averaged over PRESENT classes only)
    results['dice']      = np.mean(present_dice) if present_dice else 0.0
    results['iou']       = np.mean(present_iou)  if present_iou  else 0.0
    results['precision'] = np.mean(present_prec) if present_prec else 0.0
    results['recall']    = np.mean(present_rec)  if present_rec  else 0.0
    results['mAP']       = results['precision']  # mAP = mean precision over classes

    # Pixel accuracy
    results['accuracy'] = np.sum(pred == target) / len(target)

    # Oil Spill specific metrics (class 1) - the most important metrics
    results['oil_dice']      = results.get('dice_cls1', 0.0)
    results['oil_iou']       = results.get('iou_cls1', 0.0)
    results['oil_precision'] = results.get('prec_cls1', 0.0)
    results['oil_recall']    = results.get('rec_cls1', 0.0)

    return results


# ============================================================
# TRAINING ONE EPOCH
# ============================================================

TRACKED_METRICS = ['dice', 'iou', 'accuracy', 'mAP', 'oil_dice', 'oil_iou', 'oil_precision', 'oil_recall']

def train_one_epoch(model, loader, criterion, optimizer):
    """Train the model for one epoch. Returns average loss and metrics."""
    model.train()
    total_loss = 0
    all_metrics = {k: 0 for k in TRACKED_METRICS}
    num_batches = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        batch_metrics = compute_metrics(outputs.detach(), masks, NUM_CLASSES)
        for key in all_metrics:
            all_metrics[key] += batch_metrics.get(key, 0)
        num_batches += 1

    avg_loss = total_loss / num_batches
    for key in all_metrics:
        all_metrics[key] /= num_batches

    return avg_loss, all_metrics


# ============================================================
# VALIDATION ONE EPOCH
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion):
    """Validate the model. Returns average loss and metrics."""
    model.eval()
    total_loss = 0
    all_metrics = {k: 0 for k in TRACKED_METRICS}
    num_batches = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
        batch_metrics = compute_metrics(outputs, masks, NUM_CLASSES)
        for key in all_metrics:
            all_metrics[key] += batch_metrics.get(key, 0)
        num_batches += 1

    avg_loss = total_loss / num_batches
    for key in all_metrics:
        all_metrics[key] /= num_batches

    return avg_loss, all_metrics


# ============================================================
# FULL TRAINING LOOP FOR ONE MODEL
# ============================================================

def train_model(model, model_name, train_loader, val_loader, save_path):
    """
    Train a model for NUM_EPOCHS with early stopping patience.
    Returns training history (losses and metrics per epoch).
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    model = model.to(DEVICE)
    criterion = DiceCELoss(num_classes=NUM_CLASSES, class_weights=CLASS_WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # History tracking - now includes oil spill specific metrics
    history = {
        'train_loss': [],     'val_loss': [],
        'train_dice': [],     'val_dice': [],
        'train_iou': [],      'val_iou': [],
        'train_acc': [],      'val_acc': [],
        'train_mAP': [],      'val_mAP': [],
        'train_oil_dice': [], 'val_oil_dice': [],
        'train_oil_iou': [],  'val_oil_iou': [],
        'train_oil_prec': [], 'val_oil_prec': [],
        'train_oil_rec': [],  'val_oil_rec': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 10

    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_mAP'].append(train_metrics['mAP'])
        history['val_mAP'].append(val_metrics['mAP'])
        history['train_oil_dice'].append(train_metrics['oil_dice'])
        history['val_oil_dice'].append(val_metrics['oil_dice'])
        history['train_oil_iou'].append(train_metrics['oil_iou'])
        history['val_oil_iou'].append(val_metrics['oil_iou'])
        history['train_oil_prec'].append(train_metrics['oil_precision'])
        history['val_oil_prec'].append(val_metrics['oil_precision'])
        history['train_oil_rec'].append(train_metrics['oil_recall'])
        history['val_oil_rec'].append(val_metrics['oil_recall'])

        # Print progress - now shows Oil Spill specific metrics
        print(f"  Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}/{val_metrics['dice']:.4f} | "
              f"Oil Dice: {train_metrics['oil_dice']:.4f}/{val_metrics['oil_dice']:.4f} | "
              f"Oil IoU: {train_metrics['oil_iou']:.4f}/{val_metrics['oil_iou']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    # Load best model weights
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"  Best model saved to: {save_path}")

    return history


# ============================================================
# CONFUSION MATRIX
# ============================================================

@torch.no_grad()
def get_confusion_matrix(model, loader):
    """Compute confusion matrix over the entire validation set."""
    model.eval()
    all_preds = []
    all_targets = []

    for images, masks in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy().flatten()
        targets = masks.numpy().flatten()
        all_preds.extend(preds)
        all_targets.extend(targets)

    return confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))


# ============================================================
# COMPUTE FINAL METRICS ON VALIDATION SET
# ============================================================

@torch.no_grad()
def compute_final_metrics(model, loader):
    """Compute final metrics over the entire validation set."""
    model.eval()
    all_metrics = {k: 0 for k in TRACKED_METRICS + ['precision', 'recall']}
    num_batches = 0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)
        outputs = model(images)
        batch_metrics = compute_metrics(outputs, masks, NUM_CLASSES)
        for key in all_metrics:
            all_metrics[key] += batch_metrics.get(key, 0)
        num_batches += 1

    for key in all_metrics:
        all_metrics[key] /= num_batches

    return all_metrics


# ============================================================
# PLOTTING FUNCTIONS (Thesis Quality)
# ============================================================

# Plot style configuration
PLOT_PARAMS = {
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2,
    'figure.figsize': (10, 6),
}
plt.rcParams.update(PLOT_PARAMS)

MODEL_COLORS = {'U-Net': '#2196F3', 'Attention U-Net': '#FF5722', 'DeepLabV3+': '#4CAF50'}


def plot_loss_curves(all_histories, output_dir):
    """Plot training and validation loss curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, hist in all_histories.items():
        ax.plot(hist['train_loss'], label=name, color=MODEL_COLORS[name])
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Dice + Weighted CE)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, hist in all_histories.items():
        ax.plot(hist['val_loss'], label=name, color=MODEL_COLORS[name])
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Dice + Weighted CE)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: loss_curves.png")


def plot_metric_curve(all_histories, train_key, val_key, metric_name, filename, output_dir):
    """Plot a single metric curve (train & val) for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, hist in all_histories.items():
        ax.plot(hist[train_key], label=name, color=MODEL_COLORS[name])
    ax.set_title(f'Training {metric_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax = axes[1]
    for name, hist in all_histories.items():
        ax.plot(hist[val_key], label=name, color=MODEL_COLORS[name])
    ax.set_title(f'Validation {metric_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_confusion_matrices(all_cms, output_dir):
    """Plot confusion matrix for each model side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, cm) in zip(axes, all_cms.items()):
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')

    plt.suptitle('Confusion Matrices (Sea vs Oil Spill)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: confusion_matrices.png")


def plot_comparison_bar(all_final_metrics, output_dir):
    """Bar chart comparing final validation metrics across all models."""
    # Show both mean metrics and oil-spill specific metrics
    metric_labels = ['Mean Dice', 'Mean IoU', 'Pixel Acc', 'Oil Dice', 'Oil IoU', 'Oil Prec', 'Oil Recall']
    metric_keys   = ['dice', 'iou', 'accuracy', 'oil_dice', 'oil_iou', 'oil_precision', 'oil_recall']

    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, (name, final_m) in enumerate(all_final_metrics.items()):
        values = [final_m[k] for k in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=name, color=MODEL_COLORS[name])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Validation Metrics (Mean + Oil Spill Specific)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.15])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: model_comparison.png")


def save_comparison_csv(all_final_metrics, output_dir):
    """Save a CSV table comparing all models with per-class breakdown."""
    rows = []
    for name, m in all_final_metrics.items():
        rows.append({
            'Model': name,
            'Mean_Dice': round(m['dice'], 4),
            'Mean_IoU': round(m['iou'], 4),
            'Pixel_Accuracy': round(m['accuracy'], 4),
            'Mean_Precision': round(m['precision'], 4),
            'Mean_Recall': round(m['recall'], 4),
            'Oil_Dice': round(m['oil_dice'], 4),
            'Oil_IoU': round(m['oil_iou'], 4),
            'Oil_Precision': round(m['oil_precision'], 4),
            'Oil_Recall': round(m['oil_recall'], 4),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: model_comparison.csv")
    print(f"\n{df.to_string(index=False)}")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(RANDOM_SEED)

    # Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SAR Oil Spill Detection - Model Training")
    print("=" * 60)
    print(f"  Device:        {DEVICE}")
    print(f"  Num Classes:   {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")
    print(f"  Class Weights: {CLASS_WEIGHTS}")
    print(f"  Batch Size:    {BATCH_SIZE}")
    print(f"  Epochs:        {NUM_EPOCHS}")
    print(f"  LR:            {LEARNING_RATE}")
    print(f"  Val Split:     {VAL_SPLIT}")

    # --------------------------------------------------------
    # Load dataset paths
    # --------------------------------------------------------
    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "images", "*.png")))
    mask_paths  = sorted(glob.glob(os.path.join(DATA_DIR, "masks", "*.png")))

    print(f"\n  Total images: {len(image_paths)}")
    print(f"  Total masks:  {len(mask_paths)}")

    if len(image_paths) == 0:
        print("  ERROR: No images found! Run preprocess.py first.")
        return

    # --------------------------------------------------------
    # Scan masks to report class distribution
    # --------------------------------------------------------
    print("\n  Scanning mask class distribution (all masks)...")
    total_pixels = 0
    class_pixels = {i: 0 for i in range(NUM_CLASSES)}
    masks_with_oil = 0
    for mp in mask_paths:
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        m = np.where(m == 255, 1, 0)
        total_pixels += m.size
        for cls in range(NUM_CLASSES):
            class_pixels[cls] += np.sum(m == cls)
        if np.any(m == 1):
            masks_with_oil += 1

    for cls in range(NUM_CLASSES):
        pct = class_pixels[cls] / total_pixels * 100
        print(f"    Class {cls} ({CLASS_NAMES[cls]}): {pct:.2f}% of pixels")
    print(f"    Images with oil spill: {masks_with_oil}/{len(mask_paths)} ({masks_with_oil/len(mask_paths)*100:.1f}%)")
    print(f"    Class imbalance ratio: 1:{class_pixels[0] // max(class_pixels[1], 1)} (Sea:Oil)")

    # --------------------------------------------------------
    # Train/Validation split
    # --------------------------------------------------------
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    print(f"\n  Train: {len(train_imgs)} | Validation: {len(val_imgs)}")

    # Create datasets and dataloaders
    train_dataset = OilSpillDataset(train_imgs, train_masks, augment=True)
    val_dataset   = OilSpillDataset(val_imgs, val_masks, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # --------------------------------------------------------
    # Define models
    # --------------------------------------------------------
    models_config = {
        'U-Net': {
            'model': UNet(in_channels=1, num_classes=NUM_CLASSES),
            'path':  os.path.join(MODEL_DIR, 'unet.pth'),
        },
        'Attention U-Net': {
            'model': AttentionUNet(in_channels=1, num_classes=NUM_CLASSES),
            'path':  os.path.join(MODEL_DIR, 'attention_unet.pth'),
        },
        'DeepLabV3+': {
            'model': DeepLabV3Plus(in_channels=1, num_classes=NUM_CLASSES),
            'path':  os.path.join(MODEL_DIR, 'deeplabv3.pth'),
        },
    }

    # --------------------------------------------------------
    # Train all models
    # --------------------------------------------------------
    all_histories    = {}
    all_cms          = {}
    all_final_metrics = {}

    for name, config in models_config.items():
        # Free GPU memory before training each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Train
        history = train_model(
            model=config['model'],
            model_name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            save_path=config['path'],
        )
        all_histories[name] = history

        # Confusion matrix on validation set
        cm = get_confusion_matrix(config['model'], val_loader)
        all_cms[name] = cm

        # Final metrics
        final_metrics = compute_final_metrics(config['model'], val_loader)
        all_final_metrics[name] = final_metrics
        print(f"  {name} Final -> "
              f"Mean Dice: {final_metrics['dice']:.4f} | "
              f"Oil Dice: {final_metrics['oil_dice']:.4f} | "
              f"Oil IoU: {final_metrics['oil_iou']:.4f} | "
              f"Oil Prec: {final_metrics['oil_precision']:.4f} | "
              f"Oil Rec: {final_metrics['oil_recall']:.4f}")

        # Move model to CPU to free GPU for next model
        config['model'].cpu()

    # --------------------------------------------------------
    # Generate all plots and outputs
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Generating Evaluation Outputs")
    print(f"{'='*60}")

    # 1 & 2. Training and Validation Loss Curves
    plot_loss_curves(all_histories, OUTPUT_DIR)

    # 3. Dice Score Curve (Mean over present classes)
    plot_metric_curve(all_histories, 'train_dice', 'val_dice',
                      'Mean Dice Score', 'dice_curves.png', OUTPUT_DIR)

    # 4. IoU Curve
    plot_metric_curve(all_histories, 'train_iou', 'val_iou',
                      'Mean IoU Score', 'iou_curves.png', OUTPUT_DIR)

    # 5. mAP Curve
    plot_metric_curve(all_histories, 'train_mAP', 'val_mAP',
                      'Mean Average Precision', 'mAP_curves.png', OUTPUT_DIR)

    # 6. Accuracy Curve
    plot_metric_curve(all_histories, 'train_acc', 'val_acc',
                      'Pixel Accuracy', 'accuracy_curves.png', OUTPUT_DIR)

    # 7. Oil Spill Dice Curve (THE key metric)
    plot_metric_curve(all_histories, 'train_oil_dice', 'val_oil_dice',
                      'Oil Spill Dice Score', 'oil_dice_curves.png', OUTPUT_DIR)

    # 8. Oil Spill IoU Curve
    plot_metric_curve(all_histories, 'train_oil_iou', 'val_oil_iou',
                      'Oil Spill IoU Score', 'oil_iou_curves.png', OUTPUT_DIR)

    # 9. Oil Spill Precision & Recall
    plot_metric_curve(all_histories, 'train_oil_prec', 'val_oil_prec',
                      'Oil Spill Precision', 'oil_precision_curves.png', OUTPUT_DIR)
    plot_metric_curve(all_histories, 'train_oil_rec', 'val_oil_rec',
                      'Oil Spill Recall', 'oil_recall_curves.png', OUTPUT_DIR)

    # 10. Confusion Matrix
    plot_confusion_matrices(all_cms, OUTPUT_DIR)

    # 11. Model Comparison Bar Chart
    plot_comparison_bar(all_final_metrics, OUTPUT_DIR)

    # 12. Model Comparison Table CSV
    save_comparison_csv(all_final_metrics, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("  ALL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Models saved to:  {MODEL_DIR}/")
    print(f"  Outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
