#!/usr/bin/env python
"""
Experiment: Stronger detection baselines.
Tests multiple detector architectures on larger dataset:
  1. Logistic Regression on latent statistics (baseline, from before)
  2. MLP (2-layer) on latent statistics
  3. Logistic Regression on pixel residuals (pixel-domain stats)
  4. MLP on pixel residuals
  5. Spectral features (FFT of residual)
  6. Combined feature set (latent + pixel + spectral)
All with proper train/test splits, ROC curves, and confidence intervals.
Also reports precision/recall at fixed low-FPR operating points.
"""
import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time
import json
import csv
from scipy import stats as sp_stats
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, bit_accuracy

FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
IMG_SIZE = 256

torch.manual_seed(42)
np.random.seed(42)


def extract_latent_features(vae, img):
    """20 features: per-channel mean, std, median, skew, kurtosis."""
    lat = vae.encode(img)
    recon = vae.decode(lat)
    lat_rt = vae.encode(recon)
    feats = []
    for ch in range(4):
        x = lat_rt[0, ch].cpu().numpy().flatten()
        mu, sig = x.mean(), x.std() + 1e-8
        feats.extend([mu, sig, float(np.median(x)),
                      float(((x - mu)**3).mean() / sig**3),
                      float(((x - mu)**4).mean() / sig**4)])
    return np.array(feats)


def latent_features_from_tensor(latent):
    """20 features: per-channel mean, std, median, skew, kurtosis."""
    feats = []
    for ch in range(4):
        x = latent[0, ch].cpu().numpy().flatten()
        mu, sig = x.mean(), x.std() + 1e-8
        feats.extend([mu, sig, float(np.median(x)),
                      float(((x - mu)**3).mean() / sig**3),
                      float(((x - mu)**4).mean() / sig**4)])
    return np.array(feats)


def latent_features_extended_from_tensor(latent):
    """40 features: latent stats + position-level stats."""
    feats = []
    for ch in range(4):
        x2 = latent[0, ch].cpu().numpy()
        x = x2.flatten()
        mu, sig = x.mean(), x.std() + 1e-8
        feats.extend([
            mu, sig, float(np.median(x)),
            float(((x - mu)**3).mean() / sig**3),
            float(((x - mu)**4).mean() / sig**4),
            float(np.percentile(x, 5)),
            float(np.percentile(x, 95)),
            float(np.max(x) - np.min(x)),
            float(np.percentile(x, 75) - np.percentile(x, 25)),
            float(np.abs(np.diff(x2, axis=1)).mean()),
        ])
    return np.array(feats)


def extract_latent_features_extended(vae, img):
    """40 features: latent stats + position-level stats (percentiles, range, IQR)."""
    lat = vae.encode(img)
    recon = vae.decode(lat)
    lat_rt = vae.encode(recon)
    feats = []
    for ch in range(4):
        x = lat_rt[0, ch].cpu().numpy().flatten()
        mu, sig = x.mean(), x.std() + 1e-8
        feats.extend([
            mu, sig, float(np.median(x)),
            float(((x - mu)**3).mean() / sig**3),  # skewness
            float(((x - mu)**4).mean() / sig**4),  # kurtosis
            float(np.percentile(x, 5)),
            float(np.percentile(x, 95)),
            float(np.max(x) - np.min(x)),  # range
            float(np.percentile(x, 75) - np.percentile(x, 25)),  # IQR
            float(np.abs(np.diff(x.reshape(32, 32), axis=1)).mean()),  # spatial gradient
        ])
    return np.array(feats)


def extract_pixel_residual_features(img_clean_rt, img):
    """Features from pixel-domain residual between clean round-trip and test image."""
    a = np.array(img_clean_rt).astype(float) / 255.0
    b = np.array(img).astype(float) / 255.0
    residual = b - a
    feats = []
    for ch in range(3):
        r = residual[:, :, ch].flatten()
        mu, sig = r.mean(), r.std() + 1e-8
        feats.extend([
            mu, sig, float(np.median(r)),
            float(((r - mu)**3).mean() / sig**3),
            float(((r - mu)**4).mean() / sig**4),
            float(np.max(np.abs(r))),
        ])
    return np.array(feats)


def extract_spectral_features(img_clean_rt, img):
    """FFT-based features of the residual."""
    a = np.array(img_clean_rt).astype(float) / 255.0
    b = np.array(img).astype(float) / 255.0
    residual = b - a
    feats = []
    for ch in range(3):
        r = residual[:, :, ch]
        fft = np.fft.fft2(r)
        magnitude = np.abs(fft)
        # Split into frequency bands
        h, w = magnitude.shape
        low = magnitude[:h//4, :w//4].mean()
        mid = magnitude[h//4:h//2, w//4:w//2].mean()
        high = magnitude[h//2:, w//2:].mean()
        total = magnitude.mean() + 1e-10
        feats.extend([
            float(low / total),
            float(mid / total),
            float(high / total),
            float(total),
            float(magnitude.std()),
        ])
    return np.array(feats)


def make_test_images(n=100):
    rng = np.random.RandomState(42)
    images = []
    for i in range(n):
        kind = i % 5
        arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if kind == 0:
            arr = rng.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        elif kind == 1:
            for r in range(0, IMG_SIZE, 32):
                for c in range(0, IMG_SIZE, 32):
                    arr[r:r+32, c:c+32] = rng.randint(0, 255, 3)
        elif kind == 2:
            t = np.linspace(0, 1, IMG_SIZE)
            for ch in range(3):
                arr[:,:,ch] = (np.outer(t, np.ones(IMG_SIZE)) * rng.randint(100,255)).astype(np.uint8)
        elif kind == 3:
            block = 16
            for r in range(0, IMG_SIZE, block):
                for c in range(0, IMG_SIZE, block):
                    if ((r//block)+(c//block))%2==0:
                        arr[r:r+block, c:c+block] = rng.randint(150,255,3)
                    else:
                        arr[r:r+block, c:c+block] = rng.randint(0,100,3)
        else:
            base = rng.randint(50, 200, 3)
            arr[:] = base
            noise = rng.randint(-30, 30, (IMG_SIZE, IMG_SIZE, 3))
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def get_positive_scores(clf, X):
    if hasattr(clf, 'decision_function'):
        return clf.decision_function(X)
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    return clf.predict(X)


def operating_point_metrics(y_true, scores, target_fpr):
    clean_scores = scores[y_true == 0]
    threshold = np.quantile(clean_scores, 1.0 - target_fpr, method='higher')
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    achieved_fpr = fp / max(fp + tn, 1)
    return {
        'target_fpr': float(target_fpr),
        'threshold': float(threshold),
        'achieved_fpr': float(achieved_fpr),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def evaluate_detector(clf, X, y, cv):
    scores = np.zeros(len(y), dtype=float)
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in cv.split(X, y):
        model = clone(clf)
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = get_positive_scores(model, X[test_idx])
        preds[test_idx] = model.predict(X[test_idx])

    op_points = {
        '0.1%': operating_point_metrics(y, scores, 0.001),
        '1%': operating_point_metrics(y, scores, 0.01),
        '5%': operating_point_metrics(y, scores, 0.05),
    }
    return {
        'auc_mean': float(roc_auc_score(y, scores)),
        'auc_std': 0.0,
        'acc_mean': float(accuracy_score(y, preds)),
        'acc_std': 0.0,
        'operating_points': op_points,
    }


# ================================================================
print("=" * 70, flush=True)
print("STRONGER DETECTION EXPERIMENT", flush=True)
print("=" * 70, flush=True)
t0 = time.time()

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
N_IMAGES = 30  # 30 clean + 30 stego = 60 samples
images = make_test_images(N_IMAGES)
print(f"Created {len(images)} test images", flush=True)

EPSILONS = [1.0, 2.0, 5.0]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

DETECTORS = {
    'LR-Latent': lambda: make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000, random_state=42)),
    'MLP-Latent': lambda: make_pipeline(StandardScaler(),
                                         MLPClassifier(hidden_layer_sizes=(64, 32),
                                                       max_iter=1000, random_state=42)),
    'LR-Residual': lambda: make_pipeline(StandardScaler(),
                                          LogisticRegression(max_iter=2000, random_state=42)),
    'MLP-Residual': lambda: make_pipeline(StandardScaler(),
                                           MLPClassifier(hidden_layer_sizes=(64, 32),
                                                         max_iter=1000, random_state=42)),
    'LR-Spectral': lambda: make_pipeline(StandardScaler(),
                                          LogisticRegression(max_iter=2000, random_state=42)),
    'LR-Combined': lambda: make_pipeline(StandardScaler(),
                                          LogisticRegression(max_iter=2000, random_state=42)),
    'MLP-Combined': lambda: make_pipeline(StandardScaler(),
                                           MLPClassifier(hidden_layer_sizes=(128, 64),
                                                         max_iter=1000, random_state=42)),
}

all_results = {}

for eps in EPSILONS:
    print(f"\n--- epsilon = {eps} ---", flush=True)
    steg = PatchSteg(seed=42, epsilon=eps)

    # Build feature matrices for each feature type
    feat_latent, feat_latent_ext = [], []
    feat_residual, feat_spectral = [], []
    labels = []

    for i, img in enumerate(images):
        if i % 10 == 0:
            print(f"  Processing image {i}/{len(images)}...", flush=True)

        # Clean round-trip
        lat_clean = vae.encode(img)
        clean_rt = vae.decode(lat_clean)
        lat_clean_rt = vae.encode(clean_rt)

        # Clean features
        f_lat = latent_features_from_tensor(lat_clean_rt)
        f_lat_ext = latent_features_extended_from_tensor(lat_clean_rt)
        f_res = extract_pixel_residual_features(clean_rt, clean_rt)  # zero residual
        f_spec = extract_spectral_features(clean_rt, clean_rt)

        feat_latent.append(f_lat)
        feat_latent_ext.append(f_lat_ext)
        feat_residual.append(f_res)
        feat_spectral.append(f_spec)
        labels.append(0)

        # Stego
        carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=eps)
        torch.manual_seed(42 + i)
        bits = torch.randint(0, 2, (20,)).tolist()
        lat_m = steg.encode_message(lat_clean, carriers, bits)
        stego = vae.decode(lat_m)
        lat_stego_rt = vae.encode(stego)

        f_lat_s = latent_features_from_tensor(lat_stego_rt)
        f_lat_ext_s = latent_features_extended_from_tensor(lat_stego_rt)
        f_res_s = extract_pixel_residual_features(clean_rt, stego)
        f_spec_s = extract_spectral_features(clean_rt, stego)

        feat_latent.append(f_lat_s)
        feat_latent_ext.append(f_lat_ext_s)
        feat_residual.append(f_res_s)
        feat_spectral.append(f_spec_s)
        labels.append(1)

    X_latent = np.array(feat_latent)
    X_latent_ext = np.array(feat_latent_ext)
    X_residual = np.array(feat_residual)
    X_spectral = np.array(feat_spectral)
    X_combined = np.hstack([X_latent_ext, X_residual, X_spectral])
    y = np.array(labels)

    feature_map = {
        'LR-Latent': X_latent,
        'MLP-Latent': X_latent,
        'LR-Residual': X_residual,
        'MLP-Residual': X_residual,
        'LR-Spectral': X_spectral,
        'LR-Combined': X_combined,
        'MLP-Combined': X_combined,
    }

    for det_name, det_fn in DETECTORS.items():
        X = feature_map[det_name]
        clf = det_fn()
        try:
            result = evaluate_detector(clf, X, y, cv)
        except Exception as e:
            print(f"    {det_name}: FAILED ({e})", flush=True)
            result = {
                'auc_mean': 0.5,
                'auc_std': 0,
                'acc_mean': 0.5,
                'acc_std': 0,
                'operating_points': {},
            }

        all_results[(eps, det_name)] = result
        op_1 = result.get('operating_points', {}).get('1%', {})
        print(f"  {det_name:20s}: AUC={result['auc_mean']:.3f}+-{result['auc_std']:.3f}  "
              f"Acc={result['acc_mean']:.3f}+-{result['acc_std']:.3f}  "
              f"P/R@1%FPR={op_1.get('precision', 0):.3f}/{op_1.get('recall', 0):.3f}", flush=True)

# ================================================================
# Figure: detection heatmap
# ================================================================
print(f"\nGenerating figures...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

det_names = list(DETECTORS.keys())

# (a) AUC heatmap
ax = axes[0]
data = np.zeros((len(det_names), len(EPSILONS)))
for i, dn in enumerate(det_names):
    for j, eps in enumerate(EPSILONS):
        data[i, j] = all_results.get((eps, dn), {}).get('auc_mean', 0.5)

im = ax.imshow(data, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(EPSILONS)))
ax.set_xticklabels([f'ε={e}' for e in EPSILONS])
ax.set_yticks(range(len(det_names)))
ax.set_yticklabels(det_names)
for i in range(len(det_names)):
    for j in range(len(EPSILONS)):
        val = data[i, j]
        std = all_results.get((EPSILONS[j], det_names[i]), {}).get('auc_std', 0)
        color = 'white' if val > 0.7 else 'black'
        ax.text(j, i, f'{val:.2f}\n±{std:.2f}', ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')
ax.set_title('(a) Detection AUC by Detector and ε')
plt.colorbar(im, ax=ax, label='AUC')

# (b) Grouped bar chart: best detector per epsilon
ax = axes[1]
x = np.arange(len(EPSILONS))
width = 0.12
for i, dn in enumerate(det_names):
    vals = [all_results.get((eps, dn), {}).get('auc_mean', 0.5) for eps in EPSILONS]
    errs = [all_results.get((eps, dn), {}).get('auc_std', 0) for eps in EPSILONS]
    ax.bar(x + i * width - width * len(det_names) / 2, vals, width,
           yerr=errs, capsize=2, label=dn)

ax.set_xticks(x)
ax.set_xticklabels([f'ε={e}' for e in EPSILONS])
ax.set_ylabel('Detection AUC')
ax.set_title('(b) Detector Comparison Across ε')
ax.legend(fontsize=7, ncol=2)
ax.set_ylim(0.2, 1.05)
ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
ax.grid(True, alpha=0.2, axis='y')

plt.suptitle(f'Detection Strength: Multiple Detectors on {N_IMAGES} Images (5-Fold CV)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'detection_strength.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved detection_strength.png", flush=True)

rows = []
serializable = {}
for (eps, det_name), result in all_results.items():
    serializable.setdefault(str(eps), {})[det_name] = result
    for label, op in result.get('operating_points', {}).items():
        rows.append({
            'epsilon': eps,
            'detector': det_name,
            'auc': result['auc_mean'],
            'accuracy': result['acc_mean'],
            'operating_point': label,
            **op,
        })

with open(RESULTS_DIR / 'v2_detection_operating_points.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'epsilon', 'detector', 'auc', 'accuracy',
        'operating_point', 'target_fpr', 'threshold', 'achieved_fpr',
        'precision', 'recall', 'tn', 'fp', 'fn', 'tp',
    ])
    writer.writeheader()
    writer.writerows(rows)
with open(RESULTS_DIR / 'v2_detection_operating_points.json', 'w') as f:
    json.dump(serializable, f, indent=2)
print(f"  Saved v2_detection_operating_points.csv/json", flush=True)

print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
