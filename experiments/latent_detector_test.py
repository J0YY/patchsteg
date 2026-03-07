#!/usr/bin/env python
"""
Phase 3: Latent-space steganalysis detector.

Tests:
1. Train detector on PatchSteg eps=2 and eps=5
2. Train detector on CDF-PatchSteg
3. Cross-method evaluation matrix
4. Feature importance analysis

Generates: detector_roc_curves.png, detector_cross_method.png, detector_feature_importance.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.cdf_steganography import CDFPatchSteg
from core.detector import LatentStegDetector
from core.metrics import compute_psnr, bit_accuracy

IMG_SIZE = 256
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def get_natural_images(n=8):
    """Get natural photos from CIFAR-10."""
    from torchvision.datasets import CIFAR10
    print("  Downloading CIFAR-10 (first time only)...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=True, download=True)
    picked = []
    classes_seen = set()
    for img, label in ds:
        if label not in classes_seen and len(picked) < n:
            classes_seen.add(label)
            picked.append((img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                          ds.classes[label]))
    return [p[0] for p in picked], [p[1] for p in picked]


def generate_stego_images(vae, images, method_name):
    """Generate stego images using the specified method."""
    stego_images = []
    for i, img in enumerate(images):
        torch.manual_seed(42 + i)
        bits = torch.randint(0, 2, (20,)).tolist()
        latent = vae.encode(img)

        if method_name == 'PatchSteg eps=2':
            s = PatchSteg(seed=42, epsilon=2.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=2.0)
            lat_m = s.encode_message(latent, carriers, bits)
        elif method_name == 'PatchSteg eps=5':
            s = PatchSteg(seed=42, epsilon=5.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=5.0)
            lat_m = s.encode_message(latent, carriers, bits)
        elif method_name == 'CDF-PatchSteg':
            s = CDFPatchSteg(seed=42, sigma=1.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20)
            lat_m = s.encode_message(latent, carriers, bits)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        stego_images.append(vae.decode(lat_m))
    return stego_images


if __name__ == '__main__':
    t_total = time.time()
    print("Loading VAE...", flush=True)
    vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
    print(f"VAE loaded ({time.time()-t_total:.0f}s)", flush=True)

    nat_images, nat_names = get_natural_images(8)

    # ================================================================
    # 1. EXTRACT FEATURES FOR ALL METHODS
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 1. EXTRACTING FEATURES", flush=True)
    print("#"*60, flush=True)

    detector = LatentStegDetector()
    methods = ['PatchSteg eps=2', 'PatchSteg eps=5', 'CDF-PatchSteg']

    # Clean features
    print("  Extracting clean features...", flush=True)
    X_clean = np.array([detector.extract_features(vae, img) for img in nat_images])
    print(f"  Feature vector dimension: {X_clean.shape[1]}", flush=True)

    # Stego features per method
    X_stego = {}
    for method in methods:
        print(f"  Generating {method} stego images...", flush=True)
        stego_imgs = generate_stego_images(vae, nat_images, method)
        X_stego[method] = np.array([detector.extract_features(vae, img) for img in stego_imgs])

    # ================================================================
    # 2. WITHIN-METHOD DETECTION (CV)
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 2. WITHIN-METHOD DETECTION", flush=True)
    print("#"*60, flush=True)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    within_results = {}

    for method in methods:
        X = np.vstack([X_clean, X_stego[method]])
        y = np.concatenate([np.zeros(len(nat_images)), np.ones(len(nat_images))])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
        auc_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        within_results[method] = {
            'auc': auc_scores.mean(), 'auc_std': auc_scores.std(),
            'acc': acc_scores.mean(), 'acc_std': acc_scores.std()
        }
        print(f"  {method}: AUC={auc_scores.mean():.3f}+-{auc_scores.std():.3f} "
              f"Acc={acc_scores.mean():.3f}+-{acc_scores.std():.3f}", flush=True)

    # ================================================================
    # 3. ROC CURVES
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 3. ROC CURVES", flush=True)
    print("#"*60, flush=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {'PatchSteg eps=2': '#3498db', 'PatchSteg eps=5': '#e74c3c', 'CDF-PatchSteg': '#2ecc71'}

    for method in methods:
        X = np.vstack([X_clean, X_stego[method]])
        y = np.concatenate([np.zeros(len(nat_images)), np.ones(len(nat_images))])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
        clf.fit(X, y)
        y_score = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[method], linewidth=2,
                label=f'{method} (AUC={roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Latent-Space Steganalysis Detector')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'detector_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved detector_roc_curves.png", flush=True)

    # ================================================================
    # 4. CROSS-METHOD EVALUATION MATRIX
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 4. CROSS-METHOD DETECTION MATRIX", flush=True)
    print("#"*60, flush=True)

    # Train on one method, test on another
    cross_matrix = np.zeros((len(methods), len(methods)))

    for i, train_method in enumerate(methods):
        # Train detector on this method
        X_train = np.vstack([X_clean, X_stego[train_method]])
        y_train = np.concatenate([np.zeros(len(nat_images)), np.ones(len(nat_images))])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
        clf.fit(X_train, y_train)

        for j, test_method in enumerate(methods):
            X_test = np.vstack([X_clean, X_stego[test_method]])
            y_test = np.concatenate([np.zeros(len(nat_images)), np.ones(len(nat_images))])
            y_score = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            cross_matrix[i, j] = auc(fpr, tpr)

    print("  Cross-method AUC matrix (rows=train, cols=test):", flush=True)
    print(f"  {'':>18s} ", end='', flush=True)
    for m in methods:
        print(f"{m:>18s} ", end='', flush=True)
    print(flush=True)
    for i, train_m in enumerate(methods):
        print(f"  {train_m:>18s} ", end='', flush=True)
        for j in range(len(methods)):
            print(f"{cross_matrix[i,j]:>18.3f} ", end='', flush=True)
        print(flush=True)

    # Figure: cross-method heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cross_matrix, cmap='RdYlGn_r', vmin=0.3, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Test Method')
    ax.set_ylabel('Train Method')
    ax.set_title('Cross-Method Detection AUC Matrix')

    for i in range(len(methods)):
        for j in range(len(methods)):
            color = 'white' if cross_matrix[i, j] > 0.7 else 'black'
            ax.text(j, i, f'{cross_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, label='AUC', shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'detector_cross_method.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved detector_cross_method.png", flush=True)

    # ================================================================
    # 5. FEATURE IMPORTANCE
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 5. FEATURE IMPORTANCE", flush=True)
    print("#"*60, flush=True)

    # Train on PatchSteg eps=5 (strongest signal)
    X_train = np.vstack([X_clean, X_stego['PatchSteg eps=5']])
    y_train = np.concatenate([np.zeros(len(nat_images)), np.ones(len(nat_images))])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_scaled, y_train)

    feature_names = detector.get_feature_names()
    importances = np.abs(lr.coef_[0])
    top_k = min(15, len(importances))
    top_idx = np.argsort(importances)[-top_k:]

    print(f"  Top {top_k} features (by |coefficient|):", flush=True)
    for idx in reversed(top_idx):
        print(f"    {feature_names[idx]:>25s}: {importances[idx]:.3f}", flush=True)

    # Figure: feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_k), importances[top_idx], color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=8)
    ax.set_xlabel('|Coefficient| (Logistic Regression)')
    ax.set_title('Feature Importance for Steganalysis (Trained on PatchSteg eps=5)')
    ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'detector_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved detector_feature_importance.png", flush=True)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'#'*60}", flush=True)
    print("# SUMMARY", flush=True)
    print(f"{'#'*60}", flush=True)
    print("  Within-method detection:", flush=True)
    for m in methods:
        print(f"    {m}: AUC={within_results[m]['auc']:.3f}", flush=True)
    print(f"\n  Cross-method (trained on PatchSteg eps=5):", flush=True)
    for j, m in enumerate(methods):
        idx_train = methods.index('PatchSteg eps=5')
        print(f"    -> {m}: AUC={cross_matrix[idx_train, j]:.3f}", flush=True)
    print(f"\n  Key narrative:", flush=True)
    ps5_auc = within_results['PatchSteg eps=5']['auc']
    cdf_auc = within_results['CDF-PatchSteg']['auc']
    print(f"    PatchSteg eps=5 detectable: AUC={ps5_auc:.3f} (want > 0.9)", flush=True)
    print(f"    CDF-PatchSteg undetectable: AUC={cdf_auc:.3f} (want ~0.5)", flush=True)
    cross_ps5_to_cdf = cross_matrix[methods.index('PatchSteg eps=5'), methods.index('CDF-PatchSteg')]
    print(f"    Cross-method (PS5->CDF): AUC={cross_ps5_to_cdf:.3f} (want < 0.6)", flush=True)
    print(f"\n  Total time: {time.time()-t_total:.0f}s", flush=True)
