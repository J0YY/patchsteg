#!/usr/bin/env python
"""
Phase 2: PCA-guided perturbation directions test.

Tests:
1. Fit global PCA on VAE latents, analyze variance explained
2. Compare PCA vs random directions on accuracy
3. Compare PCA vs random on detectability

Generates: pca_components.png, pca_accuracy_comparison.png, pca_detectability.png
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

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.pca_directions import PCADirections, PCAPatchSteg
from core.metrics import compute_psnr, bit_accuracy

IMG_SIZE = 128  # 128 runs ~4x faster than 256 on CPU
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def get_natural_images(n=4):
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


def extract_latent_features(vae, img):
    """Extract statistical features from VAE latent for detection."""
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


if __name__ == '__main__':
    t_total = time.time()
    print("Loading VAE...", flush=True)
    vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
    print(f"VAE loaded ({time.time()-t_total:.0f}s)", flush=True)

    nat_images, nat_names = get_natural_images(4)

    # ================================================================
    # 1. FIT PCA ON LATENT SPACE
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 1. PCA ON VAE LATENT SPACE", flush=True)
    print("#"*60, flush=True)

    pca_dir = PCADirections(n_components=4)
    pca_dir.fit_global(vae, nat_images)

    ev_ratio = pca_dir.get_explained_variance_ratio()
    sv = pca_dir.get_singular_values()
    print(f"  Explained variance ratio: {ev_ratio}", flush=True)
    print(f"  Cumulative: {np.cumsum(ev_ratio)}", flush=True)
    print(f"  Singular values: {sv}", flush=True)
    print(f"  Top 3 explain {np.sum(ev_ratio[:3])*100:.1f}% of variance", flush=True)

    # Print PCA directions
    for i in range(4):
        d = pca_dir.get_direction(i)
        print(f"  PC{i}: {d.numpy()}", flush=True)

    # Also print the random direction for comparison
    random_steg = PatchSteg(seed=42, epsilon=5.0)
    print(f"  Random direction: {random_steg.direction.numpy()}", flush=True)

    # Figure: PCA components
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Explained variance
    ax = axes[0]
    ax.bar(range(4), ev_ratio * 100, color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12'],
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'PC{i}' for i in range(4)])
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('(a) Variance Explained per Component')
    ax.grid(True, alpha=0.2, axis='y')
    # Cumulative line
    ax2 = ax.twinx()
    ax2.plot(range(4), np.cumsum(ev_ratio) * 100, 'ko-', linewidth=2)
    ax2.set_ylabel('Cumulative (%)')
    ax2.set_ylim(0, 105)

    # (b) Component vectors
    ax = axes[1]
    components = pca_dir.global_pca.components_
    im = ax.imshow(components, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f'PC{i}' for i in range(4)])
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'Ch{i}' for i in range(4)])
    ax.set_title('(b) PCA Component Weights')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (c) Singular values
    ax = axes[2]
    ax.bar(range(4), sv, color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12'],
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'PC{i}' for i in range(4)])
    ax.set_ylabel('Singular Value')
    ax.set_title('(c) Natural Variation Scale')
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('PCA Analysis of VAE Latent Space', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pca_components.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved pca_components.png", flush=True)

    # ================================================================
    # 2. ACCURACY COMPARISON: PCA vs RANDOM DIRECTION
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 2. ACCURACY: PCA vs RANDOM", flush=True)
    print("#"*60, flush=True)

    n_carriers = 20
    results_by_method = {}

    for method_name, component in [('PCA-PC0', 0), ('PCA-PC1', 1), ('PCA-PC2', 2), ('Random', None)]:
        accs, psnrs = [], []
        for img_idx, (img, name) in enumerate(zip(nat_images, nat_names)):
            torch.manual_seed(42 + img_idx)
            bits = torch.randint(0, 2, (n_carriers,)).tolist()
            latent = vae.encode(img)

            if component is not None:
                steg = PCAPatchSteg(pca_dir, seed=42, epsilon=5.0, component=component)
            else:
                steg = PatchSteg(seed=42, epsilon=5.0)

            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=n_carriers, test_eps=5.0)
            lat_m = steg.encode_message(latent, carriers, bits)
            stego = vae.decode(lat_m)
            lat_re = vae.encode(stego)
            rec, _ = steg.decode_message(latent, lat_re, carriers)
            accs.append(bit_accuracy(bits, rec))
            psnrs.append(compute_psnr(img, stego))

        results_by_method[method_name] = {
            'acc': np.mean(accs), 'acc_std': np.std(accs),
            'psnr': np.mean(psnrs), 'psnr_std': np.std(psnrs)
        }
        print(f"  {method_name}: acc={np.mean(accs):.1f}%+-{np.std(accs):.1f} "
              f"psnr={np.mean(psnrs):.1f}+-{np.std(psnrs):.1f}dB", flush=True)

    # Figure: accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = list(results_by_method.keys())
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#95a5a6']

    ax = axes[0]
    bars = ax.bar(methods, [results_by_method[m]['acc'] for m in methods],
                  yerr=[results_by_method[m]['acc_std'] for m in methods],
                  capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Bit Accuracy (%)')
    ax.set_title('Bit Accuracy: PCA vs Random Direction')
    ax.set_ylim(40, 105)
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, [results_by_method[m]['acc'] for m in methods]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax = axes[1]
    bars = ax.bar(methods, [results_by_method[m]['psnr'] for m in methods],
                  yerr=[results_by_method[m]['psnr_std'] for m in methods],
                  capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Image Quality: PCA vs Random Direction')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, [results_by_method[m]['psnr'] for m in methods]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('PCA-Guided vs Random Perturbation Directions (eps=5.0, K=20)', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pca_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved pca_accuracy_comparison.png", flush=True)

    # ================================================================
    # 3. DETECTABILITY: PCA vs RANDOM
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 3. DETECTABILITY: PCA vs RANDOM", flush=True)
    print("#"*60, flush=True)

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    detect_results = {}

    for method_name, component in [('PCA-PC0', 0), ('PCA-PC1', 1), ('Random', None)]:
        X_all, y_all = [], []

        for i, img in enumerate(nat_images):
            X_all.append(extract_latent_features(vae, img))
            y_all.append(0)

            torch.manual_seed(42 + i)
            bits = torch.randint(0, 2, (20,)).tolist()
            latent = vae.encode(img)

            if component is not None:
                steg = PCAPatchSteg(pca_dir, seed=42, epsilon=5.0, component=component)
            else:
                steg = PatchSteg(seed=42, epsilon=5.0)

            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=5.0)
            lat_m = steg.encode_message(latent, carriers, bits)
            stego = vae.decode(lat_m)
            X_all.append(extract_latent_features(vae, stego))
            y_all.append(1)

        X = np.array(X_all)
        y = np.array(y_all)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
        auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        detect_results[method_name] = {'auc': auc.mean(), 'auc_std': auc.std()}
        print(f"  {method_name}: AUC={auc.mean():.3f}+-{auc.std():.3f}", flush=True)

    # Figure: detectability
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = list(detect_results.keys())
    aucs = [detect_results[m]['auc'] for m in methods]
    auc_errs = [detect_results[m]['auc_std'] for m in methods]
    colors = ['#e74c3c', '#2ecc71', '#95a5a6']
    bars = ax.bar(methods, aucs, yerr=auc_errs, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Chance')
    ax.axhline(0.7, color='orange', ls='--', alpha=0.5, label='Detection Threshold')
    ax.set_ylabel('Detection AUC')
    ax.set_title('Detectability: PCA vs Random Directions (eps=5.0)')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pca_detectability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved pca_detectability.png", flush=True)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'#'*60}", flush=True)
    print("# SUMMARY", flush=True)
    print(f"{'#'*60}", flush=True)
    print(f"  Top 3 PCA components explain {np.sum(ev_ratio[:3])*100:.1f}% of variance", flush=True)
    for m in results_by_method:
        print(f"  {m}: acc={results_by_method[m]['acc']:.1f}% psnr={results_by_method[m]['psnr']:.1f}dB", flush=True)
    for m in detect_results:
        print(f"  {m} detection AUC: {detect_results[m]['auc']:.3f}", flush=True)
    print(f"\n  Total time: {time.time()-t_total:.0f}s", flush=True)
