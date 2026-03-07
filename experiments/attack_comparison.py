#!/usr/bin/env python
"""
Attack comparison: does the naive logistic regression detector catch each attack?

Tests all four attack methods against a logistic regression trained only on
latent-space marginal statistics (same protocol as the original detectability
experiments). Shows which attacks evade the naive detector.

Methods tested:
  1. PatchSteg (±ε)        -- original, should be caught at eps=5.0
  2. PCA-PatchSteg         -- same ±ε but along PCA direction
  3. PSyDUCK-inspired      -- per-bit unique directions
  4. CDF-PatchSteg         -- distribution-preserving (should evade)

Generates: paper/figures/attack_comparison_auc.png
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from core.vae import StegoVAE
from core.adaptive_steganography import AdaptivePatchSteg
from core.steganography import PatchSteg
from core.cdf_steganography import CDFPatchSteg
from core.pca_directions import PCADirections, PCAPatchSteg
from core.psyduck_steganography import PSyDUCKSteg
from core.metrics import bit_accuracy, compute_psnr

IMG_SIZE = 128
N_CARRIERS = 20
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
t0 = time.time()


def get_images(n=6):
    from torchvision.datasets import CIFAR10
    print("  Loading CIFAR-10...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=True, download=True)
    imgs, classes_seen = [], set()
    for img, label in ds:
        if label not in classes_seen and len(imgs) < n:
            classes_seen.add(label)
            imgs.append(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    return imgs


def extract_features(vae, image):
    """
    Naive logistic regression feature set: per-channel marginal statistics
    after one VAE round-trip. Identical to original detectability experiment.
    20 features: 5 stats × 4 channels (mean, std, median, skew, kurtosis).
    """
    lat = vae.encode(image)
    recon = vae.decode(lat)
    lat_rt = vae.encode(recon)
    feats = []
    for ch in range(4):
        x = lat_rt[0, ch].detach().cpu().numpy().flatten()
        mu, sig = x.mean(), x.std() + 1e-8
        feats.extend([
            float(mu),
            float(sig),
            float(np.median(x)),
            float(((x - mu) ** 3).mean() / sig ** 3),   # skew
            float(((x - mu) ** 4).mean() / sig ** 4),   # kurtosis
        ])
    return np.array(feats)


print("Loading VAE...", flush=True)
vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
print(f"  VAE loaded ({time.time()-t0:.0f}s)", flush=True)

images = get_images(6)

# Fit PCA once on all images (needed for PCA-PatchSteg)
pca_dir = PCADirections(n_components=4)
pca_dir.fit_global(vae, images)
print(f"  PCA fitted", flush=True)

# ================================================================
# EVALUATE EACH ATTACK
# ================================================================
print("\n" + "#"*60, flush=True)
print("# DETECTION AUC: NAIVE LR vs EACH ATTACK", flush=True)
print("#"*60, flush=True)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
results = {}

attack_configs = [
    ("PatchSteg (±ε, ε=2)",   'patchsteg', 2.0),
    ("PatchSteg (±ε, ε=5)",   'patchsteg', 5.0),
    ("AdaptivePatchSteg (ε=5)", 'adaptive', 5.0),
    ("PCA-PatchSteg (ε=5)",   'pca',       5.0),
    ("PSyDUCK-inspired (ε=5)","psyduck",   5.0),
    ("CDF-PatchSteg",          'cdf',       None),
]

for label, method, eps in attack_configs:
    X_all, y_all, accs, psnrs = [], [], [], []

    for i, img in enumerate(images):
        np.random.seed(42 + i)
        torch.manual_seed(42 + i)
        bits = list(np.random.randint(0, 2, N_CARRIERS))
        latent = vae.encode(img)

        if method == 'patchsteg':
            steg = PatchSteg(seed=42, epsilon=eps)
            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = steg.encode_message(latent, carriers, bits)
            stego = vae.decode(lat_m)
            lat_re = vae.encode(stego)
            rec, _ = steg.decode_message(latent, lat_re, carriers)

        elif method == 'pca':
            steg = PCAPatchSteg(pca_dir, seed=42, epsilon=eps, component=0)
            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = steg.encode_message(latent, carriers, bits)
            stego = vae.decode(lat_m)
            lat_re = vae.encode(stego)
            rec, _ = steg.decode_message(latent, lat_re, carriers)

        elif method == 'adaptive':
            steg = AdaptivePatchSteg(seed=42, epsilon=eps, bits_per_symbol=1)
            pairs, _, gain_map, latent_clean = steg.select_carrier_pairs(
                vae,
                img,
                n_pairs=N_CARRIERS,
                test_eps=eps,
            )
            lat_m = steg.encode_message(
                latent_clean,
                pairs,
                bits,
                gain_map=gain_map,
                latent_reference=latent_clean,
            )
            stego = vae.decode(lat_m)
            lat_re = vae.encode(stego)
            rec, _ = steg.decode_message(latent_clean, lat_re, pairs, gain_map=gain_map)

        elif method == 'psyduck':
            steg = PSyDUCKSteg(seed=42, epsilon=eps)
            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = steg.encode_message(latent, carriers, bits)
            stego = vae.decode(lat_m)
            lat_re = vae.encode(stego)
            rec, _ = steg.decode_message(latent, lat_re, carriers)

        elif method == 'cdf':
            steg = CDFPatchSteg(seed=42, sigma=1.0)
            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = steg.encode_message(latent, carriers, bits)
            stego = vae.decode(lat_m)
            rec, _ = steg.decode_message(vae, stego, carriers)

        accs.append(bit_accuracy(bits, rec))
        psnrs.append(compute_psnr(img, stego))

        # Features: clean=0, stego=1
        X_all.append(extract_features(vae, img))
        y_all.append(0)
        X_all.append(extract_features(vae, stego))
        y_all.append(1)

    X = np.array(X_all)
    y = np.array(y_all)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')

    results[label] = {
        'auc': auc.mean(), 'auc_std': auc.std(),
        'acc': np.mean(accs), 'acc_std': np.std(accs),
        'psnr': np.mean(psnrs),
    }
    print(f"  {label}:", flush=True)
    print(f"    Bit acc: {np.mean(accs):.1f}% ± {np.std(accs):.1f}%  PSNR: {np.mean(psnrs):.1f}dB", flush=True)
    print(f"    Detection AUC: {auc.mean():.3f} ± {auc.std():.3f}", flush=True)

# ================================================================
# FIGURE
# ================================================================
print("\nGenerating figure...", flush=True)

labels = list(results.keys())
aucs = [results[l]['auc'] for l in labels]
auc_stds = [results[l]['auc_std'] for l in labels]
accs = [results[l]['acc'] for l in labels]
psnrs = [results[l]['psnr'] for l in labels]

# Color: red if detectable (AUC > 0.6), green if evades
colors = ['#e74c3c' if a > 0.6 else '#2ecc71' for a in aucs]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) Detection AUC
ax = axes[0]
bars = ax.bar(range(len(labels)), aucs, yerr=auc_stds, capsize=5,
              color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(0.5, color='gray', ls=':', alpha=0.7, label='Chance (0.5)')
ax.axhline(0.6, color='orange', ls='--', alpha=0.7, label='Detection threshold (0.6)')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Detection AUC')
ax.set_title('(a) Logistic Regression Detection\n(naive marginal-statistics detector)')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

# (b) Bit accuracy
ax = axes[1]
bars = ax.bar(range(len(labels)), accs, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title('(b) Bit Accuracy\n(higher = better attack)')
ax.set_ylim(40, 105)
ax.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')

# (c) PSNR
ax = axes[2]
bars = ax.bar(range(len(labels)), psnrs, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(30, color='orange', ls='--', alpha=0.7, label='30 dB threshold')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('PSNR (dB)')
ax.set_title('(c) Image Quality\n(higher = less visible)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, psnrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('All Attacks vs Naive Logistic Regression Detector\n'
             '(Red = detected, Green = evades)', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'attack_comparison_auc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved attack_comparison_auc.png", flush=True)

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'#'*60}", flush=True)
print("# SUMMARY", flush=True)
print(f"{'#'*60}", flush=True)
print(f"{'Attack':<30} {'AUC':>6} {'Acc':>8} {'PSNR':>8}  {'Verdict'}", flush=True)
print("-" * 70, flush=True)
for label in labels:
    r = results[label]
    verdict = "DETECTED" if r['auc'] > 0.6 else "evades"
    print(f"  {label:<28} {r['auc']:>6.3f} {r['acc']:>7.1f}% {r['psnr']:>7.1f}dB  {verdict}", flush=True)
print(f"\n  Total time: {time.time()-t0:.0f}s", flush=True)
