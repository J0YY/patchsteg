#!/usr/bin/env python
"""
Experiment: Content dependence as scientific contribution.
Studies what image statistics predict steganographic capacity:
  1. Texture energy (Gabor filter responses)
  2. Edge density (Canny edges)
  3. Entropy (Shannon entropy per patch)
  4. Frequency content (high-freq energy ratio)
  5. Semantic variation (local color variance)
  6. Correlation: each predictor vs measured bit accuracy
  7. Can we predict carrier quality before encoding?
  8. Local decoder Jacobian norm estimation at carrier positions
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
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, bit_accuracy

FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
IMG_SIZE = 256

torch.manual_seed(42)
np.random.seed(42)


def compute_edge_density(img):
    """Fraction of pixels that are edges (Canny-like via Sobel)."""
    arr = np.array(img.convert('L')).astype(float)
    sx = ndimage.sobel(arr, axis=0)
    sy = ndimage.sobel(arr, axis=1)
    edges = np.hypot(sx, sy)
    threshold = np.percentile(edges, 80)
    return (edges > threshold).mean()


def compute_entropy(img):
    """Average Shannon entropy across color channels."""
    arr = np.array(img)
    total = 0
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=256, range=(0, 255), density=True)
        hist = hist[hist > 0]
        total += -np.sum(hist * np.log2(hist + 1e-10))
    return total / 3


def compute_frequency_energy(img):
    """Ratio of high-frequency energy to total energy."""
    arr = np.array(img.convert('L')).astype(float)
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = magnitude.shape
    # Low freq = center 25%
    ch, cw = h // 2, w // 2
    r = min(h, w) // 4
    total = magnitude.sum()
    low_mask = np.zeros_like(magnitude, dtype=bool)
    for i in range(h):
        for j in range(w):
            if (i - ch)**2 + (j - cw)**2 < r**2:
                low_mask[i, j] = True
    low_energy = magnitude[low_mask].sum()
    high_ratio = 1.0 - (low_energy / (total + 1e-10))
    return high_ratio


def compute_local_variance(img):
    """Average local color variance (8x8 patches)."""
    arr = np.array(img).astype(float)
    patch_size = 8
    variances = []
    for r in range(0, arr.shape[0] - patch_size, patch_size):
        for c in range(0, arr.shape[1] - patch_size, patch_size):
            patch = arr[r:r+patch_size, c:c+patch_size, :]
            variances.append(patch.var())
    return np.mean(variances)


def compute_texture_energy(img):
    """Texture energy via gradient magnitude variance."""
    arr = np.array(img.convert('L')).astype(float)
    sx = ndimage.sobel(arr, axis=0)
    sy = ndimage.sobel(arr, axis=1)
    gradient_mag = np.hypot(sx, sy)
    return gradient_mag.var()


def estimate_local_jacobian_norm(vae, latent, position, eps_jac=0.1):
    """
    Estimate local decoder Jacobian norm at a position.
    Perturb each of 4 channels by eps_jac, measure pixel change.
    Returns Frobenius-like norm.
    """
    r, c = position
    base_img = vae.decode(latent)
    base_arr = np.array(base_img).astype(float)
    total_change = 0.0
    for ch in range(4):
        lat_pert = latent.clone()
        lat_pert[0, ch, r, c] += eps_jac
        pert_img = vae.decode(lat_pert)
        pert_arr = np.array(pert_img).astype(float)
        pixel_change = np.sqrt(((pert_arr - base_arr) ** 2).sum())
        total_change += (pixel_change / eps_jac) ** 2
    return np.sqrt(total_change)


def make_diverse_images(n=50):
    """Create images with widely varying texture/entropy/frequency."""
    rng = np.random.RandomState(42)
    images = []
    labels = []

    for i in range(n):
        kind = i % 10
        arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if kind == 0:  # solid
            arr[:] = rng.randint(50, 200, 3)
            labels.append('solid')
        elif kind == 1:  # smooth gradient
            t = np.linspace(0, 1, IMG_SIZE)
            arr[:,:,0] = (t * 200).astype(np.uint8)
            arr[:,:,1] = ((1-t) * 200).astype(np.uint8)
            arr[:,:,2] = 100
            labels.append('gradient')
        elif kind == 2:  # fine noise
            arr = rng.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            labels.append('noise')
        elif kind == 3:  # large color blocks
            for r in range(0, IMG_SIZE, 64):
                for c in range(0, IMG_SIZE, 64):
                    arr[r:r+64, c:c+64] = rng.randint(0, 255, 3)
            labels.append('blocks-64')
        elif kind == 4:  # small color blocks
            for r in range(0, IMG_SIZE, 16):
                for c in range(0, IMG_SIZE, 16):
                    arr[r:r+16, c:c+16] = rng.randint(0, 255, 3)
            labels.append('blocks-16')
        elif kind == 5:  # checkerboard
            for r in range(0, IMG_SIZE, 8):
                for c in range(0, IMG_SIZE, 8):
                    if ((r//8)+(c//8))%2==0:
                        arr[r:r+8, c:c+8] = rng.randint(200, 255, 3)
                    else:
                        arr[r:r+8, c:c+8] = rng.randint(0, 50, 3)
            labels.append('checker')
        elif kind == 6:  # sine pattern
            x = np.linspace(0, 8*np.pi, IMG_SIZE)
            pattern = ((np.sin(np.outer(x, x)) + 1) / 2 * 255).astype(np.uint8)
            arr[:,:,0] = pattern; arr[:,:,1] = pattern; arr[:,:,2] = pattern
            labels.append('sine')
        elif kind == 7:  # half solid half noise
            arr[:, :IMG_SIZE//2, :] = rng.randint(100, 150, 3)
            arr[:, IMG_SIZE//2:, :] = rng.randint(0, 255, (IMG_SIZE, IMG_SIZE//2, 3), dtype=np.uint8)
            labels.append('half-half')
        elif kind == 8:  # gaussian blobs
            for _ in range(20):
                cx, cy = rng.randint(0, IMG_SIZE, 2)
                color = rng.randint(0, 255, 3)
                for r in range(max(0,cx-30), min(IMG_SIZE,cx+30)):
                    for c in range(max(0,cy-30), min(IMG_SIZE,cy+30)):
                        d = ((r-cx)**2 + (c-cy)**2)
                        if d < 900:
                            w = np.exp(-d/200)
                            arr[r,c] = np.clip(arr[r,c].astype(float) + color*w, 0, 255).astype(np.uint8)
            labels.append('blobs')
        else:  # textured
            base = rng.randint(50, 200, 3)
            arr[:] = base
            noise = rng.randint(-50, 50, (IMG_SIZE, IMG_SIZE, 3))
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
            labels.append('textured')
        images.append(Image.fromarray(arr))
    return images, labels


# ================================================================
print("=" * 70, flush=True)
print("CONTENT DEPENDENCE SCIENCE EXPERIMENT", flush=True)
print("=" * 70, flush=True)
t0 = time.time()

vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
images, img_labels = make_diverse_images(30)
print(f"Created {len(images)} diverse test images", flush=True)

# Compute image features
print("\nComputing image features...", flush=True)
features = {
    'edge_density': [],
    'entropy': [],
    'freq_energy': [],
    'local_variance': [],
    'texture_energy': [],
}

for i, img in enumerate(images):
    if i % 10 == 0:
        print(f"  Image {i}/{len(images)}...", flush=True)
    features['edge_density'].append(compute_edge_density(img))
    features['entropy'].append(compute_entropy(img))
    features['freq_energy'].append(compute_frequency_energy(img))
    features['local_variance'].append(compute_local_variance(img))
    features['texture_energy'].append(compute_texture_energy(img))

# Measure bit accuracy for each image
print("\nMeasuring steganographic capacity...", flush=True)
eps_test = 5.0
steg = PatchSteg(seed=42, epsilon=eps_test)
accuracies = []
stability_means = []

for i, img in enumerate(images):
    if i % 10 == 0:
        print(f"  Image {i}/{len(images)}...", flush=True)
    lat = vae.encode(img)
    carriers, smap = steg.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=eps_test)
    torch.manual_seed(42 + i)
    bits = torch.randint(0, 2, (20,)).tolist()
    lat_m = steg.encode_message(lat, carriers, bits)
    st = vae.decode(lat_m)
    lat_re = vae.encode(st)
    rec, _ = steg.decode_message(lat, lat_re, carriers)
    acc = bit_accuracy(bits, rec)
    accuracies.append(acc)
    stability_means.append(smap.mean().item())

accuracies = np.array(accuracies)

# Correlation analysis
print("\n--- CORRELATION ANALYSIS ---", flush=True)
correlations = {}
for feat_name, feat_vals in features.items():
    feat_arr = np.array(feat_vals)
    r_pearson, p_pearson = pearsonr(feat_arr, accuracies)
    r_spearman, p_spearman = spearmanr(feat_arr, accuracies)
    correlations[feat_name] = {
        'pearson_r': r_pearson, 'pearson_p': p_pearson,
        'spearman_r': r_spearman, 'spearman_p': p_spearman,
    }
    print(f"  {feat_name:20s}: Pearson r={r_pearson:.3f} (p={p_pearson:.3f}), "
          f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.3f})", flush=True)

# Also correlate stability map mean with accuracy
r_stab, p_stab = pearsonr(stability_means, accuracies)
print(f"  {'stability_mean':20s}: Pearson r={r_stab:.3f} (p={p_stab:.3f})", flush=True)

# ================================================================
# Jacobian analysis at carrier positions
# ================================================================
print("\n--- LOCAL JACOBIAN ANALYSIS ---", flush=True)
print("Estimating decoder Jacobian norms at carrier vs non-carrier positions...", flush=True)

jacobian_results = {'carrier_norms': [], 'noncarrier_norms': []}
n_jac_images = 10  # subset for speed

for i in range(n_jac_images):
    img = images[i]
    lat = vae.encode(img)
    carriers, smap = steg.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=eps_test)
    carrier_set = set(carriers)

    # Sample 10 carrier and 10 non-carrier positions
    for r, c in carriers[:10]:
        jnorm = estimate_local_jacobian_norm(vae, lat, (r, c))
        jacobian_results['carrier_norms'].append(jnorm)

    non_carriers = [(r, c) for r in range(0, 32, 4) for c in range(0, 32, 4)
                    if (r, c) not in carrier_set][:10]
    for r, c in non_carriers:
        jnorm = estimate_local_jacobian_norm(vae, lat, (r, c))
        jacobian_results['noncarrier_norms'].append(jnorm)

    if i % 3 == 0:
        print(f"  Image {i}: carrier J-norm={np.mean(jacobian_results['carrier_norms'][-10:]):.1f}, "
              f"non-carrier={np.mean(jacobian_results['noncarrier_norms'][-10:]):.1f}", flush=True)

carrier_j = np.array(jacobian_results['carrier_norms'])
noncarrier_j = np.array(jacobian_results['noncarrier_norms'])
print(f"\n  Carrier Jacobian norm: {carrier_j.mean():.1f} ± {carrier_j.std():.1f}", flush=True)
print(f"  Non-carrier Jacobian norm: {noncarrier_j.mean():.1f} ± {noncarrier_j.std():.1f}", flush=True)
from scipy.stats import mannwhitneyu
u_stat, u_p = mannwhitneyu(carrier_j, noncarrier_j, alternative='two-sided')
print(f"  Mann-Whitney U test: U={u_stat:.0f}, p={u_p:.4f}", flush=True)

# ================================================================
# Figure: content dependence analysis
# ================================================================
print(f"\nGenerating figures...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Scatter plots for each feature vs accuracy
feat_items = list(features.items())
for idx, (feat_name, feat_vals) in enumerate(feat_items):
    ax = axes[idx // 3, idx % 3]
    feat_arr = np.array(feat_vals)
    c = correlations[feat_name]
    ax.scatter(feat_arr, accuracies, alpha=0.6, s=30, c='steelblue', edgecolors='navy', linewidth=0.3)

    # Fit line
    z = np.polyfit(feat_arr, accuracies, 1)
    p_line = np.poly1d(z)
    x_sorted = np.sort(feat_arr)
    ax.plot(x_sorted, p_line(x_sorted), 'r--', alpha=0.7, linewidth=2)

    ax.set_xlabel(feat_name.replace('_', ' ').title())
    ax.set_ylabel('Bit Accuracy (%)')
    ax.set_title(f"Pearson r={c['pearson_r']:.3f} (p={c['pearson_p']:.3f})")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(30, 105)

# Jacobian plot
ax = axes[1, 2]
ax.boxplot([carrier_j, noncarrier_j], labels=['Carrier', 'Non-carrier'])
ax.set_ylabel('Local Jacobian Norm')
ax.set_title(f'Decoder Jacobian at Carriers\n(U-test p={u_p:.4f})')
ax.grid(True, alpha=0.2, axis='y')

plt.suptitle(f'Content Dependence: What Image Statistics Predict Channel Capacity?\n'
             f'(ε={eps_test}, 20 carriers, {len(images)} images)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'content_science.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved content_science.png", flush=True)

# ================================================================
# Additional figure: accuracy by image type
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5))
unique_labels = list(dict.fromkeys(img_labels))
label_accs = {l: [] for l in unique_labels}
for l, a in zip(img_labels, accuracies):
    label_accs[l].append(a)

means = [np.mean(label_accs[l]) for l in unique_labels]
stds = [np.std(label_accs[l]) for l in unique_labels]
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
bars = ax.bar(range(len(unique_labels)), means, yerr=stds, capsize=4,
              color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(unique_labels)))
ax.set_xticklabels(unique_labels, rotation=30, ha='right')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title(f'Steganographic Capacity by Image Type (ε={eps_test})')
ax.set_ylim(0, 105)
ax.axhline(50, color='gray', ls=':', alpha=0.5)
ax.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'capacity_by_type.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved capacity_by_type.png", flush=True)

print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
