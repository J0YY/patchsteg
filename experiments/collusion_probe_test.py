#!/usr/bin/env python
"""
Collusion Pattern Probe vs all SOTA attacks.

Tests whether CollusionPatternProbe (C. Wang) can detect same-key stego
across all four attack methods, including CDF-PatchSteg which evades all
marginal-statistics detectors.

Detection mechanism: same seed -> same carrier positions -> shared latent
structure across images -> elevated PC1 variance in SVD of latent matrix.
This is independent of *how* values are modified, only *where*.

Attacks tested:
  - PatchSteg (±ε=5)
  - PCA-PatchSteg (ε=5)
  - PSyDUCK-inspired (ε=5)
  - CDF-PatchSteg

Generates: paper/figures/collusion_probe_results.png
"""
import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent / 'defense'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.cdf_steganography import CDFPatchSteg
from core.pca_directions import PCADirections, PCAPatchSteg
from core.psyduck_steganography import PSyDUCKSteg

# Import Cath's CollusionPatternProbe directly
from probe import CollusionPatternProbe

IMG_SIZE = 128
N_CARRIERS = 20
N_IMAGES = 8       # need enough images for SVD to be meaningful
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
t0 = time.time()


def get_images(n=N_IMAGES):
    from torchvision.datasets import CIFAR10
    print("  Loading CIFAR-10...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=True, download=True)
    imgs = []
    for img, _ in ds:
        imgs.append(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        if len(imgs) >= n:
            break
    return imgs


def make_stego_set(vae, images, attack, seed, pca_dir=None):
    """Encode all images with the given attack and seed. Returns stego images."""
    stegos = []
    for i, img in enumerate(images):
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)
        bits = list(np.random.randint(0, 2, N_CARRIERS))
        latent = vae.encode(img)

        if attack == 'PatchSteg':
            s = PatchSteg(seed=seed, epsilon=5.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = s.encode_message(latent, carriers, bits)
        elif attack == 'PCA':
            s = PCAPatchSteg(pca_dir, seed=seed, epsilon=5.0, component=0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = s.encode_message(latent, carriers, bits)
        elif attack == 'PSyDUCK':
            s = PSyDUCKSteg(seed=seed, epsilon=5.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = s.encode_message(latent, carriers, bits)
        elif attack == 'CDF':
            s = CDFPatchSteg(seed=seed, sigma=1.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = s.encode_message(latent, carriers, bits)

        stegos.append(vae.decode(lat_m))
    return stegos


print("Loading VAE...", flush=True)
vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
print(f"  VAE loaded ({time.time()-t0:.0f}s)", flush=True)

images = get_images(N_IMAGES)

pca_dir = PCADirections(n_components=4)
pca_dir.fit_global(vae, images)
print("  PCA fitted", flush=True)

probe = CollusionPatternProbe()

ATTACKS = ['PatchSteg', 'PCA', 'PSyDUCK', 'CDF']
ATTACK_LABELS = ['PatchSteg\n(±ε)', 'PCA-\nPatchSteg', 'PSyDUCK-\ninspired', 'CDF-\nPatchSteg']
SEED_A = 42
SEED_B = 99   # different key for control

print("\n" + "="*60, flush=True)
print("COLLUSION PATTERN PROBE vs ALL SOTA ATTACKS", flush=True)
print("="*60, flush=True)

# Baseline: clean images
clean_result = probe.score_set(vae, images)
print(f"\nClean images:  PC1 var = {clean_result['variance_explained_pc1']:.4f}  (baseline)", flush=True)

results = {}

for attack in ATTACKS:
    print(f"\n  {attack}:", flush=True)

    # Same-key: all images encoded with SEED_A
    same_key_stegos = make_stego_set(vae, images, attack, seed=SEED_A, pca_dir=pca_dir)
    same_result = probe.score_set(vae, same_key_stegos)

    # Different-key: half with SEED_A, half with SEED_B -> no shared carriers
    half = N_IMAGES // 2
    stegos_a = make_stego_set(vae, images[:half], attack, seed=SEED_A, pca_dir=pca_dir)
    stegos_b = make_stego_set(vae, images[half:], attack, seed=SEED_B, pca_dir=pca_dir)
    diff_stegos = stegos_a + stegos_b
    diff_result = probe.score_set(vae, diff_stegos)

    results[attack] = {
        'same_key': same_result['variance_explained_pc1'],
        'diff_key': diff_result['variance_explained_pc1'],
        'clean':    clean_result['variance_explained_pc1'],
        'sv_same':  same_result['singular_values'],
        'sv_diff':  diff_result['singular_values'],
    }

    sep = results[attack]['same_key'] - results[attack]['diff_key']
    print(f"    Same-key PC1 var = {results[attack]['same_key']:.4f}  (signal: collusion)", flush=True)
    print(f"    Diff-key PC1 var = {results[attack]['diff_key']:.4f}  (control)", flush=True)
    print(f"    Separation       = {sep:+.4f}  {'DETECTED' if sep > 0.02 else 'WEAK'}", flush=True)

# ================================================================
# FIGURE
# ================================================================
print("\nGenerating figure...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: PC1 variance bar chart ---
ax = axes[0]
x = np.arange(len(ATTACKS))
w = 0.25

same_vals = [results[a]['same_key'] for a in ATTACKS]
diff_vals = [results[a]['diff_key'] for a in ATTACKS]
clean_val = clean_result['variance_explained_pc1']

b1 = ax.bar(x - w, same_vals, w, label='Same-key stego', color='#e74c3c', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x,     diff_vals, w, label='Diff-key stego', color='#3498db', edgecolor='black', linewidth=0.5)
ax.axhline(clean_val, color='#2ecc71', ls='--', linewidth=2, label=f'Clean (baseline={clean_val:.3f})')

ax.set_xticks(x - w/2)
ax.set_xticklabels(ATTACK_LABELS, fontsize=9)
ax.set_ylabel('PC1 Variance Explained')
ax.set_title('Collusion Pattern Probe\nPC1 Variance: Same-key vs Diff-key vs Clean', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

for bar, val in zip(b1, same_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', fontsize=8, fontweight='bold', color='#c0392b')
for bar, val in zip(b2, diff_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', fontsize=8, color='#2980b9')

# --- Right: Separation (same - diff) ---
ax = axes[1]
seps = [results[a]['same_key'] - results[a]['diff_key'] for a in ATTACKS]
colors = ['#e74c3c' if s > 0.02 else '#95a5a6' for s in seps]
bars = ax.bar(range(len(ATTACKS)), seps, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(0.02, color='orange', ls='--', alpha=0.8, label='Detection threshold (0.02)')
ax.axhline(0, color='black', ls='-', linewidth=0.5)
ax.set_xticks(range(len(ATTACKS)))
ax.set_xticklabels(ATTACK_LABELS, fontsize=9)
ax.set_ylabel('PC1 Variance Separation (same − diff)')
ax.set_title('Collusion Signal Strength\n(red = detectable, grey = below threshold)', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, seps):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.001 if val >= 0 else -0.003),
            f'{val:+.3f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle("Collusion Pattern Probe: Does Shared-Key Stego Leave a Cross-Image Fingerprint?",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'collusion_probe_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved collusion_probe_results.png", flush=True)

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Clean baseline PC1 var: {clean_result['variance_explained_pc1']:.4f}", flush=True)
print(f"\n  {'Attack':<12} {'Same-key':>10} {'Diff-key':>10} {'Sep':>8}  {'Verdict'}", flush=True)
print(f"  {'-'*55}", flush=True)
for attack in ATTACKS:
    r = results[attack]
    sep = r['same_key'] - r['diff_key']
    verdict = "DETECTED" if sep > 0.02 else "not detected"
    print(f"  {attack:<12} {r['same_key']:>10.4f} {r['diff_key']:>10.4f} {sep:>+8.4f}  {verdict}", flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
