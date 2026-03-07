#!/usr/bin/env python
"""
Cath's defenses vs. all SOTA attacks.

Runs all sanitizers and anomaly detectors from experiments/defense/ against
all four attack methods at 128px (fast). Produces a combined heatmap showing
which defenses work against which attacks.

Sanitizers (from defense/sanitize.py):
  - VAE RoundTrip x1, x2
  - Noisy RoundTrip sigma=0.25, 0.5
  - Latent Quantization 8b, 6b
  - Latent Smoothing sigma=0.5, 1.0

Anomaly Detectors (from defense/anomaly.py):
  - KS Test, RT Residual, Entropy MAD, Spectral KL

Attacks:
  - PatchSteg (±ε=5)
  - PCA-PatchSteg (ε=5)
  - PSyDUCK-inspired (ε=5)
  - CDF-PatchSteg

Generates: paper/figures/cath_defense_heatmap.png
           paper/figures/cath_sanitize_all_attacks.png
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
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.cdf_steganography import CDFPatchSteg
from core.pca_directions import PCADirections, PCAPatchSteg
from core.psyduck_steganography import PSyDUCKSteg
from core.metrics import compute_psnr, bit_accuracy

# Import Cath's classes directly
sys.path.insert(0, str(Path(__file__).resolve().parent / 'defense'))
from sanitize import (VaeRoundTripSanitizer, NoisyRoundTripSanitizer,
                      LatentQuantizationSanitizer, LatentSmoothingSanitizer)
from anomaly import (KSTestDetector, RoundTripResidualDetector,
                     EntropyAnomalyDetector, SpectralAnomalyDetector)

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
    imgs, seen = [], set()
    for img, label in ds:
        if label not in seen and len(imgs) < n:
            seen.add(label)
            imgs.append(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    return imgs


def make_stego(vae, images, attack, pca_dir=None):
    """Encode 20-bit random messages with the given attack. Returns stegos, carriers, bits."""
    stegos, all_carriers, all_bits = [], [], []
    for i, img in enumerate(images):
        np.random.seed(42 + i); torch.manual_seed(42 + i)
        bits = list(np.random.randint(0, 2, N_CARRIERS))
        latent = vae.encode(img)

        if attack == 'PatchSteg':
            s = PatchSteg(seed=42, epsilon=5.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = s.encode_message(latent, carriers, bits)
        elif attack == 'PCA':
            s = PCAPatchSteg(pca_dir, seed=42, epsilon=5.0, component=0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = s.encode_message(latent, carriers, bits)
        elif attack == 'PSyDUCK':
            s = PSyDUCKSteg(seed=42, epsilon=5.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = s.encode_message(latent, carriers, bits)
        elif attack == 'CDF':
            s = CDFPatchSteg(seed=42, sigma=1.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_m = s.encode_message(latent, carriers, bits)

        stegos.append(vae.decode(lat_m))
        all_carriers.append(carriers)
        all_bits.append(bits)
    return stegos, all_carriers, all_bits


def decode_after_sanitize(vae, attack, clean_imgs, san_imgs, all_carriers, all_bits, pca_dir=None):
    """Decode message from sanitized images. Returns mean bit accuracy."""
    accs = []
    for img, san, carriers, bits in zip(clean_imgs, san_imgs, all_carriers, all_bits):
        if attack == 'PatchSteg':
            s = PatchSteg(seed=42, epsilon=5.0)
            rec, _ = s.decode_message(vae.encode(img), vae.encode(san), carriers)
        elif attack == 'PCA':
            s = PCAPatchSteg(pca_dir, seed=42, epsilon=5.0, component=0)
            rec, _ = s.decode_message(vae.encode(img), vae.encode(san), carriers)
        elif attack == 'PSyDUCK':
            s = PSyDUCKSteg(seed=42, epsilon=5.0)
            rec, _ = s.decode_message(vae.encode(img), vae.encode(san), carriers)
        elif attack == 'CDF':
            s = CDFPatchSteg(seed=42, sigma=1.0)
            rec, _ = s.decode_message(vae, san, carriers)
        accs.append(bit_accuracy(bits, rec))
    return float(np.mean(accs))


print("Loading VAE...", flush=True)
vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
print(f"  VAE loaded ({time.time()-t0:.0f}s)", flush=True)

images = get_images(6)
train_imgs = images[:3]   # calibrate detectors on clean images
test_imgs  = images[3:]   # evaluate on held-out set

# Fit PCA on all images
pca_dir = PCADirections(n_components=4)
pca_dir.fit_global(vae, images)
print("  PCA fitted", flush=True)

ATTACKS = ['PatchSteg', 'PCA', 'PSyDUCK', 'CDF']
ATTACK_LABELS = ['PatchSteg\n(±ε)', 'PCA-\nPatchSteg', 'PSyDUCK-\ninspired', 'CDF-\nPatchSteg']

sanitizers = [
    VaeRoundTripSanitizer(n_trips=1),
    VaeRoundTripSanitizer(n_trips=2),
    NoisyRoundTripSanitizer(noise_std=0.25),
    NoisyRoundTripSanitizer(noise_std=0.5),
    LatentQuantizationSanitizer(n_bits=8),
    LatentQuantizationSanitizer(n_bits=6),
    LatentSmoothingSanitizer(sigma=0.5),
    LatentSmoothingSanitizer(sigma=1.0),
]

detectors = [
    KSTestDetector(),
    RoundTripResidualDetector(),
    EntropyAnomalyDetector(patch_size=4),
    SpectralAnomalyDetector(n_bins=16),
]

# ================================================================
# CALIBRATE DETECTORS ON CLEAN IMAGES
# ================================================================
print("\nCalibrating detectors...", flush=True)
for det in detectors:
    det.fit(vae, train_imgs)
    print(f"  {det.name}", flush=True)

# ================================================================
# 1. SANITIZERS: bit accuracy after defense per attack
# ================================================================
print("\n" + "#"*60, flush=True)
print("# SANITIZERS: BIT ACCURACY AFTER DEFENSE", flush=True)
print("#"*60, flush=True)

# san_results[san_name][attack] = {'acc', 'psnr'}
san_results = {s.name: {} for s in sanitizers}
# baseline
baseline = {}

for attack in ATTACKS:
    stegos, carriers, bits = make_stego(vae, test_imgs, attack, pca_dir)
    baseline[attack] = decode_after_sanitize(vae, attack, test_imgs, stegos, carriers, bits, pca_dir)
    print(f"\n  {attack} (baseline {baseline[attack]:.1f}%):", flush=True)

    for san in sanitizers:
        san_imgs = [san.sanitize(vae, img) for img in stegos]
        acc = decode_after_sanitize(vae, attack, test_imgs, san_imgs, carriers, bits, pca_dir)
        psnr = float(np.mean([compute_psnr(c, s) for c, s in zip(test_imgs, san_imgs)]))
        san_results[san.name][attack] = {'acc': acc, 'psnr': psnr}
        print(f"    {san.name:>22s}: acc={acc:.1f}%  psnr={psnr:.1f}dB", flush=True)

# ================================================================
# 2. ANOMALY DETECTORS: AUC per attack
# ================================================================
print("\n" + "#"*60, flush=True)
print("# ANOMALY DETECTORS: AUC PER ATTACK", flush=True)
print("#"*60, flush=True)

# det_results[det_name][attack] = auc
det_results = {d.name: {} for d in detectors}

clean_scores_cache = {d.name: [d.score(vae, img) for img in test_imgs] for d in detectors}

for attack in ATTACKS:
    stegos, _, _ = make_stego(vae, test_imgs, attack, pca_dir)
    print(f"\n  {attack}:", flush=True)
    for det in detectors:
        clean_s = clean_scores_cache[det.name]
        stego_s = [det.score(vae, img) for img in stegos]
        y = [0]*len(clean_s) + [1]*len(stego_s)
        try:
            auc = roc_auc_score(y, clean_s + stego_s)
        except Exception:
            auc = 0.5
        det_results[det.name][attack] = auc
        print(f"    {det.name:>14s}: AUC={auc:.3f}", flush=True)

# ================================================================
# FIGURES
# ================================================================
print("\nGenerating figures...", flush=True)

# --- Figure 1: Sanitizer heatmap (bit accuracy) ---
san_names = [s.name for s in sanitizers]
acc_matrix = np.array([[san_results[s][a]['acc'] for a in ATTACKS] for s in san_names])
psnr_matrix = np.array([[san_results[s][a]['psnr'] for a in ATTACKS] for s in san_names])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im0 = axes[0].imshow(acc_matrix, cmap='RdYlGn_r', aspect='auto', vmin=50, vmax=100)
axes[0].set_xticks(range(len(ATTACKS)))
axes[0].set_xticklabels(ATTACK_LABELS, fontsize=9)
axes[0].set_yticks(range(len(san_names)))
axes[0].set_yticklabels(san_names, fontsize=8)
axes[0].set_title('Bit Accuracy After Sanitization (%)\n(lower = stronger defense)', fontsize=10)
plt.colorbar(im0, ax=axes[0])
for i in range(len(san_names)):
    for j in range(len(ATTACKS)):
        axes[0].text(j, i, f'{acc_matrix[i,j]:.0f}', ha='center', va='center',
                     fontsize=8, fontweight='bold',
                     color='white' if acc_matrix[i,j] < 65 else 'black')

im1 = axes[1].imshow(psnr_matrix, cmap='RdYlGn', aspect='auto', vmin=10, vmax=40)
axes[1].set_xticks(range(len(ATTACKS)))
axes[1].set_xticklabels(ATTACK_LABELS, fontsize=9)
axes[1].set_yticks(range(len(san_names)))
axes[1].set_yticklabels(san_names, fontsize=8)
axes[1].set_title('PSNR After Sanitization (dB)\n(higher = less image damage)', fontsize=10)
plt.colorbar(im1, ax=axes[1])
for i in range(len(san_names)):
    for j in range(len(ATTACKS)):
        axes[1].text(j, i, f'{psnr_matrix[i,j]:.0f}', ha='center', va='center',
                     fontsize=8, fontweight='bold',
                     color='white' if psnr_matrix[i,j] < 20 else 'black')

plt.suptitle("Cath's Sanitizers vs. All SOTA Attacks", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'cath_sanitize_all_attacks.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved cath_sanitize_all_attacks.png", flush=True)

# --- Figure 2: Detector AUC heatmap ---
det_names = [d.name for d in detectors]
auc_matrix = np.array([[det_results[d][a] for a in ATTACKS] for d in det_names])

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(auc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(ATTACKS)))
ax.set_xticklabels(ATTACK_LABELS, fontsize=10)
ax.set_yticks(range(len(det_names)))
ax.set_yticklabels(det_names, fontsize=10)
ax.set_title("Cath's Anomaly Detectors vs. All SOTA Attacks\n"
             "Detection AUC (higher = better detector, 0.5 = chance)", fontsize=11)
plt.colorbar(im, ax=ax)
for i in range(len(det_names)):
    for j in range(len(ATTACKS)):
        val = auc_matrix[i,j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if val > 0.75 or val < 0.35 else 'black')

plt.tight_layout()
plt.savefig(FIG_DIR / 'cath_defense_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved cath_defense_heatmap.png", flush=True)

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'#'*60}", flush=True)
print("# SUMMARY", flush=True)
print(f"{'#'*60}", flush=True)

print("\nSanitizer bit accuracy (baseline -> after defense):", flush=True)
for san in sanitizers:
    row = "  " + f"{san.name:>22s}: "
    row += "  ".join(f"{a}:{san_results[san.name][a]['acc']:.0f}%" for a in ATTACKS)
    print(row, flush=True)

print("\nDetector AUC (0.5=chance, 1.0=perfect):", flush=True)
print(f"  {'Detector':>14s}  " + "  ".join(f"{a:>10s}" for a in ATTACKS), flush=True)
for det in detectors:
    row = f"  {det.name:>14s}  " + "  ".join(f"{det_results[det.name][a]:>10.3f}" for a in ATTACKS)
    print(row, flush=True)

print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
