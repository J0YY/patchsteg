#!/usr/bin/env python
"""
QuantileShuffleSanitizer vs all SOTA attacks.

Tests the theoretical guarantee: CDF bit accuracy -> 50% at p=0.5.
Also tests against PatchSteg, PCA-PatchSteg, PSyDUCK for completeness.

Speed: latents and carriers are pre-computed once per image and cached.
       Each encode/decode is done only when strictly necessary.

Generates: paper/figures/quantile_sanitizer_results.png
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

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.cdf_steganography import CDFPatchSteg
from core.pca_directions import PCADirections, PCAPatchSteg
from core.psyduck_steganography import PSyDUCKSteg
from core.quantile_sanitizer import QuantileShuffleSanitizer
from core.metrics import compute_psnr, bit_accuracy

IMG_SIZE = 128
N_CARRIERS = 20
N_IMAGES = 4
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


# ================================================================
# SETUP
# ================================================================
print("Loading VAE...", flush=True)
vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
print(f"  VAE loaded ({time.time()-t0:.1f}s)", flush=True)

images = get_images(N_IMAGES)

pca_dir = PCADirections(n_components=4)
pca_dir.fit_global(vae, images)
print(f"  PCA fitted ({time.time()-t0:.1f}s)", flush=True)

# ================================================================
# PRE-COMPUTE: latents and carriers for each image x attack
# (cached -- each VAE encode is called exactly once per image per attack)
# ================================================================
print("\nPre-computing latents and carriers...", flush=True)

# Clean latents cached once
clean_latents = []
for img in images:
    clean_latents.append(vae.encode(img))
print(f"  Clean latents done ({time.time()-t0:.1f}s)", flush=True)

ATTACKS = ['PatchSteg', 'PCA', 'PSyDUCK', 'CDF']
ATTACK_LABELS = ['PatchSteg\n(±ε)', 'PCA-\nPatchSteg', 'PSyDUCK-\ninspired', 'CDF-\nPatchSteg']

cache = {}   # cache[attack][i] = {'bits', 'carriers', 'lat_stego', 'stego_img'}

for attack in ATTACKS:
    cache[attack] = []
    for i, (img, lat_clean) in enumerate(zip(images, clean_latents)):
        np.random.seed(42 + i); torch.manual_seed(42 + i)
        bits = list(np.random.randint(0, 2, N_CARRIERS))

        if attack == 'PatchSteg':
            s = PatchSteg(seed=42, epsilon=5.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_stego = s.encode_message(lat_clean, carriers, bits)
        elif attack == 'PCA':
            s = PCAPatchSteg(pca_dir, seed=42, epsilon=5.0, component=0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_stego = s.encode_message(lat_clean, carriers, bits)
        elif attack == 'PSyDUCK':
            s = PSyDUCKSteg(seed=42, epsilon=5.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_stego = s.encode_message(lat_clean, carriers, bits)
        elif attack == 'CDF':
            s = CDFPatchSteg(seed=42, sigma=1.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=N_CARRIERS)
            lat_stego = s.encode_message(lat_clean, carriers, bits)

        stego_img = vae.decode(lat_stego)
        cache[attack].append({
            'bits': bits, 'carriers': carriers,
            'lat_stego': lat_stego, 'stego_img': stego_img,
            'attacker': attack,
        })

print(f"  All stego latents cached ({time.time()-t0:.1f}s)", flush=True)


def decode_from_lat(attack, lat_clean, lat_received, carriers, pca_dir=None):
    """Decode using pre-computed latents (no extra VAE encode needed for ±ε attacks)."""
    if attack == 'PatchSteg':
        s = PatchSteg(seed=42, epsilon=5.0)
        rec, _ = s.decode_message(lat_clean, lat_received, carriers)
    elif attack == 'PCA':
        s = PCAPatchSteg(pca_dir, seed=42, epsilon=5.0, component=0)
        rec, _ = s.decode_message(lat_clean, lat_received, carriers)
    elif attack == 'PSyDUCK':
        s = PSyDUCKSteg(seed=42, epsilon=5.0)
        rec, _ = s.decode_message(lat_clean, lat_received, carriers)
    elif attack == 'CDF':
        # CDF decode needs the image (re-encodes internally) -- no clean latent needed
        s = CDFPatchSteg(seed=42, sigma=1.0)
        # lat_received here is the sanitized latent; decode needs the image
        # We pass None -- caller must use the image-based path
        return None
    return rec


# ================================================================
# BASELINE: bit accuracy before any defense
# ================================================================
print("\n" + "#"*60, flush=True)
print("# BASELINE BIT ACCURACY (no defense)", flush=True)
print("#"*60, flush=True)

baseline = {}
for attack in ATTACKS:
    accs = []
    for i, entry in enumerate(cache[attack]):
        lat_clean = clean_latents[i]
        carriers = entry['carriers']
        bits = entry['bits']

        if attack == 'CDF':
            s = CDFPatchSteg(seed=42, sigma=1.0)
            rec, _ = s.decode_message(vae, entry['stego_img'], carriers)
        else:
            s_cls = {'PatchSteg': PatchSteg, 'PCA': None, 'PSyDUCK': PSyDUCKSteg}
            if attack == 'PatchSteg':
                s = PatchSteg(seed=42, epsilon=5.0)
            elif attack == 'PCA':
                s = PCAPatchSteg(pca_dir, seed=42, epsilon=5.0, component=0)
            elif attack == 'PSyDUCK':
                s = PSyDUCKSteg(seed=42, epsilon=5.0)
            lat_re = vae.encode(entry['stego_img'])
            rec, _ = s.decode_message(lat_clean, lat_re, carriers)

        accs.append(bit_accuracy(bits, rec))

    baseline[attack] = float(np.mean(accs))
    print(f"  {attack:12s}: {baseline[attack]:.1f}%", flush=True)

# ================================================================
# QUANTILE SHUFFLE SANITIZER: sweep p values
# ================================================================
print("\n" + "#"*60, flush=True)
print("# QUANTILE SHUFFLE SANITIZER", flush=True)
print("#"*60, flush=True)

P_VALUES = [0.1, 0.2, 0.3, 0.5]
san_results = {}   # san_results[p][attack] = {'acc', 'psnr'}

for p in P_VALUES:
    san = QuantileShuffleSanitizer(p=p, seed=0)
    san_results[p] = {}
    print(f"\n  p={p}:", flush=True)

    for attack in ATTACKS:
        accs, psnrs = [], []
        for i, entry in enumerate(cache[attack]):
            img = images[i]
            lat_clean = clean_latents[i]
            carriers = entry['carriers']
            bits = entry['bits']

            # Apply sanitizer directly to stego latent (fast -- no extra encode)
            lat_san = san.sanitize_latent(entry['lat_stego'])
            san_img = vae.decode(lat_san)

            # Decode from sanitized image
            if attack == 'CDF':
                s = CDFPatchSteg(seed=42, sigma=1.0)
                rec, _ = s.decode_message(vae, san_img, carriers)
            else:
                lat_san_re = vae.encode(san_img)
                if attack == 'PatchSteg':
                    s = PatchSteg(seed=42, epsilon=5.0)
                elif attack == 'PCA':
                    s = PCAPatchSteg(pca_dir, seed=42, epsilon=5.0, component=0)
                elif attack == 'PSyDUCK':
                    s = PSyDUCKSteg(seed=42, epsilon=5.0)
                rec, _ = s.decode_message(lat_clean, lat_san_re, carriers)

            accs.append(bit_accuracy(bits, rec))
            psnrs.append(compute_psnr(img, san_img))

        san_results[p][attack] = {
            'acc': float(np.mean(accs)),
            'psnr': float(np.mean(psnrs)),
        }
        print(f"    {attack:12s}: acc={san_results[p][attack]['acc']:.1f}%  "
              f"psnr={san_results[p][attack]['psnr']:.1f}dB", flush=True)

# ================================================================
# FIGURE
# ================================================================
print("\nGenerating figure...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(ATTACKS))
width = 0.18
colors_p = ['#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

# --- Left: Bit accuracy vs p ---
ax = axes[0]
# Baseline bars
ax.bar(x - 2.5*width, [baseline[a] for a in ATTACKS], width,
       label='No defense', color='#95a5a6', edgecolor='black', linewidth=0.5)
for idx, (p, col) in enumerate(zip(P_VALUES, colors_p)):
    accs = [san_results[p][a]['acc'] for a in ATTACKS]
    ax.bar(x + (idx - 1)*width, accs, width,
           label=f'p={p}', color=col, edgecolor='black', linewidth=0.5)

ax.axhline(50, color='black', ls=':', linewidth=1.5, alpha=0.6, label='Chance (50%)')
ax.set_xticks(x)
ax.set_xticklabels(ATTACK_LABELS, fontsize=9)
ax.set_ylabel('Bit Accuracy (%)')
ax.set_ylim(0, 115)
ax.set_title('Bit Accuracy After QuantileShuffle\n(lower = stronger defense)', fontsize=10)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.2, axis='y')
for i, attack in enumerate(ATTACKS):
    for j, p in enumerate(P_VALUES):
        val = san_results[p][attack]['acc']
        bar_x = x[i] + (j - 1)*width
        ax.text(bar_x, val + 1, f'{val:.0f}', ha='center', fontsize=6, fontweight='bold')

# --- Right: PSNR vs p ---
ax = axes[1]
for idx, (p, col) in enumerate(zip(P_VALUES, colors_p)):
    psnrs = [san_results[p][a]['psnr'] for a in ATTACKS]
    ax.bar(x + (idx - 1.5)*width, psnrs, width,
           label=f'p={p}', color=col, edgecolor='black', linewidth=0.5)

ax.axhline(30, color='orange', ls='--', alpha=0.7, label='30 dB threshold')
ax.set_xticks(x)
ax.set_xticklabels(ATTACK_LABELS, fontsize=9)
ax.set_ylabel('PSNR (dB)')
ax.set_title('Image Quality After QuantileShuffle\n(higher = less damage)', fontsize=10)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.2, axis='y')
for i, attack in enumerate(ATTACKS):
    for j, p in enumerate(P_VALUES):
        val = san_results[p][attack]['psnr']
        bar_x = x[i] + (j - 1.5)*width
        ax.text(bar_x, val + 0.3, f'{val:.0f}', ha='center', fontsize=6)

plt.suptitle('QuantileShuffleSanitizer: Theoretical 50% Guarantee vs All SOTA Attacks\n'
             '(reflects latent values around channel mean with probability p)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'quantile_sanitizer_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved quantile_sanitizer_results.png", flush=True)

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'#'*60}", flush=True)
print("# SUMMARY", flush=True)
print(f"{'#'*60}", flush=True)
print(f"\n  Baseline:", flush=True)
for attack in ATTACKS:
    print(f"    {attack:12s}: {baseline[attack]:.1f}%", flush=True)

print(f"\n  After QuantileShuffle (acc% / PSNR dB):", flush=True)
print(f"  {'p':>5}  " + "  ".join(f"{a:>18s}" for a in ATTACKS), flush=True)
for p in P_VALUES:
    row = f"  {p:>5}  "
    row += "  ".join(
        f"{san_results[p][a]['acc']:5.1f}% / {san_results[p][a]['psnr']:4.1f}dB"
        for a in ATTACKS
    )
    print(row, flush=True)

print(f"\n  Theoretical prediction at p=0.5: all attacks -> 50.0% accuracy", flush=True)
print(f"  CDF actual at p=0.5: {san_results[0.5]['CDF']['acc']:.1f}%", flush=True)
print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
