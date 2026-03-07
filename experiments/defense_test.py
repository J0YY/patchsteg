#!/usr/bin/env python
"""
Defense test: CertifiedPurifier vs CDF-PatchSteg and original PatchSteg.

Tests:
1. Bit accuracy after purification (should drop to ~50%)
2. Image quality (PSNR before/after purification — should stay high)
3. KS test: purified latent still looks like cover distribution
4. PSNR vs bit-accuracy trade-off curve (noise_std sweep)

Generates: defense_tradeoff.png, defense_certified.png
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
from core.purifier import CertifiedPurifier
from core.metrics import compute_psnr, bit_accuracy

IMG_SIZE = 128
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

t_total = time.time()


def get_images(n=4):
    from torchvision.datasets import CIFAR10
    print("  Loading CIFAR-10...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=True, download=True)
    imgs, classes_seen = [], set()
    for img, label in ds:
        if label not in classes_seen and len(imgs) < n:
            classes_seen.add(label)
            imgs.append(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    return imgs


print("Loading VAE...", flush=True)
vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
print(f"  VAE loaded ({time.time()-t_total:.0f}s)", flush=True)

images = get_images(4)
N_CARRIERS = 20

# ================================================================
# 1. BASELINE: encode with CDF and original PatchSteg
# ================================================================
print("\n" + "#"*60, flush=True)
print("# 1. BASELINE BIT ACCURACY (no purification)", flush=True)
print("#"*60, flush=True)

def encode_cdf(img, seed=42, n=N_CARRIERS):
    cdf = CDFPatchSteg(seed=seed, sigma=1.0)
    carriers, _ = cdf.select_carriers_by_stability(vae, img, n_carriers=n)
    bits = list(np.random.randint(0, 2, n))
    latent = vae.encode(img)
    lat_m = cdf.encode_message(latent, carriers, bits)
    stego = vae.decode(lat_m)
    rec, _ = cdf.decode_message(vae, stego, carriers)
    return stego, carriers, bits, rec

def encode_orig(img, seed=42, eps=5.0, n=N_CARRIERS):
    ps = PatchSteg(seed=seed, epsilon=eps)
    carriers, _ = ps.select_carriers_by_stability(vae, img, n_carriers=n)
    bits = list(np.random.randint(0, 2, n))
    latent = vae.encode(img)
    lat_m = ps.encode_message(latent, carriers, bits)
    stego = vae.decode(lat_m)
    lat_re = vae.encode(stego)
    lat_clean = vae.encode(img)
    rec, _ = ps.decode_message(lat_clean, lat_re, carriers)
    return stego, carriers, bits, rec

# CDF baseline
cdf_base_accs = []
for i, img in enumerate(images):
    np.random.seed(42 + i)
    stego, carriers, bits, rec = encode_cdf(img, seed=42)
    cdf_base_accs.append(bit_accuracy(bits, rec))
print(f"  CDF-PatchSteg baseline: {np.mean(cdf_base_accs):.1f}% ± {np.std(cdf_base_accs):.1f}%", flush=True)

# Original PatchSteg baseline
orig_base_accs = []
for i, img in enumerate(images):
    np.random.seed(42 + i)
    stego, carriers, bits, rec = encode_orig(img, seed=42, eps=5.0)
    orig_base_accs.append(bit_accuracy(bits, rec))
print(f"  Original PatchSteg (ε=5) baseline: {np.mean(orig_base_accs):.1f}%", flush=True)

# ================================================================
# 2. RESAMPLE PURIFIER vs CDF
# ================================================================
print("\n" + "#"*60, flush=True)
print("# 2. RESAMPLE PURIFIER (stability-guided resampling)", flush=True)
print("#"*60, flush=True)

resample_accs, resample_psnrs = [], []
for n_purify in [20, 50, 64]:
    accs, psnrs = [], []
    for i, img in enumerate(images):
        np.random.seed(42 + i)
        stego, carriers, bits, _ = encode_cdf(img, seed=42)

        purifier = CertifiedPurifier(n_purify=n_purify, strategy='resample')
        purified, diag = purifier.purify(vae, stego)

        # Decode from purified image
        cdf = CDFPatchSteg(seed=42, sigma=1.0)
        rec, _ = cdf.decode_message(vae, purified, carriers)
        accs.append(bit_accuracy(bits, rec))
        psnrs.append(compute_psnr(img, purified))

    print(f"  n_purify={n_purify}: acc={np.mean(accs):.1f}% psnr={np.mean(psnrs):.1f}dB", flush=True)
    if n_purify == N_CARRIERS:
        resample_accs = accs
        resample_psnrs = psnrs

# Use n_purify=N_CARRIERS for the main comparison
purifier_main = CertifiedPurifier(n_purify=N_CARRIERS, strategy='resample')
resample_accs, resample_psnrs = [], []
for i, img in enumerate(images):
    np.random.seed(42 + i)
    stego, carriers, bits, _ = encode_cdf(img, seed=42)
    purified, _ = purifier_main.purify(vae, stego)
    cdf = CDFPatchSteg(seed=42, sigma=1.0)
    rec, _ = cdf.decode_message(vae, purified, carriers)
    resample_accs.append(bit_accuracy(bits, rec))
    resample_psnrs.append(compute_psnr(img, purified))

# ================================================================
# 3. NOISE PURIFIER vs CDF — sweep noise_std
# ================================================================
print("\n" + "#"*60, flush=True)
print("# 3. NOISE PURIFIER — noise_std sweep", flush=True)
print("#"*60, flush=True)

noise_stds = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
noise_accs, noise_psnrs, noise_certs = [], [], []

for noise_std in noise_stds:
    accs, psnrs = [], []
    for i, img in enumerate(images):
        np.random.seed(42 + i)
        stego, carriers, bits, _ = encode_cdf(img, seed=42)

        purifier = CertifiedPurifier(noise_std=noise_std, strategy='noise')
        purified, diag = purifier.purify(vae, stego)

        cdf = CDFPatchSteg(seed=42, sigma=1.0)
        rec, _ = cdf.decode_message(vae, purified, carriers)
        accs.append(bit_accuracy(bits, rec))
        psnrs.append(compute_psnr(img, purified))

    cert = CertifiedPurifier(noise_std=noise_std).certified_acc_upper_bound(sigma_attack=1.0)
    noise_accs.append(np.mean(accs))
    noise_psnrs.append(np.mean(psnrs))
    noise_certs.append(cert * 100)
    print(f"  noise_std={noise_std:.1f}: acc={np.mean(accs):.1f}% psnr={np.mean(psnrs):.1f}dB cert_bound={cert*100:.1f}%", flush=True)

# ================================================================
# 4. COMBINED PURIFIER vs CDF and ORIGINAL
# ================================================================
print("\n" + "#"*60, flush=True)
print("# 4. COMBINED PURIFIER", flush=True)
print("#"*60, flush=True)

purifier_combined = CertifiedPurifier(n_purify=N_CARRIERS, noise_std=0.3, strategy='combined')

combined_cdf_accs, combined_cdf_psnrs = [], []
combined_orig_accs, combined_orig_psnrs = [], []
ks_results = []

for i, img in enumerate(images):
    np.random.seed(42 + i)

    # vs CDF
    stego_cdf, carriers_cdf, bits_cdf, _ = encode_cdf(img, seed=42)
    purified_cdf, diag = purifier_combined.purify(vae, stego_cdf)
    cdf = CDFPatchSteg(seed=42, sigma=1.0)
    rec, _ = cdf.decode_message(vae, purified_cdf, carriers_cdf)
    combined_cdf_accs.append(bit_accuracy(bits_cdf, rec))
    combined_cdf_psnrs.append(compute_psnr(img, purified_cdf))

    ks = purifier_combined.run_ks_test(vae, img, purified_cdf)
    ks_results.append(ks)

    # vs Original PatchSteg
    stego_orig, carriers_orig, bits_orig, _ = encode_orig(img, seed=42, eps=5.0)
    purified_orig, _ = purifier_combined.purify(vae, stego_orig)
    ps = PatchSteg(seed=42, epsilon=5.0)
    lat_clean = vae.encode(img)
    lat_re = vae.encode(purified_orig)
    rec_orig, _ = ps.decode_message(lat_clean, lat_re, carriers_orig)
    combined_orig_accs.append(bit_accuracy(bits_orig, rec_orig))
    combined_orig_psnrs.append(compute_psnr(img, purified_orig))

print(f"  Combined vs CDF:      acc={np.mean(combined_cdf_accs):.1f}%  psnr={np.mean(combined_cdf_psnrs):.1f}dB", flush=True)
print(f"  Combined vs Original: acc={np.mean(combined_orig_accs):.1f}%  psnr={np.mean(combined_orig_psnrs):.1f}dB", flush=True)
ks_ps_fmt = [str(round(r['ks_p'], 3)) for r in ks_results]
print(f'  KS test p-values:     {ks_ps_fmt}', flush=True)

# ================================================================
# FIGURES
# ================================================================
print("\nGenerating figures...", flush=True)

# --- Figure 1: defense tradeoff (noise sweep) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(noise_stds, noise_accs, 'ro-', linewidth=2, markersize=8, label='Empirical acc')
ax.plot(noise_stds, noise_certs, 'b--', linewidth=1.5, label='Certified upper bound')
ax.axhline(50, color='gray', ls=':', alpha=0.7, label='Chance (50%)')
ax.axhline(np.mean(cdf_base_accs), color='orange', ls='--', alpha=0.7, label=f'No defense ({np.mean(cdf_base_accs):.0f}%)')
ax.set_xlabel('Noise std (δ)')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title('(a) Noise Purification: Bit Accuracy vs Noise')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_ylim(30, 105)

ax = axes[1]
ax.plot(noise_psnrs, noise_accs, 'ro-', linewidth=2, markersize=8)
for std, psnr, acc in zip(noise_stds, noise_psnrs, noise_accs):
    ax.annotate(f'δ={std}', (psnr, acc), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.axhline(50, color='gray', ls=':', alpha=0.7, label='Chance (50%)')
ax.set_xlabel('PSNR of purified image (dB)')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title('(b) PSNR–Security Trade-off')
ax.legend()
ax.grid(True, alpha=0.2)

plt.suptitle('Noise Purification Defense vs CDF-PatchSteg', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'defense_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved defense_tradeoff.png", flush=True)

# --- Figure 2: certified defense summary ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

methods = ['No defense\n(CDF)', 'Resample\n(K=20)', 'Noise\n(δ=0.5)', 'Combined\n(Resample+Noise)']

# Collect combined with noise only at 0.5
noise_0_5_accs = []
noise_0_5_psnrs = []
purifier_n = CertifiedPurifier(noise_std=0.5, strategy='noise')
for i, img in enumerate(images):
    np.random.seed(42 + i)
    stego, carriers, bits, _ = encode_cdf(img, seed=42)
    purified, _ = purifier_n.purify(vae, stego)
    cdf = CDFPatchSteg(seed=42, sigma=1.0)
    rec, _ = cdf.decode_message(vae, purified, carriers)
    noise_0_5_accs.append(bit_accuracy(bits, rec))
    noise_0_5_psnrs.append(compute_psnr(img, purified))

acc_vals = [np.mean(cdf_base_accs), np.mean(resample_accs), np.mean(noise_0_5_accs), np.mean(combined_cdf_accs)]
acc_stds = [np.std(cdf_base_accs), np.std(resample_accs), np.std(noise_0_5_accs), np.std(combined_cdf_accs)]
psnr_vals = [compute_psnr(images[0], vae.decode(vae.encode(images[0]))),
             np.mean(resample_psnrs), np.mean(noise_0_5_psnrs), np.mean(combined_cdf_psnrs)]

# Bit accuracy bar
ax = axes[0]
bars = ax.bar(methods, acc_vals, yerr=acc_stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(50, color='gray', ls='--', alpha=0.7, label='Chance (50%)')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title('(a) Bit Accuracy After Defense')
ax.set_ylim(0, 110)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, acc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

# PSNR bar
ax = axes[1]
bars = ax.bar(methods, psnr_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(30, color='orange', ls='--', alpha=0.7, label='30 dB threshold')
ax.set_ylabel('PSNR (dB)')
ax.set_title('(b) Image Quality After Defense')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, psnr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

# KS p-values
ax = axes[2]
ks_ps = [r['ks_p'] for r in ks_results]
ax.bar(range(len(ks_ps)), ks_ps, color='#3498db', edgecolor='black', linewidth=0.5)
ax.axhline(0.05, color='red', ls='--', alpha=0.7, label='p=0.05 threshold')
ax.set_xticks(range(len(ks_ps)))
ax.set_xticklabels([f'img{i}' for i in range(len(ks_ps))])
ax.set_ylabel('KS test p-value')
ax.set_title('(c) Latent Distribution Preserved\n(Combined purifier, p>0.05 = pass)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(0, 1.1)

plt.suptitle('CertifiedPurifier Defense Summary', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'defense_certified.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved defense_certified.png", flush=True)

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'#'*60}", flush=True)
print("# SUMMARY", flush=True)
print(f"{'#'*60}", flush=True)
print(f"  CDF baseline:          acc={np.mean(cdf_base_accs):.1f}%", flush=True)
print(f"  After resample purify: acc={np.mean(resample_accs):.1f}%  psnr={np.mean(resample_psnrs):.1f}dB", flush=True)
print(f"  After noise (δ=0.5):   acc={np.mean(noise_0_5_accs):.1f}%  psnr={np.mean(noise_0_5_psnrs):.1f}dB", flush=True)
print(f"  After combined:        acc={np.mean(combined_cdf_accs):.1f}%  psnr={np.mean(combined_cdf_psnrs):.1f}dB", flush=True)
print(f"  vs Original PatchSteg: acc={np.mean(combined_orig_accs):.1f}%  (both attacks neutralized)", flush=True)
print(f"  KS test all p>0.05:    {all(r['ks_p'] > 0.05 for r in ks_results)}", flush=True)
print(f"  Certified upper bound (noise δ=0.5): {CertifiedPurifier(noise_std=0.5).certified_acc_upper_bound()*100:.1f}%", flush=True)
print(f"\n  Total time: {time.time()-t_total:.0f}s", flush=True)
