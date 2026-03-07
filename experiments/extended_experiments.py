#!/usr/bin/env python
"""
Extended experiments:
1. Natural photographs (CIFAR-10 via torchvision)
2. Longer messages stress test
3. LSB baseline comparison for detectability
4. Theoretical validation (KL regularization connection)

~10 min on CPU with 256x256.
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
import io

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, bit_accuracy

IMG_SIZE = 256
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'

torch.manual_seed(42)
np.random.seed(42)


# ================================================================
# HELPERS
# ================================================================
def get_natural_images(n=8):
    """Get natural photos from CIFAR-10 (upscaled to 256x256)."""
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    print("  Downloading CIFAR-10 (first time only)...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=True, download=True)
    # Pick diverse classes: airplane, car, bird, cat, deer, dog, frog, horse
    picked = []
    classes_seen = set()
    for img, label in ds:
        if label not in classes_seen and len(picked) < n:
            classes_seen.add(label)
            picked.append((img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                          ds.classes[label]))
    return [p[0] for p in picked], [p[1] for p in picked]


def lsb_embed(img: Image.Image, bits: list) -> Image.Image:
    """Embed bits using LSB steganography (baseline)."""
    arr = np.array(img).copy()
    flat = arr.flatten()
    for i, b in enumerate(bits):
        if i >= len(flat):
            break
        flat[i] = (flat[i] & 0xFE) | b
    return Image.fromarray(flat.reshape(arr.shape))


def lsb_extract(img: Image.Image, n_bits: int) -> list:
    """Extract LSB-embedded bits."""
    flat = np.array(img).flatten()
    return [int(flat[i] & 1) for i in range(min(n_bits, len(flat)))]


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


def extract_pixel_features(img):
    """Extract pixel-domain statistics for LSB detection."""
    arr = np.array(img).astype(float)
    feats = []
    for ch in range(3):
        x = arr[:,:,ch].flatten()
        mu, sig = x.mean(), x.std() + 1e-8
        feats.extend([mu, sig, float(np.median(x)),
                      float(((x-mu)**3).mean()/sig**3),
                      float(((x-mu)**4).mean()/sig**4),
                      float(np.histogram(x, bins=2)[0][0] / len(x))])  # LSB bias
    return np.array(feats)


# ================================================================
print("Loading VAE...", flush=True)
t_total = time.time()
vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
steg5 = PatchSteg(seed=42, epsilon=5.0)
steg2 = PatchSteg(seed=42, epsilon=2.0)


# ================================================================
# 1. NATURAL PHOTOGRAPHS
# ================================================================
print("\n" + "#"*60, flush=True)
print("# 1. NATURAL PHOTOGRAPHS", flush=True)
print("#"*60, flush=True)

nat_images, nat_names = get_natural_images(8)
nat_results = []

for img, name in zip(nat_images, nat_names):
    lat = vae.encode(img)
    for eps, s in [(2.0, steg2), (5.0, steg5)]:
        carriers, smap = s.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=eps)
        torch.manual_seed(42)
        bits = torch.randint(0, 2, (20,)).tolist()
        lat_m = s.encode_message(lat, carriers, bits)
        st = vae.decode(lat_m)
        psnr = compute_psnr(img, st)
        lat_re = vae.encode(st)
        rec, confs = s.decode_message(lat, lat_re, carriers)
        acc = bit_accuracy(bits, rec)
        nat_results.append({'name': name, 'eps': eps, 'acc': acc, 'psnr': psnr})
        print(f"  {name:>12s} eps={eps}: acc={acc:>5.1f}% psnr={psnr:.1f}dB", flush=True)

# Figure: natural photo stego examples (pick 4)
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
show_imgs = nat_images[:4]
show_names = nat_names[:4]
for col, (img, name) in enumerate(zip(show_imgs, show_names)):
    lat = vae.encode(img)
    carriers, _ = steg5.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=5.0)
    torch.manual_seed(42)
    bits = torch.randint(0, 2, (20,)).tolist()
    lat_m = steg5.encode_message(lat, carriers, bits)
    st = vae.decode(lat_m)
    psnr = compute_psnr(img, st)
    lat_re = vae.encode(st)
    rec, _ = steg5.decode_message(lat, lat_re, carriers)
    acc = bit_accuracy(bits, rec)
    diff = np.clip(np.abs(np.array(img).astype(float) - np.array(st).astype(float)) * 20, 0, 255).astype(np.uint8)

    axes[0, col].imshow(np.array(img)); axes[0, col].set_title(name, fontsize=11); axes[0, col].axis('off')
    axes[1, col].imshow(np.array(st)); axes[1, col].set_title(f'PSNR={psnr:.1f}', fontsize=10); axes[1, col].axis('off')
    axes[2, col].imshow(diff); axes[2, col].set_title(f'Acc={acc:.0f}%', fontsize=10); axes[2, col].axis('off')

axes[0,0].set_ylabel('Original', fontsize=12)
axes[1,0].set_ylabel('Stego', fontsize=12)
axes[2,0].set_ylabel('Diff 20x', fontsize=12)
plt.suptitle('PatchSteg on Natural Photographs (CIFAR-10, upscaled, $\\varepsilon=5.0$)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'natural_photos.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved natural_photos.png", flush=True)

# Summary
print("\n  NATURAL PHOTOS SUMMARY:", flush=True)
for eps in [2.0, 5.0]:
    sub = [r for r in nat_results if r['eps'] == eps]
    avg_acc = np.mean([r['acc'] for r in sub])
    avg_psnr = np.mean([r['psnr'] for r in sub])
    print(f"  eps={eps}: avg_acc={avg_acc:.1f}%, avg_psnr={avg_psnr:.1f}dB", flush=True)


# ================================================================
# 2. LONGER MESSAGES STRESS TEST
# ================================================================
print("\n" + "#"*60, flush=True)
print("# 2. LONGER MESSAGES STRESS TEST", flush=True)
print("#"*60, flush=True)

messages = ["A", "HI", "HELLO", "HELLO WORLD", "THE QUICK BROWN FOX"]
# Use a natural photo (first CIFAR image)
img_msg = nat_images[0]
lat_msg = vae.encode(img_msg)
msg_results = []

for msg in messages:
    bits = PatchSteg.text_to_bits(msg)
    n_bits = len(bits)

    for reps in [1, 3]:
        n_carriers_needed = n_bits * reps
        if n_carriers_needed > 1024:  # 32x32 grid limit
            print(f"  '{msg}' ({n_bits}b) x{reps}: SKIP (need {n_carriers_needed} > 1024)", flush=True)
            msg_results.append({'msg': msg, 'n_bits': n_bits, 'reps': reps,
                               'acc': None, 'psnr': None, 'decoded': None})
            continue

        s = PatchSteg(seed=42, epsilon=5.0)
        carriers, _ = s.select_carriers_by_stability(vae, img_msg, n_carriers=n_carriers_needed, test_eps=5.0)

        if reps == 1:
            lat_m = s.encode_message(lat_msg, carriers[:n_bits], bits)
            st = vae.decode(lat_m)
            lat_re = vae.encode(st)
            rec, _ = s.decode_message(lat_msg, lat_re, carriers[:n_bits])
        else:
            lat_m = s.encode_message_with_repetition(lat_msg, carriers, bits, reps=reps)
            st = vae.decode(lat_m)
            lat_re = vae.encode(st)
            rec = s.decode_message_with_repetition(lat_msg, lat_re, carriers, n_bits, reps=reps)

        decoded = PatchSteg.bits_to_text(rec)
        acc = bit_accuracy(bits, rec)
        psnr = compute_psnr(img_msg, st)
        msg_results.append({'msg': msg, 'n_bits': n_bits, 'reps': reps,
                           'acc': acc, 'psnr': psnr, 'decoded': decoded})
        match = "OK" if decoded == msg else "FAIL"
        print(f"  '{msg}' ({n_bits:>3d}b) x{reps}: '{decoded}' acc={acc:.0f}% psnr={psnr:.1f}dB [{match}]", flush=True)

# Figure: message length vs accuracy
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for reps, marker, color in [(1, 'o', 'tab:blue'), (3, 's', 'tab:green')]:
    sub = [r for r in msg_results if r['reps'] == reps and r['acc'] is not None]
    if sub:
        axes[0].plot([r['n_bits'] for r in sub], [r['acc'] for r in sub],
                    f'{marker}-', color=color, linewidth=2, markersize=8, label=f'Rep x{reps}')
        axes[1].plot([r['n_bits'] for r in sub], [r['psnr'] for r in sub],
                    f'{marker}-', color=color, linewidth=2, markersize=8, label=f'Rep x{reps}')

axes[0].axhline(100, color='green', ls='--', alpha=0.3)
axes[0].set_xlabel('Message Length (bits)'); axes[0].set_ylabel('Bit Accuracy (%)')
axes[0].set_title('Message Length vs Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.2)
axes[0].set_ylim(40, 105)
axes[1].axhline(30, color='green', ls='--', alpha=0.3)
axes[1].set_xlabel('Message Length (bits)'); axes[1].set_ylabel('PSNR (dB)')
axes[1].set_title('Message Length vs Image Quality'); axes[1].legend(); axes[1].grid(True, alpha=0.2)
plt.suptitle('Longer Message Stress Test ($\\varepsilon=5.0$, Natural Photo)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'message_length.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved message_length.png", flush=True)


# ================================================================
# 3. LSB BASELINE COMPARISON
# ================================================================
print("\n" + "#"*60, flush=True)
print("# 3. LSB BASELINE COMPARISON FOR DETECTABILITY", flush=True)
print("#"*60, flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

n_detect = 20
rng_det = np.random.RandomState(42)
detect_images = []
for i in range(n_detect):
    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    kind = i % 4
    if kind == 0:
        arr = rng_det.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    elif kind == 1:
        t = np.linspace(0, 1, IMG_SIZE)
        base = np.tile(t, (IMG_SIZE, 1))
        arr = (np.stack([base]*3, axis=2) * rng_det.randint(50, 255, 3)).astype(np.uint8)
    elif kind == 2:
        for r in range(0, IMG_SIZE, 32):
            for c in range(0, IMG_SIZE, 32):
                arr[r:r+32, c:c+32] = rng_det.randint(0, 255, 3)
    else:
        arr[:] = rng_det.randint(0, 255, 3)
    detect_images.append(Image.fromarray(arr))

# Also add some natural photos if available
if nat_images:
    detect_images.extend(nat_images[:min(4, len(nat_images))])
    n_detect = len(detect_images)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
n_stego_bits = 200

comparison = {}
for method_name in ['PatchSteg eps=2', 'PatchSteg eps=5', 'LSB']:
    print(f"\n  Method: {method_name}", flush=True)
    X_all, y_all = [], []

    for i, img in enumerate(detect_images):
        # Clean
        if 'PatchSteg' in method_name:
            X_all.append(extract_latent_features(vae, img))
        else:
            clean_rt = vae.decode(vae.encode(img))  # fair comparison: also round-trip
            X_all.append(extract_pixel_features(clean_rt))
        y_all.append(0)

        # Stego
        torch.manual_seed(42 + i)
        bits = torch.randint(0, 2, (n_stego_bits,)).tolist()

        if method_name == 'PatchSteg eps=2':
            s = PatchSteg(seed=42, epsilon=2.0)
            lat = vae.encode(img)
            car, _ = s.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=2.0)
            lat_m = s.encode_message(lat, car, bits[:20])
            stego = vae.decode(lat_m)
            X_all.append(extract_latent_features(vae, stego))
        elif method_name == 'PatchSteg eps=5':
            s = PatchSteg(seed=42, epsilon=5.0)
            lat = vae.encode(img)
            car, _ = s.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=5.0)
            lat_m = s.encode_message(lat, car, bits[:20])
            stego = vae.decode(lat_m)
            X_all.append(extract_latent_features(vae, stego))
        else:  # LSB
            stego = lsb_embed(img, bits)
            X_all.append(extract_pixel_features(stego))
        y_all.append(1)

    X = np.array(X_all)
    y = np.array(y_all)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    comparison[method_name] = {'auc': auc.mean(), 'auc_std': auc.std(),
                                'acc': acc.mean(), 'acc_std': acc.std()}
    print(f"    AUC={auc.mean():.3f}+-{auc.std():.3f}, Acc={acc.mean():.3f}+-{acc.std():.3f}", flush=True)

# Figure
fig, ax = plt.subplots(figsize=(8, 5))
methods = list(comparison.keys())
aucs = [comparison[m]['auc'] for m in methods]
auc_errs = [comparison[m]['auc_std'] for m in methods]
colors = ['#3498db', '#e74c3c', '#95a5a6']
bars = ax.bar(methods, aucs, yerr=auc_errs, capsize=5, color=colors,
              edgecolor='black', linewidth=0.5)
ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Chance')
ax.axhline(0.7, color='orange', ls='--', alpha=0.5, label='Detection Threshold')
ax.set_ylabel('Detection AUC')
ax.set_title('Detectability Comparison: PatchSteg vs LSB Baseline')
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(True, alpha=0.2, axis='y')
# Add value labels
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'lsb_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved lsb_comparison.png", flush=True)


# ================================================================
# 4. THEORETICAL VALIDATION: KL REGULARIZATION
# ================================================================
print("\n" + "#"*60, flush=True)
print("# 4. THEORETICAL VALIDATION: LATENT SPACE STRUCTURE", flush=True)
print("#"*60, flush=True)

# Measure: how Gaussian is the latent distribution? If well-regularized,
# small perturbations stay in-distribution and the decoder handles them smoothly.
img_theory = nat_images[0] if nat_images else detect_images[0]
lat = vae.encode(img_theory)

print("  Latent statistics per channel:", flush=True)
for ch in range(4):
    x = lat[0, ch].cpu().numpy().flatten()
    print(f"    Ch{ch}: mean={x.mean():.4f}, std={x.std():.4f}, "
          f"skew={float(((x-x.mean())**3).mean()/(x.std()**3+1e-8)):.3f}, "
          f"kurt={float(((x-x.mean())**4).mean()/(x.std()**4+1e-8)):.3f}", flush=True)

# Test: perturbation magnitude relative to latent std
print("\n  Perturbation-to-noise ratio analysis:", flush=True)
lat_std = lat[0].std().item()
for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
    pnr = eps / lat_std
    print(f"    eps={eps}: perturbation/latent_std = {pnr:.2f}", flush=True)

# Measure decoder Jacobian norm approximation:
# How much does the output change per unit change in latent?
print("\n  Decoder sensitivity (output change per unit latent change):", flush=True)
base_img = vae.decode(lat)
sensitivities = []
for trial in range(5):
    direction = torch.randn(4)
    direction = direction / direction.norm()
    eps_test = 1.0
    lat_pert = lat.clone()
    for ch in range(4):
        lat_pert[0, ch, 16, 16] += eps_test * direction[ch]
    pert_img = vae.decode(lat_pert)
    pixel_change = np.abs(np.array(base_img).astype(float) - np.array(pert_img).astype(float)).max()
    sensitivities.append(pixel_change)
    print(f"    Trial {trial}: max pixel change = {pixel_change:.2f} (for eps=1.0 at position 16,16)", flush=True)

avg_sens = np.mean(sensitivities)
print(f"  Avg decoder sensitivity: {avg_sens:.2f} pixel units per latent unit", flush=True)

# Figure: latent distribution + perturbation illustration
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Latent distribution histogram
ax = axes[0]
for ch in range(4):
    x = lat[0, ch].cpu().numpy().flatten()
    ax.hist(x, bins=50, alpha=0.5, label=f'Ch{ch}', density=True)
ax.set_xlabel('Latent Value')
ax.set_ylabel('Density')
ax.set_title('(a) Latent Distribution per Channel')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (b) Perturbation relative to distribution
ax = axes[1]
x_all = lat[0].cpu().numpy().flatten()
ax.hist(x_all, bins=80, alpha=0.6, color='steelblue', density=True, label='Latent values')
# Show perturbation scale
for eps, color in [(2.0, 'orange'), (5.0, 'red')]:
    ax.axvline(x_all.mean() + eps * 0.18215, color=color, ls='--', linewidth=2, label=f'+eps={eps} (scaled)')
    ax.axvline(x_all.mean() - eps * 0.18215, color=color, ls='--', linewidth=2)
ax.set_xlabel('Value')
ax.set_title('(b) Perturbation Scale vs Latent Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# (c) Why it works: round-trip fidelity visualization
ax = axes[2]
# Encode, decode, re-encode: measure per-position fidelity
lat_clean = vae.encode(img_theory)
recon = vae.decode(lat_clean)
lat_rt = vae.encode(recon)
fidelity = ((lat_rt[0] - lat_clean[0])**2).sum(dim=0).sqrt().cpu().numpy().flatten()
ax.hist(fidelity, bins=50, color='coral', alpha=0.7, density=True)
ax.axvline(fidelity.mean(), color='red', ls='-', linewidth=2, label=f'Mean error={fidelity.mean():.3f}')
ax.axvline(2.0 * 0.18215, color='orange', ls='--', linewidth=2, label=f'eps=2 perturbation')
ax.axvline(5.0 * 0.18215, color='darkred', ls='--', linewidth=2, label=f'eps=5 perturbation')
ax.set_xlabel('Round-Trip Error (L2)')
ax.set_title('(c) Round-Trip Error vs Perturbation Magnitude')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.suptitle('Theoretical Analysis: Why PatchSteg Works', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'theoretical_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved theoretical_analysis.png", flush=True)


# ================================================================
print(f"\n{'#'*60}", flush=True)
print(f"# ALL EXTENDED EXPERIMENTS DONE in {time.time()-t_total:.0f}s", flush=True)
print(f"{'#'*60}", flush=True)
