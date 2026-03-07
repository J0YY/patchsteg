#!/usr/bin/env python
"""
Experiment: Deployment-relevant robustness.
Tests the channel under realistic distortion chains:
  1. Heavy JPEG (Q=10, Q=20, Q=30)
  2. Aggressive resize (75%, 50%, 25% then back)
  3. Crop + pad (center crop 80%, 60%)
  4. Distortion chains: JPEG->resize->JPEG
  5. Screenshot simulation (resize + JPEG + slight color shift)
  6. Re-encoding through the VAE (simulating a safety pipeline)
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
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import time
import io

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, bit_accuracy

FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
IMG_SIZE = 256

torch.manual_seed(42)
np.random.seed(42)


def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def resize_attack(img, scale):
    w, h = img.size
    small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def center_crop_pad(img, crop_frac):
    w, h = img.size
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    result = Image.new('RGB', (w, h), (128, 128, 128))
    result.paste(cropped, (left, top))
    return result


def add_noise(img, sigma):
    arr = np.array(img).astype(float)
    noise = np.random.randn(*arr.shape) * sigma * 255
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def screenshot_sim(img):
    """Simulate screenshot: slight resize, JPEG, color shift."""
    w, h = img.size
    img2 = img.resize((int(w * 0.98), int(h * 0.98)), Image.BILINEAR)
    img2 = img2.resize((w, h), Image.BILINEAR)
    img2 = jpeg_compress(img2, 85)
    enhancer = ImageEnhance.Brightness(img2)
    img2 = enhancer.enhance(1.02)
    return img2


def vae_reencode(img, vae):
    """Simulate safety pipeline: re-encode through VAE."""
    lat = vae.encode(img)
    return vae.decode(lat)


def make_test_images(n=20):
    rng = np.random.RandomState(42)
    images = []
    for i in range(n):
        kind = i % 4
        arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if kind == 0:
            arr = rng.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        elif kind == 1:
            for r in range(0, IMG_SIZE, 32):
                for c in range(0, IMG_SIZE, 32):
                    arr[r:r+32, c:c+32] = rng.randint(0, 255, 3)
        elif kind == 2:
            block = 16
            for r in range(0, IMG_SIZE, block):
                for c in range(0, IMG_SIZE, block):
                    if ((r//block)+(c//block))%2 == 0:
                        arr[r:r+block, c:c+block] = rng.randint(150, 255, 3)
                    else:
                        arr[r:r+block, c:c+block] = rng.randint(0, 100, 3)
        else:
            base = rng.randint(50, 200, 3)
            arr[:] = base
            noise = rng.randint(-30, 30, (IMG_SIZE, IMG_SIZE, 3))
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


# ================================================================
print("=" * 70, flush=True)
print("DEPLOYMENT ROBUSTNESS EXPERIMENT", flush=True)
print("=" * 70, flush=True)
t0 = time.time()

vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
images = make_test_images(10)
print(f"Created {len(images)} test images", flush=True)

# Define distortion attacks
ATTACKS = [
    ("None", lambda img: img),
    ("JPEG Q=50", lambda img: jpeg_compress(img, 50)),
    ("JPEG Q=30", lambda img: jpeg_compress(img, 30)),
    ("JPEG Q=10", lambda img: jpeg_compress(img, 10)),
    ("Resize 75%", lambda img: resize_attack(img, 0.75)),
    ("Resize 50%", lambda img: resize_attack(img, 0.50)),
    ("Resize 25%", lambda img: resize_attack(img, 0.25)),
    ("Crop 80%", lambda img: center_crop_pad(img, 0.80)),
    ("Crop 60%", lambda img: center_crop_pad(img, 0.60)),
    ("Noise σ=0.02", lambda img: add_noise(img, 0.02)),
    ("Noise σ=0.05", lambda img: add_noise(img, 0.05)),
    ("Noise σ=0.10", lambda img: add_noise(img, 0.10)),
    ("Blur r=1", lambda img: img.filter(ImageFilter.GaussianBlur(1))),
    ("Blur r=2", lambda img: img.filter(ImageFilter.GaussianBlur(2))),
    ("Screenshot sim", screenshot_sim),
    ("VAE re-encode", lambda img: vae_reencode(img, vae)),
    ("JPEG->Resize->JPEG", lambda img: jpeg_compress(resize_attack(jpeg_compress(img, 75), 0.75), 75)),
    ("JPEG->Noise->JPEG", lambda img: jpeg_compress(add_noise(jpeg_compress(img, 75), 0.02), 75)),
]

all_results = {}

for eps in [2.0, 5.0]:
    steg = PatchSteg(seed=42, epsilon=eps)
    print(f"\n--- epsilon = {eps} ---", flush=True)

    for atk_name, atk_fn in ATTACKS:
        accs = []
        for i, img in enumerate(images):
            lat = vae.encode(img)
            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=eps)
            torch.manual_seed(42 + i)
            bits = torch.randint(0, 2, (20,)).tolist()
            lat_m = steg.encode_message(lat, carriers, bits)
            stego = vae.decode(lat_m)

            # Apply attack
            attacked = atk_fn(stego)

            # Decode
            lat_clean = vae.encode(img)
            lat_recv = vae.encode(attacked)
            rec, _ = steg.decode_message(lat_clean, lat_recv, carriers)
            acc = bit_accuracy(bits, rec)
            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        key = (eps, atk_name)
        all_results[key] = {'mean': mean_acc, 'std': std_acc}
        survived = "OK" if mean_acc > 70 else "DEGRADED" if mean_acc > 55 else "FAILED"
        print(f"  {atk_name:25s}: {mean_acc:5.1f}+-{std_acc:4.1f}%  [{survived}]", flush=True)

# ================================================================
# Figure
# ================================================================
print(f"\nGenerating figure...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax_idx, eps in enumerate([2.0, 5.0]):
    ax = axes[ax_idx]
    names = [a[0] for a in ATTACKS]
    means = [all_results[(eps, n)]['mean'] for n in names]
    stds = [all_results[(eps, n)]['std'] for n in names]

    colors = []
    for m in means:
        if m > 85:
            colors.append('#2ecc71')
        elif m > 65:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')

    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=stds, capsize=3, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Bit Accuracy (%)')
    ax.set_title(f'ε = {eps}')
    ax.axvline(50, color='gray', ls=':', alpha=0.5)
    ax.axvline(90, color='green', ls='--', alpha=0.3)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis='x')

plt.suptitle('Deployment Robustness: PatchSteg Under Realistic Distortions\n'
             '(Green: >85%, Yellow: 65-85%, Red: <65%)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'deployment_robustness.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved deployment_robustness.png", flush=True)

print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
