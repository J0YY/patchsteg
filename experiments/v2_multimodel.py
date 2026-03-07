#!/usr/bin/env python
"""
Experiment: Multi-model generality.
Tests PatchSteg on multiple VAE backbones to prove the phenomenon is general.
Models tested:
  1. stabilityai/sd-vae-ft-mse (original, SD 1.x)
  2. stabilityai/sd-vae-ft-ema (EMA variant)
  3. stabilityai/sdxl-vae (SDXL VAE, if downloadable)
  4. Cross-model: encode with one VAE, decode with another
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
from diffusers import AutoencoderKL
from torchvision import transforms

from core.steganography import PatchSteg
from core.metrics import compute_psnr, bit_accuracy

FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
IMG_SIZE = 256

torch.manual_seed(42)
np.random.seed(42)


class GenericVAE:
    """Wrapper for any HuggingFace VAE."""
    def __init__(self, model_id, device='cpu', image_size=256):
        self.device = device
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.model_id = model_id
        print(f"  Loading {model_id}...", flush=True)
        self.vae = AutoencoderKL.from_pretrained(model_id).to(device).eval()
        self.scaling_factor = self.vae.config.scaling_factor
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    @torch.no_grad()
    def encode(self, image):
        if isinstance(image, Image.Image):
            x = self.to_tensor(image).unsqueeze(0).to(self.device)
        else:
            x = image
        latent = self.vae.encode(x).latent_dist.mean
        latent = latent * self.scaling_factor
        return latent

    @torch.no_grad()
    def decode(self, latent):
        latent_scaled = latent / self.scaling_factor
        pixels = self.vae.decode(latent_scaled).sample
        pixels = (pixels.clamp(-1, 1) + 1) / 2
        pixels = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(pixels)


def make_test_images(n=20):
    """Create diverse test images for evaluation."""
    images = []
    rng = np.random.RandomState(42)
    for i in range(n):
        kind = i % 5
        arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if kind == 0:  # random noise
            arr = rng.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        elif kind == 1:  # color patches
            for r in range(0, IMG_SIZE, 32):
                for c in range(0, IMG_SIZE, 32):
                    arr[r:r+32, c:c+32] = rng.randint(0, 255, 3)
        elif kind == 2:  # gradient
            t = np.linspace(0, 1, IMG_SIZE)
            for ch in range(3):
                arr[:, :, ch] = (np.outer(t, np.ones(IMG_SIZE)) * rng.randint(100, 255)).astype(np.uint8)
        elif kind == 3:  # checkerboard
            block = 16
            for r in range(0, IMG_SIZE, block):
                for c in range(0, IMG_SIZE, block):
                    if ((r // block) + (c // block)) % 2 == 0:
                        arr[r:r+block, c:c+block] = rng.randint(150, 255, 3)
                    else:
                        arr[r:r+block, c:c+block] = rng.randint(0, 100, 3)
        else:  # solid with texture
            base = rng.randint(50, 200, 3)
            arr[:] = base
            noise = rng.randint(-20, 20, (IMG_SIZE, IMG_SIZE, 3))
            arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def test_model(vae, images, model_name, epsilons=[2.0, 5.0], n_carriers=20):
    """Test PatchSteg on a given VAE model across images."""
    results = []
    for eps in epsilons:
        steg = PatchSteg(seed=42, epsilon=eps)
        accs, psnrs = [], []
        for i, img in enumerate(images):
            lat = vae.encode(img)
            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=n_carriers, test_eps=eps)
            torch.manual_seed(42 + i)
            bits = torch.randint(0, 2, (n_carriers,)).tolist()
            lat_m = steg.encode_message(lat, carriers, bits)
            st = vae.decode(lat_m)
            psnr = compute_psnr(img, st)
            lat_re = vae.encode(st)
            rec, _ = steg.decode_message(lat, lat_re, carriers)
            acc = bit_accuracy(bits, rec)
            accs.append(acc)
            psnrs.append(psnr)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)
        results.append({
            'model': model_name, 'eps': eps,
            'acc_mean': mean_acc, 'acc_std': std_acc,
            'psnr_mean': mean_psnr, 'psnr_std': std_psnr,
            'n_images': len(images)
        })
        print(f"  {model_name:30s} eps={eps}: acc={mean_acc:.1f}+-{std_acc:.1f}% "
              f"psnr={mean_psnr:.1f}+-{std_psnr:.1f}dB (n={len(images)})", flush=True)
    return results


# ================================================================
print("=" * 70, flush=True)
print("MULTI-MODEL GENERALITY EXPERIMENT", flush=True)
print("=" * 70, flush=True)
t0 = time.time()

images = make_test_images(10)
print(f"Created {len(images)} test images", flush=True)

# Model list — try each, skip if download fails
MODEL_IDS = [
    ("stabilityai/sd-vae-ft-mse", "SD-VAE-MSE"),
    ("stabilityai/sd-vae-ft-ema", "SD-VAE-EMA"),
]

# Try SDXL VAE (skip if download is too slow)
try:
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("SDXL download too slow")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60s timeout
    _ = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                       torch_dtype=torch.float32)
    signal.alarm(0)
    del _
    MODEL_IDS.append(("madebyollin/sdxl-vae-fp16-fix", "SDXL-VAE"))
    print("  SDXL-VAE available", flush=True)
except Exception as e:
    signal.alarm(0)
    print(f"  SDXL-VAE not available: {e}", flush=True)

all_results = []

for model_id, model_name in MODEL_IDS:
    print(f"\n--- Testing {model_name} ---", flush=True)
    try:
        vae = GenericVAE(model_id, device='cpu', image_size=IMG_SIZE)
        results = test_model(vae, images, model_name)
        all_results.extend(results)
        del vae
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

# ================================================================
# Cross-model test: encode with model A, decode with model B
# ================================================================
print(f"\n{'='*70}", flush=True)
print("CROSS-MODEL ROBUSTNESS", flush=True)
print("="*70, flush=True)

if len(MODEL_IDS) >= 2:
    cross_results = []
    vae_a = GenericVAE(MODEL_IDS[0][0], device='cpu', image_size=IMG_SIZE)
    vae_b = GenericVAE(MODEL_IDS[1][0], device='cpu', image_size=IMG_SIZE)

    steg = PatchSteg(seed=42, epsilon=5.0)
    cross_accs = []
    for i, img in enumerate(images[:5]):
        lat_a = vae_a.encode(img)
        carriers, _ = steg.select_carriers_by_stability(vae_a, img, n_carriers=20, test_eps=5.0)
        torch.manual_seed(42 + i)
        bits = torch.randint(0, 2, (20,)).tolist()
        lat_m = steg.encode_message(lat_a, carriers, bits)
        stego = vae_a.decode(lat_m)
        # Receiver uses MODEL B
        lat_b_clean = vae_b.encode(img)
        lat_b_recv = vae_b.encode(stego)
        rec, _ = steg.decode_message(lat_b_clean, lat_b_recv, carriers)
        acc = bit_accuracy(bits, rec)
        cross_accs.append(acc)

    print(f"  Cross-model ({MODEL_IDS[0][1]} -> {MODEL_IDS[1][1]}): "
          f"acc={np.mean(cross_accs):.1f}+-{np.std(cross_accs):.1f}%", flush=True)
    all_results.append({
        'model': f"Cross: {MODEL_IDS[0][1]}->{MODEL_IDS[1][1]}",
        'eps': 5.0,
        'acc_mean': np.mean(cross_accs), 'acc_std': np.std(cross_accs),
        'psnr_mean': 0, 'psnr_std': 0, 'n_images': len(cross_accs)
    })

    # Also test reverse
    cross_accs_rev = []
    for i, img in enumerate(images[:5]):
        lat_b = vae_b.encode(img)
        carriers, _ = steg.select_carriers_by_stability(vae_b, img, n_carriers=20, test_eps=5.0)
        torch.manual_seed(42 + i)
        bits = torch.randint(0, 2, (20,)).tolist()
        lat_m = steg.encode_message(lat_b, carriers, bits)
        stego = vae_b.decode(lat_m)
        lat_a_clean = vae_a.encode(img)
        lat_a_recv = vae_a.encode(stego)
        rec, _ = steg.decode_message(lat_a_clean, lat_a_recv, carriers)
        acc = bit_accuracy(bits, rec)
        cross_accs_rev.append(acc)

    print(f"  Cross-model ({MODEL_IDS[1][1]} -> {MODEL_IDS[0][1]}): "
          f"acc={np.mean(cross_accs_rev):.1f}+-{np.std(cross_accs_rev):.1f}%", flush=True)
    all_results.append({
        'model': f"Cross: {MODEL_IDS[1][1]}->{MODEL_IDS[0][1]}",
        'eps': 5.0,
        'acc_mean': np.mean(cross_accs_rev), 'acc_std': np.std(cross_accs_rev),
        'psnr_mean': 0, 'psnr_std': 0, 'n_images': len(cross_accs_rev)
    })

    del vae_a, vae_b
    import gc; gc.collect()

# ================================================================
# Figure: multi-model comparison
# ================================================================
print(f"\nGenerating figure...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) Accuracy by model and epsilon
same_model = [r for r in all_results if 'Cross' not in r['model']]
models = list(dict.fromkeys(r['model'] for r in same_model))
x = np.arange(len(models))
width = 0.35

eps2 = [next((r for r in same_model if r['model'] == m and r['eps'] == 2.0), None) for m in models]
eps5 = [next((r for r in same_model if r['model'] == m and r['eps'] == 5.0), None) for m in models]

ax = axes[0]
if any(e is not None for e in eps2):
    vals = [r['acc_mean'] if r else 0 for r in eps2]
    errs = [r['acc_std'] if r else 0 for r in eps2]
    ax.bar(x - width/2, vals, width, yerr=errs, label='ε=2.0', capsize=4, color='#3498db')
if any(e is not None for e in eps5):
    vals = [r['acc_mean'] if r else 0 for r in eps5]
    errs = [r['acc_std'] if r else 0 for r in eps5]
    ax.bar(x + width/2, vals, width, yerr=errs, label='ε=5.0', capsize=4, color='#e74c3c')

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title('(a) PatchSteg Accuracy Across VAE Models')
ax.legend()
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(50, color='gray', ls=':', alpha=0.5, label='Chance')

# (b) Cross-model results
ax = axes[1]
cross = [r for r in all_results if 'Cross' in r['model']]
if cross:
    labels = [r['model'].replace('Cross: ', '') for r in cross]
    vals = [r['acc_mean'] for r in cross]
    errs = [r['acc_std'] for r in cross]
    colors = ['#2ecc71', '#f39c12']
    bars = ax.bar(range(len(cross)), vals, yerr=errs, capsize=5,
                  color=colors[:len(cross)], edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(cross)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title('(b) Cross-Model Decode (ε=5.0)')
ax.set_ylim(0, 105)
ax.axhline(50, color='gray', ls=':', alpha=0.5)
ax.grid(True, alpha=0.2, axis='y')

plt.suptitle('Multi-Model Generality: PatchSteg Works Across VAE Architectures', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'multimodel.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved multimodel.png", flush=True)

print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
