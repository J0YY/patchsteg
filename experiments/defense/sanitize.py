#!/usr/bin/env python
"""
Defense (a): Image sanitization and channel capacity limitation.

Techniques:
1. VAE round-trip sanitization  — encode->decode cycle acts as a manifold projector,
   absorbing off-manifold perturbations. Repeating amplifies the effect.
2. Noisy round-trip             — add Gaussian noise to latent before decode;
   randomized sanitization that destroys carrier bit values stochastically.
3. Latent quantization          — reduce latent precision to n_bits; limits the
   information channel capacity available to a steganographer.
4. Latent spatial smoothing     — Gaussian blur across latent spatial dims;
   destroys positional encoding at specific carrier (r, c) sites.

Metrics per sanitizer:
- Bit accuracy after defense (lower = more effective defense)
- PSNR and SSIM vs original clean image (higher = less image damage)

Generates: defense_sanitize_patchsteg.png, defense_sanitize_cdf.png
"""
import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time
from scipy.ndimage import gaussian_filter

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.cdf_steganography import CDFPatchSteg
from core.metrics import compute_psnr, compute_ssim_pil, bit_accuracy

IMG_SIZE = 256
FIG_DIR = Path(__file__).resolve().parent.parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


# ─── SANITIZERS ──────────────────────────────────────────────────────────────

class VaeRoundTripSanitizer:
    """
    Trusted re-encoding: image -> latent -> decode -> re-encode.

    The VAE decoder maps latents onto its learned image manifold. Modifications
    that lie off this manifold (e.g. epsilon perturbations at carrier positions)
    are partially or fully absorbed. Each additional trip amplifies suppression.
    """
    def __init__(self, n_trips=1):
        self.n_trips = n_trips

    def sanitize(self, vae, image: Image.Image) -> Image.Image:
        img = image
        for _ in range(self.n_trips):
            _, img = vae.round_trip(img)
        return img

    @property
    def name(self):
        return f'RoundTrip x{self.n_trips}'


class NoisyRoundTripSanitizer:
    """
    Randomized sanitization: add Gaussian noise to latent before decode.

    Noise is added in latent space before decoding; the VAE decoder then
    smooths it, keeping perceptual quality high while disrupting carrier
    values. Analogous to randomized smoothing in adversarial robustness.
    """
    def __init__(self, noise_std=0.5, seed=0):
        self.noise_std = noise_std
        self.seed = seed

    def sanitize(self, vae, image: Image.Image) -> Image.Image:
        latent = vae.encode(image)
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        noise = torch.randn_like(latent, generator=rng) * self.noise_std
        return vae.decode(latent + noise)

    @property
    def name(self):
        return f'NoisyRT sigma={self.noise_std}'


class LatentQuantizationSanitizer:
    """
    Channel capacity limitation via uniform quantization of latent values.

    Reduces effective precision of each latent to `n_bits` uniform levels.
    This caps the mutual information between sender and receiver, making
    fine-grained epsilon encodings fall below the quantization floor.
    Fewer bits = stronger defense at the cost of image fidelity.
    """
    def __init__(self, n_bits=6):
        self.n_bits = n_bits

    def sanitize(self, vae, image: Image.Image) -> Image.Image:
        latent = vae.encode(image)
        lo = latent.min().item()
        hi = latent.max().item()
        levels = 2 ** self.n_bits
        latent_q = ((latent - lo) / (hi - lo + 1e-8) * (levels - 1)).round()
        latent_q = latent_q / (levels - 1) * (hi - lo) + lo
        return vae.decode(latent_q)

    @property
    def name(self):
        return f'Quantize {self.n_bits}b'


class LatentSmoothingSanitizer:
    """
    Gaussian blur in latent space (spatial smoothing per channel).

    Blurs each of the 4 latent channels independently with a Gaussian kernel.
    This destroys narrow positional spikes at carrier (r,c) sites while
    preserving coarse spatial structure. Larger sigma = stronger suppression.
    """
    def __init__(self, sigma=0.8):
        self.sigma = sigma

    def sanitize(self, vae, image: Image.Image) -> Image.Image:
        latent = vae.encode(image)
        lat_np = latent[0].cpu().numpy()
        lat_smooth = np.stack([
            gaussian_filter(lat_np[ch], sigma=self.sigma) for ch in range(4)
        ])
        latent_smooth = torch.from_numpy(lat_smooth).unsqueeze(0).to(latent.device)
        return vae.decode(latent_smooth)

    @property
    def name(self):
        return f'Smooth sigma={self.sigma}'


# ─── DATA & STEGO HELPERS ────────────────────────────────────────────────────

def get_natural_images(n=8):
    from torchvision.datasets import CIFAR10
    print("  Loading CIFAR-10...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=True, download=True)
    picked, seen = [], set()
    for img, label in ds:
        if label not in seen and len(picked) < n:
            seen.add(label)
            picked.append(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    return picked


def make_stego(vae, images, method='PatchSteg'):
    """Encode 20-bit random messages; return stegos, carriers, bits."""
    stegos, carriers_all, bits_all = [], [], []
    for i, img in enumerate(images):
        torch.manual_seed(42 + i)
        bits = torch.randint(0, 2, (20,)).tolist()
        latent = vae.encode(img)
        if method == 'PatchSteg':
            s = PatchSteg(seed=42, epsilon=5.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20)
            lat_m = s.encode_message(latent, carriers, bits)
        elif method == 'CDF':
            s = CDFPatchSteg(seed=42, sigma=1.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20)
            lat_m = s.encode_message(latent, carriers, bits)
        else:
            raise ValueError(method)
        stegos.append(vae.decode(lat_m))
        carriers_all.append(carriers)
        bits_all.append(bits)
    return stegos, carriers_all, bits_all


def decode_bits(vae, clean_imgs, sanitized_imgs, carriers_all, bits_all, method):
    """Decode from sanitized images; return mean bit accuracy."""
    accs = []
    for clean_img, san_img, carriers, bits in zip(
            clean_imgs, sanitized_imgs, carriers_all, bits_all):
        if method == 'PatchSteg':
            s = PatchSteg(seed=42, epsilon=5.0)
            lat_clean = vae.encode(clean_img)
            lat_san = vae.encode(san_img)
            decoded, _ = s.decode_message(lat_clean, lat_san, carriers)
        else:
            s = CDFPatchSteg(seed=42, sigma=1.0)
            decoded, _ = s.decode_message(vae, san_img, carriers)
        accs.append(bit_accuracy(bits, decoded))
    return float(np.mean(accs))


def eval_sanitizer(vae, sanitizer, stego_imgs, clean_imgs, carriers_all, bits_all, method):
    sanitized = [sanitizer.sanitize(vae, img) for img in stego_imgs]
    psnrs = [compute_psnr(c, s) for c, s in zip(clean_imgs, sanitized)]
    ssims = [compute_ssim_pil(c, s) for c, s in zip(clean_imgs, sanitized)]
    bacc = decode_bits(vae, clean_imgs, sanitized, carriers_all, bits_all, method)
    return {
        'psnr': float(np.mean(psnrs)),
        'ssim': float(np.mean(ssims)),
        'bit_acc': bacc,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_total = time.time()
    print("=" * 60, flush=True)
    print("DEFENSE (a): SANITIZATION & CHANNEL CAPACITY LIMITING", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading VAE...", flush=True)
    vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
    print("Loading images...", flush=True)
    images = get_natural_images(8)

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

    all_results = {}

    for steg_method in ['PatchSteg', 'CDF']:
        print(f"\n{'─'*50}", flush=True)
        print(f"Steganography: {steg_method}", flush=True)
        print(f"{'─'*50}", flush=True)

        stego_imgs, carriers_all, bits_all = make_stego(vae, images, steg_method)

        baseline_acc = decode_bits(vae, images, stego_imgs, carriers_all, bits_all, steg_method)
        print(f"  Baseline bit accuracy (no defense): {baseline_acc:.1f}%", flush=True)

        method_results = {}
        for san in sanitizers:
            m = eval_sanitizer(vae, san, stego_imgs, images, carriers_all, bits_all, steg_method)
            method_results[san.name] = m
            print(f"  {san.name:>22s}:  PSNR={m['psnr']:.1f}dB  "
                  f"SSIM={m['ssim']:.3f}  BitAcc={m['bit_acc']:.1f}%", flush=True)

        all_results[steg_method] = method_results

    # ─── FIGURES ─────────────────────────────────────────────────────────────
    for steg_method in ['PatchSteg', 'CDF']:
        mr = all_results[steg_method]
        names = list(mr.keys())
        psnrs = [mr[n]['psnr'] for n in names]
        ssims = [mr[n]['ssim'] for n in names]
        baccs = [mr[n]['bit_acc'] for n in names]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Sanitization Defense vs {steg_method}', fontsize=13, fontweight='bold')
        x = range(len(names))

        axes[0].barh(x, psnrs, color='steelblue', edgecolor='k', linewidth=0.5)
        axes[0].set_yticks(x)
        axes[0].set_yticklabels(names, fontsize=8)
        axes[0].set_xlabel('PSNR (dB)')
        axes[0].set_title('Image Quality (higher = better)')
        axes[0].axvline(30, color='r', linestyle='--', alpha=0.6, label='30 dB')
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.2, axis='x')

        axes[1].barh(x, ssims, color='seagreen', edgecolor='k', linewidth=0.5)
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(names, fontsize=8)
        axes[1].set_xlabel('SSIM')
        axes[1].set_title('Structural Similarity (higher = better)')
        axes[1].axvline(0.9, color='r', linestyle='--', alpha=0.6, label='0.9')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.2, axis='x')

        axes[2].barh(x, baccs, color='tomato', edgecolor='k', linewidth=0.5)
        axes[2].set_yticks(x)
        axes[2].set_yticklabels(names, fontsize=8)
        axes[2].set_xlabel('Bit Accuracy (%)')
        axes[2].set_title('Residual Stego Signal (lower = better defense)')
        axes[2].axvline(50, color='g', linestyle='--', alpha=0.6, label='Chance 50%')
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.2, axis='x')

        plt.tight_layout()
        fname = FIG_DIR / f'defense_sanitize_{steg_method.lower()}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved {fname.name}", flush=True)

    print(f"\nTotal time: {time.time()-t_total:.0f}s", flush=True)
