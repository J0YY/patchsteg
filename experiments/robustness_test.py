"""
Experiment 2: Robustness to Real-World Image Distortions

Key questions:
  1. Does the channel survive JPEG compression (the #1 real-world transform)?
  2. At what noise level does the channel break?
  3. Does resizing destroy the channel?
  4. Does repetition coding help recover from distortions?

Design:
  - 3 test images
  - 7 distortion types including no-distortion baseline
  - 2 epsilon values: [2.0, 5.0]
  - Both raw and 3x-repetition-coded decoding
  - 20 stability-selected carriers (60 for repetition)
  - 3 message seeds for error bars
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import io
import time

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, bit_accuracy


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.array(img).astype(float) / 255.0
    noise = np.random.randn(*arr.shape) * sigma
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def resize_and_back(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


DISTORTIONS = {
    'None':       lambda img: img,
    'JPEG Q=95':  lambda img: jpeg_compress(img, 95),
    'JPEG Q=75':  lambda img: jpeg_compress(img, 75),
    'JPEG Q=50':  lambda img: jpeg_compress(img, 50),
    'Noise 0.01': lambda img: add_gaussian_noise(img, 0.01),
    'Noise 0.05': lambda img: add_gaussian_noise(img, 0.05),
    'Resize 50%': lambda img: resize_and_back(img, 0.5),
    'Resize 25%': lambda img: resize_and_back(img, 0.25),
}


def generate_test_images():
    images = []
    grad = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    images.append(Image.fromarray(np.stack([grad, grad, grad], axis=2)))
    patches = np.zeros((512, 512, 3), dtype=np.uint8)
    rng = np.random.RandomState(99)
    for i in range(0, 512, 64):
        for j in range(0, 512, 64):
            patches[i:i+64, j:j+64] = rng.randint(0, 255, 3)
    images.append(Image.fromarray(patches))
    return images, ['Gradient', 'Color Patches']


def main():
    t_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("Loading VAE...")
    vae = StegoVAE(device=device)

    images, names = generate_test_images()
    epsilons = [2.0, 5.0]
    n_carriers_raw = 20
    n_carriers_rep = 60  # 20 effective bits with 3x repetition
    bit_seeds = [42]  # single seed for speed
    results = []

    for img_idx, (img, name) in enumerate(zip(images, names)):
        print(f"\n{'='*50}")
        print(f"Image: {name}")
        print(f"{'='*50}")

        for eps in epsilons:
            steg = PatchSteg(seed=42, epsilon=eps)

            # Get stable carriers (enough for repetition coding)
            carriers_all, _ = steg.select_carriers_by_stability(
                vae, img, n_carriers=n_carriers_rep, test_eps=eps
            )
            latent_clean = vae.encode(img)

            for dist_name, dist_fn in DISTORTIONS.items():
                accs_raw, accs_rep = [], []

                for seed in bit_seeds:
                    torch.manual_seed(seed + img_idx)
                    bits_raw = torch.randint(0, 2, (n_carriers_raw,)).tolist()

                    # --- Raw encoding (20 carriers, 20 bits) ---
                    carriers_raw = carriers_all[:n_carriers_raw]
                    latent_mod = steg.encode_message(latent_clean, carriers_raw, bits_raw)
                    stego_img = vae.decode(latent_mod)
                    distorted = dist_fn(stego_img)
                    latent_re = vae.encode(distorted)
                    recovered, _ = steg.decode_message(latent_clean, latent_re, carriers_raw)
                    accs_raw.append(bit_accuracy(bits_raw, recovered))

                    # --- Repetition-coded (60 carriers, 20 effective bits) ---
                    latent_rep = steg.encode_message_with_repetition(
                        latent_clean, carriers_all[:n_carriers_rep], bits_raw, reps=3
                    )
                    stego_rep = vae.decode(latent_rep)
                    distorted_rep = dist_fn(stego_rep)
                    latent_re_rep = vae.encode(distorted_rep)
                    decoded_rep = steg.decode_message_with_repetition(
                        latent_clean, latent_re_rep, carriers_all[:n_carriers_rep],
                        n_carriers_raw, reps=3
                    )
                    accs_rep.append(bit_accuracy(bits_raw, decoded_rep))

                raw_mean, raw_std = np.mean(accs_raw), np.std(accs_raw)
                rep_mean, rep_std = np.mean(accs_rep), np.std(accs_rep)

                print(f"  eps={eps} {dist_name:>12s}: "
                      f"raw={raw_mean:>5.1f}%+-{raw_std:.0f}  "
                      f"rep3={rep_mean:>5.1f}%+-{rep_std:.0f}")

                results.append({
                    'image': name, 'eps': eps, 'distortion': dist_name,
                    'raw_acc_mean': raw_mean, 'raw_acc_std': raw_std,
                    'rep_acc_mean': rep_mean, 'rep_acc_std': rep_std,
                })

    # ============================================================
    # Summary table
    # ============================================================
    print("\n" + "=" * 85)
    print("ROBUSTNESS RESULTS (averaged over images)")
    print("=" * 85)
    for eps in epsilons:
        print(f"\n  epsilon = {eps}")
        print(f"  {'Distortion':>12} | {'Raw Acc':>12} | {'Rep3 Acc':>12} | {'Delta':>7}")
        print(f"  {'-'*55}")
        for dist_name in DISTORTIONS:
            subset = [r for r in results if r['eps'] == eps and r['distortion'] == dist_name]
            raw = np.mean([r['raw_acc_mean'] for r in subset])
            raw_s = np.mean([r['raw_acc_std'] for r in subset])
            rep = np.mean([r['rep_acc_mean'] for r in subset])
            rep_s = np.mean([r['rep_acc_std'] for r in subset])
            delta = rep - raw
            print(f"  {dist_name:>12} | {raw:>6.1f}%+-{raw_s:>3.0f} | "
                  f"{rep:>6.1f}%+-{rep_s:>3.0f} | {delta:>+6.1f}")

    # ============================================================
    # Figure: grouped bar chart — raw vs rep3 at eps=5.0 and eps=2.0
    # ============================================================
    fig_dir = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
    dist_names_list = list(DISTORTIONS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, eps in enumerate(epsilons):
        ax = axes[ax_idx]
        raw_vals, rep_vals = [], []
        for d in dist_names_list:
            subset = [r for r in results if r['eps'] == eps and r['distortion'] == d]
            raw_vals.append(np.mean([r['raw_acc_mean'] for r in subset]))
            rep_vals.append(np.mean([r['rep_acc_mean'] for r in subset]))

        x = np.arange(len(dist_names_list))
        w = 0.35
        ax.bar(x - w/2, raw_vals, w, label='Raw (20 bits)', color='steelblue',
               edgecolor='black', linewidth=0.5)
        ax.bar(x + w/2, rep_vals, w, label='Rep3 (20 eff. bits)', color='coral',
               edgecolor='black', linewidth=0.5)
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90%')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylabel('Bit Accuracy (%)')
        ax.set_title(f'$\\varepsilon = {eps}$')
        ax.set_xticks(x)
        ax.set_xticklabels(dist_names_list, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('Robustness: Bit Accuracy Under Distortions (Stability-Selected Carriers)', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / 'robustness_bars.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir / 'robustness_bars.png'}")
    plt.close('all')
    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    return results


if __name__ == '__main__':
    results = main()
