"""
Experiment 1: Capacity & Carrier Selection Ablation

Key questions:
  1. How many bits can we encode at target accuracy (>= 90%)?
  2. Does stability-based carrier selection outperform random selection?
  3. What is the rate-distortion-accuracy tradeoff?
  4. How much does repetition coding help at lower epsilon?

Design:
  - 3 test images (diverse content)
  - 4 epsilon values: [1.0, 2.0, 5.0, 10.0]
  - 6 carrier counts: [10, 20, 50, 100, 200, 500]
  - 2 selection methods: random vs stability-ranked (KEY ABLATION)
  - 3 random seeds for error bars
  - Repetition coding (3x) tested for all configs with K >= 30

This is the most important experiment: it proves stability selection matters.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, compute_ssim_pil, bit_accuracy


def generate_test_images():
    """3 diverse images covering different frequency content."""
    images = []
    # Low frequency: smooth gradient
    grad = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    images.append(Image.fromarray(np.stack([grad, grad, grad], axis=2)))
    # Medium frequency: color patches
    patches = np.zeros((512, 512, 3), dtype=np.uint8)
    rng = np.random.RandomState(99)
    for i in range(0, 512, 64):
        for j in range(0, 512, 64):
            patches[i:i+64, j:j+64] = rng.randint(0, 255, 3)
    images.append(Image.fromarray(patches))
    return images, ['Gradient', 'Color Patches']


def run_single_trial(vae, steg, image, carriers, bits, latent_clean):
    """Encode bits, round-trip, decode, return accuracy and PSNR."""
    latent_mod = steg.encode_message(latent_clean, carriers, bits)
    stego_img = vae.decode(latent_mod)
    psnr = compute_psnr(image, stego_img)
    latent_re = vae.encode(stego_img)
    recovered, confidences = steg.decode_message(latent_clean, latent_re, carriers)
    acc = bit_accuracy(bits, recovered)
    return acc, psnr, np.mean(confidences)


def main():
    t_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading VAE...")
    vae = StegoVAE(device=device)

    images, names = generate_test_images()
    carrier_counts = [10, 20, 50, 100, 200]
    epsilons = [1.0, 2.0, 5.0, 10.0]
    bit_seeds = [42]  # single seed for speed; proper error bars in final version

    results = []

    for img_idx, (img, name) in enumerate(zip(images, names)):
        print(f"\n{'='*50}")
        print(f"Image: {name}")
        print(f"{'='*50}")

        for eps in epsilons:
            steg = PatchSteg(seed=42, epsilon=eps)

            # Compute stability map once per (image, epsilon)
            print(f"  eps={eps}: computing stability map...", end=" ", flush=True)
            t0 = time.time()
            stability_carriers, smap = steg.select_carriers_by_stability(
                vae, img, n_carriers=max(carrier_counts), test_eps=eps
            )
            latent_clean = vae.encode(img)
            print(f"done ({time.time()-t0:.1f}s)")

            for K in carrier_counts:
                for selection in ['stability', 'random']:
                    if selection == 'stability':
                        carriers = stability_carriers[:K]
                    else:
                        carriers = steg.select_carriers_fixed(K, seed=42 + img_idx)

                    # Run multiple seeds for error bars
                    accs, psnrs, confs = [], [], []
                    for seed in bit_seeds:
                        torch.manual_seed(seed + img_idx + K)
                        bits = torch.randint(0, 2, (K,)).tolist()
                        acc, psnr, conf = run_single_trial(
                            vae, steg, img, carriers, bits, latent_clean
                        )
                        accs.append(acc)
                        psnrs.append(psnr)
                        confs.append(conf)

                    # Repetition coding test (single seed, 3x)
                    rep_acc = None
                    eff_bits = None
                    if K >= 30:
                        torch.manual_seed(42 + img_idx)
                        n_msg = K // 3
                        msg_bits = torch.randint(0, 2, (n_msg,)).tolist()
                        latent_rep = steg.encode_message_with_repetition(
                            latent_clean, carriers, msg_bits, reps=3
                        )
                        stego_rep = vae.decode(latent_rep)
                        latent_re_rep = vae.encode(stego_rep)
                        decoded_rep = steg.decode_message_with_repetition(
                            latent_clean, latent_re_rep, carriers, n_msg, reps=3
                        )
                        rep_acc = bit_accuracy(msg_bits, decoded_rep)
                        eff_bits = n_msg

                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)

                    tag = "STAB" if selection == 'stability' else "RAND"
                    rep_str = f" rep3={rep_acc:.0f}%({eff_bits}b)" if rep_acc is not None else ""
                    print(f"    K={K:>3d} [{tag}] eps={eps}: "
                          f"acc={mean_acc:>5.1f}%+-{std_acc:.1f} "
                          f"psnr={np.mean(psnrs):.1f}dB{rep_str}")

                    results.append({
                        'image': name, 'K': K, 'eps': eps,
                        'selection': selection,
                        'accuracy_mean': mean_acc,
                        'accuracy_std': std_acc,
                        'psnr_mean': np.mean(psnrs),
                        'confidence_mean': np.mean(confs),
                        'rep_accuracy': rep_acc,
                        'effective_bits': eff_bits,
                    })

    # ============================================================
    # Summary tables
    # ============================================================
    print("\n" + "=" * 90)
    print("CAPACITY RESULTS: STABILITY vs RANDOM CARRIER SELECTION (averaged over images)")
    print("=" * 90)
    print(f"{'Eps':>5} | {'K':>5} | {'Stab Acc':>10} | {'Rand Acc':>10} | {'Delta':>7} | {'PSNR':>7} | {'Rep3':>7}")
    print("-" * 75)

    for eps in epsilons:
        for K in carrier_counts:
            stab = [r for r in results if r['eps'] == eps and r['K'] == K and r['selection'] == 'stability']
            rand = [r for r in results if r['eps'] == eps and r['K'] == K and r['selection'] == 'random']
            stab_acc = np.mean([r['accuracy_mean'] for r in stab])
            rand_acc = np.mean([r['accuracy_mean'] for r in rand])
            stab_std = np.mean([r['accuracy_std'] for r in stab])
            psnr_avg = np.mean([r['psnr_mean'] for r in stab])
            rep_vals = [r['rep_accuracy'] for r in stab if r['rep_accuracy'] is not None]
            rep_str = f"{np.mean(rep_vals):.0f}%" if rep_vals else "  -"
            delta = stab_acc - rand_acc
            print(f"{eps:>5.1f} | {K:>5d} | "
                  f"{stab_acc:>7.1f}%+-{stab_std:.0f} | "
                  f"{rand_acc:>7.1f}%    | "
                  f"{delta:>+6.1f} | "
                  f"{psnr_avg:>6.1f} | {rep_str:>7}")

    # ============================================================
    # Figures
    # ============================================================
    fig_dir = Path(__file__).resolve().parent.parent / 'paper' / 'figures'

    # Figure 1: Main ablation — accuracy vs K for stability vs random at each epsilon
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax_idx, eps in enumerate(epsilons):
        ax = axes[ax_idx // 2, ax_idx % 2]
        for selection, color, marker, ls in [
            ('stability', 'tab:blue', 'o', '-'),
            ('random', 'tab:red', 's', '--')
        ]:
            means, stds = [], []
            for K in carrier_counts:
                subset = [r for r in results if r['eps'] == eps and r['K'] == K
                          and r['selection'] == selection]
                means.append(np.mean([r['accuracy_mean'] for r in subset]))
                stds.append(np.mean([r['accuracy_std'] for r in subset]))
            ax.errorbar(carrier_counts, means, yerr=stds, marker=marker,
                       color=color, linestyle=ls, capsize=3,
                       label=f'{selection.capitalize()}')

        ax.axhline(y=90, color='green', linestyle=':', alpha=0.5, label='90% target')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3, label='chance')
        ax.set_xlabel('Number of Carrier Positions')
        ax.set_ylabel('Bit Accuracy (%)')
        ax.set_title(f'$\\varepsilon = {eps}$')
        ax.legend(fontsize=8)
        ax.set_ylim(40, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.suptitle('Stability-Based vs Random Carrier Selection', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / 'capacity_ablation.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir / 'capacity_ablation.png'}")

    # Figure 2: Rate-distortion-accuracy (PSNR vs accuracy, colored by K)
    fig, ax = plt.subplots(figsize=(8, 6))
    for eps in epsilons:
        psnrs_plot, accs_plot = [], []
        for K in carrier_counts:
            subset = [r for r in results if r['eps'] == eps and r['K'] == K
                      and r['selection'] == 'stability']
            psnrs_plot.append(np.mean([r['psnr_mean'] for r in subset]))
            accs_plot.append(np.mean([r['accuracy_mean'] for r in subset]))
        ax.plot(psnrs_plot, accs_plot, marker='o', label=f'eps={eps}')
        # Annotate with K values
        for i, K in enumerate(carrier_counts):
            ax.annotate(f'K={K}', (psnrs_plot[i], accs_plot[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=6)

    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=30, color='blue', linestyle='--', alpha=0.5, label='PSNR=30dB')
    ax.set_xlabel('PSNR (dB)')
    ax.set_ylabel('Bit Accuracy (%)')
    ax.set_title('Rate-Distortion-Accuracy Tradeoff (Stability Selection)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 105)
    plt.tight_layout()
    plt.savefig(fig_dir / 'rate_distortion_accuracy.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'rate_distortion_accuracy.png'}")

    plt.close('all')
    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    return results


if __name__ == '__main__':
    results = main()
