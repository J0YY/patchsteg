"""
Experiment 4: Mechanistic Analysis of the Steganographic Channel

Key questions:
  1. Which of the 4 latent channels is most important for steganography?
  2. Hypothesis: positions with low reconstruction error (without perturbation)
     should be more stable carriers. Test this.
  3. Does the stability map correlate with image spatial frequency content?
  4. Are edge/border positions systematically weaker? (boundary effects)

Design:
  - 5 test images
  - Channel ablation: perturb only ch0, only ch1, etc.
  - Reconstruction error map vs stability map correlation (Pearson + Spearman)
  - Spatial analysis: quadrant breakdown, border vs interior
  - All with proper statistics
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import stats as scipy_stats
import time

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.analysis import channel_importance, reconstruction_error_map, spatial_frequency_map


def generate_test_images():
    images = []
    names = []
    # Gradient
    grad = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    images.append(Image.fromarray(np.stack([grad, grad, grad], axis=2)))
    names.append('Gradient')
    # Color patches
    patches = np.zeros((512, 512, 3), dtype=np.uint8)
    rng = np.random.RandomState(99)
    for i in range(0, 512, 64):
        for j in range(0, 512, 64):
            patches[i:i+64, j:j+64] = rng.randint(0, 255, 3)
    images.append(Image.fromarray(patches))
    names.append('Patches')
    # Noise
    images.append(Image.fromarray(
        np.random.RandomState(42).randint(0, 255, (512, 512, 3), dtype=np.uint8)))
    names.append('Noise')
    # Solid
    images.append(Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8)))
    names.append('Solid')
    # Checkerboard
    check = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(0, 512, 32):
        for j in range(0, 512, 32):
            if ((i // 32) + (j // 32)) % 2 == 0:
                check[i:i+32, j:j+32] = 255
    images.append(Image.fromarray(check))
    names.append('Checker')
    return images, names


def main():
    t_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading VAE...")
    vae = StegoVAE(device=device)
    steg = PatchSteg(seed=42, epsilon=5.0)

    images, names = generate_test_images()
    fig_dir = Path(__file__).resolve().parent.parent / 'paper' / 'figures'

    # ================================================================
    # 1. CHANNEL IMPORTANCE ABLATION
    # ================================================================
    print("\n" + "=" * 50)
    print("1. CHANNEL IMPORTANCE ABLATION")
    print("=" * 50)

    all_ch_results = {}
    for img, name in zip(images, names):
        print(f"  {name}:", end=" ", flush=True)
        ch_res = channel_importance(vae, steg, img, eps=5.0)
        all_ch_results[name] = ch_res
        print({ch: f"{v:.1f}%" for ch, v in ch_res.items()})

    avg_ch = {ch: np.mean([all_ch_results[n][ch] for n in names]) for ch in range(4)}
    std_ch = {ch: np.std([all_ch_results[n][ch] for n in names]) for ch in range(4)}
    print(f"\n  Average: {', '.join(f'Ch{ch}: {avg_ch[ch]:.1f}%+-{std_ch[ch]:.1f}' for ch in range(4))}")

    # Identify best and worst channel
    best_ch = max(avg_ch, key=avg_ch.get)
    worst_ch = min(avg_ch, key=avg_ch.get)
    print(f"  Best channel: {best_ch} ({avg_ch[best_ch]:.1f}%), Worst: {worst_ch} ({avg_ch[worst_ch]:.1f}%)")

    # ================================================================
    # 2. RECONSTRUCTION ERROR vs STABILITY HYPOTHESIS TEST
    # ================================================================
    print("\n" + "=" * 50)
    print("2. RECONSTRUCTION ERROR vs STABILITY CORRELATION")
    print("=" * 50)
    print("  H0: No correlation between baseline reconstruction error and carrier stability")
    print("  H1: Negative correlation (low recon error = high stability)")

    pearson_rs, spearman_rs = [], []
    all_errors, all_stabilities = [], []

    for img, name in zip(images, names):
        print(f"  {name}:", end=" ", flush=True)
        error_map = reconstruction_error_map(vae, img)
        stability_map, _ = steg.compute_stability_map(vae, img, test_eps=5.0)
        all_errors.append(error_map)
        all_stabilities.append(stability_map)

        err_flat = error_map.numpy().flatten()
        stab_flat = stability_map.numpy().flatten()
        r_p, p_p = scipy_stats.pearsonr(err_flat, stab_flat)
        r_s, p_s = scipy_stats.spearmanr(err_flat, stab_flat)
        pearson_rs.append(r_p)
        spearman_rs.append(r_s)
        print(f"Pearson r={r_p:.3f} (p={p_p:.2e}), Spearman rho={r_s:.3f} (p={p_s:.2e})")

    avg_pearson = np.mean(pearson_rs)
    avg_spearman = np.mean(spearman_rs)
    # Test if mean correlation is significantly different from 0
    t_stat, t_pval = scipy_stats.ttest_1samp(pearson_rs, 0)
    print(f"\n  Avg Pearson r: {avg_pearson:.3f} (t={t_stat:.2f}, p={t_pval:.3e})")
    print(f"  Avg Spearman rho: {avg_spearman:.3f}")
    hypothesis_supported = avg_pearson < -0.05 and t_pval < 0.05
    print(f"  Hypothesis supported: {hypothesis_supported}")

    # ================================================================
    # 3. SPATIAL ANALYSIS: BORDER vs INTERIOR
    # ================================================================
    print("\n" + "=" * 50)
    print("3. SPATIAL ANALYSIS: BORDER vs INTERIOR")
    print("=" * 50)

    border_widths = [1, 2, 4, 8]
    for bw in border_widths:
        border_stabs, interior_stabs = [], []
        for smap in all_stabilities:
            s = smap.numpy()
            border_mask = np.zeros((64, 64), dtype=bool)
            border_mask[:bw, :] = True
            border_mask[-bw:, :] = True
            border_mask[:, :bw] = True
            border_mask[:, -bw:] = True
            border_stabs.append(s[border_mask].mean())
            interior_stabs.append(s[~border_mask].mean())

        avg_border = np.mean(border_stabs)
        avg_interior = np.mean(interior_stabs)
        t_bi, p_bi = scipy_stats.ttest_rel(border_stabs, interior_stabs)
        sig = "*" if p_bi < 0.05 else ""
        print(f"  Border width={bw}: border={avg_border:.3f}, interior={avg_interior:.3f}, "
              f"diff={avg_interior-avg_border:+.3f} (p={p_bi:.3f}){sig}")

    # Quadrant analysis
    print("\n  Quadrant analysis (avg over images):")
    quadrant_names = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    quadrant_slices = [(slice(0,32), slice(0,32)), (slice(0,32), slice(32,64)),
                       (slice(32,64), slice(0,32)), (slice(32,64), slice(32,64))]
    for qname, (rs, cs) in zip(quadrant_names, quadrant_slices):
        vals = [smap[rs, cs].mean().item() for smap in all_stabilities]
        print(f"    {qname}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    # ================================================================
    # 4. SPATIAL FREQUENCY CONTENT vs STABILITY
    # ================================================================
    print("\n" + "=" * 50)
    print("4. SPATIAL FREQUENCY vs STABILITY")
    print("=" * 50)

    freq_corrs = []
    for img, name, smap in zip(images, names, all_stabilities):
        latent = vae.encode(img)
        # Local variance as a proxy for high-frequency content
        local_var = torch.zeros(64, 64)
        for ch in range(4):
            ch_data = latent[0, ch].cpu()
            # Compute local variance with 3x3 window
            padded = torch.nn.functional.pad(ch_data.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='reflect')
            unfolded = padded.unfold(2, 3, 1).unfold(3, 3, 1)  # [1,1,64,64,3,3]
            local_var += unfolded.squeeze(0).squeeze(0).var(dim=(-1,-2))

        lv_flat = local_var.numpy().flatten()
        stab_flat = smap.numpy().flatten()
        r, p = scipy_stats.pearsonr(lv_flat, stab_flat)
        freq_corrs.append(r)
        print(f"  {name}: local_var vs stability Pearson r={r:.3f} (p={p:.2e})")

    avg_freq_corr = np.mean(freq_corrs)
    print(f"\n  Avg correlation: {avg_freq_corr:.3f}")

    # ================================================================
    # FIGURES
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Channel importance bar chart with error bars
    ax = axes[0, 0]
    ch_vals = [avg_ch[ch] for ch in range(4)]
    ch_errs = [std_ch[ch] for ch in range(4)]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    ax.bar(range(4), ch_vals, yerr=ch_errs, capsize=5, color=colors,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'Channel {i}' for i in range(4)])
    ax.set_ylabel('% Positions Surviving Round-Trip')
    ax.set_title('(a) Channel Importance')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis='y')

    # 2. Reconstruction error map (example image)
    ax = axes[0, 1]
    im = ax.imshow(all_errors[1].numpy(), cmap='hot')  # Color Patches
    ax.set_title(f'(b) Reconstruction Error ({names[1]})')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. Stability map (same image)
    ax = axes[0, 2]
    im = ax.imshow(all_stabilities[1].numpy(), cmap='RdYlGn')
    ax.set_title(f'(c) Stability Map ({names[1]})')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 4. Scatter: reconstruction error vs stability
    ax = axes[1, 0]
    for idx, name in enumerate(names):
        err_flat = all_errors[idx].numpy().flatten()
        stab_flat = all_stabilities[idx].numpy().flatten()
        subsample = np.random.RandomState(42).choice(len(err_flat), 300, replace=False)
        ax.scatter(err_flat[subsample], stab_flat[subsample], alpha=0.2, s=3, label=name)
    ax.set_xlabel('Reconstruction Error (L2)')
    ax.set_ylabel('Stability Score')
    ax.set_title(f'(d) Error vs Stability (avg r={avg_pearson:.3f})')
    ax.legend(fontsize=7, markerscale=3)
    ax.grid(True, alpha=0.2)

    # 5. Per-image channel importance grouped bars
    ax = axes[1, 1]
    x = np.arange(4)
    width = 0.15
    for idx, name in enumerate(names):
        vals = [all_ch_results[name][ch] for ch in range(4)]
        ax.bar(x + idx * width - 0.3, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ch{i}' for i in range(4)])
    ax.set_ylabel('% Survived')
    ax.set_title('(e) Channel Importance by Image Type')
    ax.legend(fontsize=6)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis='y')

    # 6. Correlation summary
    ax = axes[1, 2]
    corr_labels = ['Recon Error\nvs Stability', 'Local Variance\nvs Stability']
    corr_vals = [avg_pearson, avg_freq_corr]
    corr_stds = [np.std(pearson_rs), np.std(freq_corrs)]
    bar_colors = ['coral' if v < 0 else 'steelblue' for v in corr_vals]
    ax.bar(corr_labels, corr_vals, yerr=corr_stds, capsize=5,
           color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Pearson Correlation (r)')
    ax.set_title('(f) Feature Correlations with Stability')
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('Mechanistic Analysis of VAE Steganographic Channel', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / 'mechanistic_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir / 'mechanistic_analysis.png'}")
    plt.close('all')

    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    return {
        'channel_importance': avg_ch,
        'channel_importance_std': std_ch,
        'best_channel': best_ch,
        'worst_channel': worst_ch,
        'recon_error_corr': avg_pearson,
        'recon_error_corr_p': t_pval,
        'freq_corr': avg_freq_corr,
        'hypothesis_supported': hypothesis_supported,
    }


if __name__ == '__main__':
    stats = main()
