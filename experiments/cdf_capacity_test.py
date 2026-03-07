#!/usr/bin/env python
"""
Phase 1: CDF PatchSteg capacity, accuracy, and detectability tests.

Tests:
1. Bit accuracy at various carrier counts
2. PSNR comparison with original PatchSteg
3. KS test for distribution preservation
4. Detection AUC comparison (CDF vs original)

Generates: cdf_capacity_curve.png, cdf_detectability.png, cdf_distribution.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.cdf_steganography import CDFPatchSteg
from core.metrics import compute_psnr, bit_accuracy

IMG_SIZE = 256
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def get_natural_images(n=8):
    """Get natural photos from CIFAR-10."""
    from torchvision.datasets import CIFAR10
    print("  Downloading CIFAR-10 (first time only)...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=True, download=True)
    picked = []
    classes_seen = set()
    for img, label in ds:
        if label not in classes_seen and len(picked) < n:
            classes_seen.add(label)
            picked.append((img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                          ds.classes[label]))
    return [p[0] for p in picked], [p[1] for p in picked]


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


if __name__ == '__main__':
    t_total = time.time()
    print("Loading VAE...", flush=True)
    vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
    print(f"VAE loaded ({time.time()-t_total:.0f}s)", flush=True)

    nat_images, nat_names = get_natural_images(8)

    # ================================================================
    # 1. BIT ACCURACY & PSNR: CDF vs Original
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 1. CDF vs ORIGINAL: Accuracy & Quality", flush=True)
    print("#"*60, flush=True)

    carrier_counts = [5, 10, 20, 50]
    cdf_results = []
    orig_results = []

    for n_carriers in carrier_counts:
        cdf_accs, cdf_psnrs = [], []
        orig_accs, orig_psnrs = [], []

        for img_idx, (img, name) in enumerate(zip(nat_images, nat_names)):
            torch.manual_seed(42 + img_idx)
            bits = torch.randint(0, 2, (n_carriers,)).tolist()

            # CDF method
            cdf_steg = CDFPatchSteg(seed=42, sigma=1.0)
            carriers, _ = cdf_steg.select_carriers_by_stability(vae, img, n_carriers=n_carriers)
            latent = vae.encode(img)
            lat_cdf = cdf_steg.encode_message(latent, carriers, bits)
            stego_cdf = vae.decode(lat_cdf)
            rec_cdf, _ = cdf_steg.decode_message(vae, stego_cdf, carriers)
            cdf_accs.append(bit_accuracy(bits, rec_cdf))
            cdf_psnrs.append(compute_psnr(img, stego_cdf))

            # Original method (eps=5.0)
            orig_steg = PatchSteg(seed=42, epsilon=5.0)
            carriers_o, _ = orig_steg.select_carriers_by_stability(vae, img, n_carriers=n_carriers)
            lat_orig = orig_steg.encode_message(latent, carriers_o, bits)
            stego_orig = vae.decode(lat_orig)
            lat_re = vae.encode(stego_orig)
            rec_orig, _ = orig_steg.decode_message(latent, lat_re, carriers_o)
            orig_accs.append(bit_accuracy(bits, rec_orig))
            orig_psnrs.append(compute_psnr(img, stego_orig))

        cdf_results.append({
            'K': n_carriers,
            'acc': np.mean(cdf_accs), 'acc_std': np.std(cdf_accs),
            'psnr': np.mean(cdf_psnrs), 'psnr_std': np.std(cdf_psnrs)
        })
        orig_results.append({
            'K': n_carriers,
            'acc': np.mean(orig_accs), 'acc_std': np.std(orig_accs),
            'psnr': np.mean(orig_psnrs), 'psnr_std': np.std(orig_psnrs)
        })
        print(f"  K={n_carriers:>3d}: CDF acc={np.mean(cdf_accs):.1f}% psnr={np.mean(cdf_psnrs):.1f}dB | "
              f"Orig acc={np.mean(orig_accs):.1f}% psnr={np.mean(orig_psnrs):.1f}dB", flush=True)

    # Figure: capacity curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ks = [r['K'] for r in cdf_results]

    axes[0].errorbar(ks, [r['acc'] for r in cdf_results],
                     yerr=[r['acc_std'] for r in cdf_results],
                     marker='o', capsize=4, linewidth=2, label='CDF-PatchSteg')
    axes[0].errorbar(ks, [r['acc'] for r in orig_results],
                     yerr=[r['acc_std'] for r in orig_results],
                     marker='s', capsize=4, linewidth=2, label='Original (eps=5)')
    axes[0].axhline(90, color='green', ls='--', alpha=0.3, label='90% threshold')
    axes[0].set_xlabel('Number of Carriers (K)')
    axes[0].set_ylabel('Bit Accuracy (%)')
    axes[0].set_title('Bit Accuracy vs Carrier Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    axes[0].set_ylim(40, 105)

    axes[1].errorbar(ks, [r['psnr'] for r in cdf_results],
                     yerr=[r['psnr_std'] for r in cdf_results],
                     marker='o', capsize=4, linewidth=2, label='CDF-PatchSteg')
    axes[1].errorbar(ks, [r['psnr'] for r in orig_results],
                     yerr=[r['psnr_std'] for r in orig_results],
                     marker='s', capsize=4, linewidth=2, label='Original (eps=5)')
    axes[1].set_xlabel('Number of Carriers (K)')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Image Quality vs Carrier Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    plt.suptitle('CDF-PatchSteg vs Original PatchSteg: Capacity', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'cdf_capacity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved cdf_capacity_curve.png", flush=True)

    # ================================================================
    # 2. DISTRIBUTION PRESERVATION (KS TEST)
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 2. DISTRIBUTION PRESERVATION (KS TEST)", flush=True)
    print("#"*60, flush=True)

    ks_results_cdf = []
    ks_results_orig = []
    n_carriers_ks = 20

    for img_idx, (img, name) in enumerate(zip(nat_images[:4], nat_names[:4])):
        latent = vae.encode(img)
        ch_data = latent[0, 0].cpu().numpy().flatten()
        mu_ch, sigma_ch = ch_data.mean(), ch_data.std()

        # CDF method
        cdf_steg = CDFPatchSteg(seed=42, sigma=1.0)
        carriers, _ = cdf_steg.select_carriers_by_stability(vae, img, n_carriers=n_carriers_ks)
        torch.manual_seed(42 + img_idx)
        bits = torch.randint(0, 2, (n_carriers_ks,)).tolist()
        lat_cdf = cdf_steg.encode_message(latent, carriers, bits)

        # Get carrier values after CDF encoding
        cdf_carrier_vals = [lat_cdf[0, 0, r, c].item() for r, c in carriers]
        ks_stat, ks_p = scipy_stats.kstest(cdf_carrier_vals, 'norm', args=(mu_ch, sigma_ch))
        ks_results_cdf.append({'name': name, 'stat': ks_stat, 'p': ks_p})

        # Original method
        orig_steg = PatchSteg(seed=42, epsilon=5.0)
        carriers_o, _ = orig_steg.select_carriers_by_stability(vae, img, n_carriers=n_carriers_ks)
        lat_orig = orig_steg.encode_message(latent, carriers_o, bits)
        orig_carrier_vals = [lat_orig[0, 0, r, c].item() for r, c in carriers_o]
        ks_stat_o, ks_p_o = scipy_stats.kstest(orig_carrier_vals, 'norm', args=(mu_ch, sigma_ch))
        ks_results_orig.append({'name': name, 'stat': ks_stat_o, 'p': ks_p_o})

        print(f"  {name}: CDF KS p={ks_p:.4f} | Orig KS p={ks_p_o:.4f}", flush=True)

    avg_p_cdf = np.mean([r['p'] for r in ks_results_cdf])
    avg_p_orig = np.mean([r['p'] for r in ks_results_orig])
    print(f"\n  Average KS p-value: CDF={avg_p_cdf:.4f}, Original={avg_p_orig:.4f}", flush=True)

    # ================================================================
    # 3. DETECTABILITY COMPARISON
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 3. DETECTABILITY: CDF vs ORIGINAL", flush=True)
    print("#"*60, flush=True)

    n_detect = len(nat_images)
    cv = StratifiedKFold(n_splits=min(3, n_detect), shuffle=True, random_state=42)

    detect_comparison = {}
    for method_name in ['CDF-PatchSteg', 'PatchSteg eps=2', 'PatchSteg eps=5']:
        X_all, y_all = [], []

        for i, img in enumerate(nat_images):
            # Clean
            X_all.append(extract_latent_features(vae, img))
            y_all.append(0)

            # Stego
            torch.manual_seed(42 + i)
            bits = torch.randint(0, 2, (20,)).tolist()
            latent = vae.encode(img)

            if method_name == 'CDF-PatchSteg':
                s = CDFPatchSteg(seed=42, sigma=1.0)
                carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20)
                lat_m = s.encode_message(latent, carriers, bits)
                stego = vae.decode(lat_m)
            elif method_name == 'PatchSteg eps=2':
                s = PatchSteg(seed=42, epsilon=2.0)
                carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=2.0)
                lat_m = s.encode_message(latent, carriers, bits)
                stego = vae.decode(lat_m)
            else:
                s = PatchSteg(seed=42, epsilon=5.0)
                carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=5.0)
                lat_m = s.encode_message(latent, carriers, bits)
                stego = vae.decode(lat_m)

            X_all.append(extract_latent_features(vae, stego))
            y_all.append(1)

        X = np.array(X_all)
        y = np.array(y_all)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
        auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        detect_comparison[method_name] = {
            'auc': auc.mean(), 'auc_std': auc.std(),
            'acc': acc.mean(), 'acc_std': acc.std()
        }
        print(f"  {method_name}: AUC={auc.mean():.3f}+-{auc.std():.3f}, "
              f"Acc={acc.mean():.3f}+-{acc.std():.3f}", flush=True)

    # Figure: detectability comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(detect_comparison.keys())
    aucs = [detect_comparison[m]['auc'] for m in methods]
    auc_errs = [detect_comparison[m]['auc_std'] for m in methods]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(methods, aucs, yerr=auc_errs, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Chance (AUC=0.5)')
    ax.axhline(0.7, color='orange', ls='--', alpha=0.5, label='Detection Threshold')
    ax.set_ylabel('Detection AUC')
    ax.set_title('Detectability: CDF-PatchSteg vs Original PatchSteg')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'cdf_detectability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved cdf_detectability.png", flush=True)

    # ================================================================
    # 4. DISTRIBUTION VISUALIZATION
    # ================================================================
    print("\n" + "#"*60, flush=True)
    print("# 4. DISTRIBUTION VISUALIZATION", flush=True)
    print("#"*60, flush=True)

    img = nat_images[0]
    latent = vae.encode(img)
    ch_data = latent[0, 0].cpu().numpy().flatten()

    # CDF encoding
    cdf_steg = CDFPatchSteg(seed=42, sigma=1.0)
    carriers, _ = cdf_steg.select_carriers_by_stability(vae, img, n_carriers=50)
    torch.manual_seed(42)
    bits = torch.randint(0, 2, (50,)).tolist()
    lat_cdf = cdf_steg.encode_message(latent, carriers, bits)

    # Original encoding
    orig_steg = PatchSteg(seed=42, epsilon=5.0)
    carriers_o, _ = orig_steg.select_carriers_by_stability(vae, img, n_carriers=50)
    lat_orig = orig_steg.encode_message(latent, carriers_o, bits)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Clean latent distribution
    ax = axes[0]
    ax.hist(ch_data, bins=50, alpha=0.7, color='steelblue', density=True, label='Clean')
    mu, sigma = ch_data.mean(), ch_data.std()
    x_range = np.linspace(ch_data.min(), ch_data.max(), 200)
    ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label=f'N({mu:.2f},{sigma:.2f})')
    ax.set_xlabel('Latent Value (Channel 0)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Clean Latent Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (b) CDF carrier values vs clean
    ax = axes[1]
    cdf_vals = [lat_cdf[0, 0, r, c].item() for r, c in carriers]
    clean_vals = [latent[0, 0, r, c].item() for r, c in carriers]
    ax.hist(clean_vals, bins=15, alpha=0.5, color='steelblue', density=True, label='Clean carriers')
    ax.hist(cdf_vals, bins=15, alpha=0.5, color='green', density=True, label='CDF carriers')
    ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Latent Value (Channel 0)')
    ax.set_title('(b) CDF-PatchSteg: Distribution Preserved')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (c) Original carrier values vs clean
    ax = axes[2]
    orig_vals = [lat_orig[0, 0, r, c].item() for r, c in carriers_o]
    clean_vals_o = [latent[0, 0, r, c].item() for r, c in carriers_o]
    ax.hist(clean_vals_o, bins=15, alpha=0.5, color='steelblue', density=True, label='Clean carriers')
    ax.hist(orig_vals, bins=15, alpha=0.5, color='red', density=True, label='Original (eps=5)')
    ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Latent Value (Channel 0)')
    ax.set_title('(c) Original PatchSteg: Distribution Shifted')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Distribution Preservation: CDF vs Original Encoding', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'cdf_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved cdf_distribution.png", flush=True)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'#'*60}", flush=True)
    print("# SUMMARY", flush=True)
    print(f"{'#'*60}", flush=True)
    print(f"  CDF avg KS p-value: {avg_p_cdf:.4f} (want > 0.05)", flush=True)
    print(f"  Orig avg KS p-value: {avg_p_orig:.4f}", flush=True)
    print(f"  CDF detection AUC: {detect_comparison['CDF-PatchSteg']['auc']:.3f} (want 0.45-0.55)", flush=True)
    print(f"  Orig eps=5 detection AUC: {detect_comparison['PatchSteg eps=5']['auc']:.3f}", flush=True)
    for r in cdf_results:
        print(f"  CDF K={r['K']}: acc={r['acc']:.1f}% psnr={r['psnr']:.1f}dB", flush=True)
    print(f"\n  Total time: {time.time()-t_total:.0f}s", flush=True)
