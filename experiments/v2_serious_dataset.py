#!/usr/bin/env python
"""
Experiment: Serious dataset evaluation.
Tests on hundreds of natural images with confidence intervals.
Uses CIFAR-10 test set (diverse natural images) at scale.
Reports mean, std, CI for accuracy and PSNR across image categories.
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
from scipy import stats as sp_stats

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, bit_accuracy

FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
IMG_SIZE = 256

torch.manual_seed(42)
np.random.seed(42)


def get_cifar_images(n_per_class=30):
    """Get n_per_class images from each of 10 CIFAR-10 classes."""
    from torchvision.datasets import CIFAR10
    print("  Loading CIFAR-10...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=False, download=True)

    by_class = {i: [] for i in range(10)}
    for img, label in ds:
        if len(by_class[label]) < n_per_class:
            by_class[label].append(
                img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            )
    class_names = ds.classes
    images, labels, names = [], [], []
    for cls_idx in range(10):
        for img in by_class[cls_idx]:
            images.append(img)
            labels.append(cls_idx)
            names.append(class_names[cls_idx])
    return images, labels, names, class_names


# ================================================================
print("=" * 70, flush=True)
print("SERIOUS DATASET EVALUATION", flush=True)
print("=" * 70, flush=True)
t0 = time.time()

vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
N_PER_CLASS = 20  # 20 per class * 10 classes = 200 images
images, labels, names, class_names = get_cifar_images(N_PER_CLASS)
print(f"Loaded {len(images)} images across {len(class_names)} classes", flush=True)

# Test at multiple epsilon values
EPSILONS = [2.0, 5.0]
results_by_eps = {}

for eps in EPSILONS:
    print(f"\n--- epsilon = {eps} ---", flush=True)
    steg = PatchSteg(seed=42, epsilon=eps)

    all_accs = []
    all_psnrs = []
    class_accs = {c: [] for c in range(10)}
    class_psnrs = {c: [] for c in range(10)}

    for i, (img, cls) in enumerate(zip(images, labels)):
        if i % 50 == 0:
            print(f"  Processing image {i}/{len(images)}...", flush=True)

        lat = vae.encode(img)
        carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=20, test_eps=eps)
        torch.manual_seed(42 + i)
        bits = torch.randint(0, 2, (20,)).tolist()
        lat_m = steg.encode_message(lat, carriers, bits)
        st = vae.decode(lat_m)
        psnr = compute_psnr(img, st)
        lat_re = vae.encode(st)
        rec, _ = steg.decode_message(lat, lat_re, carriers)
        acc = bit_accuracy(bits, rec)

        all_accs.append(acc)
        all_psnrs.append(psnr)
        class_accs[cls].append(acc)
        class_psnrs[cls].append(psnr)

    all_accs = np.array(all_accs)
    all_psnrs = np.array(all_psnrs)

    # Bootstrap 95% CI
    n_boot = 1000
    boot_accs = []
    boot_psnrs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(all_accs), len(all_accs), replace=True)
        boot_accs.append(all_accs[idx].mean())
        boot_psnrs.append(all_psnrs[idx].mean())

    ci_acc = np.percentile(boot_accs, [2.5, 97.5])
    ci_psnr = np.percentile(boot_psnrs, [2.5, 97.5])

    print(f"\n  OVERALL (n={len(all_accs)}):", flush=True)
    print(f"    Accuracy: {all_accs.mean():.1f}% ± {all_accs.std():.1f}% "
          f"(95% CI: [{ci_acc[0]:.1f}, {ci_acc[1]:.1f}])", flush=True)
    print(f"    PSNR:     {all_psnrs.mean():.1f} ± {all_psnrs.std():.1f} dB "
          f"(95% CI: [{ci_psnr[0]:.1f}, {ci_psnr[1]:.1f}])", flush=True)

    print(f"\n  BY CLASS:", flush=True)
    for cls in range(10):
        ca = np.array(class_accs[cls])
        cp = np.array(class_psnrs[cls])
        print(f"    {class_names[cls]:12s}: acc={ca.mean():.1f}±{ca.std():.1f}% "
              f"psnr={cp.mean():.1f}±{cp.std():.1f}dB (n={len(ca)})", flush=True)

    results_by_eps[eps] = {
        'all_accs': all_accs, 'all_psnrs': all_psnrs,
        'class_accs': class_accs, 'class_psnrs': class_psnrs,
        'ci_acc': ci_acc, 'ci_psnr': ci_psnr,
    }

# ================================================================
# Figures
# ================================================================
print(f"\nGenerating figures...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for ax_idx, eps in enumerate(EPSILONS):
    res = results_by_eps[eps]

    # (top) Accuracy by class
    ax = axes[0, ax_idx]
    class_means = [np.mean(res['class_accs'][c]) for c in range(10)]
    class_stds = [np.std(res['class_accs'][c]) for c in range(10)]
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    bars = ax.bar(range(10), class_means, yerr=class_stds, capsize=3,
                  color=colors, edgecolor='black', linewidth=0.3)
    ax.set_xticks(range(10))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Bit Accuracy (%)')
    ax.set_title(f'ε={eps} (overall: {res["all_accs"].mean():.1f}% '
                 f'[{res["ci_acc"][0]:.1f}, {res["ci_acc"][1]:.1f}])')
    ax.set_ylim(0, 105)
    ax.axhline(50, color='gray', ls=':', alpha=0.5)
    ax.axhline(res['all_accs'].mean(), color='red', ls='--', alpha=0.5,
               label=f'Mean={res["all_accs"].mean():.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # (bottom) Histogram of per-image accuracy
    ax = axes[1, ax_idx]
    ax.hist(res['all_accs'], bins=20, color='steelblue', edgecolor='navy',
            alpha=0.7, density=False)
    ax.axvline(res['all_accs'].mean(), color='red', ls='--', linewidth=2,
               label=f'Mean={res["all_accs"].mean():.1f}%')
    ax.axvline(50, color='gray', ls=':', alpha=0.5, label='Chance')
    ax.set_xlabel('Bit Accuracy (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Per-Image Accuracy (ε={eps}, n={len(res["all_accs"])})')
    ax.legend()
    ax.grid(True, alpha=0.2)

plt.suptitle(f'Serious Dataset Evaluation: {len(images)} CIFAR-10 Images, '
             f'{N_PER_CLASS} per Class\n95% Bootstrap Confidence Intervals', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'serious_dataset.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved serious_dataset.png", flush=True)

print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
