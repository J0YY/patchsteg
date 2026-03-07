#!/usr/bin/env python
"""
Run the 3 remaining experiments (robustness, mechanistic, detectability)
with 256x256 images for speed. Model loaded once and shared.

Expected runtime on CPU: ~15-20 minutes total.
"""
import sys
import os
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
import io
import time
from scipy import stats as scipy_stats

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, bit_accuracy
from core.analysis import channel_importance, reconstruction_error_map

IMG_SIZE = 256
LATENT_SIZE = IMG_SIZE // 8  # 32
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def make_test_images(n=3):
    imgs, names = [], []
    # Gradient
    grad = np.tile(np.linspace(0, 255, IMG_SIZE, dtype=np.uint8), (IMG_SIZE, 1))
    imgs.append(Image.fromarray(np.stack([grad]*3, axis=2)))
    names.append('Gradient')
    # Color patches
    p = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    rng = np.random.RandomState(99)
    for i in range(0, IMG_SIZE, 32):
        for j in range(0, IMG_SIZE, 32):
            p[i:i+32, j:j+32] = rng.randint(0, 255, 3)
    imgs.append(Image.fromarray(p))
    names.append('Patches')
    if n >= 3:
        imgs.append(Image.fromarray(
            np.random.RandomState(42).randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)))
        names.append('Noise')
    return imgs, names


# ================================================================
# EXPERIMENT A: ROBUSTNESS
# ================================================================
def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def add_noise(img, sigma):
    a = np.array(img).astype(float)/255
    a = np.clip(a + np.random.randn(*a.shape)*sigma, 0, 1)
    return Image.fromarray((a*255).astype(np.uint8))

def resize_back(img, scale):
    w, h = img.size
    return img.resize((int(w*scale), int(h*scale)), Image.BILINEAR).resize((w,h), Image.BILINEAR)

DISTORTIONS = {
    'None':       lambda img: img,
    'JPEG 95':    lambda img: jpeg_compress(img, 95),
    'JPEG 75':    lambda img: jpeg_compress(img, 75),
    'JPEG 50':    lambda img: jpeg_compress(img, 50),
    'Noise .01':  lambda img: add_noise(img, 0.01),
    'Noise .05':  lambda img: add_noise(img, 0.05),
    'Resize 50%': lambda img: resize_back(img, 0.5),
}

def run_robustness(vae, images, names):
    print("\n" + "#"*60, flush=True)
    print("# EXPERIMENT: ROBUSTNESS", flush=True)
    print("#"*60, flush=True)
    t0 = time.time()

    epsilons = [2.0, 5.0]
    n_carriers = 20
    results = []

    for img_idx, (img, name) in enumerate(zip(images, names)):
        for eps in epsilons:
            steg = PatchSteg(seed=42, epsilon=eps)
            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=n_carriers, test_eps=eps)
            latent_clean = vae.encode(img)
            torch.manual_seed(42 + img_idx)
            bits = torch.randint(0, 2, (n_carriers,)).tolist()
            latent_mod = steg.encode_message(latent_clean, carriers, bits)
            stego_img = vae.decode(latent_mod)

            for dname, dfn in DISTORTIONS.items():
                distorted = dfn(stego_img)
                latent_re = vae.encode(distorted)
                recovered, confs = steg.decode_message(latent_clean, latent_re, carriers)
                acc = bit_accuracy(bits, recovered)
                print(f"  {name} eps={eps} {dname:>10s}: {acc:>5.1f}%", flush=True)
                results.append({'image': name, 'eps': eps, 'dist': dname, 'acc': acc})

    # Summary
    print(f"\n  ROBUSTNESS SUMMARY (avg over images):", flush=True)
    print(f"  {'Eps':>5} {'Distortion':>12} {'Accuracy':>10}", flush=True)
    for eps in epsilons:
        for d in DISTORTIONS:
            sub = [r for r in results if r['eps']==eps and r['dist']==d]
            print(f"  {eps:>5.1f} {d:>12s} {np.mean([r['acc'] for r in sub]):>8.1f}%", flush=True)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax_i, eps in enumerate(epsilons):
        ax = axes[ax_i]
        dnames = list(DISTORTIONS.keys())
        vals = [np.mean([r['acc'] for r in results if r['eps']==eps and r['dist']==d]) for d in dnames]
        colors = ['green' if v>=90 else 'orange' if v>=70 else 'red' for v in vals]
        ax.bar(dnames, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(90, color='green', ls='--', alpha=0.5)
        ax.axhline(50, color='gray', ls=':', alpha=0.3)
        ax.set_ylabel('Bit Accuracy (%)')
        ax.set_title(f'eps={eps}')
        ax.set_ylim(0, 105)
        plt.sca(ax)
        plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.suptitle(f'Robustness ({LATENT_SIZE}x{LATENT_SIZE} latent, 20 stability-selected carriers)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'robustness_bars.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved robustness_bars.png ({time.time()-t0:.0f}s)", flush=True)
    return results


# ================================================================
# EXPERIMENT B: MECHANISTIC ANALYSIS
# ================================================================
def run_mechanistic(vae, images, names):
    print("\n" + "#"*60, flush=True)
    print("# EXPERIMENT: MECHANISTIC ANALYSIS", flush=True)
    print("#"*60, flush=True)
    t0 = time.time()
    steg = PatchSteg(seed=42, epsilon=5.0)

    # 1. Channel importance
    print("  Channel importance...", flush=True)
    all_ch = {}
    for img, name in zip(images, names):
        ch_res = channel_importance(vae, steg, img, eps=5.0)
        all_ch[name] = ch_res
        print(f"    {name}: {dict((k, f'{v:.1f}%') for k,v in ch_res.items())}", flush=True)
    avg_ch = {ch: np.mean([all_ch[n][ch] for n in names]) for ch in range(4)}
    std_ch = {ch: np.std([all_ch[n][ch] for n in names]) for ch in range(4)}
    print(f"  Avg: {dict((k, f'{v:.1f}%') for k,v in avg_ch.items())}", flush=True)

    # 2. Reconstruction error vs stability
    print("  Recon error vs stability...", flush=True)
    pearson_rs = []
    all_errors, all_stabs = [], []
    for img, name in zip(images, names):
        err = reconstruction_error_map(vae, img)
        smap, _ = steg.compute_stability_map(vae, img, test_eps=5.0)
        all_errors.append(err)
        all_stabs.append(smap)
        r, p = scipy_stats.pearsonr(err.numpy().flatten(), smap.numpy().flatten())
        pearson_rs.append(r)
        print(f"    {name}: r={r:.3f} p={p:.2e}", flush=True)
    avg_r = np.mean(pearson_rs)
    print(f"  Avg Pearson r: {avg_r:.3f}", flush=True)

    # 3. Border vs interior
    print("  Border vs interior...", flush=True)
    S = LATENT_SIZE
    border_mask = np.zeros((S, S), dtype=bool)
    border_mask[:2, :] = border_mask[-2:, :] = border_mask[:, :2] = border_mask[:, -2:] = True
    for smap, name in zip(all_stabs, names):
        s = smap.numpy()
        print(f"    {name}: border={s[border_mask].mean():.3f} interior={s[~border_mask].mean():.3f}", flush=True)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # Channel importance
    ax = axes[0]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    ax.bar(range(4), [avg_ch[c] for c in range(4)], yerr=[std_ch[c] for c in range(4)],
           capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'Ch {i}' for i in range(4)])
    ax.set_ylabel('% Positions Surviving'); ax.set_title('(a) Channel Importance')
    ax.set_ylim(0, 105); ax.grid(True, alpha=0.2, axis='y')

    # Error vs stability scatter
    ax = axes[1]
    for idx, name in enumerate(names):
        e = all_errors[idx].numpy().flatten()
        s = all_stabs[idx].numpy().flatten()
        sub = np.random.RandomState(42).choice(len(e), min(300, len(e)), replace=False)
        ax.scatter(e[sub], s[sub], alpha=0.3, s=5, label=name)
    ax.set_xlabel('Reconstruction Error'); ax.set_ylabel('Stability')
    ax.set_title(f'(b) Error vs Stability (r={avg_r:.3f})'); ax.legend(fontsize=7, markerscale=3)
    ax.grid(True, alpha=0.2)

    # Stability + error maps side by side for one image
    ax = axes[2]
    ax.imshow(all_stabs[1].numpy(), cmap='RdYlGn')
    ax.set_title(f'(c) Stability Map ({names[1]})'); ax.set_xlabel('Col'); ax.set_ylabel('Row')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'mechanistic_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved mechanistic_analysis.png ({time.time()-t0:.0f}s)", flush=True)
    return {'channel_importance': avg_ch, 'recon_corr': avg_r}


# ================================================================
# EXPERIMENT C: DETECTABILITY
# ================================================================
def run_detectability(vae):
    print("\n" + "#"*60, flush=True)
    print("# EXPERIMENT: DETECTABILITY", flush=True)
    print("#"*60, flush=True)
    t0 = time.time()

    n_images = 15  # 15 clean + 15 stego per epsilon = 30 samples
    epsilons = [1.0, 2.0, 5.0, 10.0]
    n_carriers = 20

    # Generate images
    rng = np.random.RandomState(42)
    images = []
    for i in range(n_images):
        kind = i % 3
        if kind == 0:
            arr = rng.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        elif kind == 1:
            t = np.linspace(0, 1, IMG_SIZE)
            base = np.tile(t, (IMG_SIZE, 1))
            arr = (np.stack([base]*3, axis=2) * rng.randint(50, 255, 3)).astype(np.uint8)
        else:
            arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            for r in range(0, IMG_SIZE, 32):
                for c in range(0, IMG_SIZE, 32):
                    arr[r:r+32, c:c+32] = rng.randint(0, 255, 3)
        images.append(Image.fromarray(arr))

    def extract_feats(latent):
        feats = []
        for ch in range(4):
            x = latent[0, ch].cpu().numpy().flatten()
            mu, sig = x.mean(), x.std() + 1e-8
            feats.extend([mu, sig, float(np.median(x)),
                          float(((x-mu)**3).mean()/sig**3),
                          float(((x-mu)**4).mean()/sig**4)])
        return np.array(feats)

    # Clean features (computed once)
    print(f"  Encoding {n_images} clean round-trips...", flush=True)
    clean_feats = []
    clean_latents = []
    for i, img in enumerate(images):
        lat = vae.encode(img)
        clean_latents.append(lat)
        recon = vae.decode(lat)
        clean_feats.append(extract_feats(vae.encode(recon)))
    X_clean = np.array(clean_feats)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for small n
    det_results = {}

    for eps in epsilons:
        print(f"  eps={eps}...", end=" ", flush=True)
        steg = PatchSteg(seed=42, epsilon=eps)
        stego_feats = []
        for i, (img, lat) in enumerate(zip(images, clean_latents)):
            carriers, _ = steg.select_carriers_by_stability(vae, img, n_carriers=n_carriers, test_eps=eps)
            torch.manual_seed(42+i)
            bits = torch.randint(0, 2, (n_carriers,)).tolist()
            lat_mod = steg.encode_message(lat, carriers, bits)
            stego_img = vae.decode(lat_mod)
            stego_feats.append(extract_feats(vae.encode(stego_img)))

        X = np.vstack([X_clean, np.array(stego_feats)])
        y = np.concatenate([np.zeros(n_images), np.ones(n_images)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
        auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        print(f"AUC={auc.mean():.3f}+-{auc.std():.3f} Acc={acc.mean():.3f}+-{acc.std():.3f}", flush=True)
        det_results[eps] = {'auc': auc.mean(), 'auc_std': auc.std(),
                            'acc': acc.mean(), 'acc_std': acc.std()}

    # Summary
    print(f"\n  DETECTABILITY SUMMARY:", flush=True)
    print(f"  {'Eps':>6} {'AUC':>12} {'Accuracy':>12} {'Verdict':>10}", flush=True)
    for eps in epsilons:
        r = det_results[eps]
        v = "DETECTED" if r['auc']>0.7 else ("MARGINAL" if r['auc']>0.6 else "STEALTHY")
        print(f"  {eps:>6.1f} {r['auc']:.3f}+-{r['auc_std']:.3f} {r['acc']:.3f}+-{r['acc_std']:.3f} {v:>10}", flush=True)

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    eps_list = sorted(det_results.keys())
    aucs = [det_results[e]['auc'] for e in eps_list]
    auc_errs = [det_results[e]['auc_std'] for e in eps_list]
    ax.errorbar(eps_list, aucs, yerr=auc_errs, marker='o', capsize=4, color='tab:red', linewidth=2)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='chance')
    ax.axhline(0.7, color='orange', ls='--', alpha=0.5, label='detectable threshold')
    ax.set_xlabel('Epsilon'); ax.set_ylabel('Detection AUC')
    ax.set_title('Stealth vs Perturbation Strength')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0.35, 1.05)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'detectability_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved detectability_curve.png ({time.time()-t0:.0f}s)", flush=True)
    return det_results


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    t_total = time.time()

    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}, Latent: {LATENT_SIZE}x{LATENT_SIZE}", flush=True)
    print("Loading VAE (once)...", flush=True)
    vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
    print(f"VAE loaded ({time.time()-t_total:.0f}s)", flush=True)

    images, names = make_test_images(3)

    rob_results = run_robustness(vae, images[:2], names[:2])
    mech_results = run_mechanistic(vae, images, names)
    det_results = run_detectability(vae)

    print(f"\n{'#'*60}", flush=True)
    print(f"# ALL DONE in {time.time()-t_total:.0f}s", flush=True)
    print(f"{'#'*60}", flush=True)
