"""
Experiment 3: Detectability Analysis

Key questions:
  1. Can a statistical classifier distinguish stego from clean round-trip images?
  2. How does detectability vary with epsilon (stealth-capacity tradeoff)?
  3. Which latent statistics are most discriminative?

Design:
  - 100 diverse synthetic images
  - For each image: one clean round-trip, and stego round-trips at eps=[1.0, 2.0, 5.0, 10.0]
  - Features: per-channel mean, std, skew, kurtosis, min, max, IQR (36 features)
  - Classifier: logistic regression with 5-fold CV
  - Report: accuracy and AUC at each epsilon
  - This characterizes the stealth vs capacity frontier
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

from core.vae import StegoVAE
from core.steganography import PatchSteg


def generate_diverse_images(n=100, size=512):
    """Generate n diverse synthetic images."""
    rng = np.random.RandomState(42)
    images = []
    for i in range(n):
        kind = i % 5
        if kind == 0:
            arr = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
        elif kind == 1:
            t = np.linspace(0, 1, size)
            base = np.tile(t, (size, 1)) if rng.rand() > 0.5 else np.tile(t.reshape(-1, 1), (1, size))
            color = rng.randint(50, 255, 3)
            arr = (np.stack([base] * 3, axis=2) * color).astype(np.uint8)
        elif kind == 2:
            arr = np.full((size, size, 3), rng.randint(0, 255, 3), dtype=np.uint8)
        elif kind == 3:
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            bs = rng.choice([32, 64, 128])
            for r in range(0, size, bs):
                for c in range(0, size, bs):
                    arr[r:r+bs, c:c+bs] = rng.randint(0, 255, 3)
        else:
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            bs = rng.choice([16, 32, 64])
            for r in range(0, size, bs):
                for c in range(0, size, bs):
                    if ((r // bs) + (c // bs)) % 2 == 0:
                        arr[r:r+bs, c:c+bs] = rng.randint(100, 255, 3)
        images.append(Image.fromarray(arr))
    return images


def extract_features(latent):
    """Extract 36 statistical features from latent [1, 4, 64, 64]."""
    feats = []
    for ch in range(4):
        x = latent[0, ch].cpu().numpy().flatten()
        mu = x.mean()
        sigma = x.std() + 1e-8
        feats.extend([
            mu, sigma,
            float(np.percentile(x, 25)),
            float(np.percentile(x, 75)),
            float(np.median(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(((x - mu) ** 3).mean() / sigma ** 3),  # skewness
            float(((x - mu) ** 4).mean() / sigma ** 4),  # kurtosis
        ])
    return np.array(feats)


FEAT_NAMES = []
for ch in range(4):
    for stat in ['mean', 'std', 'q25', 'q75', 'median', 'min', 'max', 'skew', 'kurtosis']:
        FEAT_NAMES.append(f'ch{ch}_{stat}')


def main():
    t_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    n_images = 30  # 30 clean + 30 stego = 60 per epsilon
    epsilons = [1.0, 2.0, 5.0, 10.0]
    n_carriers = 20

    print(f"Generating {n_images} test images...")
    images = generate_diverse_images(n_images, size=512)

    print("Loading VAE...")
    vae = StegoVAE(device=device)

    # Collect clean features (shared across all epsilon comparisons)
    print("Encoding clean round-trips...")
    clean_features = []
    clean_latents = []
    for i, img in enumerate(images):
        if i % 20 == 0:
            print(f"  {i}/{n_images}...")
        latent = vae.encode(img)
        clean_latents.append(latent)
        recon = vae.decode(latent)
        latent_rt = vae.encode(recon)
        clean_features.append(extract_features(latent_rt))

    X_clean = np.array(clean_features)

    # For each epsilon, compute stego features and run classifier
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    detection_results = {}

    for eps in epsilons:
        print(f"\nEpsilon = {eps}")
        steg = PatchSteg(seed=42, epsilon=eps)
        stego_features = []

        for i, (img, latent_clean) in enumerate(zip(images, clean_latents)):
            if i % 20 == 0:
                print(f"  Encoding stego {i}/{n_images}...")
            carriers, _ = steg.select_carriers_by_stability(
                vae, img, n_carriers=n_carriers, test_eps=eps
            )
            torch.manual_seed(42 + i)
            bits = torch.randint(0, 2, (n_carriers,)).tolist()
            latent_mod = steg.encode_message(latent_clean, carriers, bits)
            stego_img = vae.decode(latent_mod)
            latent_stego_rt = vae.encode(stego_img)
            stego_features.append(extract_features(latent_stego_rt))

        X_stego = np.array(stego_features)

        # Build dataset: clean=0, stego=1
        X = np.vstack([X_clean, X_stego])
        y = np.concatenate([np.zeros(n_images), np.ones(n_images)])

        # Train with standardized features
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')

        print(f"  CV Accuracy: {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}")
        print(f"  CV AUC:      {auc_scores.mean():.3f} +/- {auc_scores.std():.3f}")

        # Feature importance from full fit
        clf.fit(X, y)
        coef = np.abs(clf.named_steps['logisticregression'].coef_[0])
        top5 = np.argsort(coef)[::-1][:5]
        print(f"  Top features: {[FEAT_NAMES[i] for i in top5]}")

        detection_results[eps] = {
            'acc_mean': acc_scores.mean(), 'acc_std': acc_scores.std(),
            'auc_mean': auc_scores.mean(), 'auc_std': auc_scores.std(),
            'top_features': [(FEAT_NAMES[i], coef[i]) for i in top5],
        }

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 65)
    print("DETECTABILITY RESULTS")
    print("=" * 65)
    print(f"{'Epsilon':>8} | {'CV Accuracy':>14} | {'CV AUC':>14} | {'Detectable?':>12}")
    print("-" * 58)
    for eps in epsilons:
        r = detection_results[eps]
        detectable = "YES" if r['auc_mean'] > 0.7 else ("MARGINAL" if r['auc_mean'] > 0.6 else "NO")
        print(f"{eps:>8.1f} | {r['acc_mean']:>6.3f} +/- {r['acc_std']:.3f} | "
              f"{r['auc_mean']:>6.3f} +/- {r['auc_std']:.3f} | {detectable:>12}")

    # ============================================================
    # Figures
    # ============================================================
    fig_dir = Path(__file__).resolve().parent.parent / 'paper' / 'figures'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: AUC vs epsilon (stealth-capacity frontier)
    ax = axes[0]
    eps_vals = list(detection_results.keys())
    auc_means = [detection_results[e]['auc_mean'] for e in eps_vals]
    auc_stds = [detection_results[e]['auc_std'] for e in eps_vals]
    ax.errorbar(eps_vals, auc_means, yerr=auc_stds, marker='o', capsize=4,
                color='tab:red', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='chance (AUC=0.5)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='detectable threshold')
    ax.set_xlabel('Epsilon ($\\varepsilon$)')
    ax.set_ylabel('Detection AUC')
    ax.set_title('Stealth: Detection AUC vs Perturbation Strength')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    # Plot 2: Accuracy vs epsilon
    ax = axes[1]
    acc_means = [detection_results[e]['acc_mean'] for e in eps_vals]
    acc_stds = [detection_results[e]['acc_std'] for e in eps_vals]
    ax.errorbar(eps_vals, acc_means, yerr=acc_stds, marker='s', capsize=4,
                color='tab:blue', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='chance')
    ax.set_xlabel('Epsilon ($\\varepsilon$)')
    ax.set_ylabel('Detection Accuracy')
    ax.set_title('Detection Accuracy vs Perturbation Strength')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    plt.tight_layout()
    plt.savefig(fig_dir / 'detectability_curve.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir / 'detectability_curve.png'}")
    plt.close('all')
    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    return detection_results


if __name__ == '__main__':
    results = main()
