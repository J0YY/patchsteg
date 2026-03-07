#!/usr/bin/env python
"""
Defense (c): Linear probes for collusion / modified latent detection.

Probes:

1. GlobalLatentProbe        — logistic regression on the full flattened latent
                              (4 * 32 * 32 = 4096 dims). Tests whether stego
                              modifications are globally linearly separable.

2. RoundTripResidualProbe   — logistic regression on the VAE round-trip residual
                              vector. Subtracts the on-manifold component, leaving
                              only off-manifold steganographic signal. More
                              sensitive to small epsilon perturbations.

3. PositionalCarrierProbe   — given paired (clean, stego) latents, computes a
                              per-position attribution map |mean(stego - clean)|.
                              Recovers carrier (r, c) positions without needing
                              the stego key. Evaluated by precision/recall against
                              the ground-truth carriers.

4. CollusionPatternProbe    — detects when a set of images was encoded with the
                              *same key*. Shared-key stego produces correlated
                              modifications at identical carrier positions; this
                              creates anomalously high variance in the first PC
                              of the latent matrix, compared to images with
                              independent keys or clean images.

Generates: defense_probe_roc.png, defense_probe_heatmap.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc as sklearn_auc

from core.vae import StegoVAE
from core.attacks.steganography import PatchSteg
from core.attacks.cdf_steganography import CDFPatchSteg

IMG_SIZE = 256
LATENT_SIZE = IMG_SIZE // 8   # 32
FIG_DIR = Path(__file__).resolve().parent.parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


# ─── PROBES ──────────────────────────────────────────────────────────────────

class GlobalLatentProbe:
    """
    Logistic regression on the full flattened latent tensor.

    Dimensions: 4 channels * 32 * 32 = 4096. L2-regularized (C controls
    regularization strength). High AUC means the stego modification is
    globally linearly separable from clean latents without any pre-processing.
    """

    def __init__(self, C=0.01):
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000, random_state=42)
        )

    def featurize(self, vae, image) -> np.ndarray:
        return vae.encode(image)[0].cpu().numpy().flatten()

    def fit(self, vae, clean_images, stego_images):
        X = np.stack(
            [self.featurize(vae, img) for img in clean_images] +
            [self.featurize(vae, img) for img in stego_images]
        )
        y = np.array([0] * len(clean_images) + [1] * len(stego_images))
        self.clf.fit(X, y)

    def score_image(self, vae, image) -> float:
        x = self.featurize(vae, image).reshape(1, -1)
        return float(self.clf.predict_proba(x)[0, 1])

    @property
    def name(self):
        return 'Global Latent Probe'


class RoundTripResidualProbe:
    """
    Logistic regression on the VAE round-trip residual.

    Residual = encode(decode(encode(image))) - encode(image)

    This subtracts the on-manifold component, retaining only modifications
    that the VAE cannot perfectly reconstruct. More sensitive than raw latents
    for small-epsilon encodings that survive round-trip imperfectly.
    """

    def __init__(self, C=0.01):
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000, random_state=42)
        )

    def featurize(self, vae, image) -> np.ndarray:
        latent = vae.encode(image)
        recon = vae.decode(latent)
        latent_rt = vae.encode(recon)
        return (latent_rt - latent)[0].cpu().numpy().flatten()

    def fit(self, vae, clean_images, stego_images):
        X = np.stack(
            [self.featurize(vae, img) for img in clean_images] +
            [self.featurize(vae, img) for img in stego_images]
        )
        y = np.array([0] * len(clean_images) + [1] * len(stego_images))
        self.clf.fit(X, y)

    def score_image(self, vae, image) -> float:
        x = self.featurize(vae, image).reshape(1, -1)
        return float(self.clf.predict_proba(x)[0, 1])

    @property
    def name(self):
        return 'Residual Probe'


class PositionalCarrierProbe:
    """
    Attribution-based carrier position recovery.

    Given paired (clean, stego) latents, computes:
        delta_map[ch, r, c] = mean_over_images |latent_stego - latent_clean|

    Then sums across channels to get a [32, 32] attribution heatmap.
    The top-k positions of this heatmap are the predicted carrier locations.

    Evaluated by precision and recall against ground-truth carriers.
    No classifier needed — purely uses the magnitude of latent differences.
    """

    def __init__(self):
        self.position_scores = None  # [32, 32]
        self.n_images_fitted = 0

    def fit(self, vae, clean_images, stego_images):
        H = W = LATENT_SIZE
        delta_map = np.zeros((4, H, W))
        for clean_img, stego_img in zip(clean_images, stego_images):
            lat_c = vae.encode(clean_img)[0].cpu().numpy()
            lat_s = vae.encode(stego_img)[0].cpu().numpy()
            delta_map += np.abs(lat_s - lat_c)
        delta_map /= len(clean_images)
        self.position_scores = delta_map.sum(axis=0)  # [H, W]
        self.n_images_fitted = len(clean_images)

    def heatmap(self) -> np.ndarray:
        """Normalized [0, 1] attribution map."""
        s = self.position_scores
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    def top_positions(self, k=20):
        flat = self.position_scores.flatten()
        top_idx = np.argsort(flat)[-k:][::-1]
        return [(int(idx // LATENT_SIZE), int(idx % LATENT_SIZE)) for idx in top_idx]

    def evaluate(self, true_carriers, k=20):
        predicted = set(self.top_positions(k))
        actual = set(true_carriers)
        overlap = len(predicted & actual)
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(actual) if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        return {'precision': precision, 'recall': recall, 'f1': f1, 'overlap': overlap}

    @property
    def name(self):
        return 'Positional Probe'


class CollusionPatternProbe:
    """
    Collusion detection via shared-key latent correlation.

    When multiple images are encoded with the *same* steganographic key,
    modifications land at the same (r, c) positions in every image.
    This creates a structured shared component in the latent matrix.

    Detection: compute SVD of the centered latent matrix [N, 4096].
    The fraction of variance explained by PC1 is the collusion score.
    - Same-key colluding set: high PC1 variance (shared modification pattern)
    - Different-key set or clean images: low PC1 variance (independent noise)
    """

    def score_set(self, vae, images) -> dict:
        latents = np.stack([
            vae.encode(img)[0].cpu().numpy().flatten()
            for img in images
        ])  # [N, 4096]
        centered = latents - latents.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        var_explained_pc1 = float((s[0] ** 2) / ((s ** 2).sum() + 1e-10))
        return {
            'variance_explained_pc1': var_explained_pc1,
            'collusion_score': var_explained_pc1,
            'singular_values': s[:5].tolist(),
        }

    @property
    def name(self):
        return 'Collusion Pattern Probe'


# ─── DATA & STEGO HELPERS ────────────────────────────────────────────────────

def get_natural_images(n=20):
    from torchvision.datasets import CIFAR10
    print("  Loading CIFAR-10...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=True, download=True)
    picked = []
    for img, _ in ds:
        picked.append(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        if len(picked) >= n:
            break
    return picked


def make_stego(vae, images, method='PatchSteg', seed=42):
    stegos, carriers_all = [], []
    for i, img in enumerate(images):
        torch.manual_seed(seed + i)
        bits = torch.randint(0, 2, (20,)).tolist()
        latent = vae.encode(img)
        if method == 'PatchSteg':
            s = PatchSteg(seed=seed, epsilon=5.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20)
            lat_m = s.encode_message(latent, carriers, bits)
        elif method == 'CDF':
            s = CDFPatchSteg(seed=seed, sigma=1.0)
            carriers, _ = s.select_carriers_by_stability(vae, img, n_carriers=20)
            lat_m = s.encode_message(latent, carriers, bits)
        else:
            raise ValueError(method)
        stegos.append(vae.decode(lat_m))
        carriers_all.append(carriers)
    return stegos, carriers_all


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_total = time.time()
    print("=" * 60, flush=True)
    print("DEFENSE (c): LINEAR PROBES FOR COLLUSION / LATENT MODIFICATION", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading VAE...", flush=True)
    vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
    print("Loading images...", flush=True)
    images = get_natural_images(20)
    train_imgs = images[:12]
    test_imgs = images[12:]

    # ─── 1. GLOBAL + RESIDUAL PROBES ─────────────────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("1. Global and Residual Probes (classification)", flush=True)
    print(f"{'─'*50}", flush=True)

    probe_results = {}
    steg_methods = ['PatchSteg', 'CDF']

    for steg_method in steg_methods:
        train_stegos, _ = make_stego(vae, train_imgs, steg_method)
        test_stegos, _ = make_stego(vae, test_imgs, steg_method)

        for ProbeClass in [GlobalLatentProbe, RoundTripResidualProbe]:
            probe = ProbeClass()
            probe.fit(vae, train_imgs, train_stegos)

            clean_scores = [probe.score_image(vae, img) for img in test_imgs]
            stego_scores = [probe.score_image(vae, img) for img in test_stegos]
            y = [0] * len(clean_scores) + [1] * len(stego_scores)
            scores = clean_scores + stego_scores
            try:
                probe_auc = roc_auc_score(y, scores)
            except Exception:
                probe_auc = 0.5

            key = f'{probe.name} | {steg_method}'
            probe_results[key] = {
                'auc': probe_auc,
                'clean_scores': clean_scores,
                'stego_scores': stego_scores,
            }
            print(f"  {probe.name} | {steg_method}: AUC={probe_auc:.3f}", flush=True)

    # ─── 2. POSITIONAL CARRIER PROBE ─────────────────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("2. Positional Carrier Attribution", flush=True)
    print(f"{'─'*50}", flush=True)

    pos_results = {}

    for steg_method in steg_methods:
        train_stegos, true_carriers_per_img = make_stego(vae, train_imgs, steg_method)
        # All images use same seed -> same carriers; use image 0 as ground truth
        true_carriers = true_carriers_per_img[0]

        pp = PositionalCarrierProbe()
        pp.fit(vae, train_imgs, train_stegos)
        metrics = pp.evaluate(true_carriers, k=20)

        print(f"\n  {steg_method}:", flush=True)
        print(f"    Carrier recovery: precision={metrics['precision']:.2f}  "
              f"recall={metrics['recall']:.2f}  f1={metrics['f1']:.2f}  "
              f"overlap={metrics['overlap']}/20", flush=True)

        pos_results[steg_method] = {
            'heatmap': pp.heatmap(),
            'true_carriers': true_carriers,
            'predicted_top': pp.top_positions(k=20),
            **metrics,
        }

    # ─── 3. COLLUSION PATTERN PROBE ──────────────────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("3. Collusion Detection (same-key vs diff-key vs clean)", flush=True)
    print(f"{'─'*50}", flush=True)

    collusion_probe = CollusionPatternProbe()

    # Same-key stego: all images encoded with seed=42 -> correlated carrier positions
    same_key_stegos, _ = make_stego(vae, test_imgs, 'PatchSteg', seed=42)

    # Different-key stego: half with seed=42, half with seed=99 -> no shared carriers
    diff_a, _ = make_stego(vae, test_imgs[:4], 'PatchSteg', seed=42)
    diff_b, _ = make_stego(vae, test_imgs[:4], 'PatchSteg', seed=99)
    mixed_stegos = diff_a[:2] + diff_b[:2] + diff_a[2:] + diff_b[2:]

    clean_result = collusion_probe.score_set(vae, test_imgs)
    same_key_result = collusion_probe.score_set(vae, same_key_stegos)
    diff_key_result = collusion_probe.score_set(vae, mixed_stegos)

    print(f"  Clean images      PC1 var = {clean_result['variance_explained_pc1']:.4f}  "
          f"(expected: low)", flush=True)
    print(f"  Same-key stego    PC1 var = {same_key_result['variance_explained_pc1']:.4f}  "
          f"(expected: HIGH — collusion signal)", flush=True)
    print(f"  Diff-key stego    PC1 var = {diff_key_result['variance_explained_pc1']:.4f}  "
          f"(expected: low — no collusion)", flush=True)

    # ─── FIGURE 1: Probe ROC curves ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {
        'Global Latent Probe': '#3498db',
        'Residual Probe': '#e74c3c',
    }

    for ax, steg_method in zip(axes, steg_methods):
        for probe_label, color in colors.items():
            key = f'{probe_label} | {steg_method}'
            if key not in probe_results:
                continue
            r = probe_results[key]
            y = [0] * len(r['clean_scores']) + [1] * len(r['stego_scores'])
            s = r['clean_scores'] + r['stego_scores']
            fpr, tpr, _ = roc_curve(y, s)
            roc_auc = sklearn_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{probe_label} (AUC={roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Linear Probe ROC — {steg_method}')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'defense_probe_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved defense_probe_roc.png", flush=True)

    # ─── FIGURE 2: Positional carrier heatmaps ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Positional Carrier Attribution (Linear Probe)', fontsize=13, fontweight='bold')

    for col, steg_method in enumerate(steg_methods):
        r = pos_results[steg_method]
        hm = r['heatmap']
        true_c = np.array(r['true_carriers'])
        pred_c = np.array(r['predicted_top'])

        # Row 0: attribution heatmap
        ax_hm = axes[0, col]
        im = ax_hm.imshow(hm, cmap='hot', vmin=0, vmax=1, aspect='auto')
        # Overlay true and predicted carriers
        ax_hm.scatter(true_c[:, 1], true_c[:, 0], c='cyan', s=30, marker='o',
                      label='True carriers', zorder=3, linewidths=0.5, edgecolors='white')
        ax_hm.scatter(pred_c[:, 1], pred_c[:, 0], c='lime', s=15, marker='x',
                      label='Predicted top-20', zorder=4, linewidths=1.5)
        ax_hm.set_title(f'{steg_method}: Attribution Heatmap\n'
                        f'P={r["precision"]:.2f}  R={r["recall"]:.2f}  F1={r["f1"]:.2f}',
                        fontsize=10)
        ax_hm.set_xlabel('Latent col')
        ax_hm.set_ylabel('Latent row')
        ax_hm.legend(fontsize=7, loc='upper right')
        plt.colorbar(im, ax=ax_hm, shrink=0.7)

        # Row 1: scatter of true vs predicted carrier positions
        ax_pts = axes[1, col]
        ax_pts.scatter(true_c[:, 1], true_c[:, 0], c='limegreen', s=80,
                       marker='o', label='True carriers', zorder=3)
        ax_pts.scatter(pred_c[:, 1], pred_c[:, 0], c='red', s=40,
                       marker='x', linewidths=2, label='Predicted top-20', zorder=2)
        # Highlight correctly recovered carriers
        true_set = set(map(tuple, true_c.tolist()))
        pred_set = set(map(tuple, pred_c.tolist()))
        correct = np.array(list(true_set & pred_set)) if true_set & pred_set else np.empty((0, 2))
        if len(correct):
            ax_pts.scatter(correct[:, 1], correct[:, 0], c='gold', s=120,
                           marker='*', label='Correctly recovered', zorder=5)
        ax_pts.set_xlim(-1, LATENT_SIZE)
        ax_pts.set_ylim(-1, LATENT_SIZE)
        ax_pts.invert_yaxis()
        ax_pts.set_title(f'{steg_method}: True vs Predicted Carriers', fontsize=10)
        ax_pts.legend(fontsize=7)
        ax_pts.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'defense_probe_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved defense_probe_heatmap.png", flush=True)

    # ─── SUMMARY ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    print("Classification probe AUCs:", flush=True)
    for k, v in probe_results.items():
        print(f"  {k}: AUC={v['auc']:.3f}", flush=True)

    print("\nCarrier recovery (positional probe):", flush=True)
    for m, r in pos_results.items():
        print(f"  {m}: P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1']:.2f}", flush=True)

    print("\nCollusion detection (PC1 variance explained):", flush=True)
    print(f"  Clean:      {clean_result['variance_explained_pc1']:.4f}", flush=True)
    print(f"  Same-key:   {same_key_result['variance_explained_pc1']:.4f}", flush=True)
    print(f"  Diff-key:   {diff_key_result['variance_explained_pc1']:.4f}", flush=True)

    print(f"\nTotal time: {time.time()-t_total:.0f}s", flush=True)
