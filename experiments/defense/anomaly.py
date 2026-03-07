#!/usr/bin/env python
"""
Defense (b): Anomaly detection in VAE latent space.

Detectors (all unsupervised — calibrated on clean images only):

1. KSTestDetector       — per-channel Kolmogorov-Smirnov test against a
                          calibrated N(mu, sigma) baseline. PatchSteg epsilon
                          modifications create distributional drift.

2. RoundTripResidualDetector — VAE round-trip residual L2 norm. Modifications
                          that lie off the VAE manifold produce anomalously
                          large residuals on re-encode.

3. EntropyAnomalyDetector — per-patch differential entropy of latent values.
                          Steganographic bit-setting alters local entropy at
                          carrier positions.

4. SpectralAnomalyDetector — 2D FFT power spectrum KL-divergence from a
                          clean reference. Positional carrier patterns may
                          leave frequency-domain signatures.

All detectors output a scalar anomaly score; higher = more suspicious.
We evaluate each by ROC AUC on held-out clean vs stego images.

Generates: defense_anomaly_roc.png, defense_anomaly_distributions.png
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
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, roc_curve, auc as sklearn_auc

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.cdf_steganography import CDFPatchSteg

IMG_SIZE = 256
FIG_DIR = Path(__file__).resolve().parent.parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


# ─── ANOMALY DETECTORS ───────────────────────────────────────────────────────

class KSTestDetector:
    """
    Per-channel KS test against a calibrated normal baseline.

    Fits N(mu_ch, sigma_ch) to each channel of the latent space across
    a held-out set of clean images. For each new image, computes the
    KS statistic against this reference distribution per channel.
    Score = max KS statistic across all 4 channels.
    """

    def __init__(self):
        self.channel_params = None  # list of (mu, sigma) per channel

    def fit(self, vae, clean_images):
        all_data = [[] for _ in range(4)]
        for img in clean_images:
            latent = vae.encode(img)[0].cpu().numpy()
            for ch in range(4):
                all_data[ch].extend(latent[ch].flatten().tolist())
        self.channel_params = [
            (float(np.mean(d)), float(np.std(d))) for d in all_data
        ]

    def score(self, vae, image) -> float:
        latent = vae.encode(image)[0].cpu().numpy()
        ks_stats = []
        for ch in range(4):
            x = latent[ch].flatten()
            mu, sigma = self.channel_params[ch]
            ks_stat, _ = scipy_stats.kstest(x, 'norm', args=(mu, sigma))
            ks_stats.append(ks_stat)
        return float(max(ks_stats))

    @property
    def name(self):
        return 'KS Test'


class RoundTripResidualDetector:
    """
    Round-trip residual anomaly detector.

    Encodes image -> decodes -> re-encodes. The residual (latent_rt - latent)
    captures off-manifold modifications. Clean images yield small, consistent
    residuals; stego images that perturb latents off-manifold yield larger ones.
    Score = z-score of the residual L2 norm relative to clean calibration.
    """

    def __init__(self):
        self.baseline_mean = None
        self.baseline_std = None

    def _residual_norm(self, vae, image):
        latent = vae.encode(image)
        recon = vae.decode(latent)
        latent_rt = vae.encode(recon)
        return float(torch.norm(latent_rt - latent).item())

    def fit(self, vae, clean_images):
        norms = [self._residual_norm(vae, img) for img in clean_images]
        self.baseline_mean = float(np.mean(norms))
        self.baseline_std = float(np.std(norms)) + 1e-8

    def score(self, vae, image) -> float:
        norm = self._residual_norm(vae, image)
        return (norm - self.baseline_mean) / self.baseline_std

    @property
    def name(self):
        return 'RT Residual'


class EntropyAnomalyDetector:
    """
    Per-patch differential entropy anomaly detector.

    Tiles the latent into (patch_size x patch_size) spatial patches, then
    estimates per-patch differential entropy via a Gaussian fit to each patch.
    Steganographic bits injected at carrier positions alter local entropy.
    Score = mean absolute z-score of patch entropies vs clean calibration.
    """

    def __init__(self, patch_size=4):
        self.patch_size = patch_size
        self.baseline_entropy = None
        self.baseline_std = None

    def _patch_entropies(self, vae, image):
        latent = vae.encode(image)[0].cpu().numpy()
        H, W = latent.shape[1], latent.shape[2]
        p = self.patch_size
        entropies = []
        for r in range(0, H, p):
            for c in range(0, W, p):
                for ch in range(4):
                    patch = latent[ch, r:r+p, c:c+p].flatten()
                    if len(patch) < 2:
                        continue
                    sig = patch.std() + 1e-8
                    # Gaussian differential entropy: 0.5 * log(2*pi*e*sigma^2)
                    entropies.append(float(0.5 * np.log(2 * np.pi * np.e * sig ** 2)))
        return np.array(entropies)

    def fit(self, vae, clean_images):
        all_ents = np.stack([self._patch_entropies(vae, img) for img in clean_images])
        self.baseline_entropy = all_ents.mean(axis=0)
        self.baseline_std = all_ents.std(axis=0) + 1e-8

    def score(self, vae, image) -> float:
        ents = self._patch_entropies(vae, image)
        z = np.abs((ents - self.baseline_entropy) / self.baseline_std)
        return float(z.mean())

    @property
    def name(self):
        return 'Entropy MAD'


class SpectralAnomalyDetector:
    """
    2D FFT power spectrum KL-divergence detector.

    Computes the log-magnitude power spectrum of each latent channel.
    Bins into a histogram and computes KL divergence from a clean reference.
    Positional carrier patterns leave detectable frequency-domain signatures.
    Score = sum of per-channel KL divergences from clean baseline spectra.
    """

    def __init__(self, n_bins=16):
        self.n_bins = n_bins
        self.baseline_hist = None  # list of histograms, one per channel

    def _spectral_histograms(self, vae, image):
        latent = vae.encode(image)[0].cpu()
        hists = []
        for ch in range(4):
            fft = torch.fft.fft2(latent[ch].float())
            power = torch.log1p(torch.abs(fft)).numpy().flatten()
            hist, _ = np.histogram(power, bins=self.n_bins, range=(0, 5), density=True)
            hists.append(hist + 1e-10)  # Laplace smoothing for KL
        return hists

    def fit(self, vae, clean_images):
        all_hists = [self._spectral_histograms(vae, img) for img in clean_images]
        self.baseline_hist = []
        for ch in range(4):
            ch_hists = np.stack([h[ch] for h in all_hists])
            self.baseline_hist.append(ch_hists.mean(axis=0))

    def score(self, vae, image) -> float:
        hists = self._spectral_histograms(vae, image)
        return float(sum(scipy_stats.entropy(hists[ch], self.baseline_hist[ch])
                         for ch in range(4)))

    @property
    def name(self):
        return 'Spectral KL'


# ─── DATA & STEGO HELPERS ────────────────────────────────────────────────────

def get_natural_images(n=16):
    from torchvision.datasets import CIFAR10
    print("  Loading CIFAR-10...", flush=True)
    ds = CIFAR10(root='/tmp/cifar10', train=True, download=True)
    picked = []
    for img, _ in ds:
        picked.append(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        if len(picked) >= n:
            break
    return picked


def make_stego(vae, images, method='PatchSteg'):
    stegos = []
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
    return stegos


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_total = time.time()
    print("=" * 60, flush=True)
    print("DEFENSE (b): LATENT ANOMALY DETECTION", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading VAE...", flush=True)
    vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
    print("Loading images...", flush=True)
    images = get_natural_images(16)
    train_imgs = images[:8]
    test_imgs = images[8:]

    detectors = [
        KSTestDetector(),
        RoundTripResidualDetector(),
        EntropyAnomalyDetector(patch_size=4),
        SpectralAnomalyDetector(n_bins=16),
    ]

    print("\nCalibrating detectors on clean training images...", flush=True)
    for det in detectors:
        det.fit(vae, train_imgs)
        print(f"  Calibrated: {det.name}", flush=True)

    steg_methods = ['PatchSteg', 'CDF']
    auc_table = {}
    score_table = {}  # for plotting distributions

    for steg_method in steg_methods:
        print(f"\n{'─'*50}", flush=True)
        print(f"Steganography: {steg_method}", flush=True)
        stego_imgs = make_stego(vae, test_imgs, steg_method)
        auc_table[steg_method] = {}
        score_table[steg_method] = {}

        for det in detectors:
            clean_scores = [det.score(vae, img) for img in test_imgs]
            stego_scores = [det.score(vae, img) for img in stego_imgs]
            y = [0] * len(clean_scores) + [1] * len(stego_scores)
            all_scores = clean_scores + stego_scores
            try:
                det_auc = roc_auc_score(y, all_scores)
            except Exception:
                det_auc = 0.5

            auc_table[steg_method][det.name] = det_auc
            score_table[steg_method][det.name] = {
                'clean': clean_scores, 'stego': stego_scores
            }
            print(f"  {det.name:>14s}: AUC={det_auc:.3f}  "
                  f"clean_mu={np.mean(clean_scores):.3f}  "
                  f"stego_mu={np.mean(stego_scores):.3f}", flush=True)

    # ─── FIGURE 1: ROC curves ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {
        'KS Test': '#3498db',
        'RT Residual': '#e74c3c',
        'Entropy MAD': '#2ecc71',
        'Spectral KL': '#9b59b6',
    }

    for ax, steg_method in zip(axes, steg_methods):
        stego_imgs = make_stego(vae, test_imgs, steg_method)
        for det in detectors:
            cs = score_table[steg_method][det.name]['clean']
            ss = score_table[steg_method][det.name]['stego']
            y = [0] * len(cs) + [1] * len(ss)
            scores = cs + ss
            fpr, tpr, _ = roc_curve(y, scores)
            roc_auc = sklearn_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[det.name], linewidth=2,
                    label=f'{det.name} (AUC={roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Anomaly Detection ROC — {steg_method}')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'defense_anomaly_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved defense_anomaly_roc.png", flush=True)

    # ─── FIGURE 2: Score distributions ───────────────────────────────────────
    n_det = len(detectors)
    fig, axes = plt.subplots(n_det, 2, figsize=(14, 4 * n_det))
    fig.suptitle('Anomaly Score Distributions: Clean vs Stego', fontsize=13, fontweight='bold')

    for col, steg_method in enumerate(steg_methods):
        for row, det in enumerate(detectors):
            ax = axes[row, col]
            cs = score_table[steg_method][det.name]['clean']
            ss = score_table[steg_method][det.name]['stego']
            lo = min(min(cs), min(ss))
            hi = max(max(cs), max(ss))
            margin = (hi - lo) * 0.15 + 1e-8
            bins = np.linspace(lo - margin, hi + margin, 20)
            ax.hist(cs, bins=bins, alpha=0.6, color='steelblue',
                    label='Clean', density=True)
            ax.hist(ss, bins=bins, alpha=0.6, color='tomato',
                    label='Stego', density=True)
            ax.set_title(f'{det.name} | {steg_method}  '
                         f'(AUC={auc_table[steg_method][det.name]:.2f})', fontsize=9)
            ax.set_xlabel('Anomaly Score')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'defense_anomaly_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved defense_anomaly_distributions.png", flush=True)

    # ─── SUMMARY ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY: Detection AUC", flush=True)
    print(f"{'='*60}", flush=True)
    header = f"{'Detector':>14s}  {'PatchSteg':>10s}  {'CDF':>10s}"
    print(header, flush=True)
    for det in detectors:
        ps_auc = auc_table['PatchSteg'][det.name]
        cdf_auc = auc_table['CDF'][det.name]
        print(f"  {det.name:>12s}  {ps_auc:>10.3f}  {cdf_auc:>10.3f}", flush=True)
    print(f"\nTotal time: {time.time()-t_total:.0f}s", flush=True)
