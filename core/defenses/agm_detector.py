"""
AGM-inspired steganography detector using VAE round-trip residuals.

Based on: AGM — Autoencoder-Guided Watermark Detection
          arXiv 2510.07219, 2025
          https://arxiv.org/abs/2510.07219

Core insight from the paper:
  "VAE decoder amplifies latent perturbations into pixel artifacts."
  When a latent z is perturbed (stego), the decoder renders those
  perturbations into subtle pixel-level changes. Re-encoding the decoded
  image amplifies these artifacts in a way that correlates with the
  original perturbation — producing a detectable residual.

Detection method:
  1. Compute clean round-trip:  z_rt = E(D(z))
  2. Residual:                  r = z_rt - z
  3. Analyze r: stego images have higher-energy, more structured residuals
     than clean images because the decoder's non-linearity converts
     directional latent perturbations into coherent pixel artifacts that
     survive re-encoding.

This is a key motivation for CDF-PatchSteg: by not perturbing the latent
distribution at all, CDF encoding produces residuals indistinguishable from
clean round-trip residuals.
"""
import numpy as np
import torch
from scipy.stats import kstest, norm


class AGMDetector:
    """
    Key-free steganography detector based on VAE round-trip residual analysis.

    Detects perturbation-based attacks (PatchSteg ±ε, PSyDUCK-style) by
    measuring the statistical signature of VAE decoder amplification.

    Limitation: CDF-PatchSteg preserves the latent distribution so its
    round-trip residual is indistinguishable from clean images — this is
    exactly the gap AGM motivates.
    """

    def __init__(self, threshold_energy=0.05, threshold_ks=0.05):
        """
        Args:
            threshold_energy: residual channel energy above which image is flagged
            threshold_ks: KS test p-value below which residual is flagged as non-Gaussian
        """
        self.threshold_energy = threshold_energy
        self.threshold_ks = threshold_ks

    def compute_residual(self, vae, image):
        """
        Compute the VAE round-trip latent residual.

        Returns:
            residual: [1, 4, H, W] tensor  (z_rt - z)
            z: original latent
            z_rt: round-tripped latent
        """
        z = vae.encode(image)
        recon = vae.decode(z)
        z_rt = vae.encode(recon)
        residual = z_rt - z
        return residual, z, z_rt

    def extract_features(self, vae, image):
        """
        Extract 20-dim feature vector from round-trip residual.
        Features per channel (4 channels × 5 stats):
          mean, std, mean_abs, max_abs, ks_stat_vs_gaussian
        """
        residual, z, _ = self.compute_residual(vae, image)
        r = residual[0].detach().cpu().numpy()

        feats = []
        for ch in range(4):
            vals = r[ch].flatten()
            mu, sigma = vals.mean(), vals.std() + 1e-8
            ks_stat, _ = kstest((vals - mu) / sigma, 'norm')
            feats.extend([
                float(mu),
                float(sigma),
                float(np.abs(vals).mean()),
                float(np.abs(vals).max()),
                float(ks_stat),
            ])
        return np.array(feats)

    def detect(self, vae, image):
        """
        Run detection heuristic on a single image.

        Returns:
            suspicious (bool): True if image appears to contain steganography
            score (float): 0–1 suspicion score (higher = more suspicious)
            report (dict): per-channel diagnostics
        """
        residual, z, z_rt = self.compute_residual(vae, image)
        r = residual[0].detach().cpu().numpy()

        channel_energies = []
        channel_ks_ps = []
        report = {}

        for ch in range(4):
            vals = r[ch].flatten()
            mu, sigma = vals.mean(), vals.std() + 1e-8
            energy = float(np.abs(vals).mean())
            ks_stat, ks_p = kstest((vals - mu) / sigma, 'norm')
            channel_energies.append(energy)
            channel_ks_ps.append(float(ks_p))
            report[f'ch{ch}_residual_energy'] = energy
            report[f'ch{ch}_ks_p'] = float(ks_p)
            report[f'ch{ch}_ks_stat'] = float(ks_stat)

        mean_energy = float(np.mean(channel_energies))
        min_ks_p = float(np.min(channel_ks_ps))

        # Flagged if residual energy is unusually high OR distribution is non-Gaussian
        energy_flag = mean_energy > self.threshold_energy
        ks_flag = min_ks_p < self.threshold_ks

        # Score: blend normalised energy and KS evidence
        energy_score = min(1.0, mean_energy / (self.threshold_energy * 3))
        ks_score = max(0.0, 1.0 - min_ks_p / self.threshold_ks) if ks_flag else 0.0
        score = float(np.clip(0.6 * energy_score + 0.4 * ks_score, 0, 1))

        suspicious = energy_flag or ks_flag
        report['mean_residual_energy'] = mean_energy
        report['min_ks_p'] = min_ks_p
        report['energy_flag'] = energy_flag
        report['ks_flag'] = ks_flag
        report['score'] = score

        return suspicious, score, report

    def analyze_batch(self, vae, images, labels=None):
        """
        Analyze a batch of images and optionally compute detection metrics.

        Args:
            vae: StegoVAE
            images: list of PIL Images
            labels: optional list of 0/1 ground-truth labels (0=clean, 1=stego)

        Returns:
            results: list of (suspicious, score, report) tuples
            metrics: dict with AUC/accuracy if labels provided
        """
        results = [self.detect(vae, img) for img in images]

        metrics = {}
        if labels is not None:
            from sklearn.metrics import roc_auc_score
            scores = [r[1] for r in results]
            preds = [int(r[0]) for r in results]
            try:
                metrics['auc'] = float(roc_auc_score(labels, scores))
            except Exception:
                metrics['auc'] = float('nan')
            correct = sum(p == l for p, l in zip(preds, labels))
            metrics['accuracy'] = correct / len(labels)

        return results, metrics
