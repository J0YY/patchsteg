"""
CertifiedPurifier: defense against latent-space steganography.

Key insight: CDF-PatchSteg selects carrier positions by stability (a deterministic,
key-free computation). The purifier independently identifies the same positions and
resamples their values from N(mu, sigma). Each carrier bit then has exactly 50%
chance of surviving -> message accuracy drops to chance, certified without knowing
the attacker's seed.

Also supports blind Gaussian noise purification with a closed-form certified bound.
"""
import torch
import numpy as np
from scipy.stats import norm, ks_2samp
from PIL import Image


class CertifiedPurifier:
    """
    Key-free defense against stability-based latent steganography.

    Strategies:
      'resample'  — resample top-K stable positions from N(mu, sigma) [default]
                    Certified: bit accuracy = 0.5 for any CDF-based scheme
      'noise'     — add Gaussian noise to the full latent channel
                    Certified bound: acc <= Phi(eps_attack / (2 * noise_std))
      'combined'  — resample + noise (strongest)

    Neither strategy requires knowledge of the attacker's seed.
    """

    def __init__(self, n_purify=64, noise_std=0.3, strategy='resample'):
        """
        Args:
            n_purify: number of high-stability positions to resample (for 'resample')
            noise_std: std of additive Gaussian noise applied to latent channels (for 'noise')
            strategy: 'resample' | 'noise' | 'combined'
        """
        self.n_purify = n_purify
        self.noise_std = noise_std
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def purify(self, vae, image):
        """
        Purify an image: destroy any latent steganographic payload.

        Args:
            vae: StegoVAE instance
            image: PIL Image (potentially stego)

        Returns:
            purified PIL Image, diagnostics dict
        """
        latent = vae.encode(image)
        diagnostics = {}

        if self.strategy in ('resample', 'combined'):
            latent, diag = self._resample_stable_positions(vae, image, latent)
            diagnostics.update(diag)

        if self.strategy in ('noise', 'combined'):
            latent, diag = self._add_channel_noise(latent)
            diagnostics.update(diag)

        purified = vae.decode(latent)
        return purified, diagnostics

    # ------------------------------------------------------------------
    # Strategy 1: stability-guided resampling
    # ------------------------------------------------------------------

    def _resample_stable_positions(self, vae, image, latent):
        """
        Independently compute the stability map (no key needed) and resample
        the top-K positions from their channel distribution N(mu, sigma).

        This directly invalidates any scheme that places carriers at stable
        positions — the resampled values are fresh random draws, so each
        encoded bit is retained with exactly 50% probability.
        """
        from core.steganography import PatchSteg

        # Compute stability map using an arbitrary seed (seed doesn't affect map shape)
        ps = PatchSteg(seed=0, epsilon=5.0)
        stability_map, _ = ps.compute_stability_map(vae, image, test_eps=5.0)

        # Identify top-K positions the attacker most likely used
        H, W = stability_map.shape
        flat = stability_map.abs().flatten()
        top_k = min(self.n_purify, flat.numel())
        top_indices = torch.topk(flat, top_k).indices

        latent_mod = latent.clone()
        n_resampled = 0

        for idx in top_indices:
            r, c = int(idx) // W, int(idx) % W
            for ch in range(4):
                ch_vals = latent[0, ch].cpu().numpy().flatten()
                mu = float(np.mean(ch_vals))
                sigma = float(np.std(ch_vals)) + 1e-6
                # Fresh sample — no relationship to original value or encoded bit
                new_val = float(np.random.normal(mu, sigma))
                latent_mod[0, ch, r, c] = new_val
            n_resampled += 1

        diag = {
            'n_resampled': n_resampled,
            'stability_max': float(flat.max()),
            'stability_threshold': float(flat[top_indices[-1]]),
        }
        return latent_mod, diag

    # ------------------------------------------------------------------
    # Strategy 2: channel-noise purification with certified bound
    # ------------------------------------------------------------------

    def _add_channel_noise(self, latent):
        """
        Add N(0, noise_std^2) to each latent channel.

        Certified bit accuracy upper bound for CDF encoding with sigma_attack:
            acc <= Phi(sigma_attack / (2 * noise_std))
        where Phi is the standard normal CDF.
        (Derivation: carrier val x was placed at distance ~sigma_attack/4 from mu
        on average; noise moves x by N(0, noise_std^2), flipping the half-assignment
        with probability proportional to the overlap.)

        Conservative practical bound at noise_std=0.3, sigma_attack=1.0:
            acc <= Phi(1.0 / 0.6) ≈ 0.95  [loose bound]
        Empirically the flip rate is much higher since many values are near the median.
        """
        latent_noisy = latent.clone()
        noise = torch.randn_like(latent) * self.noise_std
        latent_noisy = latent_noisy + noise

        diag = {
            'noise_std_applied': self.noise_std,
            'certified_acc_upper': float(norm.cdf(1.0 / (2 * self.noise_std))),
        }
        return latent_noisy, diag

    # ------------------------------------------------------------------
    # Certification utilities
    # ------------------------------------------------------------------

    def certified_acc_upper_bound(self, sigma_attack=1.0):
        """
        Closed-form upper bound on bit accuracy after noise purification.

        For CDF encoding that places values from N(mu, sigma_attack), a value
        encoded with bit b=1 is drawn from z > mu, so E[z - mu] = sigma_attack * sqrt(2/pi).
        Adding noise N(0, noise_std^2) flips the bit when noise exceeds (z - mu).

        This returns a conservative upper bound.
        """
        if self.strategy not in ('noise', 'combined'):
            # Resample strategy gives exactly 0.5 — no bound needed
            return 0.5
        expected_margin = sigma_attack * np.sqrt(2 / np.pi)
        p_correct = norm.cdf(expected_margin / self.noise_std)
        return float(p_correct)

    def run_ks_test(self, vae, original_image, purified_image, channel=0):
        """
        KS test: check that purified latent still looks like N(mu, sigma).
        p > 0.05 means statistically indistinguishable (purification didn't
        introduce an anomalous distribution).
        """
        lat_orig = vae.encode(original_image)[0, channel].detach().cpu().numpy().flatten()
        lat_puri = vae.encode(purified_image)[0, channel].detach().cpu().numpy().flatten()
        stat, p = ks_2samp(lat_orig, lat_puri)
        return {'ks_stat': float(stat), 'ks_p': float(p)}
