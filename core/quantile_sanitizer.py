"""
QuantileShuffleSanitizer: distribution-preserving defense against CDF-PatchSteg.

CDF-PatchSteg encodes each bit by placing a carrier's latent value in the
upper or lower Gaussian half of N(mu, sigma^2). Decoding checks which half
the re-encoded value falls in.

Defense: for each latent channel, with probability p, reflect each value
around the channel mean (new_v = 2*mu - v). This swaps upper <-> lower half.

Distribution preservation: reflection around the mean is an involution that
maps N(mu, sigma^2) to itself, so the per-channel marginal is exactly preserved.

Theoretical guarantee: each CDF carrier bit is independently destroyed with
probability p. For p=0.5, bit accuracy -> 50% (chance level), regardless of
which positions are carriers or what the hidden message is.

This guarantee is unconditional -- it holds without knowing the seed, the
carrier positions, or the encoding scheme.
"""
import numpy as np
import torch
from PIL import Image


class QuantileShuffleSanitizer:
    """
    Defeats CDF-PatchSteg with a theoretical 50% bit-accuracy guarantee.

    Parameters
    ----------
    p : float
        Probability of reflecting each latent position around its channel mean.
        p=0.5 gives the 50% accuracy floor; lower p is gentler but less effective.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.seed = seed
        self.name = f"QuantileShuffle(p={p})"

    def sanitize(self, vae, image):
        """
        Apply the sanitizer to a single PIL image. Returns a sanitized PIL image.

        Preserves: per-channel latent marginal distribution N(mu, sigma^2).
        Destroys:  upper/lower half assignment at each position (CDF encoding).
        """
        latent = vae.encode(image)           # [1, 4, H, W]
        lat = latent.clone().float()

        rng = np.random.RandomState(self.seed)

        for ch in range(lat.shape[1]):
            ch_vals = lat[0, ch].detach().cpu().numpy()   # [H, W]
            mu = float(ch_vals.mean())
            # Reflect around mu: new_v = 2*mu - v
            flip_mask = rng.rand(*ch_vals.shape) < self.p
            new_vals = np.where(flip_mask, 2.0 * mu - ch_vals, ch_vals)
            lat[0, ch] = torch.tensor(new_vals, dtype=lat.dtype)

        return vae.decode(lat)

    def sanitize_latent(self, latent):
        """
        Apply the sanitizer directly to a latent tensor [1, 4, H, W].
        Returns the modified latent (does not call VAE decode).
        Useful when latents are pre-computed and cached.
        """
        lat = latent.clone().float()
        rng = np.random.RandomState(self.seed)
        for ch in range(lat.shape[1]):
            ch_vals = lat[0, ch].detach().cpu().numpy()
            mu = float(ch_vals.mean())
            flip_mask = rng.rand(*ch_vals.shape) < self.p
            new_vals = np.where(flip_mask, 2.0 * mu - ch_vals, ch_vals)
            lat[0, ch] = torch.tensor(new_vals, dtype=lat.dtype)
        return lat
