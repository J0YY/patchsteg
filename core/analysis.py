"""Mechanistic analysis of VAE steganographic channels."""
import torch
import numpy as np
from PIL import Image


def channel_importance(vae, steg, image, eps=5.0):
    """
    Test each of the 4 latent channels independently.
    Returns dict mapping channel_idx -> bit_accuracy over all positions.
    """
    latent_clean = vae.encode(image)
    results = {}

    for ch in range(4):
        # Create a direction that only uses this channel
        direction_single = torch.zeros(4)
        direction_single[ch] = 1.0

        # Perturb all positions along this single-channel direction
        latent_test = latent_clean.clone()
        latent_test[0, ch, :, :] += eps

        # Round-trip
        recon = vae.decode(latent_test)
        latent_re = vae.encode(recon)

        # Check how many positions preserved the perturbation
        delta = (latent_re[0, ch, :, :] - latent_clean[0, ch, :, :]).cpu()
        survived = (delta > 0).float().mean().item() * 100
        results[ch] = survived

    return results


def reconstruction_error_map(vae, image):
    """
    Compute per-position reconstruction error WITHOUT any perturbation.
    Returns [64, 64] L2 error map across channels.
    """
    latent_clean = vae.encode(image)
    recon = vae.decode(latent_clean)
    latent_re = vae.encode(recon)

    # Per-position L2 across channels
    error = torch.sqrt(((latent_re[0] - latent_clean[0]) ** 2).sum(dim=0)).cpu()
    return error


def spatial_frequency_map(latent):
    """
    Compute per-position energy in frequency domain.
    Returns [64, 64] frequency energy map.
    """
    H, W = latent.shape[2], latent.shape[3]
    energy = torch.zeros(H, W)
    for ch in range(4):
        fft = torch.fft.fft2(latent[0, ch].cpu().float())
        energy += torch.abs(fft)
    return energy
