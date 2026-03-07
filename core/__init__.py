"""Core PatchSteg exports.

Structure:
  core/attacks/   — attack methods (PatchSteg, CDF, PCA, PSyDUCK, Capacity)
  core/defenses/  — defense methods (sanitizers, detectors, purifiers, probes)
  core/vae.py     — VAE backbone (shared)
  core/metrics.py — evaluation metrics (shared)
  core/analysis.py — latent analysis utilities (shared)
"""

from importlib import import_module

__all__ = [
    "AdaptivePatchSteg",
    "CapacityPatchSteg",
    "CDFPatchSteg",
    "CompactPayloadCodec",
    "PayloadCodec",
    "PCADirections",
    "PCAPatchSteg",
    "PSyDUCKSteg",
    "PatchSteg",
    "StegoVAE",
]

_EXPORT_MAP = {
    "AdaptivePatchSteg": "core.adaptive_steganography",
    "CapacityPatchSteg": "core.attacks.capacity_steganography",
    "CDFPatchSteg": "core.attacks.cdf_steganography",
    "CompactPayloadCodec": "core.attacks.payload_codec",
    "PayloadCodec": "core.attacks.payload_codec",
    "PCADirections": "core.attacks.pca_directions",
    "PCAPatchSteg": "core.attacks.pca_directions",
    "PSyDUCKSteg": "core.attacks.psyduck_steganography",
    "PatchSteg": "core.attacks.steganography",
    "StegoVAE": "core.vae",
}


def __getattr__(name):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
