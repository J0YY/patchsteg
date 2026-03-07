"""Core PatchSteg exports."""

from importlib import import_module

__all__ = [
    "CapacityPatchSteg",
    "CDFPatchSteg",
    "PayloadCodec",
    "PCADirections",
    "PCAPatchSteg",
    "PSyDUCKSteg",
    "PatchSteg",
    "StegoVAE",
]

_EXPORT_MAP = {
    "CapacityPatchSteg": "core.capacity_steganography",
    "CDFPatchSteg": "core.cdf_steganography",
    "PayloadCodec": "core.payload_codec",
    "PCADirections": "core.pca_directions",
    "PCAPatchSteg": "core.pca_directions",
    "PSyDUCKSteg": "core.psyduck_steganography",
    "PatchSteg": "core.steganography",
    "StegoVAE": "core.vae",
}


def __getattr__(name):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
