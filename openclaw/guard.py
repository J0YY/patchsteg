"""Thin adapter for using the PatchSteg universal guard inside OpenClaw."""
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

from core.defense import GuardDecision, UniversalPatchStegGuard
from core.vae import StegoVAE


def _ensure_pil_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.convert("RGB")


@dataclass
class OpenClawGuardResult:
    image: Image.Image
    suspicious: bool
    suspicion_score: float
    threshold: float
    active_fraction: float
    positions_touched: int
    top32_mean: float
    top64_mean: float
    roundtrip_p99: float
    local_p99: float

    @classmethod
    def from_decision(cls, image: Image.Image, decision: GuardDecision):
        return cls(image=image, **asdict(decision))

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("image")
        return data


class OpenClawPatchStegGuard:
    """
    Runtime wrapper for applying PatchSteg detection/sanitization in another stack.

    Typical use in a multimodal pipeline:
    1. Call `inspect()` to log a suspicion score.
    2. Call `inspect_and_filter()` before handing the image to the next model.
    """

    def __init__(
        self,
        device: str = "cpu",
        image_size: int = 256,
        strength: float = 1.0,
        detection_threshold: float = 5.0,
        activation_threshold: float = 4.0,
        min_positions: int = 48,
        max_positions: int = 160,
    ):
        self.strength = float(strength)
        self.vae = StegoVAE(device=device, image_size=image_size)
        self.guard = UniversalPatchStegGuard(
            self.vae,
            detection_threshold=detection_threshold,
            activation_threshold=activation_threshold,
            min_positions=min_positions,
            max_positions=max_positions,
        )

    def inspect(self, image: Image.Image | np.ndarray) -> OpenClawGuardResult:
        pil_image = _ensure_pil_image(image)
        decision = self.guard.inspect(pil_image)
        return OpenClawGuardResult.from_decision(pil_image, decision)

    def sanitize(
        self,
        image: Image.Image | np.ndarray,
        strength: float | None = None,
    ) -> OpenClawGuardResult:
        pil_image = _ensure_pil_image(image)
        sanitized, decision = self.guard.sanitize(
            pil_image,
            strength=self.strength if strength is None else float(strength),
        )
        return OpenClawGuardResult.from_decision(sanitized, decision)

    def inspect_and_filter(
        self,
        image: Image.Image | np.ndarray,
        strength: float | None = None,
    ) -> OpenClawGuardResult:
        pil_image = _ensure_pil_image(image)
        filtered, decision = self.guard.inspect_and_filter(
            pil_image,
            strength=self.strength if strength is None else float(strength),
        )
        return OpenClawGuardResult.from_decision(filtered, decision)
