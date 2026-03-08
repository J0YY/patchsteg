"""Seed-free PatchSteg detection and mitigation utilities."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class GuardDecision:
    suspicious: bool
    suspicion_score: float
    threshold: float
    active_fraction: float
    positions_touched: int
    top32_mean: float
    top64_mean: float
    roundtrip_p99: float
    local_p99: float


class UniversalPatchStegGuard:
    """
    Heuristic, seed-free guard for PatchSteg-style channels.

    The guard looks for two signatures that do not require the shared secret:
    1. Local latent outliers relative to a 3x3 neighborhood.
    2. Positions whose latent vectors drift unusually through one extra VAE round-trip.

    It then smooths only the most suspicious positions before handing the image
    to the downstream model.
    """

    def __init__(
        self,
        vae,
        detection_threshold=5.0,
        activation_threshold=4.0,
        min_positions=48,
        max_positions=160,
    ):
        self.vae = vae
        self.detection_threshold = float(detection_threshold)
        self.activation_threshold = float(activation_threshold)
        self.min_positions = int(min_positions)
        self.max_positions = int(max_positions)

    @staticmethod
    def _robust_positive_zscore(values):
        flat = values.flatten()
        median = flat.median()
        mad = (flat - median).abs().median().clamp_min(1e-6)
        scale = 1.4826 * mad
        return torch.relu((values - median) / scale)

    @staticmethod
    def _summarize_map(values):
        flat = values.flatten().detach().cpu().numpy()
        top32 = float(np.mean(np.sort(flat)[-32:]))
        top64 = float(np.mean(np.sort(flat)[-64:]))
        p99 = float(np.percentile(flat, 99))
        return top32, top64, p99

    def analyze_latent(self, latent):
        smooth = F.avg_pool2d(latent, kernel_size=3, stride=1, padding=1)
        local_delta = torch.linalg.norm((latent - smooth)[0].permute(1, 2, 0), dim=-1)

        roundtrip = self.vae.decode(latent)
        latent_roundtrip = self.vae.encode(roundtrip)
        roundtrip_delta = torch.linalg.norm(
            (latent_roundtrip - latent)[0].permute(1, 2, 0), dim=-1
        )

        local_score = self._robust_positive_zscore(local_delta)
        roundtrip_score = self._robust_positive_zscore(roundtrip_delta)
        suspicion_map = local_score + 1.35 * roundtrip_score

        top32_mean, top64_mean, suspicion_p99 = self._summarize_map(suspicion_map)
        _, _, local_p99 = self._summarize_map(local_delta)
        _, _, roundtrip_p99 = self._summarize_map(roundtrip_delta)

        active_fraction = float((suspicion_map > self.activation_threshold).float().mean().item())
        suspicion_score = top32_mean + 12.0 * active_fraction + 0.25 * suspicion_p99

        return {
            "smooth": smooth,
            "suspicion_map": suspicion_map,
            "suspicion_score": float(suspicion_score),
            "active_fraction": active_fraction,
            "top32_mean": top32_mean,
            "top64_mean": top64_mean,
            "roundtrip_p99": roundtrip_p99,
            "local_p99": local_p99,
        }

    def inspect(self, image):
        latent = self.vae.encode(image)
        analysis = self.analyze_latent(latent)
        return self._decision_from_analysis(analysis, positions_touched=0)

    def _decision_from_analysis(self, analysis, positions_touched):
        suspicious = analysis["suspicion_score"] >= self.detection_threshold
        return GuardDecision(
            suspicious=suspicious,
            suspicion_score=analysis["suspicion_score"],
            threshold=self.detection_threshold,
            active_fraction=analysis["active_fraction"],
            positions_touched=int(positions_touched),
            top32_mean=analysis["top32_mean"],
            top64_mean=analysis["top64_mean"],
            roundtrip_p99=analysis["roundtrip_p99"],
            local_p99=analysis["local_p99"],
        )

    def _positions_to_touch(self, suspicion_map):
        total_positions = suspicion_map.numel()
        active_positions = int((suspicion_map > self.activation_threshold).sum().item())
        positions_touched = max(self.min_positions, active_positions * 2)
        return min(self.max_positions, positions_touched, total_positions)

    def _sanitize_from_analysis(self, latent, analysis, strength):
        suspicion_map = analysis["suspicion_map"]
        smooth = analysis["smooth"]
        positions_touched = self._positions_to_touch(suspicion_map)

        top_indices = torch.argsort(suspicion_map.flatten(), descending=True)[:positions_touched]
        latent_sanitized = latent.clone()
        width = suspicion_map.shape[1]
        for idx in top_indices.tolist():
            row = idx // width
            col = idx % width
            latent_sanitized[0, :, row, col] = (
                (1.0 - strength) * latent[0, :, row, col] + strength * smooth[0, :, row, col]
            )
        return self.vae.decode(latent_sanitized), positions_touched

    def sanitize(self, image, strength=1.0):
        latent = self.vae.encode(image)
        analysis = self.analyze_latent(latent)
        sanitized, positions_touched = self._sanitize_from_analysis(
            latent, analysis, strength=strength
        )
        return sanitized, self._decision_from_analysis(analysis, positions_touched)

    def inspect_and_filter(self, image, strength=1.0):
        latent = self.vae.encode(image)
        analysis = self.analyze_latent(latent)
        decision = self._decision_from_analysis(analysis, positions_touched=0)
        if not decision.suspicious:
            return image, decision
        sanitized, positions_touched = self._sanitize_from_analysis(
            latent, analysis, strength=strength
        )
        return sanitized, self._decision_from_analysis(analysis, positions_touched)
