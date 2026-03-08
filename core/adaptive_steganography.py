"""Adaptive, detector-aware PatchSteg variant with balanced pairwise modulation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from core.payload_codec import CompactPayloadCodec, DecodedPayload, PayloadCodec, PayloadPacket
from core.steganography import PatchSteg


@dataclass
class AdaptiveEncodeResult:
    stego_image: Image.Image
    packet: PayloadPacket
    header_pairs: list[tuple[tuple[int, int], tuple[int, int]]]
    payload_pairs: list[tuple[tuple[int, int], tuple[int, int]]]
    quality_map: torch.Tensor
    gain_map: torch.Tensor


@dataclass
class AdaptiveDecodeResult:
    success: bool
    text: str | None
    error: str | None
    payload: DecodedPayload
    header_pairs: list[tuple[tuple[int, int], tuple[int, int]]]
    payload_pairs: list[tuple[tuple[int, int], tuple[int, int]]]
    header_confidence: float
    payload_confidence: float


class AdaptivePatchSteg(PatchSteg):
    """
    Stronger PatchSteg variant aimed at defeating simple residual-statistics detectors.

    Main changes relative to the baseline:
    - Pairwise differential embedding with one positive and one negative edit per symbol
    - Per-pair content-aware directions aligned to local latent geometry
    - Joint carrier scoring using stability, local texture, and clean round-trip drift
    - Seeded bit whitening so framed payload headers do not leak fixed patterns
    """

    GRAY_2BIT_TO_SYMBOL = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 1): 2,
        (1, 0): 3,
    }
    SYMBOL_TO_GRAY_2BIT = {
        0: [0, 0],
        1: [0, 1],
        2: [1, 1],
        3: [1, 0],
    }

    def __init__(
        self,
        seed: int = 42,
        epsilon: float = 2.0,
        *,
        bits_per_symbol: int = 1,
        min_gain: float = 0.25,
        max_embed_multiplier: float = 3.5,
        min_pair_distance: int = 1,
        geometry_mix: float = 0.7,
        header_scale: float = 2.0,
        header_repetitions: int = 2,
        texture_weight: float = 0.35,
        roundtrip_weight: float = 0.2,
        codec: PayloadCodec | None = None,
        use_keystream: bool = True,
    ):
        super().__init__(seed=seed, epsilon=epsilon)
        if bits_per_symbol not in (1, 2):
            raise ValueError("AdaptivePatchSteg currently supports 1 or 2 bits per symbol")
        self.bits_per_symbol = int(bits_per_symbol)
        self.min_gain = float(min_gain)
        self.max_embed_multiplier = float(max_embed_multiplier)
        self.min_pair_distance = int(min_pair_distance)
        self.geometry_mix = float(geometry_mix)
        self.header_scale = float(header_scale)
        self.header_repetitions = max(1, int(header_repetitions))
        self.texture_weight = float(texture_weight)
        self.roundtrip_weight = float(roundtrip_weight)
        self.codec = codec if codec is not None else CompactPayloadCodec()
        self.use_keystream = bool(use_keystream)

    @property
    def payload_levels(self) -> tuple[float, ...]:
        if self.bits_per_symbol == 1:
            return (-1.0, 1.0)
        return (-3.0, -1.0, 1.0, 3.0)

    @property
    def header_levels(self) -> tuple[float, float]:
        return (-self.header_scale, self.header_scale)

    @staticmethod
    def _robust_positive_zscore(values: torch.Tensor) -> torch.Tensor:
        flat = values.flatten()
        median = flat.median()
        mad = (flat - median).abs().median().clamp_min(1e-6)
        scale = 1.4826 * mad
        return torch.relu((values - median) / scale)

    def _keystream_bits(self, n_bits: int) -> list[int]:
        rng = np.random.RandomState(self.seed * 104729 + 17)
        return rng.randint(0, 2, size=n_bits).tolist()

    def _mask_bits(self, bits: Sequence[int]) -> list[int]:
        if not self.use_keystream:
            return [int(bit) for bit in bits]
        stream = self._keystream_bits(len(bits))
        return [int(bit) ^ int(mask) for bit, mask in zip(bits, stream)]

    def _unmask_bits(self, bits: Sequence[int]) -> list[int]:
        return self._mask_bits(bits)

    def _bits_to_symbol_indices(self, bits: Sequence[int]) -> list[int]:
        if self.bits_per_symbol == 1:
            return [int(bit) for bit in bits]

        padded = list(bits)
        while len(padded) % self.bits_per_symbol != 0:
            padded.append(0)

        symbols: list[int] = []
        for start in range(0, len(padded), 2):
            pair = (int(padded[start]), int(padded[start + 1]))
            symbols.append(self.GRAY_2BIT_TO_SYMBOL[pair])
        return symbols

    def _symbol_indices_to_bits(self, symbols: Sequence[int], n_bits: int) -> list[int]:
        if self.bits_per_symbol == 1:
            return [int(symbol) for symbol in symbols[:n_bits]]

        bits: list[int] = []
        for symbol in symbols:
            bits.extend(self.SYMBOL_TO_GRAY_2BIT[int(symbol)])
        return bits[:n_bits]

    def required_pairs_for_packet_bits(self, total_bits: int) -> int:
        header_pairs = self.codec.header_bits() * self.header_repetitions
        body_bits = max(0, total_bits - self.codec.header_bits())
        payload_pairs = math.ceil(body_bits / self.bits_per_symbol)
        return header_pairs + payload_pairs

    def compute_quality_map(
        self,
        vae,
        image: Image.Image,
        *,
        test_eps: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probe_eps = float(self.epsilon if test_eps is None else test_eps)
        stability_map, latent_clean = self.compute_stability_map(vae, image, test_eps=probe_eps)

        latent_smooth = F.avg_pool2d(latent_clean, kernel_size=3, stride=1, padding=1)
        texture_map = torch.linalg.norm(
            (latent_clean - latent_smooth)[0].permute(1, 2, 0),
            dim=-1,
        )

        clean_roundtrip = vae.encode(vae.decode(latent_clean))
        roundtrip_map = torch.linalg.norm(
            (clean_roundtrip - latent_clean)[0].permute(1, 2, 0),
            dim=-1,
        )

        stability_score = self._robust_positive_zscore(stability_map.clamp_min(0.0))
        texture_score = self._robust_positive_zscore(texture_map)
        roundtrip_score = self._robust_positive_zscore(roundtrip_map)

        quality_map = (
            stability_score
            + self.texture_weight * texture_score
            + self.roundtrip_weight * roundtrip_score
        )
        gain_map = (stability_map / max(probe_eps, 1e-6)).clamp_min(1e-6)
        return quality_map.cpu(), gain_map.cpu(), latent_clean

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _ordered_spread_positions(
        self,
        score_map: torch.Tensor,
        n_positions: int,
        *,
        exclude: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        side = score_map.shape[0]
        flat = score_map.flatten()
        sorted_indices = torch.argsort(flat, descending=True)
        blocked = set() if exclude is None else set(exclude)

        selected: list[tuple[int, int]] = []
        reserve: list[tuple[int, int]] = []
        for idx in sorted_indices.tolist():
            pos = (idx // side, idx % side)
            if pos in blocked:
                continue
            if all(self._distance(pos, other) > self.min_pair_distance for other in selected):
                selected.append(pos)
            else:
                reserve.append(pos)
            if len(selected) >= n_positions:
                break

        if len(selected) < n_positions:
            for pos in reserve:
                if pos not in selected:
                    selected.append(pos)
                if len(selected) >= n_positions:
                    break
        return selected[:n_positions]

    @staticmethod
    def _pair_positions(
        positions: Sequence[tuple[int, int]],
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        return [
            (positions[idx], positions[idx + 1])
            for idx in range(0, len(positions) - 1, 2)
        ]

    @staticmethod
    def _top_positions(
        score_map: torch.Tensor,
        n_positions: int,
        *,
        exclude: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        side = score_map.shape[0]
        flat = score_map.flatten()
        sorted_indices = torch.argsort(flat, descending=True)
        blocked = set() if exclude is None else set(exclude)
        positions: list[tuple[int, int]] = []
        for idx in sorted_indices.tolist():
            pos = (idx // side, idx % side)
            if pos in blocked:
                continue
            positions.append(pos)
            if len(positions) >= n_positions:
                break
        return positions

    def select_carrier_pairs(
        self,
        vae,
        image: Image.Image,
        n_pairs: int,
        *,
        test_eps: float | None = None,
    ) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], torch.Tensor, torch.Tensor, torch.Tensor]:
        quality_map, gain_map, latent_clean = self.compute_quality_map(
            vae,
            image,
            test_eps=test_eps,
        )
        positions = self._ordered_spread_positions(quality_map, n_pairs * 2)
        pairs = self._pair_positions(positions)
        return pairs[:n_pairs], quality_map, gain_map, latent_clean

    def _seeded_random_direction(self, pair_idx: int) -> torch.Tensor:
        rng = torch.Generator()
        rng.manual_seed(self.seed * 1000003 + pair_idx * 8191)
        direction = torch.randn(4, generator=rng)
        return direction / direction.norm().clamp_min(1e-6)

    def _local_geometry_direction(
        self,
        latent: torch.Tensor,
        row: int,
        col: int,
        *,
        radius: int = 1,
    ) -> torch.Tensor:
        h = latent.shape[2]
        w = latent.shape[3]
        r0 = max(0, row - radius)
        r1 = min(h, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(w, col + radius + 1)
        neighborhood = latent[0, :, r0:r1, c0:c1].permute(1, 2, 0).reshape(-1, 4)
        centered = neighborhood - neighborhood.mean(dim=0, keepdim=True)
        cov = centered.T @ centered
        eigvals, eigvecs = torch.linalg.eigh(cov)
        direction = eigvecs[:, int(torch.argmax(eigvals).item())]
        return direction / direction.norm().clamp_min(1e-6)

    def _pair_direction(
        self,
        latent_reference: torch.Tensor,
        pair: tuple[tuple[int, int], tuple[int, int]],
        pair_idx: int,
    ) -> torch.Tensor:
        (ra, ca), (rb, cb) = pair
        geom_a = self._local_geometry_direction(latent_reference, ra, ca)
        geom_b = self._local_geometry_direction(latent_reference, rb, cb)
        geom = geom_a + geom_b
        geom_norm = geom.norm().clamp_min(1e-6)
        geom = geom / geom_norm
        rand = self._seeded_random_direction(pair_idx).to(latent_reference.device)
        mixed = self.geometry_mix * geom + (1.0 - self.geometry_mix) * rand
        mixed = mixed / mixed.norm().clamp_min(1e-6)
        if torch.dot(mixed, rand) < 0:
            mixed = -mixed
        return mixed

    def _pair_amplitudes(
        self,
        gain_a: float,
        gain_b: float,
        target_level: float,
    ) -> tuple[float, float]:
        safe_a = max(gain_a, self.min_gain)
        safe_b = max(gain_b, self.min_gain)
        target_diff = self.epsilon * float(target_level)
        half_diff = 0.5 * target_diff
        max_amp = self.max_embed_multiplier * self.epsilon
        amp_a = max(-max_amp, min(max_amp, half_diff / safe_a))
        amp_b = max(-max_amp, min(max_amp, -half_diff / safe_b))
        return float(amp_a), float(amp_b)

    def _expected_pair_difference(
        self,
        gain_a: float,
        gain_b: float,
        target_level: float,
    ) -> float:
        amp_a, amp_b = self._pair_amplitudes(gain_a, gain_b, target_level)
        return gain_a * amp_a - gain_b * amp_b

    def _encode_symbol_indices(
        self,
        latent: torch.Tensor,
        carrier_pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
        symbol_indices: Sequence[int],
        *,
        gain_map: torch.Tensor,
        latent_reference: torch.Tensor,
        levels: Sequence[float],
    ) -> torch.Tensor:
        if len(carrier_pairs) != len(symbol_indices):
            raise ValueError("carrier_pairs and symbol_indices must have the same length")
        latent_mod = latent.clone()
        gain_map_dev = gain_map.to(latent.device)
        for pair_idx, (pair, symbol_idx) in enumerate(zip(carrier_pairs, symbol_indices)):
            direction = self._pair_direction(latent_reference, pair, pair_idx).to(latent.device)
            (ra, ca), (rb, cb) = pair
            gain_a = float(gain_map_dev[ra, ca].item())
            gain_b = float(gain_map_dev[rb, cb].item())
            amp_a, amp_b = self._pair_amplitudes(gain_a, gain_b, float(levels[int(symbol_idx)]))
            latent_mod[0, :, ra, ca] += amp_a * direction
            latent_mod[0, :, rb, cb] += amp_b * direction
        return latent_mod

    def _decode_symbol_indices(
        self,
        latent_clean: torch.Tensor,
        latent_received: torch.Tensor,
        carrier_pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
        *,
        gain_map: torch.Tensor,
        levels: Sequence[float],
    ) -> tuple[list[int], list[float]]:
        gain_map_dev = gain_map.to(latent_clean.device)
        decoded: list[int] = []
        confidences: list[float] = []
        for pair_idx, pair in enumerate(carrier_pairs):
            direction = self._pair_direction(latent_clean, pair, pair_idx).to(latent_clean.device)
            (ra, ca), (rb, cb) = pair
            delta_a = torch.dot(
                latent_received[0, :, ra, ca] - latent_clean[0, :, ra, ca],
                direction,
            ).item()
            delta_b = torch.dot(
                latent_received[0, :, rb, cb] - latent_clean[0, :, rb, cb],
                direction,
            ).item()
            diff = delta_a - delta_b
            gain_a = float(gain_map_dev[ra, ca].item())
            gain_b = float(gain_map_dev[rb, cb].item())
            expected = [
                self._expected_pair_difference(gain_a, gain_b, float(level))
                for level in levels
            ]
            distances = [abs(diff - target) for target in expected]
            best = int(min(range(len(expected)), key=lambda idx: distances[idx]))
            ordered = sorted(distances)
            margin = ordered[1] - ordered[0] if len(ordered) > 1 else ordered[0]
            decoded.append(best)
            confidences.append(float(max(0.0, margin)))
        return decoded, confidences

    def encode_message(
        self,
        latent: torch.Tensor,
        carrier_pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
        bits: Sequence[int],
        *,
        gain_map: torch.Tensor,
        latent_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        symbols = [int(bit) for bit in bits]
        reference = latent if latent_reference is None else latent_reference
        return self._encode_symbol_indices(
            latent,
            carrier_pairs,
            symbols,
            gain_map=gain_map,
            latent_reference=reference,
            levels=(-1.0, 1.0),
        )

    def decode_message(
        self,
        latent_clean: torch.Tensor,
        latent_received: torch.Tensor,
        carrier_pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
        *,
        gain_map: torch.Tensor,
    ) -> tuple[list[int], list[float]]:
        return self._decode_symbol_indices(
            latent_clean,
            latent_received,
            carrier_pairs,
            gain_map=gain_map,
            levels=(-1.0, 1.0),
        )

    def encode_text(
        self,
        vae,
        image: Image.Image,
        text: str,
        *,
        enable_compression: bool = True,
        compression_level: int = 9,
        test_eps: float | None = None,
    ) -> AdaptiveEncodeResult:
        packet = self.codec.pack_text(
            text,
            enable_compression=enable_compression,
            compression_level=compression_level,
        )
        masked_bits = self._mask_bits(packet.bits)
        total_pairs = self.required_pairs_for_packet_bits(len(masked_bits))
        _, quality_map, gain_map, latent_clean = self.select_carrier_pairs(
            vae,
            image,
            total_pairs,
            test_eps=test_eps,
        )

        header_len = self.codec.header_bits()
        header_bits = masked_bits[:header_len]
        body_bits = masked_bits[header_len:]
        header_symbols = []
        for bit in header_bits:
            header_symbols.extend([int(bit)] * self.header_repetitions)
        header_pair_count = len(header_symbols)
        payload_symbols = self._bits_to_symbol_indices(body_bits)
        header_positions = self._top_positions(quality_map, header_pair_count * 2)
        header_pairs = self._pair_positions(header_positions)
        payload_positions = self._ordered_spread_positions(
            quality_map,
            len(payload_symbols) * 2,
            exclude=set(header_positions),
        )
        payload_pairs = self._pair_positions(payload_positions)
        if len(payload_pairs) < len(payload_symbols):
            raise ValueError("Not enough carrier pairs selected for payload body")

        latent_mod = self._encode_symbol_indices(
            latent_clean,
            header_pairs,
            header_symbols,
            gain_map=gain_map,
            latent_reference=latent_clean,
            levels=self.header_levels,
        )
        if payload_symbols:
            latent_mod = self._encode_symbol_indices(
                latent_mod,
                payload_pairs[:len(payload_symbols)],
                payload_symbols,
                gain_map=gain_map,
                latent_reference=latent_clean,
                levels=self.payload_levels,
            )

        return AdaptiveEncodeResult(
            stego_image=vae.decode(latent_mod),
            packet=packet,
            header_pairs=header_pairs,
            payload_pairs=payload_pairs[:len(payload_symbols)],
            quality_map=quality_map,
            gain_map=gain_map,
        )

    def decode_text(
        self,
        vae,
        cover_image: Image.Image,
        stego_image: Image.Image,
        *,
        test_eps: float | None = None,
    ) -> AdaptiveDecodeResult:
        quality_map, gain_map, latent_clean = self.compute_quality_map(
            vae,
            cover_image,
            test_eps=test_eps,
        )
        latent_received = vae.encode(stego_image)

        header_len = self.codec.header_bits()
        header_pair_count = header_len * self.header_repetitions
        header_positions = self._top_positions(quality_map, header_pair_count * 2)
        header_pairs = self._pair_positions(header_positions)
        header_symbols, header_conf = self._decode_symbol_indices(
            latent_clean,
            latent_received,
            header_pairs,
            gain_map=gain_map,
            levels=self.header_levels,
        )
        raw_header_bits = [int(symbol) for symbol in header_symbols]
        masked_header_bits = []
        for idx in range(header_len):
            start = idx * self.header_repetitions
            chunk = raw_header_bits[start:start + self.header_repetitions]
            masked_header_bits.append(1 if sum(chunk) > (len(chunk) // 2) else 0)
        header_bits = self._unmask_bits(masked_header_bits)
        header_probe = self.codec.unpack_bits(header_bits)
        if not header_probe.complete:
            return AdaptiveDecodeResult(
                success=False,
                text=None,
                error=header_probe.error,
                payload=header_probe,
                header_pairs=header_pairs,
                payload_pairs=[],
                header_confidence=float(sum(header_conf) / max(len(header_conf), 1)),
                payload_confidence=0.0,
            )
        if header_probe.error is not None and header_probe.total_bits <= header_len:
            return AdaptiveDecodeResult(
                success=False,
                text=None,
                error=header_probe.error,
                payload=header_probe,
                header_pairs=header_pairs,
                payload_pairs=[],
                header_confidence=float(sum(header_conf) / max(len(header_conf), 1)),
                payload_confidence=0.0,
            )

        payload_bits = header_probe.body_bits
        payload_pairs_needed = math.ceil(payload_bits / self.bits_per_symbol)
        payload_positions = self._ordered_spread_positions(
            quality_map,
            payload_pairs_needed * 2,
            exclude=set(header_positions),
        )
        payload_pairs = self._pair_positions(payload_positions)
        payload_symbols, payload_conf = self._decode_symbol_indices(
            latent_clean,
            latent_received,
            payload_pairs,
            gain_map=gain_map,
            levels=self.payload_levels,
        )
        masked_payload_bits = self._symbol_indices_to_bits(payload_symbols, payload_bits)
        masked_bits = masked_header_bits + masked_payload_bits
        decoded_bits = self._unmask_bits(masked_bits)
        decoded = self.codec.unpack_bits(decoded_bits)
        return AdaptiveDecodeResult(
            success=decoded.success,
            text=decoded.text,
            error=decoded.error,
            payload=decoded,
            header_pairs=header_pairs,
            payload_pairs=payload_pairs,
            header_confidence=float(sum(header_conf) / max(len(header_conf), 1)),
            payload_confidence=float(sum(payload_conf) / max(len(payload_conf), 1)),
        )
