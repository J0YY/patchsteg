"""Capacity-oriented PatchSteg variant with framing, compression, and multilevel symbols."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from PIL import Image
import torch

from core.payload_codec import DecodedPayload, PayloadCodec, PayloadPacket
from core.steganography import PatchSteg


@dataclass
class CapacityEncodeResult:
    stego_image: Image.Image
    packet: PayloadPacket
    header_carriers: list[tuple[int, int]]
    payload_carriers: list[tuple[int, int]]
    gain_map: torch.Tensor


@dataclass
class CapacityDecodeResult:
    success: bool
    text: str | None
    error: str | None
    payload: DecodedPayload
    header_carriers: list[tuple[int, int]]
    payload_carriers: list[tuple[int, int]]
    header_confidence: float
    payload_confidence: float


class CapacityPatchSteg(PatchSteg):
    """
    PatchSteg variant optimized for longer payloads.

    Upgrades over base PatchSteg:
    - Compact payload framing with optional zlib compression
    - Binary header + multilevel payload symbols (2 bits/carrier by default)
    - Channel equalization using the stability map as a carrier gain estimate
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
        bits_per_symbol: int = 2,
        min_gain: float = 0.25,
        max_embed_multiplier: float = 3.0,
        codec: PayloadCodec | None = None,
    ):
        super().__init__(seed=seed, epsilon=epsilon)
        if bits_per_symbol not in (1, 2):
            raise ValueError("CapacityPatchSteg currently supports 1 or 2 bits per symbol")
        self.bits_per_symbol = int(bits_per_symbol)
        self.min_gain = float(min_gain)
        self.max_embed_multiplier = float(max_embed_multiplier)
        self.codec = codec if codec is not None else PayloadCodec()

    @property
    def payload_levels(self) -> tuple[float, ...]:
        if self.bits_per_symbol == 1:
            return (-1.0, 1.0)
        return (-3.0, -1.0, 1.0, 3.0)

    @staticmethod
    def _top_carriers_from_map(
        gain_map: torch.Tensor,
        n_carriers: int,
    ) -> list[tuple[int, int]]:
        flat = gain_map.flatten()
        side = gain_map.shape[0]
        top_indices = torch.argsort(flat, descending=True)[:n_carriers]
        return [(idx.item() // side, idx.item() % side) for idx in top_indices]

    def compute_gain_map(
        self,
        vae,
        image: Image.Image,
        *,
        test_eps: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probe_eps = float(self.epsilon if test_eps is None else test_eps)
        stability_map, latent_clean = self.compute_stability_map(vae, image, test_eps=probe_eps)
        gain_map = (stability_map / max(probe_eps, 1e-6)).clamp_min(1e-6)
        return gain_map, latent_clean

    def select_carriers_by_capacity(
        self,
        vae,
        image: Image.Image,
        n_carriers: int,
        *,
        test_eps: float | None = None,
    ) -> tuple[list[tuple[int, int]], torch.Tensor, torch.Tensor]:
        gain_map, latent_clean = self.compute_gain_map(vae, image, test_eps=test_eps)
        carriers = self._top_carriers_from_map(gain_map, n_carriers)
        return carriers, gain_map, latent_clean

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
        bits: list[int] = []
        if self.bits_per_symbol == 1:
            bits = [int(symbol) for symbol in symbols]
        else:
            for symbol in symbols:
                bits.extend(self.SYMBOL_TO_GRAY_2BIT[int(symbol)])
        return bits[:n_bits]

    def _embed_amplitude(self, gain: float, target_level: float) -> float:
        safe_gain = max(gain, self.min_gain)
        target_delta = self.epsilon * target_level
        max_amp = self.max_embed_multiplier * self.epsilon
        amp = target_delta / safe_gain
        return float(max(-max_amp, min(max_amp, amp)))

    def _expected_received(self, gain: float, target_level: float) -> float:
        return gain * self._embed_amplitude(gain, target_level)

    def _encode_symbol_indices(
        self,
        latent: torch.Tensor,
        carriers: Sequence[tuple[int, int]],
        symbol_indices: Sequence[int],
        *,
        levels: Sequence[float],
        gain_map: torch.Tensor,
    ) -> torch.Tensor:
        if len(carriers) != len(symbol_indices):
            raise ValueError("carriers and symbol_indices must have the same length")
        latent_mod = latent.clone()
        direction_dev = self.direction.to(latent.device)
        for (row, col), symbol_idx in zip(carriers, symbol_indices):
            gain = float(gain_map[row, col].item())
            amp = self._embed_amplitude(gain, float(levels[int(symbol_idx)]))
            for channel in range(4):
                latent_mod[0, channel, row, col] += amp * direction_dev[channel]
        return latent_mod

    def _decode_symbol_indices(
        self,
        latent_clean: torch.Tensor,
        latent_received: torch.Tensor,
        carriers: Sequence[tuple[int, int]],
        *,
        levels: Sequence[float],
        gain_map: torch.Tensor,
    ) -> tuple[list[int], list[float]]:
        direction_dev = self.direction.to(latent_clean.device)
        decoded: list[int] = []
        confidences: list[float] = []
        for row, col in carriers:
            gain = float(gain_map[row, col].item())
            delta = torch.dot(
                latent_received[0, :, row, col] - latent_clean[0, :, row, col],
                direction_dev,
            ).item()
            expected = [self._expected_received(gain, float(level)) for level in levels]
            distances = [abs(delta - value) for value in expected]
            best = int(min(range(len(expected)), key=lambda idx: distances[idx]))
            ordered = sorted(distances)
            margin = ordered[1] - ordered[0] if len(ordered) > 1 else ordered[0]
            decoded.append(best)
            confidences.append(float(max(0.0, margin)))
        return decoded, confidences

    def required_carriers_for_packet_bits(self, total_bits: int) -> int:
        header_carriers = self.codec.header_bits()
        body_bits = max(0, total_bits - self.codec.header_bits())
        payload_carriers = math.ceil(body_bits / self.bits_per_symbol)
        return header_carriers + payload_carriers

    def encode_text(
        self,
        vae,
        image: Image.Image,
        text: str,
        *,
        enable_compression: bool = True,
        compression_level: int = 9,
        test_eps: float | None = None,
    ) -> CapacityEncodeResult:
        packet = self.codec.pack_text(
            text,
            enable_compression=enable_compression,
            compression_level=compression_level,
        )
        total_carriers = self.required_carriers_for_packet_bits(packet.total_bits)
        carriers, gain_map, latent_clean = self.select_carriers_by_capacity(
            vae, image, total_carriers, test_eps=test_eps
        )
        header_len = self.codec.header_bits()
        header_bits = packet.bits[:header_len]
        body_bits = packet.bits[header_len:]
        header_carriers = carriers[:header_len]
        payload_carriers = carriers[header_len:]
        payload_symbol_indices = self._bits_to_symbol_indices(body_bits)
        if len(payload_carriers) < len(payload_symbol_indices):
            raise ValueError("Not enough carriers selected for payload body")

        latent_mod = self._encode_symbol_indices(
            latent_clean,
            header_carriers,
            header_bits,
            levels=(-1.0, 1.0),
            gain_map=gain_map,
        )
        if payload_symbol_indices:
            latent_mod = self._encode_symbol_indices(
                latent_mod,
                payload_carriers[:len(payload_symbol_indices)],
                payload_symbol_indices,
                levels=self.payload_levels,
                gain_map=gain_map,
            )

        return CapacityEncodeResult(
            stego_image=vae.decode(latent_mod),
            packet=packet,
            header_carriers=header_carriers,
            payload_carriers=payload_carriers[:len(payload_symbol_indices)],
            gain_map=gain_map,
        )

    def decode_text(
        self,
        vae,
        cover_image: Image.Image,
        stego_image: Image.Image,
        *,
        test_eps: float | None = None,
    ) -> CapacityDecodeResult:
        gain_map, latent_clean = self.compute_gain_map(vae, cover_image, test_eps=test_eps)
        latent_received = vae.encode(stego_image)

        header_len = self.codec.header_bits()
        header_carriers = self._top_carriers_from_map(gain_map, header_len)
        header_symbols, header_conf = self._decode_symbol_indices(
            latent_clean,
            latent_received,
            header_carriers,
            levels=(-1.0, 1.0),
            gain_map=gain_map,
        )
        header_bits = [int(symbol) for symbol in header_symbols]
        header_probe = self.codec.unpack_bits(header_bits)
        if not header_probe.complete:
            return CapacityDecodeResult(
                success=False,
                text=None,
                error=header_probe.error,
                payload=header_probe,
                header_carriers=header_carriers,
                payload_carriers=[],
                header_confidence=float(sum(header_conf) / max(len(header_conf), 1)),
                payload_confidence=0.0,
            )
        if header_probe.error is not None and header_probe.total_bits <= header_len:
            return CapacityDecodeResult(
                success=False,
                text=None,
                error=header_probe.error,
                payload=header_probe,
                header_carriers=header_carriers,
                payload_carriers=[],
                header_confidence=float(sum(header_conf) / max(len(header_conf), 1)),
                payload_confidence=0.0,
            )

        body_bits = header_probe.body_bits
        payload_symbol_count = math.ceil(body_bits / self.bits_per_symbol)
        total_carriers = header_len + payload_symbol_count
        carriers = self._top_carriers_from_map(gain_map, total_carriers)
        payload_carriers = carriers[header_len:]
        payload_symbols, payload_conf = self._decode_symbol_indices(
            latent_clean,
            latent_received,
            payload_carriers,
            levels=self.payload_levels,
            gain_map=gain_map,
        )
        payload_bits = self._symbol_indices_to_bits(payload_symbols, body_bits)
        decoded = self.codec.unpack_bits(header_bits + payload_bits)
        return CapacityDecodeResult(
            success=decoded.success,
            text=decoded.text,
            error=decoded.error,
            payload=decoded,
            header_carriers=header_carriers,
            payload_carriers=payload_carriers,
            header_confidence=float(sum(header_conf) / max(len(header_conf), 1)),
            payload_confidence=float(sum(payload_conf) / max(len(payload_conf), 1)),
        )
