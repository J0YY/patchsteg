"""
PSyDUCK-inspired steganography: per-bit seed-based trajectory divergence.

Based on: PSyDUCK — Training-Free Steganography for Latent Diffusion Models
          Jiang et al., arXiv 2501.19172 (ICML 2025 submission)
          https://arxiv.org/abs/2501.19172

Original PSyDUCK mechanism:
  During DDIM/DDPM sampling the sender uses a shared seed to choose between
  two denoising trajectories (one per bit value). The receiver runs DDIM
  inversion and detects which trajectory was taken by correlating with the
  known seed. Achieves 99.91% accuracy but requires control over the
  generation process — it is NOT post-hoc.

This post-hoc adaptation:
  The key insight from PSyDUCK is that each bit gets its own unique
  "divergence direction" (derived from seed + bit_index). This is much
  harder to detect than PatchSteg, which uses the same direction for every
  carrier — a consistent signature that detectors can find.

  Here we give each carrier position a unique 4D direction vector derived
  from (seed, position_index), then apply ±ε perturbation along that
  direction. Detectors trained on a single direction signature will fail
  because there is no single direction to find.

  The adaptation preserves the original paper's "per-bit random trajectory"
  concept while operating post-hoc on any existing image.
"""
import torch
import numpy as np


class PSyDUCKSteg:
    """
    Post-hoc adaptation of PSyDUCK's per-bit trajectory divergence concept.

    Each carrier position uses a unique direction derived from (seed, carrier_idx).
    No single shared direction across carriers — direction-based detectors fail.
    """

    def __init__(self, seed=42, epsilon=5.0):
        self.seed = seed
        self.epsilon = epsilon

    def _get_direction(self, carrier_idx):
        """Unique 4D unit direction for carrier index i."""
        rng = torch.Generator()
        rng.manual_seed(self.seed * 100003 + carrier_idx)
        d = torch.randn(4, generator=rng)
        return d / d.norm()

    def _get_key_bit(self, carrier_idx):
        """Key stream bit for XOR, derived from seed."""
        rng = np.random.RandomState(self.seed + carrier_idx * 7919)
        return int(rng.randint(0, 2))

    def select_carriers_by_stability(self, vae, image, n_carriers=20, test_eps=5.0):
        """Delegate to PatchSteg's stability selection (same carrier logic)."""
        from core.steganography import PatchSteg
        ps = PatchSteg(seed=self.seed, epsilon=test_eps)
        return ps.select_carriers_by_stability(vae, image, n_carriers, test_eps)

    def encode_message(self, latent, carriers, bits):
        """
        Encode bits into latent. Each bit gets its own unique perturbation direction.

        Args:
            latent: [1, 4, H, W] tensor
            carriers: list of (r, c) positions
            bits: list of 0/1 values

        Returns:
            Modified latent tensor (cloned)
        """
        assert len(carriers) == len(bits)
        latent_mod = latent.clone()

        for i, ((r, c), bit) in enumerate(zip(carriers, bits)):
            d = self._get_direction(i).to(latent.device)
            effective_bit = bit ^ self._get_key_bit(i)
            sign = 1.0 if effective_bit == 1 else -1.0
            for ch in range(4):
                latent_mod[0, ch, r, c] = latent_mod[0, ch, r, c] + sign * self.epsilon * d[ch]

        return latent_mod

    def decode_message(self, latent_clean, latent_received, carriers):
        """
        Decode bits by projecting latent delta onto the per-carrier direction.

        Args:
            latent_clean: [1, 4, H, W] clean reference latent
            latent_received: [1, 4, H, W] received (re-encoded stego) latent
            carriers: list of (r, c) positions

        Returns:
            (bits, confidences)
        """
        bits, confs = [], []
        for i, (r, c) in enumerate(carriers):
            d = self._get_direction(i)
            delta = (latent_received[0, :, r, c] - latent_clean[0, :, r, c]).cpu()
            proj = torch.dot(delta, d).item()
            raw_bit = 1 if proj >= 0 else 0
            decoded_bit = raw_bit ^ self._get_key_bit(i)
            bits.append(decoded_bit)
            confs.append(abs(proj))
        return bits, confs

    @staticmethod
    def text_to_bits(text):
        bits = []
        for char in text:
            byte = ord(char)
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def bits_to_text(bits):
        chars = []
        for i in range(0, len(bits) - 7, 8):
            byte = sum(bits[i + j] << (7 - j) for j in range(8))
            chars.append(chr(byte))
        return ''.join(chars)
