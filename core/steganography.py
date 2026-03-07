"""Bit encoding/decoding in SD VAE latent space."""
import torch
import numpy as np


class PatchSteg:
    def __init__(self, seed=42, epsilon=5.0):
        self.epsilon = epsilon
        self.seed = seed
        self.direction = self._make_direction(seed)

    def _make_direction(self, seed):
        """4-dim unit direction vector (one per latent channel)."""
        rng = torch.Generator()
        rng.manual_seed(seed)
        d = torch.randn(4, generator=rng)
        return d / d.norm()

    def _project(self, latent, r, c):
        """Project latent[:, :, r, c] onto direction vector."""
        return torch.dot(latent[0, :, r, c].cpu(), self.direction).item()

    def compute_stability_map(self, vae, image, test_eps=5.0):
        """
        Test all 4096 positions simultaneously.
        Perturb ALL with +eps, round-trip, check projection deltas.
        Returns [64, 64] stability map.
        """
        latent_clean = vae.encode(image)
        direction_dev = self.direction.to(latent_clean.device)

        # Perturb ALL positions with +eps
        latent_test = latent_clean.clone()
        for ch in range(4):
            latent_test[0, ch, :, :] += test_eps * direction_dev[ch]

        # Round-trip
        recon = vae.decode(latent_test)
        latent_reencoded = vae.encode(recon)

        # Vectorized projection delta computation
        proj_clean = torch.einsum('c,cij->ij', direction_dev, latent_clean[0])
        proj_reenc = torch.einsum('c,cij->ij', direction_dev, latent_reencoded[0])
        stability_map = (proj_reenc - proj_clean).cpu()

        return stability_map, latent_clean

    def select_carriers_by_stability(self, vae, image, n_carriers=20, test_eps=5.0):
        """
        Compute stability map and return top-N most reliable carrier positions.
        """
        stability_map, latent_clean = self.compute_stability_map(vae, image, test_eps)

        H = stability_map.shape[0]
        flat = stability_map.flatten()
        top_indices = torch.argsort(flat, descending=True)[:n_carriers]
        carriers = [(idx.item() // H, idx.item() % H) for idx in top_indices]

        return carriers, stability_map

    def select_carriers_fixed(self, n_carriers=20, seed=42, grid_size=32):
        """Simple deterministic carrier selection (fallback)."""
        rng = np.random.RandomState(seed)
        positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        indices = rng.choice(len(positions), n_carriers, replace=False)
        return [positions[i] for i in indices]

    def encode_message(self, latent, carriers, bits):
        """
        Encode bits into latent at carrier positions.
        Returns modified latent tensor (cloned).
        """
        assert len(carriers) == len(bits)
        latent_mod = latent.clone()
        direction_dev = self.direction.to(latent.device)
        for (r, c), bit in zip(carriers, bits):
            sign = 1.0 if bit == 1 else -1.0
            for ch in range(4):
                latent_mod[0, ch, r, c] += sign * self.epsilon * direction_dev[ch]
        return latent_mod

    def decode_message(self, latent_clean, latent_received, carriers):
        """
        Decode bits by comparing projections.
        Returns list of decoded bits, list of confidence scores.
        """
        direction_dev = self.direction.to(latent_clean.device)
        bits = []
        confidences = []
        for (r, c) in carriers:
            proj_clean = torch.dot(latent_clean[0, :, r, c], direction_dev).item()
            proj_recv = torch.dot(latent_received[0, :, r, c], direction_dev).item()
            delta = proj_recv - proj_clean
            bits.append(1 if delta > 0 else 0)
            confidences.append(abs(delta))
        return bits, confidences

    def encode_message_with_repetition(self, latent, carriers, bits, reps=3):
        """Encode each bit across `reps` carrier positions."""
        needed = len(bits) * reps
        assert len(carriers) >= needed, f"Need {needed} carriers, have {len(carriers)}"
        expanded_bits = []
        for bit in bits:
            expanded_bits.extend([bit] * reps)
        return self.encode_message(latent, carriers[:needed], expanded_bits)

    def decode_message_with_repetition(self, latent_clean, latent_received, carriers, n_bits, reps=3):
        """Decode with majority voting."""
        raw_bits, raw_conf = self.decode_message(
            latent_clean, latent_received, carriers[:n_bits * reps]
        )
        decoded = []
        for i in range(n_bits):
            chunk = raw_bits[i * reps:(i + 1) * reps]
            decoded.append(1 if sum(chunk) > reps // 2 else 0)
        return decoded

    @staticmethod
    def text_to_bits(text):
        """Convert ASCII text to list of bits."""
        bits = []
        for char in text:
            byte = ord(char)
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def bits_to_text(bits):
        """Convert list of bits back to ASCII text."""
        chars = []
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            chars.append(chr(byte))
        return ''.join(chars)
