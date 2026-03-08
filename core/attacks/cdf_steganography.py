"""Distribution-preserving steganography via inverse Gaussian CDF sampling."""
import torch
import numpy as np
from scipy.stats import norm


class CDFPatchSteg:
    """
    Encode bits by replacing carrier latent values with samples from
    upper/lower halves of N(0, sigma), making stego latents statistically
    indistinguishable from clean ones.

    Bit=1 -> sample from upper half (z > mu)
    Bit=0 -> sample from lower half (z < mu)

    Decoding: re-encode stego image, check which half each carrier falls in.
    No clean latent reference needed.
    """

    def __init__(self, seed=42, sigma=1.0):
        self.seed = seed
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)

    def _get_key_stream(self, n_bits, carrier_seed=None):
        """Generate pseudorandom key stream for XOR."""
        rng = np.random.RandomState(carrier_seed if carrier_seed is not None else self.seed)
        return rng.randint(0, 2, size=n_bits).tolist()

    def _sample_half_gaussian(self, mu, sigma, bit, rng):
        """
        Sample from upper (bit=1) or lower (bit=0) half of N(mu, sigma).
        Uses inverse CDF: for upper half, sample U ~ [0.5, 1), then ppf(U).
        """
        u = rng.uniform(0.5, 1.0)
        if bit == 1:
            return norm.ppf(u, loc=mu, scale=sigma)
        else:
            return norm.ppf(1.0 - u, loc=mu, scale=sigma)

    def select_carriers_by_stability(self, vae, image, n_carriers=20, test_eps=5.0):
        """
        Reuse PatchSteg's stability-based carrier selection.
        We import and delegate to avoid code duplication.
        """
        from core.steganography import PatchSteg
        ps = PatchSteg(seed=self.seed, epsilon=test_eps)
        return ps.select_carriers_by_stability(vae, image, n_carriers=n_carriers, test_eps=test_eps)

    def encode_message(self, latent, carriers, bits, channel=0):
        """
        Encode bits into latent at carrier positions using CDF sampling.

        For each carrier, XOR the message bit with the key stream,
        then replace the latent value with a sample from the corresponding
        half of N(mu, sigma) where mu/sigma are estimated from the latent channel.

        Args:
            latent: [1, 4, H, W] tensor
            carriers: list of (r, c) positions
            bits: list of 0/1 values
            channel: which latent channel to modify (default 0)

        Returns:
            Modified latent tensor (cloned)
        """
        assert len(carriers) == len(bits)
        latent_mod = latent.clone()

        # Estimate per-channel statistics from the latent
        ch_data = latent[0, channel].cpu().numpy().flatten()
        mu = float(np.mean(ch_data))
        sigma = float(np.std(ch_data)) if float(np.std(ch_data)) > 0.01 else self.sigma

        key_stream = self._get_key_stream(len(bits))
        rng = np.random.RandomState(self.seed + 1000)

        for i, ((r, c), bit) in enumerate(zip(carriers, bits)):
            effective_bit = bit ^ key_stream[i]
            new_val = self._sample_half_gaussian(mu, sigma, effective_bit, rng)
            latent_mod[0, channel, r, c] = new_val

        return latent_mod

    def decode_message(self, vae, stego_image, carriers, channel=0):
        """
        Decode bits from stego image. No clean latent needed.

        Re-encode the stego image, then check which half of N(mu, sigma)
        each carrier value falls in.

        Args:
            vae: StegoVAE instance
            stego_image: PIL Image (the stego output)
            carriers: list of (r, c) positions
            channel: which latent channel was modified

        Returns:
            (bits, confidences) — decoded bits and distance-from-mean confidences
        """
        latent_re = vae.encode(stego_image)
        ch_data = latent_re[0, channel].cpu().numpy().flatten()
        mu = float(np.mean(ch_data))

        key_stream = self._get_key_stream(len(carriers))
        bits = []
        confidences = []

        for i, (r, c) in enumerate(carriers):
            val = latent_re[0, channel, r, c].item()
            raw_bit = 1 if val >= mu else 0
            decoded_bit = raw_bit ^ key_stream[i]
            bits.append(decoded_bit)
            confidences.append(abs(val - mu))

        return bits, confidences

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
