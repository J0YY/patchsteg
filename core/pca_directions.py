"""PCA-guided perturbation directions for steganography."""
import torch
import numpy as np
from sklearn.decomposition import PCA
import pickle
from pathlib import Path


class PCADirections:
    """
    Fit PCA on VAE latent vectors to discover natural variation directions.
    Per-position 4D PCA: for each spatial position, fit PCA on the 4-channel
    vectors across a dataset of images.
    """

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca_models = None  # dict: (r, c) -> fitted PCA
        self.global_pca = None  # single PCA fitted on all positions

    def fit_global(self, vae, images):
        """
        Fit a single global PCA on 4D latent vectors pooled across all
        positions and images.

        Args:
            vae: StegoVAE instance
            images: list of PIL Images
        """
        all_vectors = []
        for img in images:
            latent = vae.encode(img)
            # Shape [4, H, W] -> reshape to [H*W, 4]
            vectors = latent[0].cpu().numpy().reshape(4, -1).T
            all_vectors.append(vectors)

        all_vectors = np.vstack(all_vectors)
        self.global_pca = PCA(n_components=self.n_components)
        self.global_pca.fit(all_vectors)
        return self

    def get_direction(self, component=0):
        """Get the k-th principal component as a direction vector."""
        assert self.global_pca is not None, "Must call fit_global() first"
        d = self.global_pca.components_[component]
        d = d / np.linalg.norm(d)
        return torch.tensor(d, dtype=torch.float32)

    def get_explained_variance_ratio(self):
        """Return explained variance ratio for each component."""
        assert self.global_pca is not None
        return self.global_pca.explained_variance_ratio_

    def get_singular_values(self):
        """Return singular values (scale of natural variation per component)."""
        assert self.global_pca is not None
        return self.global_pca.singular_values_

    def save(self, path):
        """Save fitted PCA model."""
        with open(path, 'wb') as f:
            pickle.dump({'global_pca': self.global_pca, 'n_components': self.n_components}, f)

    def load(self, path):
        """Load fitted PCA model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.global_pca = data['global_pca']
        self.n_components = data['n_components']
        return self


class PCAPatchSteg:
    """
    PatchSteg variant using PCA-derived perturbation directions.
    Uses the top principal component (or specified component) as the
    perturbation direction instead of a random vector.
    """

    def __init__(self, pca_directions, seed=42, epsilon=5.0, component=0):
        """
        Args:
            pca_directions: fitted PCADirections instance
            seed: random seed for carrier selection
            epsilon: perturbation strength
            component: which PCA component to use as direction
        """
        self.pca_directions = pca_directions
        self.seed = seed
        self.epsilon = epsilon
        self.component = component
        self.direction = pca_directions.get_direction(component)
        # Scale epsilon by the singular value ratio to match natural variance
        sv = pca_directions.get_singular_values()
        self.scale_factor = sv[component] / sv[0] if sv[0] > 0 else 1.0

    def _project(self, latent, r, c):
        """Project latent[:, :, r, c] onto PCA direction."""
        return torch.dot(latent[0, :, r, c].cpu(), self.direction).item()

    def select_carriers_by_stability(self, vae, image, n_carriers=20, test_eps=5.0):
        """Reuse PatchSteg's stability selection with PCA direction."""
        from core.steganography import PatchSteg
        ps = PatchSteg(seed=self.seed, epsilon=test_eps)
        # Override direction with PCA direction
        ps.direction = self.direction
        return ps.select_carriers_by_stability(vae, image, n_carriers=n_carriers, test_eps=test_eps)

    def compute_stability_map(self, vae, image, test_eps=5.0):
        """Compute stability map using PCA direction."""
        latent_clean = vae.encode(image)
        direction_dev = self.direction.to(latent_clean.device)

        latent_test = latent_clean.clone()
        for ch in range(4):
            latent_test[0, ch, :, :] += test_eps * direction_dev[ch]

        recon = vae.decode(latent_test)
        latent_reencoded = vae.encode(recon)

        proj_clean = torch.einsum('c,cij->ij', direction_dev, latent_clean[0])
        proj_reenc = torch.einsum('c,cij->ij', direction_dev, latent_reencoded[0])
        stability_map = (proj_reenc - proj_clean).cpu()

        return stability_map, latent_clean

    def encode_message(self, latent, carriers, bits):
        """Encode bits using PCA direction (same API as PatchSteg)."""
        assert len(carriers) == len(bits)
        latent_mod = latent.clone()
        direction_dev = self.direction.to(latent.device)
        for (r, c), bit in zip(carriers, bits):
            sign = 1.0 if bit == 1 else -1.0
            for ch in range(4):
                latent_mod[0, ch, r, c] += sign * self.epsilon * direction_dev[ch]
        return latent_mod

    def decode_message(self, latent_clean, latent_received, carriers):
        """Decode bits by comparing projections (same API as PatchSteg)."""
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
