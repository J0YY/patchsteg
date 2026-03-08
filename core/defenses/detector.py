"""Latent-space steganalysis detector for VAE steganography."""
import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core.analysis import spatial_frequency_map


class LatentStegDetector:
    """
    Detect steganographic modifications in VAE latent space.

    Features (~46 dimensions):
    - Per-channel residual stats: mean, std, median, skewness, kurtosis (4 ch x 5 = 20)
    - Per-channel spectral energy stats: mean, std (4 ch x 2 = 8)
    - Cross-channel correlations: 6 pairs
    - Per-channel histogram bin counts: 2 bins per channel (4 x 3 = 12)
    Total: 46 features
    """

    def __init__(self):
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42)
        )
        self.fitted = False

    def extract_features(self, vae, image):
        """
        Extract detection features from an image via VAE round-trip residual.

        Args:
            vae: StegoVAE instance
            image: PIL Image

        Returns:
            numpy array of features
        """
        latent = vae.encode(image)
        recon = vae.decode(latent)
        latent_rt = vae.encode(recon)

        # Residual: difference between original encoding and round-trip
        residual = (latent_rt[0] - latent[0]).cpu().numpy()

        feats = []

        # Per-channel residual statistics (20 features)
        for ch in range(4):
            x = residual[ch].flatten()
            mu = x.mean()
            sig = x.std() + 1e-8
            feats.extend([
                mu, sig,
                float(np.median(x)),
                float(((x - mu) ** 3).mean() / sig ** 3),  # skewness
                float(((x - mu) ** 4).mean() / sig ** 4),  # kurtosis
            ])

        # Spectral energy features (8 features)
        spec = spatial_frequency_map(latent_rt)
        for ch in range(4):
            import torch
            fft = torch.fft.fft2(latent_rt[0, ch].cpu().float())
            energy = torch.abs(fft).numpy()
            feats.extend([float(energy.mean()), float(energy.std())])

        # Cross-channel correlations (6 features)
        channels = [residual[ch].flatten() for ch in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                r, _ = scipy_stats.pearsonr(channels[i], channels[j])
                feats.append(float(r))

        # Histogram features (12 features)
        for ch in range(4):
            x = residual[ch].flatten()
            hist, _ = np.histogram(x, bins=3, density=True)
            feats.extend([float(h) for h in hist])

        return np.array(feats)

    def fit(self, X, y):
        """
        Train the detector.

        Args:
            X: feature matrix [n_samples, n_features]
            y: labels (0=clean, 1=stego)
        """
        self.clf.fit(X, y)
        self.fitted = True

    def predict(self, X):
        """Predict labels."""
        assert self.fitted
        return self.clf.predict(X)

    def predict_proba(self, X):
        """Predict probabilities."""
        assert self.fitted
        return self.clf.predict_proba(X)

    def get_feature_names(self):
        """Return feature names for interpretation."""
        names = []
        for ch in range(4):
            for stat in ['mean', 'std', 'median', 'skew', 'kurt']:
                names.append(f'ch{ch}_residual_{stat}')
        for ch in range(4):
            for stat in ['spec_mean', 'spec_std']:
                names.append(f'ch{ch}_{stat}')
        for i in range(4):
            for j in range(i + 1, 4):
                names.append(f'corr_ch{i}_ch{j}')
        for ch in range(4):
            for b in range(3):
                names.append(f'ch{ch}_hist_bin{b}')
        return names
