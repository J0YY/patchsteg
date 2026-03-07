# Reference Papers

Key papers for the PatchSteg project. Read these to understand the landscape.

## Distribution-Preserving Watermarking (Core to Phase 1: CDF-PatchSteg)

| File | Paper | Venue | Why It Matters |
|------|-------|-------|----------------|
| `gaussian_shading_cvpr2024.pdf` | Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models (Yang et al.) | CVPR 2024 | **Primary inspiration for CDF-PatchSteg.** Uses CDF partitioning of Gaussian prior for provably undetectable watermarks. Our CDF encoding adapts this idea to VAE latent space. |
| `tree_ring_watermarks_neurips2023.pdf` | Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust (Wen et al.) | NeurIPS 2023 | Embeds watermarks in Fourier space of initial noise. Demonstrates that distribution-preserving methods in latent space achieve robustness + undetectability. |
| `stable_signature_iccv2023.pdf` | The Stable Signature: Rooting Watermarks in Latent Diffusion Models (Fernandez et al.) | ICCV 2023 | Fine-tunes the VAE decoder to embed watermarks. Requires training (unlike PatchSteg) but achieves strong robustness. Good contrast point for our training-free approach. |

## Steganography Baselines (Phase 4: Benchmarking)

| File | Paper | Venue | Why It Matters |
|------|-------|-------|----------------|
| `rosteals_cvprw2023.pdf` | RoSteALS: Robust Steganography using Autoencoder Latent Space (Bui et al.) | CVPR-W 2023 | Closest prior work: trains an encoder on top of a frozen VAE for steganography. Key comparison baseline for PatchSteg (we need no training). |
| `trustmark_2023.pdf` | TrustMark: Universal Watermarking for Arbitrary Resolution Images (Bui et al.) | arXiv 2023 / ICCV 2025 | GAN-based watermarking with good robustness. Benchmark baseline for Phase 4 comparison. |

## Theoretical Foundations

| File | Paper | Venue | Why It Matters |
|------|-------|-------|----------------|
| `vae_pca_directions_cvpr2019.pdf` | Variational Autoencoders Pursue PCA Directions (by Accident) (Rolinek et al.) | CVPR 2019 | **Theoretical basis for Phase 2 (PCA directions).** Shows VAE latent directions align with PCA of the data, explaining why PCA-guided perturbations follow natural variation. |

## AI Safety / Threat Model

| File | Paper | Venue | Why It Matters |
|------|-------|-------|----------------|
| `secret_collusion_motwani2024.pdf` | Secret Collusion among AI Agents: Multi-Agent Deception via Steganography (Motwani et al.) | arXiv 2024 | **Defines our threat model.** Shows LLM agents can develop covert text channels. PatchSteg extends this to the vision domain. |
