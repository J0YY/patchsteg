# PatchSteg: Defeating SOTA Steganographic Attacks in Latent Space

**Can a defender reliably neutralize covert image channels between AI agents — even when the attack is theoretically undetectable?**

PatchSteg studies this question by implementing a progression of increasingly strong steganographic attacks (each grounded in a published paper) and building defenses against them. All attacks are training-free, operating directly on the latent space of a pretrained Stable Diffusion VAE. No fine-tuning, no auxiliary networks.

---

## Attacks Implemented

Each attack encodes a binary message into an image so that two agents sharing a secret seed can communicate covertly. Attacks are ordered by detectability (easiest to hardest to defend):

| Attack | Paper | Method | Detection AUC | Status |
|--------|-------|--------|---------------|--------|
| **PatchSteg (±ε)** | *This work* | Add ±ε to carrier latent values along a secret direction vector | 0.93 (detectable) | In demo |
| **CapacityPatchSteg** | *This work* | Compression + framed payloads + multilevel symbols + channel equalization for longer text/paragraphs | Not yet benchmarked | Experimental |
| **AdaptivePatchSteg** | *This work* | Balanced pairwise modulation + geometry-aware directions + bit whitening + joint carrier scoring to suppress linear residual signatures | Not yet benchmarked | Experimental |
| **PCA-PatchSteg** | [Rolinek et al., CVPR 2019](https://arxiv.org/abs/1812.03463) — *VAEs Pursue PCA Directions (by Accident)* | Perturb along principal directions of the latent distribution — follows natural variation, harder to flag | ~0.6–0.8 | In demo |
| **CDF-PatchSteg** | [Yang et al., CVPR 2024](https://arxiv.org/abs/2404.04956) — *Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models* | Replace carrier values with samples from upper/lower half of N(μ,σ) via inverse CDF — distribution-preserving, theoretically undetectable | ~0.50 (chance) | In demo |
| **PSyDUCK-inspired** | [Jiang et al., arXiv 2501.19172](https://arxiv.org/abs/2501.19172) — *PSyDUCK: Training-Free Steganography for Latent Diffusion Models* | Per-bit unique direction vectors (no shared signature across carriers) — direction-based detectors fail. Post-hoc adaptation of PSyDUCK's trajectory-divergence concept. | ~0.6–0.8 | In demo |

**Note:** Original PSyDUCK requires generation-time DDIM seed control (not post-hoc). Our adaptation captures its per-bit unique-direction concept in a post-hoc VAE setting. AGM (arXiv 2510.07219) explains *why* ±ε is detectable (VAE decoder amplifies latent perturbations) and is implemented as a detector (see Defenses).

**Status note:** The paper's quantitative claims currently cover PatchSteg, CDF-PatchSteg, PCA-PatchSteg, PSyDUCK-inspired, and the defense stack. `CapacityPatchSteg` and `AdaptivePatchSteg` are repo extensions for stronger practical channels, but they are not yet part of the paper-quality benchmark suite.

---

## Defenses Implemented

| Defense | Paper | Approach | Defeats |
|---------|-------|----------|---------|
| **UniversalPatchStegGuard** | This work | Latent anomaly scoring + spatial smoothing (no key needed) | PatchSteg (±ε) |
| **AGMDetector** | [arXiv 2510.07219](https://arxiv.org/abs/2510.07219) — *AGM: Autoencoder-Guided Watermark Detection* | VAE round-trip residual analysis: stego latents produce structured residuals that clean images don't | PatchSteg (±ε), PSyDUCK-inspired |
| **CertifiedPurifier** | This work | Stability-guided resampling + channel noise — certified bit accuracy bound | All four attacks |

---

## Running Each Attack on an Image

```python
from PIL import Image
from core.vae import StegoVAE
vae = StegoVAE(device='cpu', image_size=128)
image = Image.open("your_image.png").convert("RGB").resize((128, 128))
message = "hello"
```

### PatchSteg (±ε) — `core/steganography.py`
> Directional latent perturbation. Fast, high accuracy, detectable at large ε.
```python
from core.steganography import PatchSteg
steg = PatchSteg(seed=42, epsilon=5.0)
carriers, _ = steg.select_carriers_by_stability(vae, image, n_carriers=48)
bits = PatchSteg.text_to_bits(message)
latent_stego = steg.encode_message(vae.encode(image), carriers[:len(bits)], bits)
stego = vae.decode(latent_stego)
# Decode:
rec, _ = steg.decode_message(vae.encode(image), vae.encode(stego), carriers[:len(bits)])
print(PatchSteg.bits_to_text(rec))
```

### CapacityPatchSteg — `core/capacity_steganography.py`
> Capacity-oriented PatchSteg variant for longer sentences and paragraphs. Keeps the same cover-image decode assumption, but improves throughput via UTF-8 payload framing, optional zlib compression, binary headers, quaternary payload symbols, and stability-map equalization.

This should be read as a payload-layer upgrade over the same post-hoc latent channel, not as a separate fully benchmarked attack family. It still assumes the receiver can reproduce the same carrier set from the clean cover image and shared secret.
```python
from core.capacity_steganography import CapacityPatchSteg
steg = CapacityPatchSteg(seed=42, epsilon=2.0, bits_per_symbol=2)
result = steg.encode_text(
    vae,
    image,
    "This is a much longer paragraph than raw ASCII PatchSteg can usually carry.",
    enable_compression=True,
)
stego = result.stego_image

decoded = steg.decode_text(vae, image, stego)
print(decoded.text)
print(decoded.payload.compressed, decoded.payload.original_bytes, decoded.payload.stored_bytes)
```

### AdaptivePatchSteg — `core/adaptive_steganography.py`
> Stronger post-hoc attack aimed at defeating simple residual-statistics detectors. It uses pairwise differential embedding, content-aware per-pair directions, seeded payload whitening, a compact packet header, and a joint carrier score over stability, texture, and clean round-trip drift.

This is the most aggressive perturbation-based attack in the repo. It still assumes the receiver can reproduce the clean cover latent, but it removes the baseline's single-direction global signature and keeps the packet header from appearing as a fixed bit pattern.

The raw pairwise bit channel is the verified part today. The framed text helpers are still experimental and should be treated as a tuning surface, especially at `128x128`; for longer text, use `256x256` or larger images and benchmark the operating point you want.
```python
from core.adaptive_steganography import AdaptivePatchSteg

steg = AdaptivePatchSteg(seed=42, epsilon=2.5, bits_per_symbol=1)
result = steg.encode_text(
    vae,
    image,
    "Transmit a longer instruction while keeping the latent residual signature balanced.",
    enable_compression=True,
)
stego = result.stego_image

decoded = steg.decode_text(vae, image, stego)
print(decoded.text)
print(decoded.header_confidence, decoded.payload_confidence)
```

### PCA-PatchSteg — `core/pca_directions.py`
> Perturbs along principal directions of the latent distribution (Rolinek et al., CVPR 2019). Harder to detect via direction-based detectors than random ±ε.
```python
from core.pca_directions import PCADirections, PCAPatchSteg
pca_dir = PCADirections(n_components=4)
pca_dir.fit_global(vae, [image])          # fit on available images
steg = PCAPatchSteg(pca_dir, seed=42, epsilon=5.0, component=0)
carriers, _ = steg.select_carriers_by_stability(vae, image, n_carriers=48)
bits = PCAPatchSteg.text_to_bits(message)  # same API as PatchSteg
latent_stego = steg.encode_message(vae.encode(image), carriers[:len(bits)], bits)
stego = vae.decode(latent_stego)
rec, _ = steg.decode_message(vae.encode(image), vae.encode(stego), carriers[:len(bits)])
print(PCAPatchSteg.bits_to_text(rec))
```

### PSyDUCK-inspired — `core/psyduck_steganography.py`
> Per-bit unique direction vectors (Jiang et al., arXiv 2501.19172). No shared direction signature — direction-based detectors trained on a single direction fail.
```python
from core.psyduck_steganography import PSyDUCKSteg
steg = PSyDUCKSteg(seed=42, epsilon=5.0)
carriers, _ = steg.select_carriers_by_stability(vae, image, n_carriers=48)
bits = PSyDUCKSteg.text_to_bits(message)
latent_stego = steg.encode_message(vae.encode(image), carriers[:len(bits)], bits)
stego = vae.decode(latent_stego)
rec, _ = steg.decode_message(vae.encode(image), vae.encode(stego), carriers[:len(bits)])
print(PSyDUCKSteg.bits_to_text(rec))
```

### CDF-PatchSteg — `core/cdf_steganography.py`
> Distribution-preserving encoding (Yang et al., CVPR 2024). Replaces carrier values with fresh samples from N(μ,σ). Statistically indistinguishable from clean; no clean reference needed to decode.
```python
from core.cdf_steganography import CDFPatchSteg
steg = CDFPatchSteg(seed=42, sigma=1.0)
carriers, _ = steg.select_carriers_by_stability(vae, image, n_carriers=20)
bits = CDFPatchSteg.text_to_bits(message)
latent_stego = steg.encode_message(vae.encode(image), carriers[:len(bits)], bits)
stego = vae.decode(latent_stego)
# Blind decode — no cover image needed:
rec, _ = steg.decode_message(vae, stego, carriers[:len(bits)])
print(CDFPatchSteg.bits_to_text(rec))
```

### AGM Detector (defense) — `core/agm_detector.py`
> VAE round-trip residual analysis (arXiv 2510.07219). Detects perturbation-based attacks; blind to CDF-PatchSteg.
```python
from core.agm_detector import AGMDetector
det = AGMDetector()
suspicious, score, report = det.detect(vae, stego)
print(f"Suspicious: {suspicious}  Score: {score:.3f}")
print(f"Residual energy: {report['mean_residual_energy']:.4f}")
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Bit accuracy (structured images) | 100% at ε=5.0 |
| Bit accuracy (natural photos, CIFAR-10) | 98.8% avg |
| Visual quality (ε=2.0) | PSNR > 38 dB |
| Max message length demonstrated | 152 bits (19 ASCII chars) |
| Stealth vs LSB baseline (ε=2.0) | AUC 0.35 vs 0.65 |
| JPEG robustness | 100% accuracy at Q=50 |
| Carrier selection overhead | 1 additional encode-decode cycle |

---

## How It Works

### Encoding (Sender)
1. **Encode** the cover image through the SD VAE to get a 4×32×32 latent tensor
2. **Select carriers** — rank all spatial positions by perturbation survival through round-trip; pick top-K
3. **Embed bits** — at each carrier position, add `+ε·d` for bit=1 or `−ε·d` for bit=0, where `d` is a shared 4D direction vector
4. **Decode** the modified latent back to pixel space → stego image

### Decoding (Receiver)
1. **Re-encode** the received stego image through the same VAE
2. **Compare** projections at carrier positions against the clean round-trip baseline
3. **Read bits** — positive projection delta → 1, negative → 0
4. **(Optional)** Apply majority-vote decoding if repetition coding was used

### Why It Works
The VAE's KL regularization creates a locally smooth, approximately invertible latent space. Small directional perturbations produce coherent pixel changes that survive re-encoding because the decoder-encoder composition is nearly identity for in-distribution latent vectors. The perturbation-to-noise ratio at ε=2.0 (~2.3σ) sits at the boundary of statistical distinguishability, enabling stealth.

---

## Project Structure

```
patchsteg/
├── core/                          # Core library
│   ├── vae.py                     # StegoVAE: SD VAE wrapper
│   ├── steganography.py           # PatchSteg: ±epsilon encoding
│   ├── cdf_steganography.py       # CDFPatchSteg: distribution-preserving encoding
│   ├── pca_directions.py          # PCA-guided perturbation directions
│   ├── detector.py                # Latent-space steganalysis detector (46 features)
│   ├── analysis.py                # Mechanistic analysis helpers
│   └── metrics.py                 # PSNR, SSIM, bit accuracy
│
├── experiments/                   # Reproducible experiment scripts
│   ├── cdf_capacity_test.py       # CDF accuracy, KS test, detectability
│   ├── pca_test.py                # PCA directions analysis
│   ├── latent_detector_test.py    # Cross-method detection matrix
│   ├── v2_*.py                    # V2 experiments (multi-model, deployment, etc.)
│   └── *.py                       # V1 experiments (stability, capacity, etc.)
│
├── paper/                         # LaTeX paper
│   ├── main.tex                   # Full paper source
│   ├── build.sh                   # Build script (uses tectonic)
│   └── figures/                   # All generated figures
│
├── references/                    # Key research papers (PDFs + INDEX.md)
├── demo/                          # Gradio interactive demo
├── requirements.txt               # Python dependencies
├── CLAUDE.md                      # AI agent instructions (read by Claude Code)
└── EXPERIMENT_LOG.md              # Experiment results log
```

---

## Installation

### Prerequisites
- Python 3.10+
- ~1GB disk for the SD VAE model weights (downloaded automatically)
- CPU is sufficient; GPU optional (set `device='cuda'` for speedup)

### Setup

```bash
# Clone
git clone https://github.com/J0YY/patchsteg.git
cd patchsteg

# Create environment (conda recommended)
conda create -n patchsteg python=3.11 -y
conda activate patchsteg

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `torch` — tensor operations and VAE inference
- `diffusers` — Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`)
- `torchvision` — image transforms, CIFAR-10 dataset
- `Pillow` — image I/O
- `numpy`, `scipy` — numerical computation
- `matplotlib` — figure generation
- `scikit-learn` — detection classifiers
- `scikit-image` — SSIM metric
- `gradio` — interactive demo (optional)

---

## Quick Start

### Encode a message into an image

```python
from PIL import Image
from core.vae import StegoVAE
from core.steganography import PatchSteg

# Load model
vae = StegoVAE(device='cpu', image_size=256)
steg = PatchSteg(seed=42, epsilon=5.0)

# Load cover image
cover = Image.open("your_image.png").convert("RGB")

# Select stable carrier positions
carriers, stability_map = steg.select_carriers_by_stability(
    vae, cover, n_carriers=48, test_eps=5.0
)

# Encode a text message (with 3x repetition coding)
message = "HI"
bits = PatchSteg.text_to_bits(message)  # 16 bits
latent = vae.encode(cover)
latent_stego = steg.encode_message_with_repetition(latent, carriers, bits, reps=3)
stego_image = vae.decode(latent_stego)
stego_image.save("stego_image.png")
```

### Decode a message from a stego image

```python
# Receiver has the same seed (shared secret)
vae = StegoVAE(device='cpu', image_size=256)
steg = PatchSteg(seed=42, epsilon=5.0)

# Load the stego image and the original cover
stego = Image.open("stego_image.png").convert("RGB")
cover = Image.open("your_image.png").convert("RGB")

# Re-select carriers (deterministic given same seed and image)
carriers, _ = steg.select_carriers_by_stability(
    vae, cover, n_carriers=48, test_eps=5.0
)

# Decode
latent_clean = vae.encode(cover)
latent_received = vae.encode(stego)
decoded_bits = steg.decode_message_with_repetition(
    latent_clean, latent_received, carriers, n_bits=16, reps=3
)
decoded_text = PatchSteg.bits_to_text(decoded_bits)
print(f"Decoded: {decoded_text}")  # "HI"
```

### Run the interactive demo

```bash
# WSL (recommended — uses venv outside repo for speed)
source ~/patchseg-venv/bin/activate
python demo/app.py
# Opens Gradio interface at http://localhost:7860

# conda alternative
conda activate patchsteg
python demo/app.py
```

The demo has a **"Attack method"** selector in the UI:
- **Original PatchSteg (±ε)** — classic directional perturbation (detectable at high ε)
- **CDF-PatchSteg (SOTA, undetectable)** — distribution-preserving encoding, theoretically undetectable (AUC ≈ 0.5)

A static HTML summary of the demo is also saved at `demo/demo_summary.html` for offline presentation.

---

## SOTA Attacks

### CDF-PatchSteg — Distribution-Preserving Steganography

The original PatchSteg uses ±ε perturbations which are statistically detectable (AUC 0.93 at ε=5.0). **CDF-PatchSteg** is our state-of-the-art replacement: instead of shifting values by ε, it resamples each carrier value from the upper or lower half of the latent distribution N(μ, σ) using inverse Gaussian CDF. The resulting latent is statistically indistinguishable from a clean encode (target detection AUC ≈ 0.5).

**Key distinction vs prior SOTA** (Gaussian Shading, PRC, PSyDUCK): all existing distribution-preserving methods require *control over the diffusion noise seed during generation*. CDF-PatchSteg works **post-hoc** on existing images — the sender did not generate them.

#### Python API

```python
from core.vae import StegoVAE
from core.cdf_steganography import CDFPatchSteg

vae = StegoVAE(device='cpu', image_size=128)
cdf = CDFPatchSteg(seed=42, sigma=1.0)

# Encode
carriers, _ = cdf.select_carriers_by_stability(vae, cover_image, n_carriers=20)
latent = vae.encode(cover_image)
bits = [1, 0, 1, 1, 0, 1, 0, 0]  # your message bits
latent_stego = cdf.encode_message(latent, carriers, bits)
stego_image = vae.decode(latent_stego)

# Decode (no clean latent needed — blind decode)
decoded_bits, confidences = cdf.decode_message(vae, stego_image, carriers)
```

#### Run the experiment

```bash
source ~/patchseg-venv/bin/activate
python experiments/cdf_capacity_test.py
# Generates: paper/figures/cdf_capacity_curve.png, cdf_detectability.png, cdf_distribution.png
```

---

## Experiments

### Running experiments

```bash
conda activate patchsteg

# V1 core experiments
python experiments/run_remaining.py        # Robustness, mechanistic, detectability
python experiments/extended_experiments.py  # Natural photos, messages, LSB, theory

# V2 expanded experiments
python experiments/v2_run_all.py           # Multi-model, deployment, content science

# Phase 1-3 (new)
python experiments/cdf_capacity_test.py    # CDF distribution-preserving steganography
python experiments/pca_test.py             # PCA-guided perturbation directions
python experiments/latent_detector_test.py # Latent-space steganalysis detector
```

### Experiment Summary

| Experiment | Script | Key Finding |
|-----------|--------|-------------|
| Round-trip validation | `roundtrip_test.py` | 100% accuracy at ε=5.0 with 20 carriers |
| Carrier stability | `carrier_stability.py` | Stability is spatially non-uniform (range 0.98–3.48) |
| Capacity analysis | `capacity_test.py` | Up to 500 bits at 99.4% raw accuracy |
| Robustness | `robustness_test.py` | Survives JPEG Q=50, noise σ=0.01 |
| Detectability | `detectability_test.py` | AUC ranges from 0.44 (ε=1) to 0.97 (ε=10) |
| Natural photographs | `extended_experiments.py` | 98.8% accuracy on CIFAR-10 images |
| **CDF-PatchSteg** | `cdf_capacity_test.py` | Distribution-preserving encoding (target AUC≈0.5) |
| **PCA directions** | `pca_test.py` | Data-driven perturbation directions from latent PCA |
| **Steganalysis detector** | `latent_detector_test.py` | 46-feature detector + cross-method eval matrix |

---

## Core API Reference

### `StegoVAE(device='cpu', image_size=256)`
Wrapper around the Stable Diffusion VAE with proper scaling factor handling.

| Method | Description |
|--------|-------------|
| `encode(image) → Tensor` | PIL Image → latent tensor [1, 4, H/8, H/8] |
| `decode(latent) → Image` | Latent tensor → PIL Image |
| `round_trip(image) → (Tensor, Image)` | Encode → decode, returns both |

### `PatchSteg(seed=42, epsilon=5.0)`
Steganographic encoder/decoder using directional latent perturbations.

| Method | Description |
|--------|-------------|
| `compute_stability_map(vae, image)` | Returns [H/8, H/8] stability scores |
| `select_carriers_by_stability(vae, image, n_carriers)` | Top-N most stable positions |
| `encode_message(latent, carriers, bits)` | Embed bits into latent |
| `decode_message(latent_clean, latent_received, carriers)` | Extract bits |
| `encode_message_with_repetition(latent, carriers, bits, reps)` | Embed with R× coding |
| `decode_message_with_repetition(...)` | Decode with majority vote |
| `text_to_bits(text)` | ASCII → bit list |
| `bits_to_text(bits)` | Bit list → ASCII |

### `CapacityPatchSteg(seed=42, epsilon=2.0, bits_per_symbol=2)`
Capacity-oriented PatchSteg extension for longer textual payloads.

| Method | Description |
|--------|-------------|
| `encode_text(vae, image, text, enable_compression=True)` | Compress + frame UTF-8 text, then embed it |
| `decode_text(vae, cover_image, stego_image)` | Recover and validate framed text payload |
| `compute_gain_map(vae, image)` | Estimate per-carrier channel gain from the stability map |
| `select_carriers_by_capacity(vae, image, n_carriers)` | Top-N strongest carriers for multilevel coding |

### `AdaptivePatchSteg(seed=42, epsilon=2.5, bits_per_symbol=1)`
Detector-aware PatchSteg extension with balanced pairwise modulation.

| Method | Description |
|--------|-------------|
| `encode_text(vae, image, text, enable_compression=True)` | Frame, whiten, and embed text with pairwise differential symbols |
| `decode_text(vae, cover_image, stego_image)` | Recover and validate the framed payload |
| `compute_quality_map(vae, image)` | Joint carrier score over stability, texture, and round-trip drift |
| `select_carrier_pairs(vae, image, n_pairs)` | Build a spread-out schedule of balanced carrier pairs |

---

## Parameter Guide

| Parameter | Recommended | Range | Effect |
|-----------|-------------|-------|--------|
| `epsilon` | 2.0–5.0 | 0.5–10.0 | Higher = more reliable but more detectable |
| `n_carriers` | 20–100 | 1–1024 | More carriers = more bits but lower PSNR |
| `reps` | 3 | 1–5 | Higher = more error correction, fewer effective bits |
| `seed` | Any integer | — | Shared secret between sender and receiver |
| `image_size` | 256 | 256, 512 | Larger = more carrier positions (up to 4096) |

### Operating Points

| Profile | ε | Carriers | Reps | Bits | PSNR | Detection AUC |
|---------|---|----------|------|------|------|---------------|
| **Stealth** | 2.0 | 60 | 3 | 20 | ~30 dB | ~0.35 |
| **Balanced** | 5.0 | 48 | 3 | 16 | ~28 dB | ~0.81 |
| **High capacity** | 5.0 | 200 | 1 | 200 | ~21 dB | ~0.93 |

---

## Paper

The full paper is at `paper/main.pdf`. To rebuild after edits:

```bash
cd paper
# Requires tectonic (https://tectonic-typesetting.github.io/)
bash build.sh
```

The current paper reports quantitative results for the baseline binary PatchSteg channel and the evaluated CDF/PCA/PSyDUCK variants. `CapacityPatchSteg` is included in the codebase and this README as an experimental throughput-oriented extension, but its throughput/detectability numbers are not yet folded into the paper tables.

The paper includes:
- 7 tables with quantitative results
- 18 figures (pipeline diagrams, stability maps, Pareto frontiers, detection comparisons, etc.)
- Mechanistic analysis of why the channel works
- Comparison against LSB steganography baseline

---

## Threat Model

PatchSteg demonstrates that two AI agents sharing:
1. The same pretrained VAE model
2. A shared secret seed

can communicate covertly through images that appear identical to human observers. This is relevant to:
- **Multi-agent AI safety**: Agents passing images (screenshots, generated content, camera feeds) could hide coordination signals
- **AI alignment monitoring**: Current pixel-level monitors would miss latent-space steganography
- **Watermarking**: The same mechanism could embed provenance information in AI-generated images

### Limitations
- Both agents must share the exact same VAE checkpoint
- Under the paper-evaluated binary protocol, capacity is limited to tens to hundreds of bits per image
- Statistical detection is possible at higher perturbation strengths
- Requires the original cover image or an equivalent clean latent reference for decoding (not blind steganography)
- `CapacityPatchSteg` is experimental and not yet benchmarked in the paper-quality evaluation suite

---

## Citation

If you use PatchSteg in your research:

```bibtex
@article{patchsteg2026,
  title={PatchSteg: Covert Steganographic Communication Between Vision Models via Latent Space Perturbation},
  author={Team PatchSteg},
  year={2026}
}
```

---

## License

This project is released for research and educational purposes. Use responsibly.
