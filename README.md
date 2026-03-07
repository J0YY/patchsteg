# PatchSteg: Covert Steganographic Communication Between Vision Models via Latent Space Perturbation

**Training-free steganographic framework for hiding binary messages in images using pretrained VAE latent spaces.**

PatchSteg enables two AI agents sharing a secret key to embed and recover binary messages in images with negligible visual artifacts. No model fine-tuning, no auxiliary networks, no adversarial training — just directional perturbations in the latent space of a pretrained autoencoder.

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
│   ├── vae.py                     # StegoVAE: SD VAE wrapper with proper scaling
│   ├── steganography.py           # PatchSteg: encoding, decoding, carrier selection
│   ├── analysis.py                # Mechanistic analysis (channel importance, error maps)
│   └── metrics.py                 # PSNR, SSIM, bit accuracy
│
├── experiments/                   # Reproducible experiment scripts
│   ├── carrier_stability.py       # Stability map computation and visualization
│   ├── capacity_test.py           # Capacity vs quality tradeoff
│   ├── robustness_test.py         # JPEG, noise, resize robustness
│   ├── detectability_test.py      # Statistical detectability analysis
│   ├── mechanistic_analysis.py    # Channel importance, error correlation
│   ├── extended_experiments.py    # Natural photos, longer messages, LSB baseline, theory
│   ├── generate_figures.py        # Publication-quality figure generation
│   ├── run_all.py                 # Run all experiments sequentially
│   └── run_remaining.py           # Run subset of experiments
│
├── paper/                         # LaTeX paper
│   ├── main.tex                   # Full paper source
│   ├── main.pdf                   # Compiled PDF
│   ├── build.sh                   # Build script (uses tectonic)
│   └── figures/                   # All generated figures (18 PNGs)
│
├── demo/                          # Interactive demo
│   └── app.py                     # Gradio app (encode/decode/analyze)
│
├── roundtrip_test.py              # Initial validation test
├── requirements.txt               # Python dependencies
└── EXPERIMENT_LOG.md              # Detailed experiment results log
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
conda activate patchsteg
python demo/app.py
# Opens Gradio interface at http://localhost:7860
```

---

## Experiments

### Running all experiments

```bash
conda activate patchsteg

# Run the initial round-trip validation
python roundtrip_test.py

# Run core experiments (carrier stability, capacity, robustness, detectability, mechanistic)
python experiments/run_remaining.py

# Run extended experiments (natural photos, longer messages, LSB baseline, theory)
python experiments/extended_experiments.py

# Generate publication figures
python experiments/generate_figures.py
```

### Experiment Summary

| Experiment | Script | Key Finding |
|-----------|--------|-------------|
| Round-trip validation | `roundtrip_test.py` | 100% accuracy at ε=5.0 with 20 carriers |
| Carrier stability | `carrier_stability.py` | Stability is spatially non-uniform (range 0.98–3.48) |
| Capacity analysis | `capacity_test.py` | Up to 500 bits at 99.4% raw accuracy |
| Robustness | `robustness_test.py` | Survives JPEG Q=50, noise σ=0.01 |
| Detectability | `detectability_test.py` | AUC ranges from 0.44 (ε=1) to 0.97 (ε=10) |
| Mechanistic analysis | `mechanistic_analysis.py` | All 4 channels contribute equally; recon error ≠ stability |
| Natural photographs | `extended_experiments.py` | 98.8% accuracy on CIFAR-10 images |
| Message stress test | `extended_experiments.py` | 152-bit messages at 100% accuracy |
| LSB comparison | `extended_experiments.py` | PatchSteg ε=2 stealthier than LSB (AUC 0.35 vs 0.65) |
| Theoretical analysis | `extended_experiments.py` | Perturbation/noise ratio explains detectability threshold |

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
- Channel capacity is limited (tens to hundreds of bits per image)
- Statistical detection is possible at higher perturbation strengths
- Requires the original cover image for decoding (not blind steganography)

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
