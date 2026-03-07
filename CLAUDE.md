# PatchSteg - Claude Code Instructions

## Team
- 3-person team for Varsity Hackathon 2026
- All teammates use AI agents (Claude Code) — this file keeps everyone aligned
- Always pull before starting work and push after finishing to avoid conflicts

## Project Structure
- `core/` — Library code (steganography, VAE wrapper, analysis, metrics, detector)
- `experiments/` — Experiment scripts (each self-contained, runnable standalone)
- `paper/` — LaTeX paper (`main.tex`), figures in `paper/figures/`
- `demo/` — Gradio demo app
- `references/` — Key research papers as PDFs (see `references/INDEX.md` for summaries)

## References (READ FIRST)
The `references/` folder contains 7 PDFs of papers central to this project. **Read `references/INDEX.md`** for a table mapping each paper to why it matters and which phase it relates to. Key papers:
- **Gaussian Shading** (CVPR 2024) — inspiration for CDF-PatchSteg (Phase 1)
- **VAEs Pursue PCA Directions** (CVPR 2019) — theoretical basis for PCA directions (Phase 2)
- **RoSteALS** (CVPR-W 2023) — closest prior work, key comparison baseline
- **Secret Collusion** (Motwani et al.) — defines our threat model
- **Tree-Ring Watermarks**, **Stable Signature**, **TrustMark** — benchmarking baselines

## Conventions
- **Seeds**: Always use `seed=42` unless testing seed variation
- **Printing**: Use `flush=True` on all print statements
- **Matplotlib**: Always set `matplotlib.use('Agg')` before importing pyplot
- **Figures**: Save to `paper/figures/` as PNG at `dpi=150`
- **Image size**: Default `IMG_SIZE = 256` for experiments (32x32 latent grid)
- **Path setup**: Experiment scripts insert project root into sys.path:
  ```python
  sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
  ```

## Paper Workflow
- **Update `paper/main.tex`** whenever new experiments/findings are added:
  - Add new sections, tables, figures as needed
  - Reference figures with `\includegraphics{figures/<name>.png}`
  - Use `\label` and `\ref` for cross-references
- **Build PDF**: `cd paper && bash build.sh` (uses tectonic)
- **Figures directory**: `paper/figures/` — all experiment figures go here

## Experiment Log
- Keep `EXPERIMENT_LOG.md` in sync with every new experiment
- Log format: experiment name, key results, figure filenames, status

## Key APIs
- `StegoVAE(device, image_size)` — `.encode(img)`, `.decode(latent)`, `.round_trip(img)`
- `PatchSteg(seed, epsilon)` — `.encode_message()`, `.decode_message()`, `.select_carriers_by_stability()`
- `CDFPatchSteg(seed, sigma)` — `.encode_message()`, `.decode_message()` (distribution-preserving, no clean latent needed for decode)
- `PCAPatchSteg(pca_directions, seed, epsilon, component)` — PCA-guided variant, same API as PatchSteg
- `PCADirections(n_components)` — `.fit_global(vae, images)`, `.get_direction(k)`, `.save()/.load()`
- `LatentStegDetector()` — `.extract_features(vae, img)`, `.fit(X, y)`, `.predict(X)` (46-feature steganalysis)
- `compute_psnr(img1, img2)`, `compute_ssim_pil(img1, img2)`, `bit_accuracy(sent, received)`

## Evolution Roadmap (Current State)
| Phase | Description | Status | Files |
|-------|-------------|--------|-------|
| 1 | CDF-PatchSteg (distribution-preserving) | Code done, experiments pending | `core/cdf_steganography.py`, `experiments/cdf_capacity_test.py` |
| 2 | PCA-guided perturbation directions | Code done, experiments pending | `core/pca_directions.py`, `experiments/pca_test.py` |
| 3 | Latent-space steganalysis detector | Code done, experiments pending | `core/detector.py`, `experiments/latent_detector_test.py` |
| 4 | Extended benchmarking vs baselines | Not started | — |

Phases 1 & 2 are independent. Phase 3 needs Phase 1. Phase 4 needs all.

## Dependencies
- Listed in `requirements.txt`
- Add new deps there when needed
