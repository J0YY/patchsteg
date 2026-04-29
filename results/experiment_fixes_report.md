# Experiment Fixes Run Log

Date run: 2026-04-29  
Repo: `/Users/joyyang/Projects/patchsteg`  
Paper LaTeX files edited in this pass: none

## Summary

I ran the shared experiment fixes from `fixes.md` that could be executed locally without adding new API-dependent demos or new external high-resolution datasets:

- P1: added and ran PSNR + SSIM + LPIPS reporting for the 300-image CIFAR-10 evaluation.
- P1: reran the multi-/cross-VAE evaluation with PSNR + SSIM + LPIPS.
- P2: reran detector reporting with precision/recall/confusion matrices at fixed low-FPR operating points.
- P3: reran robustness with realistic deployment distortions and moved JPEG Q10 into an appendix-labeled stress baseline.
- Added a new top-k/random-k/bottom-k carrier ablation with held-out direction validation support.
- Added a new composite Enc(Dec(z)) directional-gain analysis script.

Not run in this pass:

- P4 agent-in-loop demo: requires a VLM-agent API setup, captions, monitor prompts, and cost/latency tracking.
- P5 sanitizer Pareto: not run yet; needs the 100-image sanitizer sweep across latent smoothing, quantile reflection, latent Gaussian noise, JPEG, and optional SDEdit.
- P6 adaptive sender: not run yet; requires implementing/training the gradient-aware detector-evading sender.
- P7 native high-resolution validation: not run yet; requires collecting/standardizing 100 native 512x512 COCO/LAION-style images.
- P8 composite Jacobian: optional and not run.

## Code/Dependency Changes

- `requirements.txt`
  - Added `lpips`
  - Added `pytorch-msssim`
- `core/metrics.py`
  - Added `compute_lpips_pil`
  - Added `compute_quality_metrics`
- `experiments/v2_serious_dataset.py`
  - Changed CIFAR-10 scale from 20/class to 30/class, i.e. 300 total images.
  - Added PSNR, SSIM, and LPIPS result export.
  - Added MPS device support through `PATCHSTEG_DEVICE`.
- `experiments/v2_multimodel.py`
  - Added PSNR, SSIM, and LPIPS result export for same-model and cross-model settings.
  - Added MPS device support through `PATCHSTEG_DEVICE`.
- `experiments/v2_detection_strength.py`
  - Added out-of-fold detector scores.
  - Added precision, recall, thresholds, achieved FPR, and confusion matrices at target FPRs 0.1%, 1%, and 5%.
  - Cached latent feature extraction to avoid redundant VAE passes.
- `experiments/v2_robustness_deployment.py`
  - Replaced the main robustness suite with realistic distortions from `fixes.md`.
  - Cached stego samples per image/epsilon so distortions reuse the same carrier selections.
- `experiments/topk_bottomk_ablation.py`
  - New CLI experiment for top-k/random-k/bottom-k carrier comparisons.
  - Supports held-out validation through separate stability, payload, and evaluation direction seeds.
  - Exposes `compute_stability_map`, `select_carriers_from_map`, and `evaluate_carriers` boundaries.
- `experiments/composite_directional_gain.py`
  - New CLI experiment estimating local directional gain of the composite VAE round-trip map.
  - Compares local directional gain to the existing global PatchSteg stability proxy.
  - Optionally evaluates top-gain/random/bottom-gain carrier recovery.

## P1: 300-CIFAR Quality Metrics

Command:

```bash
PATCHSTEG_DEVICE=mps python experiments/v2_serious_dataset.py
```

Artifacts:

- Raw per-image CSV: `results/v2_serious_dataset_metrics.csv`
- Summary JSON: `results/v2_serious_dataset_metrics.json`
- Figure: `paper/figures/serious_dataset.png`

Results:

| epsilon | n | bit acc mean | bit acc 95% CI | PSNR mean | SSIM mean | LPIPS mean |
|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 300 | 97.83% | [97.37, 98.25] | 38.67 dB | 0.9805 | 0.0390 |
| 5.0 | 300 | 99.05% | [98.82, 99.30] | 28.83 dB | 0.9472 | 0.2260 |

Notes:

- This supersedes the older 200-image run in `experiments/v2_serious_dataset.py`.
- LPIPS used the AlexNet backbone.

## P1: Multi-/Cross-VAE Quality Metrics

Command:

```bash
PATCHSTEG_DEVICE=mps python experiments/v2_multimodel.py
```

Artifacts:

- CSV: `results/v2_multimodel_metrics.csv`
- JSON: `results/v2_multimodel_metrics.json`
- Figure: `paper/figures/multimodel.png`

Results:

| model | epsilon | n | bit acc mean | PSNR mean | SSIM mean | LPIPS mean |
|---|---:|---:|---:|---:|---:|---:|
| SD-VAE-MSE | 2.0 | 10 | 98.0% | 26.49 dB | 0.6192 | 0.1160 |
| SD-VAE-MSE | 5.0 | 10 | 99.5% | 21.05 dB | 0.5731 | 0.3172 |
| SD-VAE-EMA | 2.0 | 10 | 99.0% | 26.49 dB | 0.6126 | 0.0869 |
| SD-VAE-EMA | 5.0 | 10 | 98.5% | 20.91 dB | 0.5688 | 0.2883 |
| SDXL-VAE | 2.0 | 10 | 98.5% | 22.99 dB | 0.6101 | 0.2356 |
| SDXL-VAE | 5.0 | 10 | 89.0% | 18.63 dB | 0.5620 | 0.4415 |
| Cross: SD-VAE-MSE -> SD-VAE-EMA | 5.0 | 5 | 100.0% | 20.54 dB | 0.5726 | 0.3337 |
| Cross: SD-VAE-EMA -> SD-VAE-MSE | 5.0 | 5 | 99.0% | 20.77 dB | 0.5687 | 0.2968 |

## P2: Detector Operating Points

Command:

```bash
python experiments/v2_detection_strength.py
```

Artifacts:

- CSV: `results/v2_detection_operating_points.csv`
- JSON: `results/v2_detection_operating_points.json`
- Figure: `paper/figures/detection_strength.png`

Protocol:

- 30 synthetic test images.
- One clean and one stego sample per image.
- Epsilon values: 1.0, 2.0, 5.0.
- Detectors: LR/MLP latent, LR/MLP residual, LR spectral, LR/MLP combined.
- Reported operating points at target FPRs 0.1%, 1%, and 5%.

Important caveat:

- With only 30 clean negatives, the smallest nonzero achievable FPR is 1/30 = 3.33%. The fixed-FPR thresholds are therefore quantized; the CSV/JSON include both target FPR and achieved FPR.

Selected 1% target-FPR results:

| epsilon | detector | AUC | accuracy | precision | recall | achieved FPR | confusion matrix |
|---:|---|---:|---:|---:|---:|---:|---|
| 1.0 | LR-Latent | 0.217 | 0.300 | 0.000 | 0.000 | 0.033 | TN=29 FP=1 FN=30 TP=0 |
| 1.0 | LR-Combined | 1.000 | 1.000 | 0.968 | 1.000 | 0.033 | TN=29 FP=1 FN=0 TP=30 |
| 2.0 | LR-Latent | 0.868 | 0.750 | 0.929 | 0.433 | 0.033 | TN=29 FP=1 FN=17 TP=13 |
| 2.0 | LR-Combined | 1.000 | 1.000 | 0.968 | 1.000 | 0.033 | TN=29 FP=1 FN=0 TP=30 |
| 5.0 | LR-Latent | 1.000 | 0.983 | 0.968 | 1.000 | 0.033 | TN=29 FP=1 FN=0 TP=30 |
| 5.0 | LR-Combined | 1.000 | 1.000 | 0.968 | 1.000 | 0.033 | TN=29 FP=1 FN=0 TP=30 |

## P3: Realistic Robustness Refresh

Command:

```bash
python experiments/v2_robustness_deployment.py
```

Artifacts:

- CSV: `results/v2_robustness_realistic.csv`
- JSON: `results/v2_robustness_realistic.json`
- Figure: `paper/figures/deployment_robustness.png`

Protocol:

- 10 synthetic test images.
- Epsilon values: 2.0 and 5.0.
- Main distortions: no transform, JPEG Q70 roundtrip, social-media-style max-edge downscale + JPEG Q80, screenshot simulation, additive noise sigma=0.10, VAE re-encode.
- JPEG Q10 retained only as `Appendix stress: JPEG Q10`.

Results:

| epsilon | distortion | bit acc mean | bit acc std |
|---:|---|---:|---:|
| 2.0 | None | 99.5% | 1.5 |
| 2.0 | JPEG Q70 roundtrip | 99.0% | 2.0 |
| 2.0 | Social downscale+Q80 | 99.0% | 2.0 |
| 2.0 | Screenshot sim | 97.0% | 4.58 |
| 2.0 | Noise sigma=0.10 | 87.5% | 11.88 |
| 2.0 | VAE re-encode | 99.5% | 1.5 |
| 2.0 | Appendix stress: JPEG Q10 | 81.0% | 15.30 |
| 5.0 | None | 99.0% | 2.0 |
| 5.0 | JPEG Q70 roundtrip | 99.0% | 2.0 |
| 5.0 | Social downscale+Q80 | 98.5% | 3.20 |
| 5.0 | Screenshot sim | 98.5% | 3.20 |
| 5.0 | Noise sigma=0.10 | 98.5% | 2.29 |
| 5.0 | VAE re-encode | 99.0% | 2.0 |
| 5.0 | Appendix stress: JPEG Q10 | 96.5% | 3.91 |

## Generated Files To Use Later

- `results/v2_serious_dataset_metrics.csv`
- `results/v2_serious_dataset_metrics.json`
- `results/v2_multimodel_metrics.csv`
- `results/v2_multimodel_metrics.json`
- `results/v2_detection_operating_points.csv`
- `results/v2_detection_operating_points.json`
- `results/v2_robustness_realistic.csv`
- `results/v2_robustness_realistic.json`
- `paper/figures/serious_dataset.png`
- `paper/figures/multimodel.png`
- `paper/figures/detection_strength.png`
- `paper/figures/deployment_robustness.png`

## New Carrier Ablation Script

Script:

- `experiments/topk_bottomk_ablation.py`

Purpose:

- Tests whether positions selected by the PatchSteg stability map recover bits better than random or bottom-ranked positions under matched image, VAE, epsilon, payload size, and decode protocol.
- The held-out mode estimates stability using one direction seed set and evaluates recovery with different held-out direction seeds.

Main CLI args:

```bash
python experiments/topk_bottomk_ablation.py \
  --dataset cifar10 \
  --num_images 50 \
  --resolution 256 \
  --epsilon 2.0 \
  -k 20 \
  --num_trials 3 \
  --num_stability_directions 4 \
  --num_eval_directions 4 \
  --heldout_eval true \
  --output_dir results/topk_bottomk_ablation \
  --device mps
```

Smoke test command run:

```bash
python experiments/topk_bottomk_ablation.py \
  --dataset cifar10 \
  --num_images 3 \
  --resolution 128 \
  --epsilon 2.0 \
  -k 8 \
  --num_trials 1 \
  --num_stability_directions 2 \
  --num_eval_directions 2 \
  --seed 123 \
  --stability_seed 123 \
  --payload_seed 456 \
  --direction_seed 789 \
  --heldout_direction_seed 987 \
  --heldout_eval true \
  --output_dir results/topk_bottomk_smoke \
  --device mps
```

Smoke test artifacts:

- `results/topk_bottomk_smoke/topk_bottomk_ablation.csv`
- `results/topk_bottomk_smoke/topk_bottomk_ablation_summary.json`
- `results/topk_bottomk_smoke/topk_bottomk_bit_accuracy.png`
- `results/topk_bottomk_smoke/topk_bottomk_psnr.png`

Smoke test results:

| mode | n | bit acc mean | bit acc std | PSNR mean |
|---|---:|---:|---:|---:|
| topk | 6 | 100.00% | 0.00 | 29.98 dB |
| random | 6 | 100.00% | 0.00 | 31.02 dB |
| bottomk | 6 | 95.83% | 9.32 | 32.06 dB |

Determinism check:

- Re-ran the same command with output directory `results/topk_bottomk_smoke_repeat`.
- `diff -u` on the two CSV files returned no differences.

## New Composite Directional-Gain Script

Script:

- `experiments/composite_directional_gain.py`

Purpose:

- Estimates local directional gain for the VAE round-trip map `F(z) = Enc(Dec(z))`:
  `gain(r,c,d) = <F(z + epsilon*d_{r,c}) - F(z), d_{r,c}> / epsilon`.
- High positive gain means the round-trip preserves sign/magnitude of the local latent perturbation along `d`.
- The script also records the existing PatchSteg stability score. That score is a related global perturb-all-positions proxy, not mathematically identical to the local gain estimate.

Main CLI args:

```bash
python experiments/composite_directional_gain.py \
  --dataset cifar10 \
  --num_images 20 \
  --resolution 256 \
  --epsilon 2.0 \
  --num_positions 128 \
  --num_directions 4 \
  --k 20 \
  --output_dir results/composite_directional_gain \
  --device mps
```

Smoke test command run:

```bash
python experiments/composite_directional_gain.py \
  --dataset cifar10 \
  --num_images 3 \
  --resolution 128 \
  --epsilon 2.0 \
  --num_positions 12 \
  --num_directions 1 \
  --k 4 \
  --seed 321 \
  --output_dir results/composite_gain_smoke \
  --device mps
```

Smoke test artifacts:

- `results/composite_gain_smoke/composite_directional_gain.csv`
- `results/composite_gain_smoke/composite_directional_gain_recovery.csv`
- `results/composite_gain_smoke/composite_directional_gain_summary.json`
- `results/composite_gain_smoke/directional_gain_heatmap.png`
- `results/composite_gain_smoke/gain_vs_stability_scatter.png`

Smoke test results:

- Gain vs existing stability proxy, Pearson: `r=0.7054`, `p=1.55e-06`, `n=36`.
- Gain vs existing stability proxy, Spearman: `r=-0.1753`, `p=0.3065`, `n=36`.
- Recovery smoke result saturated at 100% for top_gain, random, and bottom_gain with this tiny 3-image/4-carrier run; this smoke is only an end-to-end validation, not the scientific result.

Determinism check:

- Re-ran the same command with output directory `results/composite_gain_smoke_repeat`.
- `diff -u` on both gain CSV and recovery CSV returned no differences.
