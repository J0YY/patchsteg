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
- Added and ran a new top-k/random-k/bottom-k carrier ablation with held-out direction validation support.
- Added and ran a new composite Enc(Dec(z)) directional-gain analysis script.
- Added and ran a monitor benchmark to replace the proposed agent-in-loop demo with directly reportable monitor capability tiers.

Not run in this pass:

- P4 agent-in-loop demo: intentionally replaced by the monitor benchmark below; no separate VLM/API demo was run.
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
- `experiments/monitor_benchmark.py`
  - New CLI experiment generating matched clean/stego images and benchmarking monitor feature families.
  - Monitors include pixel-only no-reference, VAE round-trip, paired/reference pixel residual, and combined feature monitors.
  - Reports AUC, accuracy, low-FPR precision/recall, and confusion matrices.

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
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_summary.json`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_operating_points.csv`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_sample_scores.csv`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_pair_quality.csv`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_auc.png`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_operating_point.png`
- `results/monitor_benchmark_smoke/monitor_benchmark_summary.json`
- `results/monitor_benchmark_smoke/monitor_benchmark_operating_points.csv`
- `results/monitor_benchmark_smoke/monitor_benchmark_sample_scores.csv`
- `results/monitor_benchmark_smoke/monitor_benchmark_pair_quality.csv`
- `results/monitor_benchmark_smoke/monitor_benchmark_auc.png`
- `results/monitor_benchmark_smoke/monitor_benchmark_operating_point.png`
- `results/topk_bottomk_cifar50_eps2_k20_heldout/topk_bottomk_ablation.csv`
- `results/topk_bottomk_cifar50_eps2_k20_heldout/topk_bottomk_ablation_summary.json`
- `results/topk_bottomk_cifar50_eps2_k20_heldout/topk_bottomk_bit_accuracy.png`
- `results/topk_bottomk_cifar50_eps2_k20_heldout/topk_bottomk_psnr.png`
- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/composite_directional_gain.csv`
- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/composite_directional_gain_recovery.csv`
- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/composite_directional_gain_summary.json`
- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/directional_gain_heatmap.png`
- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/gain_vs_stability_scatter.png`

## New Monitor Benchmark

Script:

- `experiments/monitor_benchmark.py`

Purpose:

- Replaces the proposed agent-in-loop demo with a cleaner benchmark that can be integrated directly into the paper.
- Generates matched benign/stego images under the same VAE, epsilon, payload size, seed, and image distribution.
- Evaluates monitor capability tiers rather than a single demo prompt:
  - `pixel_noref`: sees only posted pixels and uses no clean reference or VAE.
  - `vae_roundtrip`: can run the VAE and inspect latent round-trip residuals.
  - `paired_pixel_residual`: has access to the benign clean-roundtrip image for the same cover.
  - `combined`: concatenates all feature families.

Main CLI args:

```bash
python experiments/monitor_benchmark.py \
  --dataset cifar10 \
  --num_images 50 \
  --resolution 256 \
  --epsilon 2.0 \
  -k 20 \
  --seed 42 \
  --cv_splits 5 \
  --output_dir results/monitor_benchmark \
  --device mps
```

Outputs:

- `monitor_benchmark_summary.json`
- `monitor_benchmark_operating_points.csv`
- `monitor_benchmark_sample_scores.csv`
- `monitor_benchmark_pair_quality.csv`
- `monitor_benchmark_auc.png`
- `monitor_benchmark_operating_point.png`

Full run command:

```bash
python experiments/monitor_benchmark.py \
  --dataset cifar10 \
  --num_images 100 \
  --resolution 256 \
  --epsilon 2.0 \
  -k 20 \
  --seed 42 \
  --cv_splits 5 \
  --output_dir results/monitor_benchmark_cifar100_eps2_k20 \
  --device mps
```

Full run artifacts:

- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_summary.json`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_operating_points.csv`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_sample_scores.csv`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_pair_quality.csv`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_auc.png`
- `results/monitor_benchmark_cifar100_eps2_k20/monitor_benchmark_operating_point.png`

Full run pair quality:

| metric | value |
|---|---:|
| bit accuracy mean | 97.90% |
| bit accuracy std | 3.48 |
| PSNR mean | 38.60 dB |
| SSIM mean | 0.9799 |
| LPIPS mean | 0.0404 |

Full run monitor results at target 1% FPR:

| monitor | AUC | accuracy | precision | recall | achieved FPR | confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| pixel_noref | 0.605 | 0.570 | 0.750 | 0.030 | 0.010 | TN=99 FP=1 FN=97 TP=3 |
| vae_roundtrip | 0.782 | 0.730 | 0.000 | 0.000 | 0.010 | TN=99 FP=1 FN=100 TP=0 |
| paired_pixel_residual | 1.000 | 1.000 | 0.833 | 1.000 | 0.200 | TN=80 FP=20 FN=0 TP=100 |
| combined | 1.000 | 1.000 | 0.990 | 1.000 | 0.010 | TN=99 FP=1 FN=0 TP=100 |

Full run caveats:

- With 100 clean negatives, the smallest nonzero achieved FPR is 1%, so the 0.1% target-FPR operating point is still quantized.
- The `paired_pixel_residual` monitor is intentionally strong because it has a clean paired reference for the same cover. Its 1% target-FPR row achieved 20% FPR in this run, so it should be interpreted as a capability-tier result rather than a calibrated low-FPR deployment point.
- The key qualitative result is that no-reference pixel monitoring is weak, VAE round-trip features help but do not recover stego samples at the strict 1% point, and the combined/reference-aware monitor is highly effective.

Smoke test command run:

```bash
python experiments/monitor_benchmark.py \
  --dataset cifar10 \
  --num_images 3 \
  --resolution 128 \
  --epsilon 2.0 \
  -k 8 \
  --seed 222 \
  --cv_splits 3 \
  --output_dir results/monitor_benchmark_smoke \
  --device mps
```

Smoke test artifacts:

- `results/monitor_benchmark_smoke/monitor_benchmark_summary.json`
- `results/monitor_benchmark_smoke/monitor_benchmark_operating_points.csv`
- `results/monitor_benchmark_smoke/monitor_benchmark_sample_scores.csv`
- `results/monitor_benchmark_smoke/monitor_benchmark_pair_quality.csv`
- `results/monitor_benchmark_smoke/monitor_benchmark_auc.png`
- `results/monitor_benchmark_smoke/monitor_benchmark_operating_point.png`

Smoke test pair quality:

| metric | value |
|---|---:|
| bit accuracy mean | 95.83% |
| bit accuracy std | 5.89 |
| PSNR mean | 30.41 dB |
| SSIM mean | 0.9287 |
| LPIPS mean | 0.1131 |

Smoke test monitor results:

| monitor | AUC | accuracy | precision at target 1% FPR | recall at target 1% FPR | achieved FPR |
|---|---:|---:|---:|---:|---:|
| pixel_noref | 0.556 | 0.333 | 0.500 | 0.333 | 0.333 |
| vae_roundtrip | 0.000 | 0.167 | 0.000 | 0.000 | 0.333 |
| paired_pixel_residual | 1.000 | 1.000 | 0.750 | 1.000 | 0.333 |
| combined | 0.667 | 0.500 | 0.000 | 0.000 | 0.333 |

Smoke test caveat:

- This is only a 3-image deterministic smoke test. The achieved FPR is quantized at 1/3 because there are only three clean negatives, so these numbers should not be treated as paper claims.

Determinism check:

- Re-ran the same command with output directory `results/monitor_benchmark_smoke_repeat`.
- `diff -u` on operating-point, sample-score, and pair-quality CSV files returned no differences.

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

Full held-out run command:

```bash
python experiments/topk_bottomk_ablation.py \
  --dataset cifar10 \
  --num_images 50 \
  --resolution 256 \
  --epsilon 2.0 \
  -k 20 \
  --num_trials 2 \
  --num_stability_directions 3 \
  --num_eval_directions 3 \
  --seed 123 \
  --stability_seed 123 \
  --payload_seed 456 \
  --direction_seed 789 \
  --heldout_direction_seed 987 \
  --heldout_eval true \
  --output_dir results/topk_bottomk_cifar50_eps2_k20_heldout \
  --device mps
```

Full held-out artifacts:

- `results/topk_bottomk_cifar50_eps2_k20_heldout/topk_bottomk_ablation.csv`
- `results/topk_bottomk_cifar50_eps2_k20_heldout/topk_bottomk_ablation_summary.json`
- `results/topk_bottomk_cifar50_eps2_k20_heldout/topk_bottomk_bit_accuracy.png`
- `results/topk_bottomk_cifar50_eps2_k20_heldout/topk_bottomk_psnr.png`

Full held-out results:

| mode | n | bit acc mean | bit acc 95% CI | bit acc std | PSNR mean | PSNR std |
|---|---:|---:|---:|---:|---:|---:|
| topk | 300 | 97.77% | [97.28, 98.18] | 4.00 | 35.62 dB | 3.36 |
| random | 300 | 99.85% | [99.73, 99.95] | 0.95 | 36.52 dB | 3.25 |
| bottomk | 300 | 95.53% | [94.70, 96.27] | 6.96 | 37.61 dB | 3.29 |

Full held-out interpretation:

- This is a mixed or negative result for the stability-selection story under held-out directions. Random carriers outperformed top-k carriers in bit recovery and also had higher PSNR.
- Bottom-k carriers were clearly worse than random, so the stability map still contains some useful information about poor carrier regions.
- The result suggests that the current top-k stability proxy may overfit to the measured direction set or trade off recovery against stronger perturbation visibility under this held-out protocol. It should not be used as a clean claim that top-k carrier selection improves recovery on CIFAR-10 without additional analysis.

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

Full run command:

```bash
python experiments/composite_directional_gain.py \
  --dataset cifar10 \
  --num_images 20 \
  --resolution 256 \
  --epsilon 2.0 \
  --num_positions 128 \
  --num_directions 2 \
  --k 20 \
  --seed 321 \
  --output_dir results/composite_directional_gain_cifar20_eps2_pos128_dir2 \
  --device mps
```

Full run artifacts:

- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/composite_directional_gain.csv`
- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/composite_directional_gain_recovery.csv`
- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/composite_directional_gain_summary.json`
- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/directional_gain_heatmap.png`
- `results/composite_directional_gain_cifar20_eps2_pos128_dir2/gain_vs_stability_scatter.png`

Full run results:

| metric | value |
|---|---:|
| sampled gain rows | 5120 |
| Pearson r, gain vs existing stability | 0.3475 |
| Pearson p-value | 3.27e-145 |
| Spearman r, gain vs existing stability | 0.4298 |
| Spearman p-value | 2.56e-229 |

Full run carrier recovery by directional-gain mode:

| mode | n | bit acc mean | bit acc std |
|---|---:|---:|---:|
| top_gain | 20 | 100.00% | 0.00 |
| random | 20 | 99.75% | 1.09 |
| bottom_gain | 20 | 99.00% | 2.55 |

Full run interpretation:

- Local directional gain is positively correlated with the existing PatchSteg stability proxy across sampled positions and directions.
- Recovery is nearly saturated at epsilon 2.0 with 20 carriers on this subset, so the carrier-recovery rows are less discriminative than the gain/stability correlation.
- The heatmap and scatter plot are the useful paper-facing artifacts from this run.

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

## Non-CIFAR Natural-Image Dataset Evaluation

Script:

- `experiments/natural_dataset_eval.py`

Purpose:

- Adds a CPU-feasible non-CIFAR evaluation on native object images.
- This is meant to strengthen the paper beyond CIFAR-only evidence by testing the same PatchSteg encode/decode path on a larger, variable-resolution natural-image corpus.

Dataset research and choice:

- Selected Caltech101 for the CPU-bound run. The official CaltechDATA record describes the dataset as pictures of objects from 101 categories, roughly 40 to 800 images per category, roughly 300 x 200 pixels, and a 137.4 MB archive: https://data.caltech.edu/records/mzrjq-6wc02
- TensorFlow Datasets lists Caltech101 as roughly 9k images, 101 object classes plus background clutter, variable image sizes with typical edge lengths of 200-300 pixels, and a 131.05 MiB download: https://www.tensorflow.org/datasets/catalog/caltech101
- I did not choose COCO, LAION, or ImageNet-style options for this pass because the user constraint was CPU-only with about 12 hours available. Those datasets are better paper-scale follow-ups, but the download and wall-clock cost are much larger than needed for an immediate robustness check.
- I did not choose STL10 as the primary result because it is closer to CIFAR-style small images. It remains supported by the script as `--dataset stl10`.

Implementation details:

- Loads Caltech101 through `torchvision.datasets.Caltech101`.
- Uses stratified sampling over class labels so the selected subset covers many classes instead of taking a contiguous slice.
- Reuses existing PatchSteg components: `StegoVAE`, `PatchSteg.select_carriers_by_stability`, `PatchSteg.encode_message`, `PatchSteg.decode_message`, `bit_accuracy`, `compute_psnr`, and `compute_ssim_pil`.
- LPIPS was skipped for these CPU runs to keep runtime bounded.

Main run command:

```bash
python experiments/natural_dataset_eval.py \
  --dataset caltech101 \
  --num_images 202 \
  --resolution 128 \
  --epsilon 2.0 5.0 \
  --num_carriers 20 \
  --device cpu \
  --output_dir results/caltech101_eval_cpu202_r128_eps2_5 \
  --skip_lpips \
  --bootstrap 1000
```

Main run artifacts:

- `results/caltech101_eval_cpu202_r128_eps2_5/natural_dataset_eval.csv`
- `results/caltech101_eval_cpu202_r128_eps2_5/natural_dataset_eval_summary.json`
- `results/caltech101_eval_cpu202_r128_eps2_5/natural_dataset_eval_summary.png`
- `results/caltech101_eval_cpu202_r128_eps2_5/natural_dataset_eval_accuracy_hist.png`

Main run result:

| epsilon | n images | k | bit acc mean | bit acc 95% CI | PSNR mean | PSNR 95% CI | SSIM mean | SSIM 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 202 | 20 | 99.53% | [99.16, 99.80] | 21.54 dB | [21.21, 21.87] | 0.707 | [0.690, 0.724] |
| 5.0 | 202 | 20 | 99.48% | [99.23, 99.70] | 16.61 dB | [16.43, 16.79] | 0.589 | [0.572, 0.606] |

Runtime:

- `930.8s` for 202 images x 2 epsilon values on CPU after the dataset was cached.
- The one-time Caltech101 download took most of the earlier smoke-test runtime.

Smoke test command:

```bash
python experiments/natural_dataset_eval.py \
  --dataset caltech101 \
  --num_images 3 \
  --resolution 128 \
  --epsilon 2.0 \
  --num_carriers 8 \
  --device cpu \
  --output_dir results/caltech101_smoke_cpu_r128 \
  --skip_lpips \
  --bootstrap 100
```

Smoke test artifacts:

- `results/caltech101_smoke_cpu_r128/natural_dataset_eval.csv`
- `results/caltech101_smoke_cpu_r128/natural_dataset_eval_summary.json`
- `results/caltech101_smoke_cpu_r128/natural_dataset_eval_summary.png`
- `results/caltech101_smoke_cpu_r128/natural_dataset_eval_accuracy_hist.png`

Smoke test result:

| epsilon | n images | k | bit acc mean | PSNR mean | SSIM mean |
|---|---:|---:|---:|---:|---:|
| 2.0 | 3 | 8 | 100.00% | 25.50 dB | 0.729 |

Additional sanity run:

```bash
python experiments/natural_dataset_eval.py \
  --dataset caltech101 \
  --num_images 80 \
  --resolution 128 \
  --epsilon 2.0 5.0 \
  --num_carriers 20 \
  --device cpu \
  --output_dir results/caltech101_eval_cpu80_r128_eps2_5 \
  --skip_lpips \
  --bootstrap 1000
```

Additional sanity artifacts:

- `results/caltech101_eval_cpu80_r128_eps2_5/natural_dataset_eval.csv`
- `results/caltech101_eval_cpu80_r128_eps2_5/natural_dataset_eval_summary.json`
- `results/caltech101_eval_cpu80_r128_eps2_5/natural_dataset_eval_summary.png`
- `results/caltech101_eval_cpu80_r128_eps2_5/natural_dataset_eval_accuracy_hist.png`

Additional sanity result:

| epsilon | n images | k | bit acc mean | bit acc 95% CI | PSNR mean | SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| 2.0 | 80 | 20 | 99.81% | [99.62, 100.00] | 21.60 dB | 0.704 |
| 5.0 | 80 | 20 | 99.25% | [98.81, 99.69] | 16.67 dB | 0.587 |

Interpretation:

- PatchSteg recovery remains very high on Caltech101 under the same VAE round-trip protocol, which is useful because these images are not CIFAR-sized 32 x 32 inputs and cover many object categories.
- Epsilon 5.0 does not materially improve recovery because accuracy is already saturated at epsilon 2.0; it mainly increases visible distortion, reflected in lower PSNR and SSIM.
- For the paper, the strongest use is a robustness table: CIFAR result plus Caltech101 native-image result, with Caltech101 framed as a CPU-feasible natural-image stress test.

## ImageNet-Family 1000-Image GPU Evaluation

Question:

- The Caltech101 paper-facing result above evaluated `202` images.
- We then tested a larger ImageNet-family subset using the Athena GPU launcher specified in `/Users/joyyang/CURSOR_REMOTE_SRUN_MANUAL_GENERIC.md`.

Dataset found on cluster:

- `/datasets/imagenet-full-fall2011/images`
- This is an ImageNet-style synset-folder tree with `21841` top-level synset directories.
- This is not the canonical ILSVRC2012 50k validation split. It is still a useful ImageNet-family natural-image stress test, and the run below samples `1000` synset folders with one random image per sampled synset.

Implementation changes needed:

- Updated `experiments/natural_dataset_eval.py` so `--image_dir` supports recursive/ImageFolder-style class subdirectories.
- Avoided exhaustive recursive scans over ImageNet-scale trees by sampling class folders first and only listing files inside selected folders.
- Made SSIM optional for this script because the remote `safesae` environment did not have `scikit-image`; the ImageNet run records bit accuracy and PSNR, with SSIM/LPIPS omitted.

Remote setup:

- Remote project directory: `~/patchsteg`
- Remote launcher: `~/remote_srun.sh`
- Slurm allocation used by the successful runs: `gpu`, `gpu:1`, `8` CPUs, `32G` memory.
- Successful jobs ran on: `c1-g4-04`

Dry run command:

```bash
~/remote_srun.sh --dry-run ~/patchsteg \
  python experiments/natural_dataset_eval.py \
    --image_dir /datasets/imagenet-full-fall2011/images \
    --num_images 1000 \
    --resolution 128 \
    --epsilon 2.0 \
    --num_carriers 20 \
    --device cuda \
    --output_dir results/imagenet_fall2011_subset1000_gpu_r128_eps2 \
    --skip_lpips \
    --bootstrap 1000
```

Main run command:

```bash
~/remote_srun.sh --github-test --git-pull --log ~/patchsteg \
  python experiments/natural_dataset_eval.py \
    --image_dir /datasets/imagenet-full-fall2011/images \
    --num_images 1000 \
    --resolution 128 \
    --epsilon 2.0 \
    --num_carriers 20 \
    --device cuda \
    --output_dir results/imagenet_fall2011_subset1000_gpu_r128_eps2 \
    --skip_lpips \
    --bootstrap 1000
```

Main run artifacts:

- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/natural_dataset_eval.csv`
- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/natural_dataset_eval_summary.json`
- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/natural_dataset_eval_summary.png`
- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/natural_dataset_eval_accuracy_hist.png`
- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/remote_run.log`

Main run result:

| source | n images | epsilon | k | bit acc mean | bit acc 95% CI | PSNR mean | PSNR 95% CI | SSIM | LPIPS | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| ImageNet full Fall 2011 synset subset | 1000 | 2.0 | 20 | 99.80% | [99.72, 99.87] | 21.56 dB | [21.40, 21.71] | skipped, dependency absent | skipped | 173.1s |

Smoke test command:

```bash
~/remote_srun.sh --github-test --git-pull --log ~/patchsteg \
  python experiments/natural_dataset_eval.py \
    --image_dir /datasets/imagenet-full-fall2011/images \
    --num_images 6 \
    --resolution 128 \
    --epsilon 2.0 \
    --num_carriers 8 \
    --device cuda \
    --output_dir results/imagenet_fall2011_smoke_gpu6_r128_eps2 \
    --skip_lpips \
    --bootstrap 100
```

Smoke test artifacts:

- `results/imagenet_fall2011_smoke_gpu6_r128_eps2/natural_dataset_eval.csv`
- `results/imagenet_fall2011_smoke_gpu6_r128_eps2/natural_dataset_eval_summary.json`
- `results/imagenet_fall2011_smoke_gpu6_r128_eps2/natural_dataset_eval_summary.png`
- `results/imagenet_fall2011_smoke_gpu6_r128_eps2/natural_dataset_eval_accuracy_hist.png`
- `results/imagenet_fall2011_smoke_gpu6_r128_eps2/remote_run.log`

Smoke test result:

| source | n images | epsilon | k | bit acc mean | PSNR mean | runtime |
|---|---:|---:|---:|---:|---:|---:|
| ImageNet full Fall 2011 synset subset | 6 | 2.0 | 8 | 100.00% | 24.58 dB | 6.2s |

Failed attempts and fixes:

- First ImageNet smoke attempt was stopped because the initial `ImageFolder` loader recursively enumerated the full ImageNet tree before sampling.
- Second smoke attempt was stopped because it still inspected all synset folders before selecting a tiny subset.
- Third smoke attempt loaded images correctly but failed because `skimage` was absent in the remote environment. The script now treats SSIM as optional for this experiment path.

Interpretation:

- This is the strongest non-CIFAR robustness result so far: `1000` ImageNet-family natural images, sampled across `1000` synset folders, with `99.80%` mean bit recovery at epsilon `2.0`.
- The result is more paper-useful than the 202-image Caltech101 run for the “larger dataset” claim, but it should be described precisely as an ImageNet full Fall 2011 synset subset unless we later obtain the canonical ILSVRC2012 validation split.
