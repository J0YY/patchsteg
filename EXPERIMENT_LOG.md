# PatchSteg Experiment Log

## ALL EXPERIMENTS COMPLETE

### 1. roundtrip_test.py — DONE
- Confirmed SD VAE round-trip works
- eps=5.0 / 20 carriers = 100% bit accuracy, PSNR 30.7 dB
- eps=2.0 / 20 carriers = 75% (motivates stability selection)

### 2. carrier_stability.py — DONE
- 100% of positions positive stability at eps=5.0 (averaged)
- Variance range 0.98–3.48 in stability score
- Figures: carrier_stability_avg.png, carrier_stability_heatmap.png

### 3. capacity_test.py — DONE (old version, 512x512)
- Stability-selected carriers: 100% acc at K=20, eps=2.0 on patches
- Up to 500 carriers tested, 99.4% raw acc with rep3=100%
- PSNR degrades predictably: 32 dB at K=20 → 24 dB at K=500
- Figures: capacity_curve.png, capacity_ablation.png, rate_distortion_accuracy.png
- Note: ablation version ran but stdout was lost to buffering

### 4. robustness_test.py — DONE (256x256, run_remaining.py)
- Color patches: 100% acc survives JPEG Q=50, noise sigma=0.01 at eps=5.0
- Gradient images: near-chance even without distortion (~45-55%)
- Key finding: image content strongly determines channel reliability
- Figure: robustness_bars.png

### 5. mechanistic_analysis.py — DONE (256x256, run_remaining.py)
- Channel importance: all 4 channels similar (97.5%–99.7%)
- Reconstruction error vs stability: Pearson r = 0.048 (no correlation!)
- Border vs interior: no systematic difference
- Figure: mechanistic_analysis.png

### 6. detectability_test.py — DONE (256x256, run_remaining.py)
- eps=1.0: AUC=0.44 (STEALTHY, below chance)
- eps=2.0: AUC=0.68 (MARGINAL)
- eps=5.0: AUC=0.93 (DETECTED)
- eps=10.0: AUC=0.97 (DETECTED)
- Clear stealth-capacity tradeoff
- Figure: detectability_curve.png

## Paper
- main.tex fully populated with all results, tables, figures
- PDF compiled successfully
- All 8 figures referenced

### 7. extended_experiments.py — DONE
- Natural photos (CIFAR-10, 8 images): eps=2.0 avg 98.8% acc / 38.1dB, eps=5.0 avg 98.8% acc / 29.4dB
- Longer messages: up to 152 bits ("THE QUICK BROWN FOX") at 100% acc with rep=1
- LSB baseline: PatchSteg eps=2 AUC=0.349 (stealthier than LSB AUC=0.651)
- Theoretical analysis: latent stats, perturbation/noise ratios, decoder sensitivity (~10.8 max pixel change at eps=1)
- Figures: natural_photos.png, message_length.png, lsb_comparison.png, theoretical_analysis.png

## Paper
- main.tex updated with all extended results (4 new subsections, 4 new figures, 3 new tables)
- PDF compiled successfully (12 figures, 7 tables total)

## Phase 1: CDF-PatchSteg (Distribution-Preserving)

### 8. cdf_capacity_test.py — DONE
- CDF K=5: 92.5% acc, 40.3 dB | K=10: 88.8%, 38.5 dB | K=20: 88.1%, 36.9 dB | K=50: 87.8%, 34.2 dB
- Average KS p-value: 0.179 (distribution preserved; original PatchSteg p=0.000)
- Detection AUC: 0.148 (effectively undetectable); original eps=5 AUC=0.722
- Figures: cdf_capacity_curve.png, cdf_detectability.png, cdf_distribution.png

## Phase 2: PCA-Guided Directions

### 9. pca_test.py — DONE
- Top 3 PCA components explain 96.1% of variance
- PC0 (64.6%): 100% acc, 24.6 dB PSNR, AUC=0.574 (stealthiest)
- PC1 (17.0%): 96.2% acc, 23.7 dB, AUC=0.833
- PC2 (14.5%): 98.8% acc, 28.2 dB
- Random baseline: 99.4% acc, 29.4 dB, AUC=0.722
- Figures: pca_components.png, pca_accuracy_comparison.png, pca_detectability.png

## Phase 3: Latent-Space Steganalysis Detector

### 10. latent_detector_test.py — DONE
- Within-method: PatchSteg eps=2 AUC=0.625, eps=5 AUC=1.000, CDF AUC=0.875
- Cross-method (PS eps=5 → CDF): AUC=0.688 (weak transfer)
- Top features: spectral means, residual skewness
- Figures: detector_roc_curves.png, detector_cross_method.png, detector_feature_importance.png

## V2 Experiments

### 11. v2_multimodel.py — DONE
- SD-VAE-MSE: eps=2 98.0%, eps=5 99.5%
- SD-VAE-EMA: eps=2 98.5%, eps=5 98.5%
- SDXL-VAE: eps=2 98.5%, eps=5 90.0%
- Cross-model MSE→EMA: 100%, EMA→MSE: 99%
- Figure: multimodel.png

### 12. v2_serious_dataset.py — DONE
- 200 CIFAR-10 images (20/class)
- eps=2.0: 98.2%±3.3% acc (95% CI [97.8, 98.7]), PSNR 38.6±1.5 dB
- eps=5.0: 98.5%±3.0% acc (95% CI [98.0, 98.9]), PSNR 28.9±1.6 dB
- Figure: serious_dataset.png

### 13. v2_detection_strength.py — DONE
- 7 detectors × 3 epsilon values
- Key finding: pixel-residual and spectral detectors achieve AUC=1.0 even at eps=1.0
- Only latent-statistics LR struggles at eps=1 (AUC=0.172)
- Figure: detection_strength.png

### 14. v2_content_science.py — DONE
- Entropy (r=0.555) and freq_energy (r=0.534) most correlated with capacity
- Carrier positions have higher Jacobian norms (2050 vs 1684, p=0.058)
- Figure: content_science.png, capacity_by_type.png

### 15. v2_robustness_deployment.py — DONE
- eps=5 survives: JPEG Q=10 (96.5%), resize 25% (92%), noise σ=0.10 (98%)
- VAE re-encode: 99%, screenshot sim: 98.5%
- Only center crop degrades significantly (60%: 60% acc)
- Figure: deployment_robustness.png

## Paper
- main.tex fully updated with all Phase 1-3 and V2 results
- PDF compiled successfully (7.19 MiB)
- All figures generated from real experimental data

## Remaining work
- [ ] Gradio demo (demo/app.py written but not tested end-to-end)
- [ ] Phase 4: baseline comparison (RoSteALS, TrustMark, Tree-Ring)
