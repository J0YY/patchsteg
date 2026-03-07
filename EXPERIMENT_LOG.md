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

## Remaining work
- [ ] Gradio demo (demo/app.py written but not tested end-to-end)
- [ ] Could re-run capacity ablation (random vs stability) with unbuffered output to capture numbers
