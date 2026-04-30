# Natural Dataset Runs Report

Date: 2026-04-30

This report summarizes the new non-CIFAR dataset experiments for PatchSteg. It is intentionally separate from the LaTeX paper sources.

## Summary

| dataset | setting | images | epsilon | k | device | bit accuracy | PSNR | runtime |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Caltech101 | CPU natural-image run | 202 | 2.0 | 20 | CPU | 99.53% | 21.54 dB | 930.5s |
| Caltech101 | CPU natural-image run | 202 | 5.0 | 20 | CPU | 99.48% | 16.61 dB | 930.5s total |
| ImageNet full Fall 2011 synset subset | GPU large natural-image run | 1000 | 2.0 | 20 | CUDA | 99.80% | 21.56 dB | 172.7s |

Main takeaway: PatchSteg recovery remains near-saturated on larger natural-image datasets, not just CIFAR-style inputs. The ImageNet-family run is the strongest robustness result so far because it covers 1000 sampled synset folders.

## Experiment 1: Caltech101 Natural-Image Run

Purpose:

- Test PatchSteg on a CPU-feasible non-CIFAR natural-image dataset.
- Add a medium-scale robustness result with native object images and many categories.

Command:

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

Artifacts:

- `results/caltech101_eval_cpu202_r128_eps2_5/natural_dataset_eval.csv`
- `results/caltech101_eval_cpu202_r128_eps2_5/natural_dataset_eval_summary.json`
- `results/caltech101_eval_cpu202_r128_eps2_5/natural_dataset_eval_summary.png`
- `results/caltech101_eval_cpu202_r128_eps2_5/natural_dataset_eval_accuracy_hist.png`

Results:

| epsilon | images | k | bit accuracy mean | bit accuracy 95% CI | PSNR mean | PSNR 95% CI | SSIM mean | SSIM 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 202 | 20 | 99.53% | [99.16, 99.80] | 21.54 dB | [21.21, 21.87] | 0.707 | [0.690, 0.724] |
| 5.0 | 202 | 20 | 99.48% | [99.23, 99.70] | 16.61 dB | [16.43, 16.79] | 0.589 | [0.572, 0.606] |

Interpretation:

- The channel remains highly recoverable on Caltech101, showing the CIFAR result is not an artifact of tiny 32x32 images.
- Epsilon 5.0 did not materially improve bit accuracy because epsilon 2.0 was already near saturation.
- Epsilon 5.0 mainly increases distortion, visible in the lower PSNR and SSIM.
- For the paper, the cleanest use is a robustness table entry: Caltech101, 202 images, epsilon 2.0, k=20, 99.53% recovery.

## Experiment 2: ImageNet-Family GPU Run

Purpose:

- Test PatchSteg on a substantially larger ImageNet-style natural-image dataset.
- Use the cluster GPU path to get a stronger dataset-scale result within a short runtime.

Dataset:

- Cluster path: `/datasets/imagenet-full-fall2011/images`
- Structure: ImageNet-style synset folders.
- Count observed on cluster: `21841` top-level synset directories.
- Important caveat: this is not the canonical ILSVRC2012 50k validation split. It should be described as an ImageNet full Fall 2011 synset-folder subset.
- Sampling: `1000` images, one random image from each sampled synset folder.

Remote execution:

- Launcher: `~/remote_srun.sh`
- Remote project: `~/patchsteg`
- Slurm setup: `gpu`, `gpu:1`, `8` CPUs, `32G` memory
- Successful node: `c1-g4-04`

Command:

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

Artifacts:

- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/natural_dataset_eval.csv`
- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/natural_dataset_eval_summary.json`
- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/natural_dataset_eval_summary.png`
- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/natural_dataset_eval_accuracy_hist.png`
- `results/imagenet_fall2011_subset1000_gpu_r128_eps2/remote_run.log`

Results:

| epsilon | images | k | bit accuracy mean | bit accuracy 95% CI | PSNR mean | PSNR 95% CI | SSIM | LPIPS | runtime |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| 2.0 | 1000 | 20 | 99.80% | [99.72, 99.87] | 21.56 dB | [21.40, 21.71] | skipped | skipped | 172.7s |

SSIM was skipped because the remote `safesae` environment did not have `scikit-image`. LPIPS was intentionally skipped for the bounded GPU run.

Interpretation:

- This is currently the strongest dataset-scale robustness result: 1000 ImageNet-family images across 1000 sampled synset folders.
- The high recovery rate supports the claim that PatchSteg is not narrowly tuned to CIFAR or Caltech101.
- The PSNR is essentially aligned with Caltech101 at epsilon 2.0, which suggests the same perturbation scale behaves similarly across more diverse natural images.
- The result should be phrased carefully: "ImageNet full Fall 2011 synset subset" rather than "ImageNet-1k validation" unless we later run the canonical ILSVRC2012 validation split.

## Practical Paper Use

Recommended table rows:

| dataset | images | epsilon | k | bit accuracy | PSNR |
|---|---:|---:|---:|---:|---:|
| Caltech101 | 202 | 2.0 | 20 | 99.53% | 21.54 dB |
| ImageNet full Fall 2011 synset subset | 1000 | 2.0 | 20 | 99.80% | 21.56 dB |

Recommended wording:

- "We further evaluate PatchSteg on non-CIFAR natural images. On 202 Caltech101 images, PatchSteg recovers 99.53% of bits at epsilon 2.0. On a 1000-image ImageNet full Fall 2011 synset subset sampled across 1000 synset folders, recovery is 99.80%, indicating that the covert channel persists across substantially more diverse natural-image inputs."

Limitations to state:

- The ImageNet-family experiment is not the canonical ILSVRC2012 validation split.
- SSIM/LPIPS were not computed for the GPU ImageNet-family run due to the remote dependency/runtime setup.
- Both new dataset runs use 128px resized images, so a full-resolution study remains a future robustness check.
