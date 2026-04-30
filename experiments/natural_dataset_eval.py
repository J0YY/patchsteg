#!/usr/bin/env python
"""
PatchSteg evaluation on non-CIFAR natural-image datasets.

This is meant to add a CPU-feasible larger/native-image result alongside the
existing CIFAR-10 evaluation. Caltech101 is the default because it has thousands
of native object images across many categories without the size/cost of COCO or
LAION-scale downloads.

Examples:
  python experiments/natural_dataset_eval.py --dataset caltech101 \
      --num_images 80 --resolution 128 --epsilon 2.0 5.0 \
      --num_carriers 20 --device cpu \
      --output_dir results/caltech101_eval_cpu80_r128

  python experiments/natural_dataset_eval.py --image_dir /path/to/imagenet/val \
      --num_images 1000 --resolution 128 --epsilon 2.0 \
      --num_carriers 20 --device cuda \
      --output_dir results/imagenet1k_val_subset_r128_eps2
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from core.metrics import bit_accuracy, compute_lpips_pil, compute_psnr, compute_ssim_pil
from core.steganography import PatchSteg
from core.vae import StegoVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PatchSteg on non-CIFAR natural-image datasets.")
    parser.add_argument("--dataset", default="caltech101", choices=["caltech101", "stl10"])
    parser.add_argument(
        "--image_dir",
        default=None,
        help=(
            "Optional directory of images; overrides --dataset. Supports recursive "
            "ImageFolder-style class subdirectories, e.g. ImageNet val."
        ),
    )
    parser.add_argument("--num_images", type=int, default=80)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--epsilon", nargs="+", type=float, default=[2.0, 5.0])
    parser.add_argument("--num_carriers", "-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="results/natural_dataset_eval")
    parser.add_argument("--dataset_root", default="/tmp/patchsteg_datasets")
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS if a very fast CPU run is needed.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def load_image_dir(image_dir, num_images, resolution, seed):
    root = Path(image_dir).expanduser()
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if not paths:
        raise ValueError(f"No supported image files found under {root}")

    class_names = sorted({p.parent.name for p in paths})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    labels = [class_to_idx[p.parent.name] for p in paths]
    if len(class_names) > 1:
        indices = stratified_indices(labels, num_images, seed)
    else:
        indices = list(range(min(num_images, len(paths))))

    rows = []
    for image_index, idx in enumerate(indices):
        path = paths[idx]
        class_name = path.parent.name if len(class_names) > 1 else "image_dir"
        img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        try:
            rel = path.relative_to(root).with_suffix("")
            image_id = str(rel).replace(os.sep, "__")
        except ValueError:
            image_id = path.stem
        rows.append({
            "image_id": image_id,
            "class_index": class_to_idx.get(class_name, -1),
            "class_name": class_name,
            "image_index": image_index,
            "image": img,
        })
    return rows


def stratified_indices(labels, num_images, seed):
    rng = np.random.RandomState(seed)
    by_class = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)
    for idxs in by_class.values():
        rng.shuffle(idxs)

    selected = []
    class_ids = sorted(by_class)
    cursor = {c: 0 for c in class_ids}
    while len(selected) < num_images:
        made_progress = False
        for c in class_ids:
            if len(selected) >= num_images:
                break
            if cursor[c] < len(by_class[c]):
                selected.append(by_class[c][cursor[c]])
                cursor[c] += 1
                made_progress = True
        if not made_progress:
            break
    return selected


def load_dataset(args):
    if args.image_dir:
        return load_image_dir(args.image_dir, args.num_images, args.resolution, args.seed)

    if args.dataset == "caltech101":
        from torchvision.datasets import Caltech101
        ds = Caltech101(root=args.dataset_root, target_type="category", download=True)
        categories = list(ds.categories)
        labels = [int(ds[i][1]) for i in range(len(ds))]
        indices = stratified_indices(labels, args.num_images, args.seed)
        rows = []
        for idx in indices:
            img, label = ds[idx]
            img = img.convert("RGB").resize((args.resolution, args.resolution), Image.BILINEAR)
            rows.append({
                "image_id": f"caltech101_{idx:05d}_{categories[int(label)]}",
                "class_index": int(label),
                "class_name": categories[int(label)],
                "image": img,
            })
        return rows

    if args.dataset == "stl10":
        from torchvision.datasets import STL10
        ds = STL10(root=args.dataset_root, split="test", download=True)
        labels = [int(ds[i][1]) for i in range(len(ds))]
        indices = stratified_indices(labels, args.num_images, args.seed)
        rows = []
        for idx in indices:
            img, label = ds[idx]
            img = img.convert("RGB").resize((args.resolution, args.resolution), Image.BILINEAR)
            rows.append({
                "image_id": f"stl10_{idx:05d}_{ds.classes[int(label)]}",
                "class_index": int(label),
                "class_name": ds.classes[int(label)],
                "image": img,
            })
        return rows

    raise ValueError(args.dataset)


def quality_metrics(clean, stego, device, skip_lpips):
    metrics = {
        "psnr": float(compute_psnr(clean, stego)),
        "ssim": float(compute_ssim_pil(clean, stego)),
    }
    metrics["lpips"] = None if skip_lpips else float(compute_lpips_pil(clean, stego, device=device))
    return metrics


def bootstrap_ci(values, seed, n_boot):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [None, None]
    rng = np.random.RandomState(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def summarize(rows, epsilons, seed, n_boot):
    summary = {}
    for eps in epsilons:
        subset = [r for r in rows if float(r["epsilon"]) == float(eps)]
        summary[str(eps)] = {"n_images": len(subset)}
        for metric in ["bit_accuracy", "psnr", "ssim", "lpips"]:
            vals = [r[metric] for r in subset if r[metric] is not None]
            if not vals:
                continue
            summary[str(eps)][f"{metric}_mean"] = float(np.mean(vals))
            summary[str(eps)][f"{metric}_std"] = float(np.std(vals))
            summary[str(eps)][f"{metric}_ci95"] = bootstrap_ci(vals, seed + int(eps * 100), n_boot)
    return summary


def save_figures(rows, summary, output_dir, dataset_name):
    epsilons = sorted(summary.keys(), key=float)
    acc_means = [summary[e]["bit_accuracy_mean"] for e in epsilons]
    acc_err = [
        [summary[e]["bit_accuracy_mean"] - summary[e]["bit_accuracy_ci95"][0] for e in epsilons],
        [summary[e]["bit_accuracy_ci95"][1] - summary[e]["bit_accuracy_mean"] for e in epsilons],
    ]
    psnr_means = [summary[e]["psnr_mean"] for e in epsilons]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar([f"eps={e}" for e in epsilons], acc_means, yerr=acc_err, capsize=4, color="#3182bd")
    axes[0].axhline(50, color="black", linestyle=":", linewidth=1)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Bit accuracy (%)")
    axes[0].set_title("Recovery")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([f"eps={e}" for e in epsilons], psnr_means, color="#756bb1")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Distortion")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(f"PatchSteg on {dataset_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "natural_dataset_eval_summary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for eps in epsilons:
        vals = [r["bit_accuracy"] for r in rows if str(r["epsilon"]) == eps]
        ax.hist(vals, bins=np.linspace(0, 100, 21), alpha=0.45, label=f"eps={eps}")
    ax.axvline(50, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Per-image bit accuracy (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-image recovery distribution: {dataset_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "natural_dataset_eval_accuracy_hist.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Natural-image dataset evaluation", flush=True)
    print(f"  dataset={args.dataset} image_dir={args.image_dir}", flush=True)
    print(f"  device={args.device} resolution={args.resolution} n={args.num_images}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    images = load_dataset(args)
    print(f"  loaded {len(images)} images", flush=True)

    vae = StegoVAE(device=args.device, image_size=args.resolution)
    result_rows = []
    for eps in args.epsilon:
        print(f"\n--- epsilon={eps} ---", flush=True)
        steg = PatchSteg(seed=args.seed, epsilon=eps)
        for idx, row in enumerate(images):
            if idx % 10 == 0:
                print(f"  image {idx}/{len(images)}", flush=True)
            img = row["image"]
            latent = vae.encode(img)
            carriers, _ = steg.select_carriers_by_stability(
                vae, img, n_carriers=args.num_carriers, test_eps=eps
            )
            bit_rng = np.random.RandomState(args.seed + idx + int(eps * 1000))
            bits = bit_rng.randint(0, 2, size=args.num_carriers).astype(int).tolist()
            latent_mod = steg.encode_message(latent, carriers, bits)
            stego = vae.decode(latent_mod)
            latent_recv = vae.encode(stego)
            recovered, _ = steg.decode_message(latent, latent_recv, carriers)
            metrics = quality_metrics(img, stego, args.device, args.skip_lpips)
            result_rows.append({
                "dataset": args.dataset if not args.image_dir else "image_dir",
                "image_id": row["image_id"],
                "image_index": idx,
                "class_index": row["class_index"],
                "class_name": row["class_name"],
                "epsilon": eps,
                "k": args.num_carriers,
                "seed": args.seed,
                "resolution": args.resolution,
                "vae_name": "stabilityai/sd-vae-ft-mse",
                "device": args.device,
                "bit_accuracy": float(bit_accuracy(bits, recovered)),
                **metrics,
            })

    fieldnames = [
        "dataset", "image_id", "image_index", "class_index", "class_name",
        "epsilon", "k", "seed", "resolution", "vae_name", "device",
        "bit_accuracy", "psnr", "ssim", "lpips",
    ]
    with open(output_dir / "natural_dataset_eval.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    summary = {
        "config": vars(args),
        "dataset_note": (
            "Caltech101 was chosen as a CPU-feasible native-image dataset with "
            "many more categories and higher native resolution than CIFAR-10."
        ),
        "results": summarize(result_rows, args.epsilon, args.seed, args.bootstrap),
        "runtime_seconds": time.time() - t0,
    }
    with open(output_dir / "natural_dataset_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_figures(result_rows, summary["results"], output_dir, args.dataset if not args.image_dir else "image_dir")
    print(f"\nWrote {output_dir / 'natural_dataset_eval.csv'}", flush=True)
    print(f"Wrote {output_dir / 'natural_dataset_eval_summary.json'}", flush=True)
    print(f"Wrote figures in {output_dir}", flush=True)
    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
