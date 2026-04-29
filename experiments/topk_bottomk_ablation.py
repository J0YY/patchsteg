#!/usr/bin/env python
"""
Top-k/random-k/bottom-k carrier ablation for PatchSteg.

This experiment tests whether the PatchSteg stability map predicts recovery:
carriers ranked high by stability should recover better than random carriers,
and carriers ranked low should recover worse under the same VAE, image set,
epsilon, payload size, and decode protocol.

Example smoke run:
  python experiments/topk_bottomk_ablation.py --dataset cifar10 --num_images 3 \
      --resolution 128 --epsilon 2.0 -k 8 --num_trials 1 \
      --output_dir results/topk_bottomk_smoke --device cpu
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms

from core.metrics import bit_accuracy, compute_quality_metrics
from core.steganography import PatchSteg
from core.vae import StegoVAE


DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"


class NamedVAE:
    """Small wrapper that keeps non-default VAE support local to this script."""

    def __init__(self, vae_name, device="cpu", image_size=256):
        self.device = device
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.model_id = vae_name
        self.vae = AutoencoderKL.from_pretrained(vae_name, low_cpu_mem_usage=False).to(device).eval()
        self.scaling_factor = self.vae.config.scaling_factor
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def encode(self, image):
        x = self.to_tensor(image).unsqueeze(0).to(self.device)
        latent = self.vae.encode(x).latent_dist.mean
        return latent * self.scaling_factor

    @torch.no_grad()
    def decode(self, latent):
        pixels = self.vae.decode(latent / self.scaling_factor).sample
        pixels = (pixels.clamp(-1, 1) + 1) / 2
        arr = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def default_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_vae(vae_name, device, resolution):
    if vae_name == DEFAULT_VAE:
        return StegoVAE(device=device, image_size=resolution)
    return NamedVAE(vae_name, device=device, image_size=resolution)


def load_images(dataset, image_dir, num_images, resolution):
    images = []
    if image_dir:
        paths = sorted(
            p for p in Path(image_dir).expanduser().iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )
        for path in paths[:num_images]:
            img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
            images.append((path.stem, img))
        return images

    if dataset != "cifar10":
        raise ValueError("Only --dataset cifar10 is implemented when --image_dir is not supplied.")

    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root="/tmp/cifar10", train=False, download=True)
    for idx, (img, label) in enumerate(ds):
        img = img.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        images.append((f"cifar10_{idx:05d}_{ds.classes[label]}", img))
        if len(images) >= num_images:
            break
    return images


def direction_seeds(base_seed, num_directions):
    return [int(base_seed) + i for i in range(num_directions)]


def compute_stability_map(vae, image, epsilon, direction_seed, num_stability_directions):
    """Average PatchSteg's existing directional stability score across seeds."""
    maps = []
    for seed in direction_seeds(direction_seed, num_stability_directions):
        steg = PatchSteg(seed=seed, epsilon=epsilon)
        smap, _ = steg.compute_stability_map(vae, image, test_eps=epsilon)
        maps.append(smap.float().cpu().numpy())
    return np.mean(np.stack(maps, axis=0), axis=0)


def select_carriers_from_map(stability_map, k, mode, rng):
    h, w = stability_map.shape
    flat = stability_map.reshape(-1)
    if k > flat.size:
        raise ValueError(f"k={k} exceeds available latent positions={flat.size}")
    if mode == "topk":
        idx = np.argsort(flat)[::-1][:k]
    elif mode == "bottomk":
        idx = np.argsort(flat)[:k]
    elif mode == "random":
        idx = rng.choice(flat.size, size=k, replace=False)
    else:
        raise ValueError(f"Unknown carrier mode: {mode}")
    return [(int(i // w), int(i % w)) for i in idx]


def evaluate_carriers(vae, image, latent_clean, carriers, bits, epsilon, direction_seed, quality_device):
    steg = PatchSteg(seed=direction_seed, epsilon=epsilon)
    latent_mod = steg.encode_message(latent_clean, carriers, bits)
    stego = vae.decode(latent_mod)
    latent_recv = vae.encode(stego)
    recovered, _ = steg.decode_message(latent_clean, latent_recv, carriers)
    quality = compute_quality_metrics(image, stego, device=quality_device)
    return {
        "bit_accuracy": float(bit_accuracy(bits, recovered)),
        "psnr": quality["psnr"],
        "ssim": quality["ssim"],
        "lpips": quality["lpips"],
    }


def bootstrap_ci(values, seed, n_boot=1000):
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return [None, None]
    rng = np.random.RandomState(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def summarize(rows, seed):
    summary = {}
    for mode in sorted({r["carrier_mode"] for r in rows}):
        subset = [r for r in rows if r["carrier_mode"] == mode]
        accs = [r["bit_accuracy"] for r in subset]
        psnrs = [r["psnr"] for r in subset]
        summary[mode] = {
            "n": len(subset),
            "bit_accuracy_mean": float(np.mean(accs)),
            "bit_accuracy_std": float(np.std(accs)),
            "bit_accuracy_ci95": bootstrap_ci(accs, seed),
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
        }
    return summary


def save_bar_plot(summary, output_dir):
    modes = [m for m in ["topk", "random", "bottomk"] if m in summary]
    means = [summary[m]["bit_accuracy_mean"] for m in modes]
    stds = [summary[m]["bit_accuracy_std"] for m in modes]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"topk": "#2c7fb8", "random": "#7f7f7f", "bottomk": "#d95f0e"}
    ax.bar(modes, means, yerr=stds, capsize=4, color=[colors[m] for m in modes])
    ax.axhline(50, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_ylabel("Bit accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Carrier Stability Ablation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "topk_bottomk_bit_accuracy.png", dpi=160)
    plt.close(fig)


def save_psnr_plot(summary, output_dir):
    modes = [m for m in ["topk", "random", "bottomk"] if m in summary]
    means = [summary[m]["psnr_mean"] for m in modes]
    stds = [summary[m]["psnr_std"] for m in modes]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(modes, means, yerr=stds, capsize=4, color="#8c96c6")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Carrier Ablation Image Quality")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "topk_bottomk_psnr.png", dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="PatchSteg top-k/random-k/bottom-k carrier ablation.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--vae_name", default=DEFAULT_VAE)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--num_carriers", "-k", type=int, default=20)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/topk_bottomk_ablation")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--carrier_modes", nargs="+", default=["topk", "random", "bottomk"],
                        choices=["topk", "random", "bottomk"])
    parser.add_argument("--save_csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stability_seed", type=int, default=None)
    parser.add_argument("--payload_seed", type=int, default=None)
    parser.add_argument("--direction_seed", type=int, default=None)
    parser.add_argument("--heldout_direction_seed", type=int, default=None)
    parser.add_argument("--num_stability_directions", type=int, default=1)
    parser.add_argument("--num_eval_directions", type=int, default=1)
    parser.add_argument("--heldout_eval", type=parse_bool, default=True)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stability_seed = args.seed if args.stability_seed is None else args.stability_seed
    payload_seed = args.seed + 10_000 if args.payload_seed is None else args.payload_seed
    direction_seed = args.seed if args.direction_seed is None else args.direction_seed
    heldout_direction_seed = args.seed + 20_000 if args.heldout_direction_seed is None else args.heldout_direction_seed
    eval_base_seed = heldout_direction_seed if args.heldout_eval else direction_seed
    eval_protocol = "heldout_direction" if args.heldout_eval else "same_direction"

    print("Top-k/random-k/bottom-k carrier ablation")
    print(f"  Run command: python experiments/topk_bottomk_ablation.py --dataset cifar10 --num_images {args.num_images} -k {args.num_carriers}")
    print(f"  Output dir: {output_dir}")
    print(f"  Eval protocol: {eval_protocol}")

    vae = load_vae(args.vae_name, args.device, args.resolution)
    images = load_images(args.dataset, args.image_dir, args.num_images, args.resolution)
    rows = []

    for image_idx, (image_id, image) in enumerate(images):
        print(f"  image {image_idx + 1}/{len(images)}: {image_id}", flush=True)
        stability_map = compute_stability_map(
            vae,
            image,
            args.epsilon,
            direction_seed=direction_seed,
            num_stability_directions=args.num_stability_directions,
        )
        latent_clean = vae.encode(image)

        for trial in range(args.num_trials):
            random_rng = np.random.RandomState(stability_seed + image_idx * 10_000 + trial)
            carriers_by_mode = {
                mode: select_carriers_from_map(stability_map, args.num_carriers, mode, random_rng)
                for mode in args.carrier_modes
            }

            for eval_idx, eval_seed in enumerate(direction_seeds(eval_base_seed + trial * 1000, args.num_eval_directions)):
                payload_rng = np.random.RandomState(payload_seed + image_idx * 100_000 + trial * 1000 + eval_idx)
                bits = payload_rng.randint(0, 2, size=args.num_carriers).astype(int).tolist()

                for mode in args.carrier_modes:
                    metrics = evaluate_carriers(
                        vae,
                        image,
                        latent_clean,
                        carriers_by_mode[mode],
                        bits,
                        args.epsilon,
                        eval_seed,
                        args.device,
                    )
                    rows.append({
                        "image_id": image_id,
                        "image_index": image_idx,
                        "trial": trial,
                        "eval_direction_index": eval_idx,
                        "carrier_mode": mode,
                        "epsilon": args.epsilon,
                        "k": args.num_carriers,
                        "seed": args.seed,
                        "stability_seed": stability_seed,
                        "payload_seed": payload_seed,
                        "direction_seed": direction_seed,
                        "eval_direction_seed": eval_seed,
                        "heldout_direction_seed": heldout_direction_seed,
                        "num_stability_directions": args.num_stability_directions,
                        "num_eval_directions": args.num_eval_directions,
                        "heldout_eval": args.heldout_eval,
                        "eval_protocol": eval_protocol,
                        "vae_name": args.vae_name,
                        "resolution": args.resolution,
                        **metrics,
                    })

    summary = {
        "config": {
            **vars(args),
            "stability_seed": stability_seed,
            "payload_seed": payload_seed,
            "direction_seed": direction_seed,
            "heldout_direction_seed": heldout_direction_seed,
            "eval_protocol": eval_protocol,
        },
        "by_mode": summarize(rows, args.seed),
    }

    if args.save_csv:
        csv_path = output_dir / "topk_bottomk_ablation.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {csv_path}")

    summary_path = output_dir / "topk_bottomk_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {summary_path}")

    if args.save_figures:
        save_bar_plot(summary["by_mode"], output_dir)
        save_psnr_plot(summary["by_mode"], output_dir)
        print(f"  wrote figures in {output_dir}")

    print("How to rerun: use the command above, or pass --image_dir for custom covers.")


if __name__ == "__main__":
    main()
