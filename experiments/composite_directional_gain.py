#!/usr/bin/env python
"""
Composite Enc∘Dec directional-gain analysis for PatchSteg.

Scientific interpretation:
For F(z) = Enc(Dec(z)), high positive gain at position (r,c) along direction d
means the VAE round-trip preserves the sign and magnitude of a local latent
perturbation along d. PatchSteg should work best at positions where this local
directional gain is high.

This differs slightly from the existing PatchSteg stability map. The existing
map perturbs all spatial positions simultaneously and compares F(z + all_eps*d)
to z. This script estimates a local one-position gain and compares it to the
existing score when requested.

Example smoke run:
  python experiments/composite_directional_gain.py --dataset cifar10 \
      --num_images 3 --resolution 128 --epsilon 2.0 --num_positions 24 \
      --num_directions 1 --k 8 --output_dir results/composite_gain_smoke \
      --device cpu
"""
import argparse
import csv
import json
import os
import sys
import warnings
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
from scipy.stats import pearsonr, spearmanr
from torchvision import transforms

from core.metrics import bit_accuracy, compute_psnr
from core.steganography import PatchSteg
from core.vae import StegoVAE


DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"


class NamedVAE:
    def __init__(self, vae_name, device="cpu", image_size=256):
        self.device = device
        self.image_size = image_size
        self.latent_size = image_size // 8
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
        return self.vae.encode(x).latent_dist.mean * self.scaling_factor

    @torch.no_grad()
    def decode(self, latent):
        pixels = self.vae.decode(latent / self.scaling_factor).sample
        pixels = (pixels.clamp(-1, 1) + 1) / 2
        arr = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)


def default_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_vae(vae_name, device, resolution):
    if vae_name == DEFAULT_VAE:
        return StegoVAE(device=device, image_size=resolution)
    return NamedVAE(vae_name, device=device, image_size=resolution)


def load_images(dataset, image_dir, num_images, resolution):
    if image_dir:
        images = []
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
    images = []
    for idx, (img, label) in enumerate(ds):
        img = img.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        images.append((f"cifar10_{idx:05d}_{ds.classes[label]}", img))
        if len(images) >= num_images:
            break
    return images


def choose_positions(latent_h, latent_w, num_positions, rng):
    all_positions = [(r, c) for r in range(latent_h) for c in range(latent_w)]
    if num_positions == "all":
        return all_positions
    n = int(num_positions)
    if n > len(all_positions):
        raise ValueError(f"--num_positions {n} exceeds latent positions {len(all_positions)}")
    idx = rng.choice(len(all_positions), size=n, replace=False)
    return [all_positions[int(i)] for i in idx]


def compute_directional_gain_for_image(vae, image, image_id, args, rng):
    latent = vae.encode(image)
    z_base = vae.encode(vae.decode(latent))
    _, _, latent_h, latent_w = latent.shape
    positions = choose_positions(latent_h, latent_w, args.num_positions, rng)
    rows = []
    gain_map_sum = np.zeros((latent_h, latent_w), dtype=float)
    gain_map_count = np.zeros((latent_h, latent_w), dtype=float)

    for direction_id in range(args.num_directions):
        direction_seed = args.seed + direction_id
        steg = PatchSteg(seed=direction_seed, epsilon=args.epsilon)
        direction = steg.direction.to(latent.device)
        # Existing score perturbs all positions simultaneously and subtracts z,
        # so it is a related global directional-persistence proxy, not identical
        # to the local gain below.
        stability_map, _ = steg.compute_stability_map(vae, image, test_eps=args.epsilon)
        stability_np = stability_map.float().cpu().numpy()

        for r, c in positions:
            z_pert = latent.clone()
            z_pert[0, :, r, c] += args.epsilon * direction
            z_pert_rt = vae.encode(vae.decode(z_pert))
            local_delta = z_pert_rt[0, :, r, c] - z_base[0, :, r, c]
            gain = float(torch.dot(local_delta, direction).item() / args.epsilon)
            gain_map_sum[r, c] += gain
            gain_map_count[r, c] += 1.0
            rows.append({
                "image_id": image_id,
                "r": int(r),
                "c": int(c),
                "direction_id": direction_id,
                "direction_seed": direction_seed,
                "gain": gain,
                "stability_score_if_available": float(stability_np[r, c]),
            })

    gain_map = np.full((latent_h, latent_w), np.nan)
    mask = gain_map_count > 0
    gain_map[mask] = gain_map_sum[mask] / gain_map_count[mask]
    return rows, gain_map, latent


def select_from_gain_map(gain_map, k, mode, rng):
    valid = np.flatnonzero(np.isfinite(gain_map.reshape(-1)))
    if k > len(valid):
        raise ValueError(f"k={k} exceeds sampled finite gain positions={len(valid)}")
    flat = gain_map.reshape(-1)
    if mode == "top_gain":
        idx = valid[np.argsort(flat[valid])[::-1][:k]]
    elif mode == "bottom_gain":
        idx = valid[np.argsort(flat[valid])[:k]]
    elif mode == "random":
        idx = rng.choice(valid, size=k, replace=False)
    else:
        raise ValueError(mode)
    width = gain_map.shape[1]
    return [(int(i // width), int(i % width)) for i in idx]


def evaluate_gain_carriers(vae, image, latent_clean, gain_map, args, image_idx):
    rows = []
    rng = np.random.RandomState(args.seed + 30_000 + image_idx)
    bit_rng = np.random.RandomState(args.seed + 40_000 + image_idx)
    bits = bit_rng.randint(0, 2, size=args.k).astype(int).tolist()
    eval_seed = args.seed + 50_000
    steg = PatchSteg(seed=eval_seed, epsilon=args.epsilon)
    for mode in ["top_gain", "random", "bottom_gain"]:
        carriers = select_from_gain_map(gain_map, args.k, mode, rng)
        z_mod = steg.encode_message(latent_clean, carriers, bits)
        stego = vae.decode(z_mod)
        z_recv = vae.encode(stego)
        recovered, _ = steg.decode_message(latent_clean, z_recv, carriers)
        rows.append({
            "mode": mode,
            "bit_accuracy": float(bit_accuracy(bits, recovered)),
            "psnr": float(compute_psnr(image, stego)),
            "eval_direction_seed": eval_seed,
        })
    return rows


def safe_corr(x, y, fn):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return {"r": None, "p": None, "n": int(mask.sum())}
    r, p = fn(x[mask], y[mask])
    return {"r": float(r), "p": float(p), "n": int(mask.sum())}


def save_heatmap(gain_maps, output_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_gain = np.nanmean(np.stack(gain_maps, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(avg_gain, cmap=cmap)
    ax.set_title("Average Local Directional Gain")
    ax.set_xlabel("latent c")
    ax.set_ylabel("latent r")
    fig.colorbar(im, ax=ax, label="gain")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "directional_gain_heatmap.png", dpi=160)
    plt.close(fig)


def save_scatter(rows, output_dir):
    gains = [r["gain"] for r in rows]
    stability = [r["stability_score_if_available"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gains, stability, s=12, alpha=0.55)
    ax.set_xlabel("local directional gain")
    ax.set_ylabel("existing stability score")
    ax.set_title("Gain vs Existing Stability Proxy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "gain_vs_stability_scatter.png", dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Composite Enc(Dec(z)) directional-gain analysis.")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--vae_name", default=DEFAULT_VAE)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--num_positions", default="64", help="'all' or integer sample count")
    parser.add_argument("--num_directions", type=int, default=1)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output_dir", default="results/composite_directional_gain")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Composite Enc(Dec(z)) directional-gain analysis")
    print(f"  Output dir: {output_dir}")
    print("  High positive gain means the VAE round trip preserves local latent perturbations along d.")

    vae = load_vae(args.vae_name, args.device, args.resolution)
    images = load_images(args.dataset, args.image_dir, args.num_images, args.resolution)
    rng = np.random.RandomState(args.seed)

    all_rows = []
    all_gain_maps = []
    recovery_rows = []
    for image_idx, (image_id, image) in enumerate(images):
        print(f"  image {image_idx + 1}/{len(images)}: {image_id}", flush=True)
        rows, gain_map, latent = compute_directional_gain_for_image(vae, image, image_id, args, rng)
        all_rows.extend(rows)
        all_gain_maps.append(gain_map)
        for rec in evaluate_gain_carriers(vae, image, latent, gain_map, args, image_idx):
            rec.update({
                "image_id": image_id,
                "image_index": image_idx,
                "epsilon": args.epsilon,
                "k": args.k,
                "seed": args.seed,
                "vae_name": args.vae_name,
                "resolution": args.resolution,
            })
            recovery_rows.append(rec)

    gain_csv = output_dir / "composite_directional_gain.csv"
    with open(gain_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    recovery_csv = output_dir / "composite_directional_gain_recovery.csv"
    with open(recovery_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(recovery_rows[0].keys()))
        writer.writeheader()
        writer.writerows(recovery_rows)

    summary = {
        "config": vars(args),
        "gain_vs_existing_stability": {
            "pearson": safe_corr(
                [r["gain"] for r in all_rows],
                [r["stability_score_if_available"] for r in all_rows],
                pearsonr,
            ),
            "spearman": safe_corr(
                [r["gain"] for r in all_rows],
                [r["stability_score_if_available"] for r in all_rows],
                spearmanr,
            ),
        },
        "recovery_by_mode": {},
    }
    for mode in sorted({r["mode"] for r in recovery_rows}):
        vals = [r["bit_accuracy"] for r in recovery_rows if r["mode"] == mode]
        summary["recovery_by_mode"][mode] = {
            "n": len(vals),
            "bit_accuracy_mean": float(np.mean(vals)),
            "bit_accuracy_std": float(np.std(vals)),
        }

    summary_json = output_dir / "composite_directional_gain_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    save_heatmap(all_gain_maps, output_dir)
    save_scatter(all_rows, output_dir)
    print(f"  wrote {gain_csv}")
    print(f"  wrote {recovery_csv}")
    print(f"  wrote {summary_json}")
    print("How to rerun: use --num_positions all for exhaustive maps, or an integer for a faster sample.")


if __name__ == "__main__":
    main()
