#!/usr/bin/env python
"""
Monitor benchmark for PatchSteg.

This replaces the heavier agent-in-loop demo with a paper-ready monitor
benchmark. It generates matched clean/stego image pairs and evaluates monitors
with different capabilities:

  pixel_noref          sees only posted pixels
  vae_roundtrip        can run the VAE and inspect round-trip latent residuals
  paired_pixel_residual has access to the clean/benign round-trip reference
  combined             concatenates all feature families

The benchmark reports AUC plus low-FPR operating points and confusion matrices.
"""
import argparse
import csv
import io
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
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from core.metrics import bit_accuracy, compute_quality_metrics
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
        paths = sorted(
            p for p in Path(image_dir).expanduser().iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        )
        images = []
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


def moment_features(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    mu = x.mean()
    sig = x.std() + 1e-8
    return [
        float(mu),
        float(sig),
        float(np.median(x)),
        float(((x - mu) ** 3).mean() / sig ** 3),
        float(((x - mu) ** 4).mean() / sig ** 4),
        float(np.percentile(x, 5)),
        float(np.percentile(x, 95)),
        float(np.max(x) - np.min(x)),
    ]


def jpeg_roundtrip(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def pixel_noref_features(img):
    arr = np.asarray(img.convert("RGB")).astype(float) / 255.0
    gray = arr.mean(axis=2)
    feats = []
    for ch in range(3):
        feats.extend(moment_features(arr[:, :, ch]))

    hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=False)
    probs = hist / max(hist.sum(), 1)
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    fft = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = gray.shape
    center = fft[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    high_energy = (fft.sum() - center.sum()) / (fft.sum() + 1e-8)

    jpg = np.asarray(jpeg_roundtrip(img, 90)).astype(float) / 255.0
    recompress_residual = arr - jpg

    feats.extend([
        float(entropy),
        float(grad.mean()),
        float(grad.std()),
        float(high_energy),
    ])
    feats.extend(moment_features(recompress_residual))
    return np.array(feats)


def latent_roundtrip_features(vae, img):
    latent = vae.encode(img)
    recon = vae.decode(latent)
    latent_rt = vae.encode(recon)
    residual = (latent_rt - latent)[0].detach().cpu().numpy()
    latent_np = latent[0].detach().cpu().numpy()
    feats = []
    for ch in range(4):
        feats.extend(moment_features(latent_np[ch]))
        feats.extend(moment_features(residual[ch]))
        fft = np.abs(np.fft.fft2(residual[ch]))
        feats.extend([float(fft.mean()), float(fft.std()), float(fft.max())])
    return np.array(feats)


def paired_pixel_residual_features(reference_img, posted_img):
    ref = np.asarray(reference_img.convert("RGB")).astype(float) / 255.0
    posted = np.asarray(posted_img.convert("RGB")).astype(float) / 255.0
    residual = posted - ref
    feats = []
    for ch in range(3):
        feats.extend(moment_features(residual[:, :, ch]))
        fft = np.abs(np.fft.fft2(residual[:, :, ch]))
        feats.extend([float(fft.mean()), float(fft.std()), float(fft.max())])
    return np.array(feats)


def generate_pairs(vae, images, epsilon, k, seed, quality_device):
    rows = []
    samples = []
    steg = PatchSteg(seed=seed, epsilon=epsilon)
    for image_idx, (image_id, image) in enumerate(images):
        latent = vae.encode(image)
        clean_rt = vae.decode(latent)
        carriers, _ = steg.select_carriers_by_stability(vae, image, n_carriers=k, test_eps=epsilon)
        bit_rng = np.random.RandomState(seed + image_idx)
        bits = bit_rng.randint(0, 2, size=k).astype(int).tolist()
        latent_mod = steg.encode_message(latent, carriers, bits)
        stego = vae.decode(latent_mod)
        latent_recv = vae.encode(stego)
        recovered, _ = steg.decode_message(latent, latent_recv, carriers)
        quality = compute_quality_metrics(image, stego, device=quality_device)
        recovery = float(bit_accuracy(bits, recovered))

        samples.append({
            "image_id": image_id,
            "image_index": image_idx,
            "label": 0,
            "sample_type": "clean_roundtrip",
            "posted_image": clean_rt,
            "reference_image": clean_rt,
        })
        samples.append({
            "image_id": image_id,
            "image_index": image_idx,
            "label": 1,
            "sample_type": "patchsteg",
            "posted_image": stego,
            "reference_image": clean_rt,
        })
        rows.append({
            "image_id": image_id,
            "image_index": image_idx,
            "epsilon": epsilon,
            "k": k,
            "bit_accuracy": recovery,
            "psnr": quality["psnr"],
            "ssim": quality["ssim"],
            "lpips": quality["lpips"],
        })
    return samples, rows


def build_feature_matrices(vae, samples, monitor_names):
    features = {name: [] for name in monitor_names}
    labels = []
    metadata = []
    for sample in samples:
        posted = sample["posted_image"]
        reference = sample["reference_image"]
        pieces = {}
        if "pixel_noref" in monitor_names or "combined" in monitor_names:
            pieces["pixel_noref"] = pixel_noref_features(posted)
        if "vae_roundtrip" in monitor_names or "combined" in monitor_names:
            pieces["vae_roundtrip"] = latent_roundtrip_features(vae, posted)
        if "paired_pixel_residual" in monitor_names or "combined" in monitor_names:
            pieces["paired_pixel_residual"] = paired_pixel_residual_features(reference, posted)

        for name in monitor_names:
            if name == "combined":
                features[name].append(np.concatenate([
                    pieces["pixel_noref"],
                    pieces["vae_roundtrip"],
                    pieces["paired_pixel_residual"],
                ]))
            else:
                features[name].append(pieces[name])

        labels.append(sample["label"])
        metadata.append({
            "image_id": sample["image_id"],
            "image_index": sample["image_index"],
            "label": sample["label"],
            "sample_type": sample["sample_type"],
        })
    return {k: np.asarray(v) for k, v in features.items()}, np.asarray(labels), metadata


def positive_scores(model, x):
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def operating_point(y_true, scores, target_fpr):
    clean_scores = scores[y_true == 0]
    threshold = np.quantile(clean_scores, 1.0 - target_fpr, method="higher")
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "target_fpr": float(target_fpr),
        "threshold": float(threshold),
        "achieved_fpr": float(fp / max(fp + tn, 1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_monitor(x, y, cv_splits, seed):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=float)
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in cv.split(x, y):
        model = clone(clf)
        model.fit(x[train_idx], y[train_idx])
        scores[test_idx] = positive_scores(model, x[test_idx])
        preds[test_idx] = model.predict(x[test_idx])
    return {
        "auc": float(roc_auc_score(y, scores)),
        "accuracy": float(accuracy_score(y, preds)),
        "operating_points": {
            "0.1%": operating_point(y, scores, 0.001),
            "1%": operating_point(y, scores, 0.01),
            "5%": operating_point(y, scores, 0.05),
        },
        "scores": scores,
        "preds": preds,
    }


def save_auc_plot(results, output_dir):
    monitors = list(results.keys())
    aucs = [results[m]["auc"] for m in monitors]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(monitors, aucs, color=["#8c8c8c", "#2c7fb8", "#41ab5d", "#756bb1"][:len(monitors)])
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("PatchSteg Monitor Benchmark")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "monitor_benchmark_auc.png", dpi=160)
    plt.close(fig)


def save_recall_plot(results, output_dir, operating_label="1%"):
    monitors = list(results.keys())
    recalls = [results[m]["operating_points"][operating_label]["recall"] for m in monitors]
    precisions = [results[m]["operating_points"][operating_label]["precision"] for m in monitors]
    x = np.arange(len(monitors))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, recalls, width=0.36, label="recall", color="#3182bd")
    ax.bar(x + 0.18, precisions, width=0.36, label="precision", color="#de2d26")
    ax.set_xticks(x)
    ax.set_xticklabels(monitors, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"Operating Point Metrics at Target FPR {operating_label}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "monitor_benchmark_operating_point.png", dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark PatchSteg monitors on matched clean/stego pairs.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--num_images", type=int, default=30)
    parser.add_argument("--vae_name", default=DEFAULT_VAE)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--num_carriers", "-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output_dir", default="results/monitor_benchmark")
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--monitors", nargs="+",
                        default=["pixel_noref", "vae_roundtrip", "paired_pixel_residual", "combined"],
                        choices=["pixel_noref", "vae_roundtrip", "paired_pixel_residual", "combined"])
    parser.add_argument("--save_csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_figures", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("PatchSteg monitor benchmark")
    print(f"  Output dir: {output_dir}")
    print("  Monitor tiers: pixel-only, VAE-aware, paired-reference, combined")

    vae = load_vae(args.vae_name, args.device, args.resolution)
    images = load_images(args.dataset, args.image_dir, args.num_images, args.resolution)
    if args.cv_splits > len(images):
        args.cv_splits = len(images)
    if args.cv_splits < 2:
        raise ValueError("Need at least two images for cross-validated monitor benchmarking.")

    samples, pair_rows = generate_pairs(
        vae,
        images,
        epsilon=args.epsilon,
        k=args.num_carriers,
        seed=args.seed,
        quality_device=args.device,
    )
    feature_mats, y, metadata = build_feature_matrices(vae, samples, args.monitors)

    monitor_results = {}
    sample_score_rows = []
    operating_rows = []
    for monitor_name in args.monitors:
        print(f"  evaluating {monitor_name}", flush=True)
        result = evaluate_monitor(feature_mats[monitor_name], y, args.cv_splits, args.seed)
        monitor_results[monitor_name] = {
            "auc": result["auc"],
            "accuracy": result["accuracy"],
            "operating_points": result["operating_points"],
            "n_samples": int(len(y)),
            "n_clean": int((y == 0).sum()),
            "n_stego": int((y == 1).sum()),
        }
        for row, score, pred in zip(metadata, result["scores"], result["preds"]):
            sample_score_rows.append({
                **row,
                "monitor": monitor_name,
                "score": float(score),
                "predicted_label": int(pred),
                "epsilon": args.epsilon,
                "k": args.num_carriers,
                "seed": args.seed,
                "vae_name": args.vae_name,
                "resolution": args.resolution,
            })
        for op_name, op in result["operating_points"].items():
            operating_rows.append({
                "monitor": monitor_name,
                "epsilon": args.epsilon,
                "k": args.num_carriers,
                "auc": result["auc"],
                "accuracy": result["accuracy"],
                "operating_point": op_name,
                **op,
            })

    summary = {
        "config": vars(args),
        "monitor_results": monitor_results,
        "pair_quality": {
            "bit_accuracy_mean": float(np.mean([r["bit_accuracy"] for r in pair_rows])),
            "bit_accuracy_std": float(np.std([r["bit_accuracy"] for r in pair_rows])),
            "psnr_mean": float(np.mean([r["psnr"] for r in pair_rows])),
            "ssim_mean": float(np.mean([r["ssim"] for r in pair_rows])),
            "lpips_mean": float(np.mean([r["lpips"] for r in pair_rows])),
        },
    }

    with open(output_dir / "monitor_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_csv:
        with open(output_dir / "monitor_benchmark_operating_points.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(operating_rows[0].keys()))
            writer.writeheader()
            writer.writerows(operating_rows)
        with open(output_dir / "monitor_benchmark_sample_scores.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_score_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_score_rows)
        with open(output_dir / "monitor_benchmark_pair_quality.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)

    if args.save_figures:
        save_auc_plot(monitor_results, output_dir)
        save_recall_plot(monitor_results, output_dir, "1%")

    print(f"  wrote {output_dir / 'monitor_benchmark_summary.json'}")
    print("How to rerun: increase --num_images for paper-ready estimates; use --image_dir for custom covers.")


if __name__ == "__main__":
    main()
