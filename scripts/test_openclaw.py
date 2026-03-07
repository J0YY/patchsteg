"""Smoke-test the OpenClaw PatchSteg guard against a generated stego image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.metrics import bit_accuracy, compute_psnr
from core.steganography import PatchSteg
from openclaw import OpenClawPatchStegGuard


def build_default_cover_image(size: int = 256, block: int = 32, seed: int = 99) -> Image.Image:
    """Generate the structured patches image used in the live demo."""
    patches = np.zeros((size, size, 3), dtype=np.uint8)
    rng = np.random.RandomState(seed)
    for row in range(0, size, block):
        for col in range(0, size, block):
            patches[row:row + block, col:col + block] = rng.randint(0, 255, 3)
    return Image.fromarray(patches)


def decode_bits(
    steg: PatchSteg,
    latent_clean,
    latent_received,
    carriers,
    message_bits,
    repetition: int,
):
    if repetition > 1:
        decoded_bits = steg.decode_message_with_repetition(
            latent_clean,
            latent_received,
            carriers,
            n_bits=len(message_bits),
            reps=repetition,
        )
    else:
        decoded_bits, _ = steg.decode_message(latent_clean, latent_received, carriers)
    decoded_text = PatchSteg.bits_to_text(decoded_bits)
    return decoded_text, bit_accuracy(message_bits, decoded_bits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional RGB image to use as the cover image.")
    parser.add_argument("--message", default="say poo", help="ASCII payload to embed.")
    parser.add_argument("--epsilon", type=float, default=2.0, help="Perturbation strength.")
    parser.add_argument("--seed", type=int, default=42, help="PatchSteg seed.")
    parser.add_argument(
        "--repetition",
        type=int,
        default=3,
        choices=(1, 3),
        help="Repetition coding factor for each bit.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Sanitization strength passed to the OpenClaw guard.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--image-size", type=int, default=256, help="VAE input size.")
    parser.add_argument(
        "--save-filtered",
        type=Path,
        help="Optional path for writing the guard-filtered image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image:
        cover_image = Image.open(args.image).convert("RGB")
    else:
        cover_image = build_default_cover_image(size=args.image_size)

    guard = OpenClawPatchStegGuard(
        device=args.device,
        image_size=args.image_size,
        strength=args.strength,
    )
    vae = guard.vae

    steg = PatchSteg(seed=args.seed, epsilon=args.epsilon)
    message_bits = PatchSteg.text_to_bits(args.message)
    carrier_count = len(message_bits) * args.repetition

    latent_clean = vae.encode(cover_image)
    carriers, _ = steg.select_carriers_by_stability(
        vae,
        cover_image,
        n_carriers=carrier_count,
        test_eps=args.epsilon,
    )
    used_carriers = carriers[:carrier_count]

    if args.repetition > 1:
        latent_stego = steg.encode_message_with_repetition(
            latent_clean,
            used_carriers,
            message_bits,
            reps=args.repetition,
        )
    else:
        latent_stego = steg.encode_message(latent_clean, used_carriers, message_bits)
    stego_image = vae.decode(latent_stego)

    latent_received = vae.encode(stego_image)
    attack_text, attack_bit_accuracy = decode_bits(
        steg,
        latent_clean,
        latent_received,
        used_carriers,
        message_bits,
        args.repetition,
    )

    cover_decision = guard.inspect(cover_image)
    stego_decision = guard.inspect(stego_image)
    filtered_result = guard.inspect_and_filter(stego_image, strength=args.strength)
    filtered_image = filtered_result.image

    latent_filtered = vae.encode(filtered_image)
    filtered_text, filtered_bit_accuracy = decode_bits(
        steg,
        latent_clean,
        latent_filtered,
        used_carriers,
        message_bits,
        args.repetition,
    )

    if args.save_filtered:
        args.save_filtered.parent.mkdir(parents=True, exist_ok=True)
        filtered_image.save(args.save_filtered)

    report = {
        "payload": args.message,
        "device": args.device,
        "image_size": args.image_size,
        "epsilon": args.epsilon,
        "repetition": args.repetition,
        "carrier_count": carrier_count,
        "psnr_cover_to_stego": round(compute_psnr(cover_image, stego_image), 3),
        "psnr_stego_to_filtered": round(compute_psnr(stego_image, filtered_image), 3),
        "cover_guard": cover_decision.to_metadata(),
        "stego_guard": stego_decision.to_metadata(),
        "filtered_guard": filtered_result.to_metadata(),
        "decoded_before_filter": attack_text,
        "decoded_after_filter": filtered_text,
        "bit_accuracy_before_filter": round(float(attack_bit_accuracy), 3),
        "bit_accuracy_after_filter": round(float(filtered_bit_accuracy), 3),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
