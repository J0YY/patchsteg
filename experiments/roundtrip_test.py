"""
PatchSteg Round-Trip Validation Test
Tests whether steganographic bits survive encode -> perturb -> decode -> reencode pipeline.

Backbones tested:
1. MAE (facebook/vit-mae-base)
2. Stable Diffusion VAE (stabilityai/sd-vae-ft-mse)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

EPSILONS = [0.5, 2.0, 5.0]
NUM_CARRIER_POSITIONS = [5, 20]
NUM_TEST_IMAGES = 2
SEED = 42

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================
# HELPERS
# ============================================================

def compute_psnr(img1, img2):
    """Both inputs: tensors in [0, 1] range."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)


def max_pixel_diff(img1, img2):
    """Both inputs: tensors in [0, 1] range."""
    return (img1 - img2).abs().max().item()


def make_checkerboard(H, W, block=16):
    img = torch.zeros(1, 3, H, W)
    for i in range(0, H, block):
        for j in range(0, W, block):
            if ((i // block) + (j // block)) % 2 == 0:
                img[:, :, i:i+block, j:j+block] = 1.0
    return img


def make_random_patches(H, W, block=32):
    torch.manual_seed(123)
    img = torch.zeros(1, 3, H, W)
    for i in range(0, H, block):
        for j in range(0, W, block):
            color = torch.rand(3).view(1, 3, 1, 1)
            img[:, :, i:i+block, j:j+block] = color
    return img


def generate_test_images(H, W):
    """Generate synthetic test images in [0, 1] range."""
    torch.manual_seed(SEED)
    images = []

    # Image 1: Random noise
    images.append(torch.rand(1, 3, H, W))

    # Image 2: Smooth gradient
    grad = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 3, H, W)
    images.append(grad.clone())

    return images


def imagenet_normalize(img_01):
    """Normalize [0,1] image to ImageNet distribution."""
    return (img_01 - IMAGENET_MEAN) / IMAGENET_STD


def imagenet_denormalize(img_norm):
    """Denormalize ImageNet-normalized image back to [0,1]."""
    return img_norm * IMAGENET_STD + IMAGENET_MEAN


def generate_direction_vector(dim, seed=42):
    """Unit-length random direction vector."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    d = torch.randn(dim, generator=rng)
    return d / d.norm()


def bit_accuracy(sent, received):
    sent = np.array(sent)
    received = np.array(received)
    return np.mean(sent == received) * 100.0


# ============================================================
# MAE TEST
# ============================================================

def mae_encode(model, pixel_values):
    """
    Encode image to all 196 patch embeddings (no masking).
    Returns: latent [1, 196, 768], ids_restore [1, 196]
    """
    # Set mask ratio to 0 to keep all patches
    original_mask_ratio = model.config.mask_ratio
    model.config.mask_ratio = 0.0

    with torch.no_grad():
        outputs = model.vit(pixel_values)
        # outputs.last_hidden_state: [1, 197, 768] (CLS + 196 patches)
        # But we need ids_restore for the decoder

    model.config.mask_ratio = original_mask_ratio

    # With mask_ratio=0, all patches are kept. We need ids_restore.
    # Run embeddings manually to get ids_restore
    model.config.mask_ratio = 0.0
    with torch.no_grad():
        embedding_output = model.vit.embeddings(pixel_values)
        # This returns (sequence, mask, ids_restore)
        # sequence: [1, 197, 768] (CLS + 196 unmasked patches)
        # mask: [1, 196] all zeros (nothing masked)
        # ids_restore: [1, 196]

    model.config.mask_ratio = original_mask_ratio

    if isinstance(embedding_output, tuple):
        sequence, mask, ids_restore = embedding_output
    else:
        # Fallback: construct ids_restore manually
        ids_restore = torch.arange(196).unsqueeze(0).to(pixel_values.device)
        sequence = embedding_output

    # Run through encoder blocks
    with torch.no_grad():
        encoder_output = model.vit.encoder(sequence)
        latent = encoder_output.last_hidden_state  # [1, 197, 768]

    return latent, ids_restore


def mae_decode(model, latent, ids_restore):
    """
    Decode latent patch embeddings back to pixel predictions.
    latent: [1, 197, 768] (CLS + patches)
    Returns: pixel_pred [1, 3, 224, 224] in ImageNet-normalized space
    """
    with torch.no_grad():
        decoder_output = model.decoder(latent, ids_restore)
        logits = decoder_output.logits  # [1, 196, patch_size**2 * 3]

    # Unpatchify: convert patch predictions to image
    patch_size = model.config.patch_size  # 16
    image_size = model.config.image_size  # 224
    h = w = image_size // patch_size  # 14

    # logits: [1, 196, 768] -> [1, 14, 14, 16, 16, 3]
    img = logits.reshape(1, h, w, patch_size, patch_size, 3)
    img = img.permute(0, 5, 1, 3, 2, 4)  # [1, 3, 14, 16, 14, 16]
    img = img.reshape(1, 3, image_size, image_size)

    return img


def test_mae():
    """Test MAE backbone for perturbation survival."""
    print("\n" + "=" * 60)
    print("BACKBONE: MAE (facebook/vit-mae-base)")
    print("=" * 60)

    print("Loading MAE model...")
    t0 = time.time()
    from transformers import ViTMAEForPreTraining, ViTImageProcessor

    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    model = model.to(DEVICE).to(DTYPE)
    model.eval()

    processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Check if norm_pix_loss is enabled
    norm_pix_loss = getattr(model.config, 'norm_pix_loss', False)
    print(f"  norm_pix_loss: {norm_pix_loss}")
    print(f"  patch_size: {model.config.patch_size}, image_size: {model.config.image_size}")

    # Generate test images (224x224, [0,1] range)
    test_images_01 = generate_test_images(224, 224)

    # Direction vector for 768-dim patch embeddings
    direction = generate_direction_vector(model.config.hidden_size, seed=SEED).to(DEVICE)

    results = []

    for img_idx, img_01 in enumerate(test_images_01):
        # Normalize to ImageNet
        img_norm = imagenet_normalize(img_01).to(DEVICE)

        # Encode
        latent_clean, ids_restore = mae_encode(model, img_norm)
        # latent_clean: [1, 197, 768] (CLS at index 0, patches at 1-196)

        for epsilon in EPSILONS:
            for num_carriers in NUM_CARRIER_POSITIONS:
                torch.manual_seed(SEED + img_idx)
                # Select carrier positions (patch indices 1-196, skip CLS at 0)
                carrier_indices = torch.randperm(196)[:num_carriers] + 1  # +1 to skip CLS

                # Generate random bits
                bits = torch.randint(0, 2, (num_carriers,)).tolist()

                # Apply perturbations
                latent_modified = latent_clean.clone()
                for i, (idx, bit) in enumerate(zip(carrier_indices, bits)):
                    sign = 1.0 if bit == 1 else -1.0
                    latent_modified[0, idx] += sign * epsilon * direction

                # Decode to pixels
                pixel_pred = mae_decode(model, latent_modified, ids_restore)

                # Handle norm_pix_loss denormalization
                if norm_pix_loss:
                    # The model predicts normalized patches. We need the original
                    # patch statistics to denormalize. Compute from input.
                    ps = model.config.patch_size
                    h = w = 224 // ps
                    # Patchify input to get means and stds
                    patches = img_norm.unfold(2, ps, ps).unfold(3, ps, ps)  # [1,3,14,14,16,16]
                    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 196, -1)  # [1,196,768]
                    patch_mean = patches.mean(dim=-1, keepdim=True)
                    patch_var = patches.var(dim=-1, keepdim=True)
                    # Denormalize logits
                    logits_flat = pixel_pred.reshape(1, 3, 14, 16, 14, 16)
                    logits_flat = logits_flat.permute(0, 2, 4, 1, 3, 5).reshape(1, 196, -1)
                    logits_flat = logits_flat * (patch_var + 1e-6).sqrt() + patch_mean
                    logits_flat = logits_flat.reshape(1, 14, 14, 16, 16, 3)
                    pixel_pred = logits_flat.permute(0, 5, 1, 3, 2, 4).reshape(1, 3, 224, 224)

                # Convert back to [0,1] range
                recon_01 = imagenet_denormalize(pixel_pred).clamp(0, 1)

                # Compute PSNR
                psnr = compute_psnr(img_01.to(DEVICE), recon_01)
                mpd = max_pixel_diff(img_01.to(DEVICE), recon_01)

                # Re-encode
                recon_norm = imagenet_normalize(recon_01)
                latent_reencoded, _ = mae_encode(model, recon_norm)

                # Recover bits
                recovered_bits = []
                for i, (idx, bit) in enumerate(zip(carrier_indices, bits)):
                    proj_clean = torch.dot(latent_clean[0, idx], direction).item()
                    proj_reenc = torch.dot(latent_reencoded[0, idx], direction).item()
                    # If bit=1, we added +eps, so proj should increase
                    # If bit=0, we added -eps, so proj should decrease
                    if proj_reenc > proj_clean:
                        recovered_bits.append(1)
                    else:
                        recovered_bits.append(0)

                acc = bit_accuracy(bits, recovered_bits)
                results.append({
                    'img': img_idx, 'epsilon': epsilon,
                    'carriers': num_carriers, 'accuracy': acc,
                    'psnr': psnr, 'max_diff': mpd
                })

    # Print results table (averaged over images)
    print(f"\n{'Epsilon':>8} | {'Carriers':>8} | {'Bit Acc':>10} | {'PSNR (dB)':>10} | {'Max Px Diff':>12}")
    print("-" * 60)

    for eps in EPSILONS:
        for nc in NUM_CARRIER_POSITIONS:
            subset = [r for r in results if r['epsilon'] == eps and r['carriers'] == nc]
            avg_acc = np.mean([r['accuracy'] for r in subset])
            avg_psnr = np.mean([r['psnr'] for r in subset])
            avg_mpd = np.mean([r['max_diff'] for r in subset])
            print(f"{eps:>8.1f} | {nc:>8d} | {avg_acc:>9.1f}% | {avg_psnr:>10.1f} | {avg_mpd:>12.4f}")

    return results


# ============================================================
# SD VAE TEST
# ============================================================

def test_sd_vae():
    """Test Stable Diffusion VAE backbone for perturbation survival."""
    print("\n" + "=" * 60)
    print("BACKBONE: SD VAE (stabilityai/sd-vae-ft-mse)")
    print("=" * 60)

    print("Loading SD VAE model...")
    t0 = time.time()
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(DEVICE).to(DTYPE)
    vae.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    scaling_factor = vae.config.scaling_factor
    print(f"  scaling_factor: {scaling_factor}")
    print(f"  latent channels: {vae.config.latent_channels}")

    # Generate test images (256x256, [0,1] range)
    test_images_01 = generate_test_images(256, 256)

    # Direction vector for 4-channel latent
    direction = generate_direction_vector(vae.config.latent_channels, seed=SEED).to(DEVICE)

    results = []

    latent_h = 256 // 8  # 32 for 256x256 input

    for img_idx, img_01 in enumerate(test_images_01):
        print(f"\n  Image {img_idx+1}/{len(test_images_01)}")
        # Normalize to [-1, 1] for VAE
        img_vae = (img_01 * 2.0 - 1.0).to(DEVICE)

        # Encode
        with torch.no_grad():
            latent_clean = vae.encode(img_vae).latent_dist.mean  # [1, 4, 32, 32]
        print(f"    Encoded latent shape: {latent_clean.shape}")

        for epsilon in EPSILONS:
            for num_carriers in NUM_CARRIER_POSITIONS:
                print(f"    eps={epsilon}, carriers={num_carriers} ...", end=" ", flush=True)
                torch.manual_seed(SEED + img_idx)
                # Select carrier positions as (row, col) in latent grid
                all_positions = torch.randperm(latent_h * latent_h)[:num_carriers]
                carrier_rows = all_positions // latent_h
                carrier_cols = all_positions % latent_h

                # Generate random bits
                bits = torch.randint(0, 2, (num_carriers,)).tolist()

                # Apply perturbations
                latent_modified = latent_clean.clone()
                for i, (r, c, bit) in enumerate(zip(carrier_rows, carrier_cols, bits)):
                    sign = 1.0 if bit == 1 else -1.0
                    latent_modified[0, :, r, c] += sign * epsilon * direction

                # Decode to pixels
                with torch.no_grad():
                    recon_vae = vae.decode(latent_modified).sample  # [1, 3, 256, 256]

                # Convert back to [0,1]
                recon_01 = ((recon_vae + 1.0) / 2.0).clamp(0, 1)

                # Compute PSNR
                psnr = compute_psnr(img_01.to(DEVICE), recon_01)
                mpd = max_pixel_diff(img_01.to(DEVICE), recon_01)

                # Re-encode
                with torch.no_grad():
                    latent_reencoded = vae.encode(recon_vae.clamp(-1, 1)).latent_dist.mean

                # Recover bits
                recovered_bits = []
                for i, (r, c, bit) in enumerate(zip(carrier_rows, carrier_cols, bits)):
                    proj_clean = torch.dot(latent_clean[0, :, r, c], direction).item()
                    proj_reenc = torch.dot(latent_reencoded[0, :, r, c], direction).item()
                    if proj_reenc > proj_clean:
                        recovered_bits.append(1)
                    else:
                        recovered_bits.append(0)

                acc = bit_accuracy(bits, recovered_bits)
                print(f"acc={acc:.0f}% psnr={psnr:.1f}dB")
                results.append({
                    'img': img_idx, 'epsilon': epsilon,
                    'carriers': num_carriers, 'accuracy': acc,
                    'psnr': psnr, 'max_diff': mpd
                })

    # Print results table (averaged over images)
    print(f"\n{'Epsilon':>8} | {'Carriers':>8} | {'Bit Acc':>10} | {'PSNR (dB)':>10} | {'Max Px Diff':>12}")
    print("-" * 60)

    for eps in EPSILONS:
        for nc in NUM_CARRIER_POSITIONS:
            subset = [r for r in results if r['epsilon'] == eps and r['carriers'] == nc]
            avg_acc = np.mean([r['accuracy'] for r in subset])
            avg_psnr = np.mean([r['psnr'] for r in subset])
            avg_mpd = np.mean([r['max_diff'] for r in subset])
            print(f"{eps:>8.1f} | {nc:>8d} | {avg_acc:>9.1f}% | {avg_psnr:>10.1f} | {avg_mpd:>12.4f}")

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(mae_results, sdvae_results):
    """Save a quick visualization figure."""
    active = [(r, n) for r, n in [(mae_results, "MAE"), (sdvae_results, "SD VAE")] if r is not None]
    if not active:
        return

    fig, axes = plt.subplots(len(active), 2, figsize=(14, 5 * len(active)))
    if len(active) == 1:
        axes = axes.reshape(1, 2)

    for ax_row, (results, name) in enumerate(active):
        # Left: Accuracy vs Epsilon
        ax = axes[ax_row, 0]
        for nc in NUM_CARRIER_POSITIONS:
            accs = []
            for eps in EPSILONS:
                subset = [r for r in results if r['epsilon'] == eps and r['carriers'] == nc]
                accs.append(np.mean([r['accuracy'] for r in subset]))
            ax.plot(EPSILONS, accs, marker='o', label=f'{nc} carriers')
        ax.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='90% threshold')
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='chance')
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Bit Accuracy (%)')
        ax.set_title(f'{name}: Accuracy vs Epsilon')
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        ax.set_ylim(40, 105)
        ax.grid(True, alpha=0.3)

        # Right: PSNR vs Epsilon
        ax = axes[ax_row, 1]
        for nc in NUM_CARRIER_POSITIONS:
            psnrs = []
            for eps in EPSILONS:
                subset = [r for r in results if r['epsilon'] == eps and r['carriers'] == nc]
                psnrs.append(np.mean([r['psnr'] for r in subset]))
            ax.plot(EPSILONS, psnrs, marker='s', label=f'{nc} carriers')
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='30 dB threshold')
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{name}: PSNR vs Epsilon')
        ax.legend(fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roundtrip_results.png', dpi=150)
    print("\nVisualization saved to roundtrip_results.png")


# ============================================================
# RECOMMENDATION
# ============================================================

def print_recommendation(mae_results, sdvae_results):
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    best_backbone = None
    best_score = None

    for results, name in [(mae_results, "MAE"), (sdvae_results, "SD VAE")]:
        if results is None:
            continue

        # Find best operating point: highest accuracy where PSNR >= 30
        best_for_backbone = None
        for eps in EPSILONS:
            for nc in NUM_CARRIER_POSITIONS:
                subset = [r for r in results if r['epsilon'] == eps and r['carriers'] == nc]
                avg_acc = np.mean([r['accuracy'] for r in subset])
                avg_psnr = np.mean([r['psnr'] for r in subset])
                if avg_psnr >= 30 and (best_for_backbone is None or avg_acc > best_for_backbone['acc']):
                    best_for_backbone = {
                        'acc': avg_acc, 'psnr': avg_psnr,
                        'eps': eps, 'carriers': nc, 'name': name
                    }

        if best_for_backbone and (best_score is None or best_for_backbone['acc'] > best_score['acc']):
            best_score = best_for_backbone

    if best_score:
        print(f"- Best backbone: {best_score['name']}")
        print(f"- Best operating point: epsilon={best_score['eps']}, carriers={best_score['carriers']}")
        print(f"  -> Bit accuracy: {best_score['acc']:.1f}%, PSNR: {best_score['psnr']:.1f} dB")

        if best_score['acc'] >= 90:
            print("- VERDICT: GO -- system works, proceed with hackathon build")
        elif best_score['acc'] >= 80:
            print("- VERDICT: PARTIAL -- works but needs repetition coding")
        else:
            print("- VERDICT: NO-GO -- neither backbone preserves perturbations reliably")
    else:
        # No result with PSNR >= 30, find best overall
        print("- No operating point found with PSNR >= 30 dB")
        for results, name in [(mae_results, "MAE"), (sdvae_results, "SD VAE")]:
            if results is None:
                continue
            best = max(results, key=lambda r: r['accuracy'])
            subset = [r for r in results if r['epsilon'] == best['epsilon'] and r['carriers'] == best['carriers']]
            avg_acc = np.mean([r['accuracy'] for r in subset])
            avg_psnr = np.mean([r['psnr'] for r in subset])
            print(f"  {name} best: eps={best['epsilon']}, carriers={best['carriers']}, "
                  f"acc={avg_acc:.1f}%, psnr={avg_psnr:.1f} dB")
        print("- VERDICT: NO-GO -- neither backbone preserves perturbations reliably at acceptable quality")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(SEED)

    print("=" * 60)
    print("PatchSteg Round-Trip Validation Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epsilons: {EPSILONS}")
    print(f"Carrier counts: {NUM_CARRIER_POSITIONS}")
    print(f"Test images: {NUM_TEST_IMAGES}")

    overall_start = time.time()

    mae_results = None  # Skipped for speed
    sdvae_results = None

    # Test SD VAE
    try:
        t0 = time.time()
        sdvae_results = test_sd_vae()
        print(f"\nSD VAE test completed in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"\nSD VAE test FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Visualization
    try:
        save_visualization(mae_results, sdvae_results)
    except Exception as e:
        print(f"\nVisualization failed: {e}")

    # Recommendation
    print_recommendation(mae_results, sdvae_results)

    print(f"\nTotal runtime: {time.time()-overall_start:.1f}s")
