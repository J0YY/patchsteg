"""
Experiment: Map which latent positions are stable steganographic carriers.

Generates stability heatmap and statistics across test images.
Saves figure to paper/figures/carrier_stability_heatmap.png
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time

from core.vae import StegoVAE
from core.steganography import PatchSteg


def generate_test_images(n=5, size=512):
    """Generate diverse synthetic 512x512 test images as PIL."""
    images = []
    # 1: Random noise
    images.append(Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)))
    # 2: Smooth gradient
    grad = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    images.append(Image.fromarray(np.stack([grad, grad, grad], axis=2)))
    # 3: Checkerboard
    check = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, 32):
        for j in range(0, size, 32):
            if ((i // 32) + (j // 32)) % 2 == 0:
                check[i:i+32, j:j+32] = 255
    images.append(Image.fromarray(check))
    # 4: Solid mid-gray
    images.append(Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8)))
    # 5: Color patches
    patches = np.zeros((size, size, 3), dtype=np.uint8)
    rng = np.random.RandomState(99)
    for i in range(0, size, 64):
        for j in range(0, size, 64):
            patches[i:i+64, j:j+64] = rng.randint(0, 255, 3)
    images.append(Image.fromarray(patches))
    return images[:n]


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading VAE...")
    vae = StegoVAE(device=device)
    steg = PatchSteg(seed=42, epsilon=5.0)

    images = generate_test_images(5)
    image_names = ['Noise', 'Gradient', 'Checkerboard', 'Solid Gray', 'Color Patches']

    all_maps = []
    for idx, (img, name) in enumerate(zip(images, image_names)):
        print(f"  Computing stability map for '{name}'...")
        t0 = time.time()
        smap, _ = steg.compute_stability_map(vae, img, test_eps=5.0)
        print(f"    Done in {time.time()-t0:.1f}s, range: [{smap.min():.2f}, {smap.max():.2f}]")
        all_maps.append(smap)

    avg_map = torch.stack(all_maps).mean(dim=0)

    # Statistics
    print("\n=== Stability Statistics ===")
    print(f"Average stability: {avg_map.mean():.3f}")
    print(f"Std stability: {avg_map.std():.3f}")
    print(f"Min: {avg_map.min():.3f}, Max: {avg_map.max():.3f}")

    # What fraction of positions have positive stability (perturbation survived)?
    frac_positive = (avg_map > 0).float().mean().item() * 100
    print(f"Positions with positive stability (avg): {frac_positive:.1f}%")

    # Per-image stats
    for smap, name in zip(all_maps, image_names):
        frac = (smap > 0).float().mean().item() * 100
        print(f"  {name}: {frac:.1f}% positive")

    # Spatial analysis: edges vs center
    center_mask = torch.zeros(64, 64, dtype=torch.bool)
    center_mask[16:48, 16:48] = True
    edge_mask = ~center_mask
    center_mean = avg_map[center_mask].mean().item()
    edge_mean = avg_map[edge_mask].mean().item()
    print(f"\nCenter (16:48) avg stability: {center_mean:.3f}")
    print(f"Edge avg stability: {edge_mean:.3f}")

    # Top-20 and bottom-20 positions
    flat = avg_map.flatten()
    top20_idx = torch.argsort(flat, descending=True)[:20]
    bot20_idx = torch.argsort(flat)[:20]
    top20_pos = [(idx.item() // 64, idx.item() % 64) for idx in top20_idx]
    bot20_pos = [(idx.item() // 64, idx.item() % 64) for idx in bot20_idx]
    print(f"\nTop-20 carrier positions (most stable): {top20_pos[:5]}...")
    print(f"Bottom-20 positions (least stable): {bot20_pos[:5]}...")

    # === Visualization ===
    fig_dir = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Main figure: 2x3 grid (5 individual maps + average)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, (smap, name) in enumerate(zip(all_maps, image_names)):
        ax = axes[idx // 3, idx % 3]
        im = ax.imshow(smap.numpy(), cmap='RdYlGn', vmin=avg_map.min().item(), vmax=avg_map.max().item())
        ax.set_title(f'{name}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Average map with top/bottom markers
    ax = axes[1, 2]
    im = ax.imshow(avg_map.numpy(), cmap='RdYlGn')
    # Mark top-20 in red circles
    for r, c in top20_pos:
        ax.plot(c, r, 'r^', markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    # Mark bottom-20 in blue
    for r, c in bot20_pos:
        ax.plot(c, r, 'bv', markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    ax.set_title('Average (top-20=red, bot-20=blue)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Carrier Stability Maps (projection delta after round-trip, eps=5.0)', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / 'carrier_stability_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {fig_dir / 'carrier_stability_heatmap.png'}")

    # Also save just the average map as a clean figure for the paper
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    im2 = ax2.imshow(avg_map.numpy(), cmap='RdYlGn')
    for r, c in top20_pos:
        ax2.plot(c, r, 'r^', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    ax2.set_title('Average Carrier Stability (top-20 marked)')
    ax2.set_xlabel('Latent Column')
    ax2.set_ylabel('Latent Row')
    plt.colorbar(im2, ax=ax2, label='Projection Delta')
    plt.tight_layout()
    plt.savefig(fig_dir / 'carrier_stability_avg.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_dir / 'carrier_stability_avg.png'}")

    plt.close('all')

    # Return stats for paper update
    return {
        'frac_positive': frac_positive,
        'center_mean': center_mean,
        'edge_mean': edge_mean,
        'avg_stability_mean': avg_map.mean().item(),
        'avg_stability_std': avg_map.std().item(),
        'per_image_positive': {name: (smap > 0).float().mean().item() * 100
                               for smap, name in zip(all_maps, image_names)},
    }


if __name__ == '__main__':
    stats = main()
    print("\n=== Summary for paper ===")
    print(f"Overall: {stats['frac_positive']:.1f}% positions stable")
    print(f"Center: {stats['center_mean']:.3f}, Edge: {stats['edge_mean']:.3f}")
