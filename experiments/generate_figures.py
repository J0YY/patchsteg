#!/usr/bin/env python
"""
Generate publication-quality figures for the paper.
All 256x256 for speed. ~5-8 min on CPU.

Generates:
  1. hero_pipeline.png — full pipeline visualization
  2. stego_examples.png — side-by-side original/stego/diff for multiple images
  3. stability_by_content.png — per-image stability heatmaps
  4. pareto_frontier.png — stealth vs accuracy vs epsilon (money plot)
  5. carrier_ablation.png — random vs stability selection head-to-head
  6. text_demo.png — end-to-end "HELLO" encode/decode proof
  7. direction_ablation.png — does direction vector choice matter?
"""
import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path
import time

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, bit_accuracy

IMG_SIZE = 256
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper' / 'figures'

torch.manual_seed(42)
np.random.seed(42)


def make_images():
    imgs, names = {}, []
    # Gradient
    g = np.tile(np.linspace(0, 255, IMG_SIZE, dtype=np.uint8), (IMG_SIZE, 1))
    imgs['Gradient'] = Image.fromarray(np.stack([g]*3, axis=2))
    # Color patches
    p = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    rng = np.random.RandomState(99)
    for i in range(0, IMG_SIZE, 32):
        for j in range(0, IMG_SIZE, 32):
            p[i:i+32, j:j+32] = rng.randint(0, 255, 3)
    imgs['Patches'] = Image.fromarray(p)
    # Noise
    imgs['Noise'] = Image.fromarray(
        np.random.RandomState(42).randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    # Checkerboard
    c = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i in range(0, IMG_SIZE, 32):
        for j in range(0, IMG_SIZE, 32):
            if ((i//32)+(j//32))%2==0: c[i:i+32, j:j+32] = 255
    imgs['Checker'] = Image.fromarray(c)
    return imgs


# ================================================================
print("Loading VAE...", flush=True)
t0 = time.time()
vae = StegoVAE(device='cpu', image_size=IMG_SIZE)
steg = PatchSteg(seed=42, epsilon=5.0)
images = make_images()
print(f"Loaded in {time.time()-t0:.0f}s", flush=True)


# ================================================================
# 1. HERO PIPELINE FIGURE
# ================================================================
print("1. Hero pipeline figure...", flush=True)
img = images['Patches']
latent_clean = vae.encode(img)
carriers, smap = steg.select_carriers_by_stability(vae, img, n_carriers=20)
torch.manual_seed(42)
bits = torch.randint(0, 2, (20,)).tolist()
latent_mod = steg.encode_message(latent_clean, carriers, bits)
stego = vae.decode(latent_mod)
latent_re = vae.encode(stego)
recovered, confs = steg.decode_message(latent_clean, latent_re, carriers)
acc = bit_accuracy(bits, recovered)
psnr_val = compute_psnr(img, stego)

# Difference image
diff_arr = np.abs(np.array(img).astype(float) - np.array(stego).astype(float))
diff_amplified = np.clip(diff_arr * 20, 0, 255).astype(np.uint8)

fig = plt.figure(figsize=(16, 4.5))
gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.8], wspace=0.15)

ax = fig.add_subplot(gs[0])
ax.imshow(np.array(img)); ax.set_title('(a) Original Image', fontsize=11)
ax.axis('off')

ax = fig.add_subplot(gs[1])
im = ax.imshow(smap.numpy(), cmap='RdYlGn')
for r, c in carriers:
    ax.plot(c, r, 'r+', markersize=6, markeredgewidth=1.5)
ax.set_title('(b) Stability Map\n+ Carrier Positions', fontsize=11)
ax.set_xlabel(f'{len(carriers)} carriers selected')

ax = fig.add_subplot(gs[2])
ax.imshow(np.array(stego)); ax.set_title(f'(c) Stego Image\nPSNR={psnr_val:.1f} dB', fontsize=11)
ax.axis('off')

ax = fig.add_subplot(gs[3])
ax.imshow(diff_amplified); ax.set_title('(d) Difference (20x)', fontsize=11)
ax.axis('off')

ax = fig.add_subplot(gs[4])
ax.barh(range(20), confs[::-1], color=['green' if recovered[19-i]==bits[19-i] else 'red' for i in range(20)],
        height=0.7)
ax.set_yticks(range(20))
ax.set_yticklabels([f'b{19-i}={bits[19-i]}' for i in range(20)], fontsize=6)
ax.set_xlabel('Confidence')
ax.set_title(f'(e) Decoded Bits\nAcc={acc:.0f}%', fontsize=11)

plt.suptitle('PatchSteg Pipeline: Encode $\\rightarrow$ Perturb $\\rightarrow$ Decode $\\rightarrow$ Recover', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'hero_pipeline.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved hero_pipeline.png", flush=True)


# ================================================================
# 2. STEGO EXAMPLES — multiple images side by side
# ================================================================
print("2. Stego examples...", flush=True)
example_names = ['Gradient', 'Patches', 'Noise', 'Checker']
fig, axes = plt.subplots(3, 4, figsize=(14, 10))

for col, name in enumerate(example_names):
    img = images[name]
    lat = vae.encode(img)
    car, _ = steg.select_carriers_by_stability(vae, img, n_carriers=20)
    torch.manual_seed(42)
    b = torch.randint(0, 2, (20,)).tolist()
    lat_m = steg.encode_message(lat, car, b)
    st = vae.decode(lat_m)
    psnr_v = compute_psnr(img, st)
    d = np.abs(np.array(img).astype(float) - np.array(st).astype(float))
    d_amp = np.clip(d * 20, 0, 255).astype(np.uint8)

    # Re-encode and check accuracy
    lat_re = vae.encode(st)
    rec, _ = steg.decode_message(lat, lat_re, car)
    a = bit_accuracy(b, rec)

    axes[0, col].imshow(np.array(img)); axes[0, col].set_title(f'{name}', fontsize=11)
    axes[0, col].axis('off')
    axes[1, col].imshow(np.array(st)); axes[1, col].set_title(f'Stego (PSNR={psnr_v:.1f})', fontsize=10)
    axes[1, col].axis('off')
    axes[2, col].imshow(d_amp); axes[2, col].set_title(f'Diff 20x (Acc={a:.0f}%)', fontsize=10)
    axes[2, col].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=12)
axes[1, 0].set_ylabel('Stego', fontsize=12)
axes[2, 0].set_ylabel('Difference', fontsize=12)
plt.suptitle('Stego Examples Across Image Types ($\\varepsilon=5.0$, 20 carriers)', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'stego_examples.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved stego_examples.png", flush=True)


# ================================================================
# 3. STABILITY BY CONTENT — per-image heatmaps
# ================================================================
print("3. Stability by content...", flush=True)
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
smaps = {}
for ax, name in zip(axes, ['Gradient', 'Patches', 'Noise', 'Checker']):
    sm, _ = steg.compute_stability_map(vae, images[name], test_eps=5.0)
    smaps[name] = sm
    im = ax.imshow(sm.numpy(), cmap='RdYlGn', vmin=0, vmax=5)
    pos_pct = (sm > 0).float().mean().item() * 100
    ax.set_title(f'{name}\nmean={sm.mean():.2f}, std={sm.std():.2f}\n{pos_pct:.0f}% positive', fontsize=9)
plt.colorbar(im, ax=axes[-1], fraction=0.046, label='Stability Score')
plt.suptitle('Carrier Stability is Content-Dependent', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'stability_by_content.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved stability_by_content.png", flush=True)


# ================================================================
# 4. PARETO FRONTIER — stealth vs accuracy (the money plot)
# ================================================================
print("4. Pareto frontier...", flush=True)
# Need: at each epsilon, what's the accuracy AND the detection AUC?
# Use results from our experiments:
# Detection AUC from detectability test
det_auc = {1.0: 0.44, 2.0: 0.68, 5.0: 0.93, 10.0: 0.97}
# Bit accuracy (patches, 20 stability carriers) - compute fresh
pareto_eps = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
pareto_accs = []
pareto_psnrs = []
img_p = images['Patches']
lat_p = vae.encode(img_p)

for eps in pareto_eps:
    s = PatchSteg(seed=42, epsilon=eps)
    car, _ = s.select_carriers_by_stability(vae, img_p, n_carriers=20, test_eps=eps)
    torch.manual_seed(42)
    b = torch.randint(0, 2, (20,)).tolist()
    lat_m = s.encode_message(lat_p, car, b)
    st = vae.decode(lat_m)
    lat_re = vae.encode(st)
    rec, _ = s.decode_message(lat_p, lat_re, car)
    pareto_accs.append(bit_accuracy(b, rec))
    pareto_psnrs.append(compute_psnr(img_p, st))
    print(f"  eps={eps}: acc={pareto_accs[-1]:.0f}%, psnr={pareto_psnrs[-1]:.1f}", flush=True)

fig, ax1 = plt.subplots(figsize=(8, 5))
color1, color2 = 'tab:blue', 'tab:red'

# Accuracy curve
ax1.plot(pareto_eps, pareto_accs, 'o-', color=color1, linewidth=2, markersize=8, label='Bit Accuracy (%)')
ax1.set_xlabel('Epsilon ($\\varepsilon$)', fontsize=12)
ax1.set_ylabel('Bit Accuracy (%)', color=color1, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(40, 105)
ax1.axhline(90, color=color1, ls=':', alpha=0.3)

# Detection AUC on second axis
ax2 = ax1.twinx()
det_eps = sorted(det_auc.keys())
det_vals = [det_auc[e] for e in det_eps]
ax2.plot(det_eps, [v*100 for v in det_vals], 's--', color=color2, linewidth=2, markersize=8, label='Detection AUC (%)')
ax2.set_ylabel('Detection AUC (%)', color=color2, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(40, 105)
ax2.axhline(70, color=color2, ls=':', alpha=0.3)

# Shade the "sweet spot"
ax1.axvspan(1.5, 3.5, alpha=0.1, color='green', label='Sweet spot ($\\varepsilon \\approx 2$)')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)

ax1.set_title('Stealth--Capacity Pareto Frontier\n(Color Patches, 20 stability-selected carriers)', fontsize=12)
ax1.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(FIG_DIR / 'pareto_frontier.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved pareto_frontier.png", flush=True)


# ================================================================
# 5. CARRIER SELECTION ABLATION — random vs stability
# ================================================================
print("5. Carrier selection ablation...", flush=True)
K_vals = [5, 10, 20, 50]
eps_vals = [2.0, 5.0]
ablation_results = []
img_abl = images['Patches']
lat_abl = vae.encode(img_abl)

for eps in eps_vals:
    s = PatchSteg(seed=42, epsilon=eps)
    stab_carriers, _ = s.select_carriers_by_stability(vae, img_abl, n_carriers=max(K_vals), test_eps=eps)
    rand_carriers = s.select_carriers_fixed(max(K_vals), seed=42, grid_size=IMG_SIZE//8)

    for K in K_vals:
        for method, carriers in [('Stability', stab_carriers[:K]), ('Random', rand_carriers[:K])]:
            torch.manual_seed(42)
            b = torch.randint(0, 2, (K,)).tolist()
            lat_m = s.encode_message(lat_abl, carriers, b)
            st = vae.decode(lat_m)
            lat_re = vae.encode(st)
            rec, _ = s.decode_message(lat_abl, lat_re, carriers)
            acc = bit_accuracy(b, rec)
            ablation_results.append({'eps': eps, 'K': K, 'method': method, 'acc': acc})
            print(f"  eps={eps} K={K:>2d} {method:>9s}: {acc:.0f}%", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax_i, eps in enumerate(eps_vals):
    ax = axes[ax_i]
    for method, color, marker in [('Stability', 'tab:blue', 'o'), ('Random', 'tab:red', 's')]:
        accs = [r['acc'] for r in ablation_results if r['eps']==eps and r['method']==method]
        ax.plot(K_vals, accs, f'{marker}-', color=color, linewidth=2, markersize=8, label=method)
    ax.axhline(90, color='green', ls='--', alpha=0.4, label='90% target')
    ax.axhline(50, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Number of Carriers')
    ax.set_ylabel('Bit Accuracy (%)')
    ax.set_title(f'$\\varepsilon = {eps}$')
    ax.legend(fontsize=9)
    ax.set_ylim(30, 105)
    ax.grid(True, alpha=0.2)

plt.suptitle('Stability-Based vs Random Carrier Selection (Color Patches)', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'carrier_ablation.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved carrier_ablation.png", flush=True)


# ================================================================
# 6. END-TO-END TEXT DEMO
# ================================================================
print("6. End-to-end text demo...", flush=True)
message = "HI"
msg_bits = PatchSteg.text_to_bits(message)
n_bits = len(msg_bits)  # 16 bits for "HI"

img_demo = images['Patches']
lat_demo = vae.encode(img_demo)
# Use 3x repetition: need 48 carriers for 16 bits
steg_demo = PatchSteg(seed=42, epsilon=5.0)
carriers_demo, smap_demo = steg_demo.select_carriers_by_stability(vae, img_demo, n_carriers=n_bits*3, test_eps=5.0)
lat_mod_demo = steg_demo.encode_message_with_repetition(lat_demo, carriers_demo, msg_bits, reps=3)
stego_demo = vae.decode(lat_mod_demo)
psnr_demo = compute_psnr(img_demo, stego_demo)

# Decode
lat_re_demo = vae.encode(stego_demo)
decoded_bits = steg_demo.decode_message_with_repetition(lat_demo, lat_re_demo, carriers_demo, n_bits, reps=3)
decoded_text = PatchSteg.bits_to_text(decoded_bits)
demo_acc = bit_accuracy(msg_bits, decoded_bits)

print(f"  Sent: '{message}' -> {msg_bits}", flush=True)
print(f"  Recv: '{decoded_text}' -> {decoded_bits}", flush=True)
print(f"  Accuracy: {demo_acc:.0f}%, PSNR: {psnr_demo:.1f} dB", flush=True)

diff_demo = np.clip(np.abs(np.array(img_demo).astype(float) - np.array(stego_demo).astype(float)) * 30, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(np.array(img_demo)); axes[0].set_title('Original Image', fontsize=11); axes[0].axis('off')
axes[1].imshow(np.array(stego_demo)); axes[1].set_title(f'Stego Image (PSNR={psnr_demo:.1f} dB)', fontsize=11); axes[1].axis('off')
axes[2].imshow(diff_demo); axes[2].set_title('Difference (30x amplified)', fontsize=11); axes[2].axis('off')

# Bit visualization
ax = axes[3]
colors_sent = ['#2196F3' if b==1 else '#F44336' for b in msg_bits]
colors_recv = ['#2196F3' if b==1 else '#F44336' for b in decoded_bits]
x = np.arange(n_bits)
ax.bar(x - 0.2, msg_bits, 0.35, color=colors_sent, alpha=0.7, label='Sent', edgecolor='black', linewidth=0.3)
ax.bar(x + 0.2, decoded_bits, 0.35, color=colors_recv, alpha=0.7, label='Received', edgecolor='black', linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels([f'{i}' for i in range(n_bits)], fontsize=7)
ax.set_xlabel('Bit Index')
ax.set_ylabel('Bit Value')
ax.set_title(f'"{message}" -> "{decoded_text}" ({demo_acc:.0f}%)', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(-0.1, 1.4)

plt.suptitle(f'End-to-End Message Transmission: "{message}" encoded with 3x repetition coding', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'text_demo.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved text_demo.png", flush=True)


# ================================================================
# 7. DIRECTION VECTOR ABLATION
# ================================================================
print("7. Direction vector ablation...", flush=True)
seeds_to_test = [1, 7, 42, 100, 256, 999, 1337, 2024, 3141, 9999]
dir_results = []
img_dir = images['Patches']
lat_dir = vae.encode(img_dir)

for seed in seeds_to_test:
    s = PatchSteg(seed=seed, epsilon=5.0)
    car, _ = s.select_carriers_by_stability(vae, img_dir, n_carriers=20, test_eps=5.0)
    torch.manual_seed(42)
    b = torch.randint(0, 2, (20,)).tolist()
    lat_m = s.encode_message(lat_dir, car, b)
    st = vae.decode(lat_m)
    lat_re = vae.encode(st)
    rec, _ = s.decode_message(lat_dir, lat_re, car)
    acc = bit_accuracy(b, rec)
    dir_results.append({'seed': seed, 'acc': acc, 'direction': s.direction.tolist()})
    print(f"  seed={seed:>5d}: acc={acc:.0f}%, d={s.direction.numpy().round(2)}", flush=True)

fig, ax = plt.subplots(figsize=(8, 4))
seeds_str = [str(r['seed']) for r in dir_results]
accs = [r['acc'] for r in dir_results]
colors = ['green' if a >= 90 else 'orange' if a >= 70 else 'red' for a in accs]
ax.bar(seeds_str, accs, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(90, color='green', ls='--', alpha=0.4)
ax.axhline(np.mean(accs), color='blue', ls='-', alpha=0.5, label=f'mean={np.mean(accs):.1f}%')
ax.set_xlabel('Direction Seed')
ax.set_ylabel('Bit Accuracy (%)')
ax.set_title(f'Direction Vector Ablation ($\\varepsilon=5.0$, 20 carriers, Color Patches)\nMean={np.mean(accs):.1f}% $\\pm$ {np.std(accs):.1f}%')
ax.legend()
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / 'direction_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved direction_ablation.png", flush=True)


# ================================================================
print(f"\nAll figures generated in {time.time()-t0:.0f}s", flush=True)
print(f"Figures in: {FIG_DIR}", flush=True)
