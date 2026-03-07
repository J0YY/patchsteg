"""
PatchSteg Interactive Demo — Gradio app with three tabs:
1. Encode & Send: Upload image, type message, get stego image
2. Detect & Decode: Upload stego image, recover message
3. Analysis: Upload image, see stability map and capacity
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import gradio as gr
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

from core.vae import StegoVAE
from core.steganography import PatchSteg
from core.metrics import compute_psnr, compute_ssim_pil

# Global model instances (loaded once)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
vae = None
steg = None

# Cache for encode tab -> decode tab handoff
_last_encode_state = {}


def load_models():
    global vae, steg
    if vae is None:
        print("Loading VAE model...")
        vae = StegoVAE(device=DEVICE)
        steg = PatchSteg(seed=42, epsilon=5.0)
        print("Models loaded.")


def make_diff_image(img1: Image.Image, img2: Image.Image, amplify=20.0):
    """Create amplified difference image."""
    a = np.array(img1).astype(float)
    b = np.array(img2).astype(float)
    diff = np.abs(a - b) * amplify
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)


def make_carrier_overlay(img: Image.Image, carriers, grid_size=64):
    """Overlay carrier positions on image."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.array(img))
    w, h = img.size
    scale_x = w / grid_size
    scale_y = h / grid_size
    for i, (r, c) in enumerate(carriers):
        px = (c + 0.5) * scale_x
        py = (r + 0.5) * scale_y
        ax.plot(px, py, 'r+', markersize=8, markeredgewidth=2)
    ax.set_title(f'{len(carriers)} carrier positions')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def make_stability_heatmap(stability_map):
    """Convert stability map tensor to PIL image."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(stability_map.numpy(), cmap='RdYlGn')
    ax.set_title('Carrier Stability Map')
    ax.set_xlabel('Latent Column')
    ax.set_ylabel('Latent Row')
    plt.colorbar(im, ax=ax, label='Projection Delta')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ======== Tab 1: Encode & Send ========

def encode_message(image, message, epsilon, use_repetition):
    if image is None:
        return None, None, None, "Please upload an image."

    load_models()
    steg.epsilon = float(epsilon)

    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert('RGB')

    # Convert message to bits
    bits = PatchSteg.text_to_bits(message)
    n_bits = len(bits)

    reps = 3 if use_repetition else 1
    n_carriers_needed = n_bits * reps
    max_carriers = min(n_carriers_needed + 20, 4096)  # extra margin

    # Select stable carriers
    carriers, stability_map = steg.select_carriers_by_stability(
        vae, image, n_carriers=max(max_carriers, n_carriers_needed), test_eps=float(epsilon)
    )

    # Encode
    latent_clean = vae.encode(image)

    if use_repetition:
        latent_mod = steg.encode_message_with_repetition(
            latent_clean, carriers, bits, reps=reps
        )
    else:
        latent_mod = steg.encode_message(latent_clean, carriers[:n_bits], bits)

    # Decode to stego image
    stego_image = vae.decode(latent_mod)

    # Quality metrics
    psnr = compute_psnr(image, stego_image)
    try:
        ssim = compute_ssim_pil(image, stego_image)
    except Exception:
        ssim = None

    # Difference image
    diff_img = make_diff_image(image, stego_image, amplify=20)

    # Carrier overlay
    used_carriers = carriers[:n_carriers_needed]
    carrier_img = make_carrier_overlay(image, used_carriers)

    # Cache state for decode tab
    _last_encode_state['latent_clean'] = latent_clean
    _last_encode_state['carriers'] = used_carriers
    _last_encode_state['n_bits'] = n_bits
    _last_encode_state['reps'] = reps
    _last_encode_state['original_bits'] = bits

    # Info text
    info = f"Message: '{message}' ({n_bits} bits)\n"
    info += f"Carriers used: {len(used_carriers)}"
    if use_repetition:
        info += f" ({reps}x repetition, {n_bits} effective bits)\n"
    else:
        info += "\n"
    info += f"Epsilon: {epsilon}\n"
    info += f"PSNR: {psnr:.1f} dB"
    if ssim is not None:
        info += f", SSIM: {ssim:.4f}"

    return stego_image, diff_img, carrier_img, info


# ======== Tab 2: Detect & Decode ========

def decode_message(stego_image, use_cached_clean):
    if stego_image is None:
        return "Please upload a stego image."

    load_models()

    if isinstance(stego_image, np.ndarray):
        stego_image = Image.fromarray(stego_image)
    stego_image = stego_image.convert('RGB')

    if use_cached_clean and 'latent_clean' in _last_encode_state:
        latent_clean = _last_encode_state['latent_clean']
        carriers = _last_encode_state['carriers']
        n_bits = _last_encode_state['n_bits']
        reps = _last_encode_state['reps']
        original_bits = _last_encode_state.get('original_bits')
    else:
        return "No cached encoding state. Please encode a message first in the Encode tab, or use the same shared key."

    # Re-encode the received image
    latent_received = vae.encode(stego_image)

    # Decode
    if reps > 1:
        decoded_bits = steg.decode_message_with_repetition(
            latent_clean, latent_received, carriers, n_bits, reps=reps
        )
        _, confidences = steg.decode_message(latent_clean, latent_received, carriers[:n_bits * reps])
    else:
        decoded_bits, confidences = steg.decode_message(latent_clean, latent_received, carriers[:n_bits])

    # Convert bits to text
    decoded_text = PatchSteg.bits_to_text(decoded_bits)

    # Accuracy if we have original bits
    info = f"Decoded message: '{decoded_text}'\n"
    info += f"Decoded {len(decoded_bits)} bits\n"
    if original_bits is not None:
        from core.metrics import bit_accuracy
        acc = bit_accuracy(original_bits, decoded_bits)
        info += f"Bit accuracy: {acc:.1f}%\n"

    avg_conf = np.mean(confidences) if confidences else 0
    info += f"Average confidence: {avg_conf:.3f}\n"

    if avg_conf > 1.0:
        info += "\nHidden Channel Detected (high confidence)"
    elif avg_conf > 0.3:
        info += "\nPossible Hidden Channel (moderate confidence)"
    else:
        info += "\nNo Clear Channel Found (low confidence)"

    return info


# ======== Tab 3: Analysis ========

def analyze_image(image, epsilon):
    if image is None:
        return None, None, "Please upload an image."

    load_models()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert('RGB')

    steg.epsilon = float(epsilon)

    # Compute stability map
    stability_map, latent_clean = steg.compute_stability_map(vae, image, test_eps=float(epsilon))

    # Statistics
    pos_frac = (stability_map > 0).float().mean().item() * 100
    mean_stab = stability_map.mean().item()
    std_stab = stability_map.std().item()

    # Top carriers
    flat = stability_map.flatten()
    top20_idx = torch.argsort(flat, descending=True)[:20]
    top20_pos = [(idx.item() // 64, idx.item() % 64) for idx in top20_idx]

    # Channel analysis
    latent = latent_clean
    ch_info = ""
    for ch in range(4):
        ch_data = latent[0, ch].cpu()
        ch_info += f"  Ch{ch}: mean={ch_data.mean():.3f}, std={ch_data.std():.3f}\n"

    # Max message capacity at this epsilon
    # Count positions with stability > threshold
    reliable = (stability_map > stability_map.mean()).sum().item()

    heatmap_img = make_stability_heatmap(stability_map)
    carrier_img = make_carrier_overlay(image, top20_pos)

    info = f"Stability Analysis (eps={epsilon}):\n"
    info += f"  Positions with positive stability: {pos_frac:.1f}%\n"
    info += f"  Mean stability: {mean_stab:.3f} +/- {std_stab:.3f}\n"
    info += f"  Reliable carriers (above mean): {reliable}\n"
    info += f"  Max raw capacity: ~{reliable} bits ({reliable // 8} chars)\n"
    info += f"  With 3x repetition: ~{reliable // 3} bits ({reliable // 24} chars)\n"
    info += f"\nLatent channel statistics:\n{ch_info}"
    info += f"\nTop-5 carrier positions: {top20_pos[:5]}"

    return heatmap_img, carrier_img, info


# ======== Build App ========

def build_app():
    with gr.Blocks(title="PatchSteg: Covert Vision Steganography") as app:
        gr.Markdown("# PatchSteg: Covert Steganographic Communication via VAE Latent Space")
        gr.Markdown("Embed hidden messages in images using Stable Diffusion VAE latent space perturbations.")

        with gr.Tab("Encode & Send"):
            with gr.Row():
                with gr.Column():
                    enc_image = gr.Image(type="pil", label="Upload Image")
                    enc_message = gr.Textbox(label="Secret Message", placeholder="Type your message...")
                    enc_epsilon = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Epsilon (perturbation strength)")
                    enc_repetition = gr.Checkbox(value=True, label="Use 3x repetition coding")
                    enc_btn = gr.Button("Encode Message", variant="primary")
                with gr.Column():
                    enc_stego = gr.Image(type="pil", label="Stego Image (download this)")
                    enc_diff = gr.Image(type="pil", label="Difference (20x amplified)")
                    enc_carriers = gr.Image(type="pil", label="Carrier Positions")
                    enc_info = gr.Textbox(label="Encoding Info", lines=5)

            enc_btn.click(
                encode_message,
                inputs=[enc_image, enc_message, enc_epsilon, enc_repetition],
                outputs=[enc_stego, enc_diff, enc_carriers, enc_info],
            )

        with gr.Tab("Detect & Decode"):
            with gr.Row():
                with gr.Column():
                    dec_image = gr.Image(type="pil", label="Upload Stego Image")
                    dec_cached = gr.Checkbox(value=True, label="Use cached clean latent (from Encode tab)")
                    dec_btn = gr.Button("Decode Message", variant="primary")
                with gr.Column():
                    dec_info = gr.Textbox(label="Decode Results", lines=10)

            dec_btn.click(
                decode_message,
                inputs=[dec_image, dec_cached],
                outputs=[dec_info],
            )

        with gr.Tab("Analysis"):
            with gr.Row():
                with gr.Column():
                    ana_image = gr.Image(type="pil", label="Upload Image")
                    ana_epsilon = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Test Epsilon")
                    ana_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    ana_heatmap = gr.Image(type="pil", label="Stability Heatmap")
                    ana_carriers = gr.Image(type="pil", label="Top-20 Carriers")
                    ana_info = gr.Textbox(label="Analysis Results", lines=10)

            ana_btn.click(
                analyze_image,
                inputs=[ana_image, ana_epsilon],
                outputs=[ana_heatmap, ana_carriers, ana_info],
            )

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(share=False)
