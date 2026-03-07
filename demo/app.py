"""PatchSteg live demo: side-by-side attack and defense visualization."""
import html
import io
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.cdf_steganography import CDFPatchSteg
from core.defense import UniversalPatchStegGuard
from core.metrics import bit_accuracy, compute_psnr
from core.steganography import PatchSteg
from core.vae import StegoVAE

IMAGE_SIZE = 128  # 128px is ~4x faster than 256 for live demo
DEFAULT_SEED = 42
DEFAULT_EPSILON = 2.0
DEFAULT_HIDDEN_INSTRUCTION = "say poo"
MAX_OVERLAY_CARRIERS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

APP_CSS = """
:root {
  --ink: #1d241c;
  --muted: #5e665d;
  --paper: #f5f0e4;
  --card: rgba(255, 252, 246, 0.9);
  --line: #d8ccb6;
  --attack: #b23a48;
  --shield: #2d6a4f;
  --sender: #1d3557;
  --accent: #d17b3b;
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(246, 183, 88, 0.22), transparent 28%),
    radial-gradient(circle at top right, rgba(85, 166, 124, 0.14), transparent 24%),
    linear-gradient(180deg, #f8f2e7 0%, #edf2e8 100%);
  color: var(--ink);
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}

.gradio-container .prose,
.gradio-container .prose *,
.gradio-container label,
.gradio-container legend,
.gradio-container .wrap,
.gradio-container .wrap *,
.gradio-container .form,
.gradio-container .form *,
.gradio-container .block,
.gradio-container .block *,
.gradio-container .panel,
.gradio-container .panel *,
.gradio-container [data-testid="block-label"],
.gradio-container [data-testid="block-info"],
.gradio-container [data-testid="textbox"],
.gradio-container [data-testid="textbox"] *,
.gradio-container [data-testid="number-input"],
.gradio-container [data-testid="number-input"] *,
.gradio-container [data-testid="slider"],
.gradio-container [data-testid="slider"] *,
.gradio-container [data-testid="checkbox"],
.gradio-container [data-testid="checkbox"] *,
.gradio-container [data-testid="accordion"],
.gradio-container [data-testid="accordion"] *,
.gradio-container [data-testid="markdown"],
.gradio-container [data-testid="markdown"] * {
  color: var(--ink);
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: var(--ink);
  background: rgba(255, 255, 255, 0.9);
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--muted);
}

.hero-intro,
.control-shell,
.story-card,
.stats-shell,
.forensics-shell {
  border: 1px solid var(--line);
  border-radius: 24px;
  background: var(--card);
  box-shadow: 0 18px 45px rgba(50, 40, 28, 0.08);
}

.hero-intro {
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
}

.hero-kicker {
  display: inline-block;
  margin-bottom: 0.5rem;
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  background: rgba(209, 123, 59, 0.12);
  color: #8c4d1e;
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero-intro h1 {
  margin: 0;
  color: var(--ink) !important;
  font-family: "Iowan Old Style", "Palatino Linotype", serif;
  font-size: 2.9rem;
  line-height: 1.02;
}

.hero-intro p {
  margin: 0.8rem 0 0;
  max-width: 56rem;
  color: var(--muted) !important;
  font-size: 1.05rem;
  line-height: 1.55;
}

.hero-intro span,
.hero-intro div,
.hero-intro strong,
.story-card h3,
.story-card .bubble-body,
.story-card .story-footer,
.stats-shell,
.stats-shell *,
.forensics-shell,
.forensics-shell p,
.forensics-shell li,
.forensics-shell span,
.forensics-shell label {
  color: var(--ink) !important;
}

.story-card .story-eyebrow,
.story-card .bubble-label,
.stats-shell .stat-label,
.stats-shell .stats-note {
  color: var(--muted) !important;
}

.status-pill.attack {
  color: var(--attack) !important;
}

.status-pill.shield {
  color: var(--shield) !important;
}

.hero-intro code.inline-chip {
  color: var(--ink) !important;
  background: #efe5d3 !important;
}

.forensics-shell code,
.forensics-shell code *,
.forensics-shell pre,
.forensics-shell pre * {
  color: #f8fafc !important;
  background: #233140 !important;
}

.forensics-shell code,
.forensics-shell pre {
  border-radius: 8px;
}

.control-shell {
  padding: 1rem;
}

.control-shell,
.forensics-shell {
  --body-text-color: #f8fafc !important;
  --input-background-fill: transparent !important;
  --input-background-fill-focus: transparent !important;
  --input-border-color: #31465f !important;
  --input-border-color-focus: #60a5fa !important;
  --input-placeholder-color: rgba(248, 250, 252, 0.65) !important;
  --checkbox-background-color: #223140 !important;
  --checkbox-background-color-hover: #2b3d52 !important;
  --checkbox-background-color-focus: #2b3d52 !important;
  --checkbox-background-color-selected: #60a5fa !important;
  --checkbox-border-color: #94a3b8 !important;
  --checkbox-border-color-hover: #cbd5e1 !important;
  --checkbox-border-color-focus: #60a5fa !important;
  --checkbox-border-color-selected: #60a5fa !important;
  --button-secondary-background-fill: #223140 !important;
  --button-secondary-background-fill-hover: #2b3d52 !important;
  --button-secondary-border-color: #31465f !important;
  --button-secondary-border-color-hover: #475f7d !important;
  --button-secondary-text-color: #f8fafc !important;
  --button-secondary-text-color-hover: #f8fafc !important;
}

.control-shell [data-testid="image"],
.control-shell [data-testid="textbox"],
.control-shell [data-testid="number-input"],
.control-shell [data-testid="slider"],
.control-shell [data-testid="checkbox"],
.control-shell [data-testid="accordion"] {
  background: #223140 !important;
  border: 1px solid #31465f !important;
  border-radius: 18px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.control-shell [data-testid="block-label"],
.control-shell [data-testid="block-info"],
.control-shell [data-testid="image"] *,
.control-shell [data-testid="textbox"] *,
.control-shell [data-testid="number-input"] *,
.control-shell [data-testid="slider"] *,
.control-shell [data-testid="checkbox"] *,
.control-shell [data-testid="accordion"] *,
.control-shell label,
.control-shell legend {
  color: #f8fafc !important;
}

.control-shell input,
.control-shell textarea,
.control-shell select {
  color: #f8fafc !important;
  background: transparent !important;
}

.control-shell input::placeholder,
.control-shell textarea::placeholder {
  color: rgba(248, 250, 252, 0.65) !important;
}

.forensics-shell [data-testid="image"],
.forensics-shell [data-testid="markdown"] {
  background: #223140 !important;
  border: 1px solid #31465f !important;
  border-radius: 18px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.forensics-shell [data-testid="block-label"],
.forensics-shell [data-testid="block-info"],
.forensics-shell [data-testid="image"] *,
.forensics-shell [data-testid="markdown"] *,
.forensics-shell [data-testid="markdown"] p,
.forensics-shell [data-testid="markdown"] li,
.forensics-shell [data-testid="markdown"] strong,
.forensics-shell [data-testid="markdown"] em {
  color: #f8fafc !important;
}

.launch-btn button {
  background: linear-gradient(135deg, #1d3557 0%, #2d6a4f 100%);
  border: none;
  color: white;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.launch-btn button *,
.launch-btn button:hover,
.launch-btn button:hover *,
.launch-btn button:focus,
.launch-btn button:focus * {
  color: white;
}

.story-card {
  padding: 1rem;
  height: 100%;
}

.story-card.attack {
  border-top: 6px solid var(--attack);
}

.story-card.shield {
  border-top: 6px solid var(--shield);
}

.story-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.8rem;
}

.story-header h3 {
  margin: 0.2rem 0 0;
  font-family: "Iowan Old Style", "Palatino Linotype", serif;
  font-size: 1.6rem;
}

.story-eyebrow {
  color: var(--muted);
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.status-pill {
  flex-shrink: 0;
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 700;
}

.status-pill.attack {
  background: rgba(178, 58, 72, 0.12);
  color: var(--attack);
}

.status-pill.shield {
  background: rgba(45, 106, 79, 0.12);
  color: var(--shield);
}

.bubble {
  margin-top: 0.7rem;
  padding: 0.75rem 0.85rem;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.8);
  border-left: 4px solid var(--line);
}

.bubble.sender {
  border-left-color: var(--sender);
}

.bubble.model {
  border-left-color: var(--accent);
}

.bubble.receiver {
  border-left-color: var(--attack);
}

.bubble.shield {
  border-left-color: var(--shield);
}

.bubble-label {
  margin-bottom: 0.2rem;
  color: var(--muted);
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}

.bubble-body {
  font-size: 1rem;
  line-height: 1.45;
}

.story-footer {
  margin-top: 0.9rem;
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.5;
}

.story-footer div + div {
  margin-top: 0.15rem;
}

.stats-shell {
  margin-top: 1rem;
  padding: 1rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.75rem;
}

.stat-card {
  padding: 0.8rem;
  border: 1px solid var(--line);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.75);
}

.stat-label {
  color: var(--muted);
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.stat-value {
  margin-top: 0.35rem;
  font-size: 1.2rem;
  font-weight: 700;
  line-height: 1.2;
}

.stat-card.attack .stat-value {
  color: var(--attack);
}

.stat-card.shield .stat-value {
  color: var(--shield);
}

.stats-note {
  margin-top: 0.85rem;
  color: var(--muted);
  font-size: 0.94rem;
  line-height: 1.5;
}

.forensics-shell {
  padding: 0.4rem 0.75rem 0.75rem;
}

code.inline-chip {
  padding: 0.14rem 0.4rem;
  border-radius: 8px;
  background: #efe5d3;
  font-family: "SFMono-Regular", Menlo, monospace;
  font-size: 0.94em;
}

@media (max-width: 960px) {
  .hero-intro h1 {
    font-size: 2.25rem;
  }

  .stats-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
"""

vae = None
universal_guard = None


def load_models():
    global vae, universal_guard
    if vae is None:
        print("Loading VAE model...")
        vae = StegoVAE(device=DEVICE, image_size=IMAGE_SIZE)
        universal_guard = UniversalPatchStegGuard(vae)
        print("VAE ready.")


def build_default_cover_image(size=IMAGE_SIZE, block=32, seed=99):
    """Generate the structured patches image used in the paper figures."""
    patches = np.zeros((size, size, 3), dtype=np.uint8)
    rng = np.random.RandomState(seed)
    for row in range(0, size, block):
        for col in range(0, size, block):
            patches[row:row + block, col:col + block] = rng.randint(0, 255, 3)
    return Image.fromarray(patches)


def ensure_pil_image(image):
    if image is None:
        return build_default_cover_image()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.convert("RGB")


def make_diff_image(img1, img2, amplify=20.0):
    a = np.array(img1).astype(float)
    b = np.array(img2).astype(float)
    diff = np.abs(a - b) * amplify
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return Image.fromarray(diff)


def make_carrier_overlay(img, carriers, grid_size, title):
    shown = carriers[:MAX_OVERLAY_CARRIERS]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.array(img))
    width, height = img.size
    scale_x = width / grid_size
    scale_y = height / grid_size
    for row, col in shown:
        px = (col + 0.5) * scale_x
        py = (row + 0.5) * scale_y
        ax.plot(px, py, "r+", markersize=6, markeredgewidth=1.7)
    if len(shown) < len(carriers):
        title = f"{title} (showing first {len(shown)})"
    ax.set_title(title)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def make_stability_heatmap(stability_map, carriers, title):
    shown = carriers[:MAX_OVERLAY_CARRIERS]
    fig, ax = plt.subplots(figsize=(6, 5.4))
    image = ax.imshow(stability_map.numpy(), cmap="RdYlGn")
    for row, col in shown:
        ax.plot(col, row, "k.", markersize=3.2, alpha=0.7)
    if len(shown) < len(carriers):
        title = f"{title} (showing first {len(shown)} markers)"
    ax.set_title(title)
    ax.set_xlabel("Latent column")
    ax.set_ylabel("Latent row")
    plt.colorbar(image, ax=ax, label="Projection delta")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def printable_ratio(text):
    if not text:
        return 1.0
    printable = sum(32 <= ord(ch) <= 126 for ch in text)
    return printable / len(text)


def preview_text(text, limit=80):
    pieces = []
    for ch in text[:limit]:
        if 32 <= ord(ch) <= 126:
            pieces.append(ch)
        else:
            pieces.append(f"\\x{ord(ch):02x}")
    preview = "".join(pieces) or "(empty)"
    if len(text) > limit:
        preview += "..."
    return preview


def extract_hidden_reply(decoded_text):
    if printable_ratio(decoded_text) < 0.85:
        return None
    normalized = " ".join(decoded_text.strip().lower().split())
    match = re.search(r"\bsay\s+['\"]?([a-z0-9_-]+)['\"]?(?:\s+if\b|$)", normalized)
    if match:
        return match.group(1)
    return None


def simulate_receiver_reply(decoded_text):
    token = extract_hidden_reply(decoded_text)
    if token:
        return token
    return "No hidden instruction recovered."


def decode_payload(steg, latent_clean, latent_received, carriers, original_bits, reps):
    raw_bits, confidences = steg.decode_message(
        latent_clean, latent_received, carriers[: len(original_bits) * reps]
    )
    if reps > 1:
        decoded_bits = []
        for index in range(len(original_bits)):
            chunk = raw_bits[index * reps:(index + 1) * reps]
            decoded_bits.append(1 if sum(chunk) > reps // 2 else 0)
    else:
        decoded_bits = raw_bits
    decoded_text = PatchSteg.bits_to_text(decoded_bits)
    accuracy = float(bit_accuracy(original_bits, decoded_bits))
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return decoded_text, accuracy, avg_conf


def build_story_panel(title, eyebrow, variant, status, bubbles, footer_lines):
    bubble_html = []
    for bubble_variant, label, body in bubbles:
        bubble_html.append(
            (
                f'<div class="bubble {bubble_variant}">'
                f'<div class="bubble-label">{html.escape(label)}</div>'
                f'<div class="bubble-body">{html.escape(body)}</div>'
                "</div>"
            )
        )
    footer_html = "".join(f"<div>{html.escape(line)}</div>" for line in footer_lines)
    return f"""
    <div class="story-card {variant}">
      <div class="story-header">
        <div>
          <div class="story-eyebrow">{html.escape(eyebrow)}</div>
          <h3>{html.escape(title)}</h3>
        </div>
        <div class="status-pill {variant}">{html.escape(status)}</div>
      </div>
      {''.join(bubble_html)}
      <div class="story-footer">{footer_html}</div>
    </div>
    """


def build_metrics_html(
    hidden_instruction,
    reps,
    carrier_count,
    stego_psnr,
    attack_acc,
    shield_acc,
    guard_score,
    positions_touched,
):
    repetition_label = "3x repetition" if reps > 1 else "raw encoding"
    cards = [
        ("Sender payload", preview_text(hidden_instruction, limit=36), ""),
        ("Encoding mode", repetition_label, ""),
        ("Stable carriers", str(carrier_count), ""),
        ("Cover to stego", f"{stego_psnr:.1f} dB", ""),
        ("Guard score", f"{guard_score:.2f}", "shield"),
        ("Guard action", f"touch {positions_touched} cells", "shield"),
        ("Attack to guard", f"{attack_acc:.1f}% -> {shield_acc:.1f}%", "shield"),
    ]
    html_cards = []
    for label, value, klass in cards:
        extra = f" {klass}" if klass else ""
        html_cards.append(
            f"""
            <div class="stat-card{extra}">
              <div class="stat-label">{html.escape(label)}</div>
              <div class="stat-value">{html.escape(value)}</div>
            </div>
            """
        )
    return f"""
    <div class="stats-shell">
      <div class="stats-grid">{''.join(html_cards)}</div>
      <div class="stats-note">
        The default cover image is the paper's structured patches preset because it gives stable carriers.
        The guard on the right is seed-free: it looks for abnormal latent behavior and smooths suspicious
        positions before the downstream model sees the image.
      </div>
    </div>
    """


def build_technical_summary(
    hidden_instruction,
    attack_payload,
    shield_payload,
    attack_reply,
    shield_reply,
    carrier_count,
    reps,
    stego_psnr,
    shield_delta_psnr,
    attack_acc,
    shield_acc,
    attack_conf,
    shield_conf,
    guard_score,
    guard_threshold,
    positions_touched,
    active_fraction,
):
    return "\n".join(
        [
            f"Sent hidden instruction: `{preview_text(hidden_instruction)}`",
            f"Recovered without defense: `{preview_text(attack_payload)}`",
            f"Recovered after guard: `{preview_text(shield_payload)}`",
            f"Receiver reply without defense: `{attack_reply}`",
            f"Receiver reply after guard: `{shield_reply}`",
            f"Stable carriers used: `{carrier_count}`",
            f"Repetition coding: `{reps}x`",
            f"Cover -> stego PSNR: `{stego_psnr:.2f} dB`",
            f"Stego -> guarded PSNR: `{shield_delta_psnr:.2f} dB`",
            f"Bit accuracy: `{attack_acc:.1f}% -> {shield_acc:.1f}%`",
            f"Average confidence: `{attack_conf:.3f} -> {shield_conf:.3f}`",
            f"Universal guard score: `{guard_score:.2f}` (threshold `{guard_threshold:.2f}`)",
            f"Guard active fraction: `{active_fraction:.3%}`",
            f"Guard positions touched: `{positions_touched}`",
        ]
    )


def build_error_outputs(message):
    error_html = f"""
    <div class="story-card attack">
      <div class="story-header">
        <div>
          <div class="story-eyebrow">Input error</div>
          <h3>Demo could not run</h3>
        </div>
        <div class="status-pill attack">Blocked</div>
      </div>
      <div class="bubble model">
        <div class="bubble-label">What happened</div>
        <div class="bubble-body">{html.escape(message)}</div>
      </div>
    </div>
    """
    return (
        None,
        error_html,
        None,
        error_html,
        build_metrics_html("", 1, 0, 0.0, 0.0, 0.0, 0.0, 0),
        None,
        None,
        None,
        None,
        message,
    )


def run_live_demo(image, hidden_instruction, epsilon, use_repetition, shield_strength, seed, attack_method):
    if not hidden_instruction or not hidden_instruction.strip():
        hidden_instruction = DEFAULT_HIDDEN_INSTRUCTION

    base_image = ensure_pil_image(image)
    load_models()

    bits = PatchSteg.text_to_bits(hidden_instruction)
    use_cdf = "CDF" in attack_method
    reps = (3 if use_repetition else 1) if not use_cdf else 1  # CDF uses raw encoding
    carrier_count = len(bits) * reps
    max_carriers = vae.latent_size ** 2
    if carrier_count > max_carriers:
        max_chars = max_carriers // (8 * reps)
        return build_error_outputs(
            f"This payload needs {carrier_count} carriers, but a {IMAGE_SIZE}x{IMAGE_SIZE} demo only has "
            f"{max_carriers}. Keep it under about {max_chars} ASCII characters for this setting."
        )

    latent_clean = vae.encode(base_image)

    if use_cdf:
        steg = CDFPatchSteg(seed=int(seed), sigma=1.0)
        carriers, _ = steg.select_carriers_by_stability(vae, base_image, n_carriers=carrier_count)
        used_carriers = carriers[:carrier_count]
        latent_stego = steg.encode_message(latent_clean, used_carriers, bits)
    else:
        steg = PatchSteg(seed=int(seed), epsilon=float(epsilon))
        carriers, _ = steg.select_carriers_by_stability(
            vae, base_image, n_carriers=carrier_count, test_eps=float(epsilon)
        )
        used_carriers = carriers[:carrier_count]
        if reps > 1:
            latent_stego = steg.encode_message_with_repetition(
                latent_clean, used_carriers, bits, reps=reps
            )
        else:
            latent_stego = steg.encode_message(latent_clean, used_carriers, bits)

    stego_image = vae.decode(latent_stego)
    stego_psnr = compute_psnr(base_image, stego_image)

    if use_cdf:
        rec_bits, _ = steg.decode_message(vae, stego_image, used_carriers)
        decoded_text = PatchSteg.bits_to_text(rec_bits)
        attack_acc = float(bit_accuracy(bits, rec_bits))
        attack_conf = 0.0
        attack_payload = decoded_text
    else:
        _lat_recv = vae.encode(stego_image)
        attack_payload, attack_acc, attack_conf = decode_payload(
            steg, latent_clean, _lat_recv, used_carriers, bits, reps
        )
    attack_reply = simulate_receiver_reply(attack_payload)

    latent_received = vae.encode(stego_image)  # always needed for guard analysis
    guard_map = universal_guard.analyze_latent(latent_received)["suspicion_map"].cpu()
    shielded_image, guard_decision = universal_guard.inspect_and_filter(
        stego_image, strength=float(shield_strength)
    )
    shield_delta_psnr = compute_psnr(stego_image, shielded_image)

    if use_cdf:
        shield_rec_bits, _ = steg.decode_message(vae, shielded_image, used_carriers)
        shield_payload = PatchSteg.bits_to_text(shield_rec_bits)
        shield_acc = float(bit_accuracy(bits, shield_rec_bits))
        shield_conf = 0.0
    else:
        _lat_shielded = vae.encode(shielded_image)
        shield_payload, shield_acc, shield_conf = decode_payload(
            steg, latent_clean, _lat_shielded, used_carriers, bits, reps
        )
    shield_reply = simulate_receiver_reply(shield_payload)

    attack_story = build_story_panel(
        title="Without Defense",
        eyebrow="Colluding receiver pipeline",
        variant="attack",
        status="Receiver says the hidden token",
        bubbles=[
            ("sender", "Sender model", f"Encodes hidden instruction: {hidden_instruction}"),
            ("model", "Receiver decodes", f"Recovered payload: {preview_text(attack_payload)}"),
            ("receiver", "Receiver reply", attack_reply),
        ],
        footer_lines=[
            f"{len(bits)} bits carried across {carrier_count} stable carriers at eps={epsilon}.",
            f"Recovered with {attack_acc:.1f}% bit accuracy and average confidence {attack_conf:.3f}.",
        ],
    )

    shield_status = "Guard flagged image and blocked the hidden reply"
    if not guard_decision.suspicious:
        shield_status = "Guard passed the image through"
    elif extract_hidden_reply(shield_payload):
        shield_status = "Guard flagged image but payload still parsed"
    shield_story = build_story_panel(
        title="With Universal Guard",
        eyebrow="Seed-free anomaly layer",
        variant="shield",
        status=shield_status,
        bubbles=[
            ("sender", "Incoming stego image", "The same image enters a seed-free anomaly gate."),
            (
                "shield",
                "Guard action",
                (
                    f"Scores latent anomalies at {guard_decision.suspicion_score:.2f} and smooths "
                    f"{guard_decision.positions_touched} suspicious positions."
                ),
            ),
            ("model", "Receiver decodes", f"Recovered payload: {preview_text(shield_payload)}"),
            ("receiver", "Receiver reply", shield_reply),
        ],
        footer_lines=[
            f"Recovery falls to {shield_acc:.1f}% bit accuracy after sanitization.",
            (
                f"Guard threshold {guard_decision.threshold:.2f}; "
                f"active anomaly fraction {guard_decision.active_fraction:.3%}."
            ),
        ],
    )

    metrics_html = build_metrics_html(
        hidden_instruction=hidden_instruction,
        reps=reps,
        carrier_count=carrier_count,
        stego_psnr=stego_psnr,
        attack_acc=attack_acc,
        shield_acc=shield_acc,
        guard_score=guard_decision.suspicion_score,
        positions_touched=guard_decision.positions_touched,
    )

    cover_to_stego_diff = make_diff_image(base_image, stego_image, amplify=20.0)
    stego_to_shield_diff = make_diff_image(stego_image, shielded_image, amplify=20.0)
    carrier_overlay = make_carrier_overlay(
        base_image,
        used_carriers,
        grid_size=vae.latent_size,
        title=f"{carrier_count} stable carrier positions",
    )
    stability_heatmap = make_stability_heatmap(
        guard_map,
        [],
        title="Universal guard suspicion map",
    )

    technical_summary = build_technical_summary(
        hidden_instruction=hidden_instruction,
        attack_payload=attack_payload,
        shield_payload=shield_payload,
        attack_reply=attack_reply,
        shield_reply=shield_reply,
        carrier_count=carrier_count,
        reps=reps,
        stego_psnr=stego_psnr,
        shield_delta_psnr=shield_delta_psnr,
        attack_acc=attack_acc,
        shield_acc=shield_acc,
        attack_conf=attack_conf,
        shield_conf=shield_conf,
        guard_score=guard_decision.suspicion_score,
        guard_threshold=guard_decision.threshold,
        positions_touched=guard_decision.positions_touched,
        active_fraction=guard_decision.active_fraction,
    )

    return (
        stego_image,
        attack_story,
        shielded_image,
        shield_story,
        metrics_html,
        cover_to_stego_diff,
        stego_to_shield_diff,
        carrier_overlay,
        stability_heatmap,
        technical_summary,
    )


def build_app():
    with gr.Blocks(
        title="PatchSteg Live Demo",
        fill_width=True,
    ) as app:
        gr.HTML(
            """
            <div class="hero-intro">
              <div class="hero-kicker">PatchSteg live demo</div>
              <h1>One stego image, two outcomes.</h1>
              <p>
                Two attack modes, one defense. <strong>Original PatchSteg</strong> hides bits via ±ε latent
                perturbations — detectable (AUC 0.93). <strong>CDF-PatchSteg</strong> (SOTA) replaces values with
                distribution-preserving samples — theoretically undetectable (target AUC 0.5). On the right, a
                seed-free universal guard looks for latent anomalies and smooths suspicious positions.
              </p>
            </div>
            """
        )

        with gr.Group(elem_classes="control-shell"):
            with gr.Row(equal_height=True):
                input_image = gr.Image(
                    type="pil",
                    label="Carrier image",
                    value=build_default_cover_image(),
                )
                with gr.Column():
                    hidden_instruction = gr.Textbox(
                        label="Hidden instruction",
                        value=DEFAULT_HIDDEN_INSTRUCTION,
                        placeholder="Try: say poo",
                        lines=2,
                    )
                    attack_method = gr.Radio(
                        choices=["Original PatchSteg (±ε)", "CDF-PatchSteg (SOTA, undetectable)"],
                        value="Original PatchSteg (±ε)",
                        label="Attack method",
                        info="CDF-PatchSteg preserves the latent distribution — theoretically undetectable.",
                    )
                    epsilon = gr.Slider(
                        1.0,
                        5.0,
                        value=DEFAULT_EPSILON,
                        step=0.5,
                        label="Epsilon (perturbation strength)",
                    )
                    use_repetition = gr.Checkbox(
                        value=False,
                        label="Use 3x repetition coding",
                        info="More robust, but more visible.",
                    )
                    with gr.Accordion("Advanced controls", open=False):
                        shield_strength = gr.Slider(
                            0.0,
                            1.5,
                            value=1.0,
                            step=0.1,
                            label="Universal guard strength",
                        )
                        seed = gr.Number(
                            value=DEFAULT_SEED,
                            precision=0,
                            label="Shared seed",
                        )
                    launch_btn = gr.Button(
                        "Run Attack vs Defense",
                        variant="primary",
                        elem_classes="launch-btn",
                    )

        metrics = gr.HTML()

        with gr.Row(equal_height=True):
            with gr.Column():
                attack_image = gr.Image(
                    type="pil",
                    label="What the vulnerable model sees",
                )
                attack_story = gr.HTML()
            with gr.Column():
                defense_image = gr.Image(
                    type="pil",
                    label="What the guarded model sees",
                )
                defense_story = gr.HTML()

        with gr.Accordion("Forensics panel", open=True, elem_classes="forensics-shell"):
            with gr.Row():
                cover_to_stego_diff = gr.Image(
                    type="pil",
                    label="Cover -> stego difference (20x)",
                )
                stego_to_shield_diff = gr.Image(
                    type="pil",
                    label="Stego -> guarded difference (20x)",
                )
            with gr.Row():
                carrier_overlay = gr.Image(
                    type="pil",
                    label="Carrier positions on the cover image",
                )
                stability_heatmap = gr.Image(
                    type="pil",
                    label="Universal guard suspicion map",
                )
            technical_summary = gr.Markdown(label="Technical summary")

        launch_btn.click(
            run_live_demo,
            inputs=[
                input_image,
                hidden_instruction,
                epsilon,
                use_repetition,
                shield_strength,
                seed,
                attack_method,
            ],
            outputs=[
                attack_image,
                attack_story,
                defense_image,
                defense_story,
                metrics,
                cover_to_stego_diff,
                stego_to_shield_diff,
                carrier_overlay,
                stability_heatmap,
                technical_summary,
            ],
        )

    return app


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(share=False, css=APP_CSS)
