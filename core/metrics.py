"""Image quality and bit accuracy metrics."""
import torch
import numpy as np
import math
from PIL import Image

_LPIPS_MODEL = None


def compute_psnr(img1, img2):
    """PSNR between two [0,1] tensors or PIL images."""
    if isinstance(img1, Image.Image):
        img1 = torch.tensor(np.array(img1)).float() / 255.0
    if isinstance(img2, Image.Image):
        img2 = torch.tensor(np.array(img2)).float() / 255.0
    mse = torch.mean((img1.float() - img2.float()) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)


def compute_ssim_pil(img1: Image.Image, img2: Image.Image):
    """SSIM between two PIL images using skimage."""
    from skimage.metrics import structural_similarity as ssim
    a = np.array(img1)
    b = np.array(img2)
    return ssim(a, b, channel_axis=2, data_range=255)


def _pil_to_lpips_tensor(img: Image.Image, device='cpu'):
    arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device)


def compute_lpips_pil(img1: Image.Image, img2: Image.Image, device='cpu'):
    """LPIPS distance using the AlexNet backbone."""
    global _LPIPS_MODEL
    import lpips

    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    with torch.no_grad():
        x = _pil_to_lpips_tensor(img1, device=device)
        y = _pil_to_lpips_tensor(img2, device=device)
        return float(_LPIPS_MODEL(x, y).item())


def compute_quality_metrics(img1: Image.Image, img2: Image.Image, device='cpu'):
    """Return the paper-facing image-quality metrics for two PIL images."""
    return {
        'psnr': float(compute_psnr(img1, img2)),
        'ssim': float(compute_ssim_pil(img1, img2)),
        'lpips': float(compute_lpips_pil(img1, img2, device=device)),
    }


def bit_accuracy(sent, received):
    sent = np.array(sent)
    received = np.array(received)
    return np.mean(sent == received) * 100.0


def max_pixel_diff_pil(img1: Image.Image, img2: Image.Image):
    a = np.array(img1).astype(float)
    b = np.array(img2).astype(float)
    return np.max(np.abs(a - b))
