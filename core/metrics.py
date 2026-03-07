"""Image quality and bit accuracy metrics."""
import torch
import numpy as np
import math
from PIL import Image


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


def bit_accuracy(sent, received):
    sent = np.array(sent)
    received = np.array(received)
    return np.mean(sent == received) * 100.0


def max_pixel_diff_pil(img1: Image.Image, img2: Image.Image):
    a = np.array(img1).astype(float)
    b = np.array(img2).astype(float)
    return np.max(np.abs(a - b))
