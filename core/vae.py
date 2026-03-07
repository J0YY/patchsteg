"""SD VAE encoding/decoding with proper scaling."""
import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
from torchvision import transforms


class StegoVAE:
    def __init__(self, device='cpu', image_size=256):
        self.device = device
        self.image_size = image_size
        self.latent_size = image_size // 8  # 32 for 256, 64 for 512
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            low_cpu_mem_usage=False,
        ).to(device).eval()
        self.scaling_factor = self.vae.config.scaling_factor  # 0.18215

        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """Image -> latent [1, 4, image_size/8, image_size/8]."""
        x = self.to_tensor(image).unsqueeze(0).to(self.device)
        latent = self.vae.encode(x).latent_dist.mean
        latent = latent * self.scaling_factor
        return latent

    @torch.no_grad()
    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Tensor in [-1,1] [1,3,512,512] -> latent [1,4,64,64]"""
        latent = self.vae.encode(x.to(self.device)).latent_dist.mean
        latent = latent * self.scaling_factor
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """Latent [1, 4, image_size/8, image_size/8] -> Image."""
        latent_scaled = latent / self.scaling_factor
        pixels = self.vae.decode(latent_scaled).sample
        pixels = (pixels.clamp(-1, 1) + 1) / 2  # -> [0, 1]
        pixels = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(pixels)

    @torch.no_grad()
    def decode_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent [1, 4, 64, 64] -> tensor [1, 3, 512, 512] in [-1, 1]"""
        latent_scaled = latent / self.scaling_factor
        pixels = self.vae.decode(latent_scaled).sample
        return pixels.clamp(-1, 1)

    @torch.no_grad()
    def round_trip(self, image: Image.Image):
        """Encode -> decode -> return both latent and reconstructed image."""
        latent = self.encode(image)
        recon = self.decode(latent)
        return latent, recon
