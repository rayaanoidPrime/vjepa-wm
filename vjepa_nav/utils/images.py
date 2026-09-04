"""Image helpers for visualization / evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """(C, H, W) or (B, C, H, W) in [0,1]-normalized space -> uint8 tensor."""
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    y = x.detach() * std + mean
    y = torch.clamp(y, 0.0, 1.0)
    return y


def to_pil(x: torch.Tensor):
    from PIL import Image

    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = (denormalize(x) * 255.0).clamp(0, 255).to(torch.uint8)
    return [Image.fromarray(t.permute(1, 2, 0).cpu().numpy()) for t in x]


def save_tile(rows: list[list[torch.Tensor]], title: str, out_dir: Path, fname: str = "sample.png") -> Path:
    """rows: list of rows; each row is a list of (C,H,W) tensors."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pads = [max(r) for r in zip(*[(t.shape[1], t.shape[2]) for row in rows for t in row])]
    H, W = pads
    cols = max(len(r) for r in rows)
    grid = Image.new("RGB", (W * cols, H * len(rows) + 24), (255, 255, 255))
    from PIL import ImageDraw

    d = ImageDraw.Draw(grid)
    d.text((8, 4), title, fill=(0, 0, 0))
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            im = to_pil(t)[0].resize((W, H))
            grid.paste(im, (c * W, 24 + r * H))
    path = out_dir / fname
    grid.save(path)
    return path


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(max_val**2 / mse)))


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    import math

    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w.expand(1, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    import torch.nn.functional as F

    a = a.float().unsqueeze(0) if a.ndim == 3 else a.float()
    b = b.float().unsqueeze(0) if b.ndim == 3 else b.float()
    win = _gaussian_window().to(a.device)
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    mu_a = F.conv2d(a, win, padding=11 // 2)
    mu_b = F.conv2d(b, win, padding=11 // 2)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    sig_a2 = F.conv2d(a * a, win, padding=11 // 2) - mu_a2
    sig_b2 = F.conv2d(b * b, win, padding=11 // 2) - mu_b2
    sig_ab = F.conv2d(a * b, win, padding=11 // 2) - mu_ab
    s = ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    )
    return float(s.mean().item())
