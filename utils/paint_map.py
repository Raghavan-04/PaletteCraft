"""
paint_map.py — Dynamic Paint-Map Visualiser

Scans every pixel of the calibrated image, computes CIE76 ΔE against the
selected target colour, and returns a composite image where:
  • Matching pixels (ΔE ≤ threshold) receive a vivid neon overlay
  • Non-matching pixels are desaturated / dimmed for contrast

All operations are fully vectorised with NumPy for performance.
"""

import numpy as np
from PIL import Image

from .color_utils import rgb_image_to_lab, delta_e_image, rgb_to_lab


# ──────────────────────────── Image helpers ───────────────────────────────────

def _desaturate(arr: np.ndarray, factor: float) -> np.ndarray:
    """
    Blend image with its grayscale version.

    factor=0 → fully grayscale | factor=1 → original colour
    """
    # BT.601 luma weights
    gray = (
        0.299 * arr[:, :, 0].astype(float) +
        0.587 * arr[:, :, 1].astype(float) +
        0.114 * arr[:, :, 2].astype(float)
    )
    gray_rgb = np.stack([gray, gray, gray], axis=2)
    blended = factor * arr.astype(float) + (1.0 - factor) * gray_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


# ─────────────────────────── Public interface ─────────────────────────────────

# Preset neon highlight colours (RGB)
NEON_COLOURS = {
    "Neon Pink"  : (255, 20,  147),
    "Neon Green" : (57,  255,  20),
    "Neon Yellow": (255, 255,   0),
    "Neon Cyan"  : (0,   255, 255),
    "Neon Orange": (255, 100,   0),
    "Neon Purple": (180,  0,  255),
}


def generate_paint_map(
    arr: np.ndarray,
    target_rgb: list,
    threshold: float = 8.0,
    overlay_color: tuple = (255, 20, 147),   # Neon Pink
    overlay_alpha: float = 0.65,
    dim_factor: float = 0.25,
    max_size: int = 800,
) -> Image.Image:
    """
    Generate a paint-map overlay image.

    Pixels whose ΔE to `target_rgb` is within `threshold` get a vivid colour
    overlay; all others are desaturated to draw the artist's eye to the
    target areas.

    Args:
        arr          : uint8 RGB image array (H, W, 3)
        target_rgb   : [R, G, B] reference colour
        threshold    : ΔE76 cutoff for a "match"  (try 5-15)
        overlay_color: RGB tuple for the neon highlight
        overlay_alpha: blending weight for the overlay on matched pixels (0-1)
        dim_factor   : saturation level for non-matched pixels (0=grey, 1=full)
        max_size     : resize image to this max dimension before processing

    Returns:
        PIL.Image.Image — the composited paint map
    """
    # ── Resize for performance ──────────────────────────────────────────────
    h, w = arr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        pil_tmp = Image.fromarray(arr).resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )
        work = np.array(pil_tmp, dtype=np.uint8)
    else:
        work = arr.copy()

    # ── Lab conversion (vectorised) ─────────────────────────────────────────
    lab_img = rgb_image_to_lab(work)                     # (H, W, 3) float32
    target_lab = rgb_to_lab(target_rgb).astype(np.float32)

    # ── Compute per-pixel ΔE ───────────────────────────────────────────────
    de_map = delta_e_image(lab_img, target_lab)          # (H, W) float32
    mask = de_map <= threshold                            # (H, W) bool

    # ── Build composite ────────────────────────────────────────────────────
    # Start with desaturated base
    base = _desaturate(work, dim_factor).astype(float)

    # Apply neon overlay to matched pixels
    neon = np.array(overlay_color, dtype=float)
    overlay_contrib = overlay_alpha * neon + (1.0 - overlay_alpha) * work[mask].astype(float)
    base[mask] = overlay_contrib

    result = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def get_match_percentage(
    arr: np.ndarray,
    target_rgb: list,
    threshold: float = 8.0,
) -> float:
    """
    Return the percentage of image pixels that match the target colour within ΔE threshold.

    Args:
        arr       : uint8 RGB image array (H, W, 3)
        target_rgb: [R, G, B] reference colour
        threshold : ΔE76 cutoff

    Returns:
        float (0-100) representing coverage percentage
    """
    # Subsample for speed
    pixels = arr.reshape(-1, 3)
    if len(pixels) > 20000:
        idx = np.random.default_rng(0).choice(len(pixels), 20000, replace=False)
        pixels = pixels[idx]

    # Compute Lab for subset
    h = len(pixels)
    tmp = pixels.reshape(h, 1, 3)
    lab = rgb_image_to_lab(tmp)[:, 0, :]                # (N, 3)
    target_lab = rgb_to_lab(target_rgb).astype(np.float32)

    diff = lab - target_lab
    de = np.sqrt(np.sum(diff ** 2, axis=1))
    matched = (de <= threshold).sum()
    return float(matched) / len(de) * 100.0