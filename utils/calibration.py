"""
calibration.py — Automated Color Constancy / Auto White Balance Engine

Implements:
  1. Gray-World Assumption  (primary)
  2. Max-RGB / White-Patch  (fallback)
  3. Combined hybrid mode

These eliminate lighting-cast bias from the inspiration photo so the
extracted palette reflects true pigment hues rather than ambient light colour.
"""

import numpy as np
from PIL import Image


# ──────────────────────────── AWB algorithms ─────────────────────────────────

def gray_world_awb(arr: np.ndarray) -> np.ndarray:
    """
    Gray-World Assumption AWB.

    Assumes the average reflectance of all surfaces in a natural scene is
    achromatic (neutral gray).  Scale each channel so that
        R_avg = G_avg = B_avg = (R_avg + G_avg + B_avg) / 3
    """
    arr = arr.astype(np.float64)
    r_avg, g_avg, b_avg = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    overall = (r_avg + g_avg + b_avg) / 3.0

    # Guard against division by zero for very dark channels
    sr = overall / r_avg if r_avg > 1 else 1.0
    sg = overall / g_avg if g_avg > 1 else 1.0
    sb = overall / b_avg if b_avg > 1 else 1.0

    out = arr.copy()
    out[:, :, 0] = np.clip(arr[:, :, 0] * sr, 0, 255)
    out[:, :, 1] = np.clip(arr[:, :, 1] * sg, 0, 255)
    out[:, :, 2] = np.clip(arr[:, :, 2] * sb, 0, 255)
    return out.astype(np.uint8)


def max_rgb_awb(arr: np.ndarray) -> np.ndarray:
    """
    Max-RGB / White-Patch AWB.

    Assumes the brightest patch in the scene is the illuminant.
    Scales each channel to its maximum value so white patches become white.
    """
    arr = arr.astype(np.float64)
    max_r = arr[:, :, 0].max()
    max_g = arr[:, :, 1].max()
    max_b = arr[:, :, 2].max()
    global_max = max(max_r, max_g, max_b, 1)

    sr = global_max / max_r if max_r > 0 else 1.0
    sg = global_max / max_g if max_g > 0 else 1.0
    sb = global_max / max_b if max_b > 0 else 1.0

    out = arr.copy()
    out[:, :, 0] = np.clip(arr[:, :, 0] * sr, 0, 255)
    out[:, :, 1] = np.clip(arr[:, :, 1] * sg, 0, 255)
    out[:, :, 2] = np.clip(arr[:, :, 2] * sb, 0, 255)
    return out.astype(np.uint8)


def combined_awb(arr: np.ndarray, gw_weight: float = 0.7) -> np.ndarray:
    """
    Hybrid AWB: weighted blend of Gray-World and Max-RGB results.
    Gray-World (70% default) is more natural; Max-RGB adds punch.
    """
    gw = gray_world_awb(arr).astype(np.float64)
    mr = max_rgb_awb(arr).astype(np.float64)
    blended = gw_weight * gw + (1.0 - gw_weight) * mr
    return np.clip(blended, 0, 255).astype(np.uint8)


# ──────────────────────────── Public interface ────────────────────────────────

def auto_white_balance(arr: np.ndarray, method: str = "gray_world") -> np.ndarray:
    """
    Automated AWB with smart fallback.

    Args:
        arr   : uint8 RGB image array (H, W, 3)
        method: "gray_world" | "max_rgb" | "combined"

    Returns:
        White-balanced uint8 RGB array (H, W, 3)
    """
    if method == "max_rgb":
        return max_rgb_awb(arr)
    elif method == "combined":
        return combined_awb(arr)
    else:
        result = gray_world_awb(arr)
        # Sanity check: if any channel becomes wildly unbalanced after correction,
        # fall back to Max-RGB.
        avgs = [result[:, :, c].mean() for c in range(3)]
        ratio = max(avgs) / (min(avgs) + 1e-6)
        if ratio > 4.0:          # heuristic threshold
            return max_rgb_awb(arr)
        return result


def get_image_array(image: Image.Image, max_size: int = 1200) -> np.ndarray:
    """
    Convert PIL Image to RGB uint8 numpy array, resizing large images for
    performance while preserving aspect ratio.

    Args:
        image   : PIL Image (any mode)
        max_size: max width or height in pixels

    Returns:
        uint8 numpy array (H, W, 3)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return np.array(image, dtype=np.uint8)