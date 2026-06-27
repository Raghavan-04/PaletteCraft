"""
color_utils.py — Core color science utilities for PaletteCraft
Handles RGB ↔ XYZ ↔ CIELAB conversions, Delta-E, and Kubelka-Munk subtractive mixing.
"""

import numpy as np


# ──────────────────────────────── Hex / RGB ───────────────────────────────────

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert '#RRGGBB' hex string to (R, G, B) int tuple (0-255)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(r, g, b) -> str:
    """Convert R, G, B integers (0-255) to '#rrggbb' hex string."""
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


# ──────────────────────────── Gamma / Linearisation ──────────────────────────

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB [0-1] → linear light [0-1] (remove gamma)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear light [0-1] → sRGB [0-1] (apply gamma)."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


def rgb_to_linear(rgb) -> np.ndarray:
    """Convert RGB uint8 array/list [0-255] to linear float [0-1]."""
    return _srgb_to_linear(np.asarray(rgb, dtype=float) / 255.0)


def linear_to_rgb255(linear: np.ndarray) -> np.ndarray:
    """Convert linear float [0-1] to uint8 [0-255]."""
    return np.clip(_linear_to_srgb(linear) * 255.0, 0, 255).astype(np.uint8)


# ──────────────────────────── RGB → XYZ → CIELAB ────────────────────────────

# sRGB to CIE XYZ D65 matrix
_M_RGB_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

# D65 white-point
_D65 = np.array([0.95047, 1.00000, 1.08883])


def _f_lab(t: np.ndarray) -> np.ndarray:
    """CIE Lab f() transfer function."""
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t), t / (3.0 * delta ** 2) + 4.0 / 29.0)


def rgb_to_lab(rgb) -> np.ndarray:
    """
    Convert a single RGB color [0-255] to CIELAB [L*, a*, b*].

    Args:
        rgb: array-like of length 3 (R, G, B)

    Returns:
        np.ndarray([L, a, b])
    """
    linear = _srgb_to_linear(np.asarray(rgb, dtype=float) / 255.0)
    xyz = _M_RGB_XYZ @ linear
    xyz_n = xyz / _D65
    f = _f_lab(xyz_n)
    L = 116.0 * f[1] - 16.0
    a = 500.0 * (f[0] - f[1])
    b = 200.0 * (f[1] - f[2])
    return np.array([L, a, b], dtype=float)


def rgb_image_to_lab(arr: np.ndarray) -> np.ndarray:
    """
    Vectorised RGB image (H, W, 3) uint8 → CIELAB (H, W, 3) float32.
    Optimised for full-image Delta-E comparisons.
    """
    h, w = arr.shape[:2]
    flat = arr.reshape(-1, 3).astype(np.float64) / 255.0

    linear = _srgb_to_linear(flat)                    # (N, 3)
    xyz = (linear @ _M_RGB_XYZ.T)                      # (N, 3)
    xyz_n = xyz / _D65                                  # (N, 3)
    f = _f_lab(xyz_n)                                   # (N, 3)

    L = 116.0 * f[:, 1] - 16.0
    a_ch = 500.0 * (f[:, 0] - f[:, 1])
    b_ch = 200.0 * (f[:, 1] - f[:, 2])

    lab = np.stack([L, a_ch, b_ch], axis=1).astype(np.float32)
    return lab.reshape(h, w, 3)


# ─────────────────────────────── Delta-E ─────────────────────────────────────

def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIE76 Euclidean Delta-E between two Lab colors."""
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))


def delta_e_image(lab_image: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
    """
    Vectorised CIE76 Delta-E between every pixel and a target Lab color.

    Args:
        lab_image : (H, W, 3) float32 array
        target_lab: (3,) float array [L, a, b]

    Returns:
        (H, W) float32 array of per-pixel Delta-E values
    """
    diff = lab_image - target_lab.astype(np.float32)
    return np.sqrt(np.sum(diff ** 2, axis=2))


# ─────────────────── Kubelka-Munk Subtractive Mixing ─────────────────────────

def km_mix_subtractive(colors: list, weights) -> np.ndarray:
    """
    Mix physical pigment colors using the Kubelka-Munk (K-M) theory.

    K-M models paint as having absorption (K) and scattering (S) coefficients.
    Per channel: K/S = (1 - R)² / (2R),   mix K/S linearly,
    then back to reflectance: R = 1 + K/S − √((K/S)² + 2·K/S)

    This gives a perceptually superior result to simple RGB averaging
    for subtractive (pigment) mixing.

    Args:
        colors : list of array-like RGB [0-255], one per pigment
        weights: list/array of mixing weights (need not sum to 1)

    Returns:
        uint8 np.ndarray [R, G, B] of the simulated mixture
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    km_channels = []
    for color in colors:
        lin = rgb_to_linear(color)
        lin = np.clip(lin, 1e-6, 1.0 - 1e-6)
        ks = (1.0 - lin) ** 2 / (2.0 * lin)   # per-channel K/S
        km_channels.append(ks)

    mixed_ks = sum(wi * ks for wi, ks in zip(w, km_channels))

    # Saunderson/K-M inversion
    mixed_lin = 1.0 + mixed_ks - np.sqrt(mixed_ks ** 2 + 2.0 * mixed_ks)
    mixed_lin = np.clip(mixed_lin, 0.0, 1.0)

    return linear_to_rgb255(mixed_lin)