"""
mixing_engine.py — Subtractive Paint Mixing Recipe Calculator

Algorithm:
  1. Convert target colour to CIELAB (perceptually uniform space).
  2. Pre-filter inventory to the N closest colours to target (prunes search space).
  3. For every combination of 1, 2, or 3 paints from the pre-filtered set:
       - Use scipy SLSQP to minimise ΔE₇₆ between Kubelka-Munk mixture and target.
       - Multiple starting points are tried for robustness.
  4. Return the globally best recipe with human-readable parts notation.

Kubelka-Munk (K-M) subtractive mixing is significantly more accurate than
simple RGB averaging for physical pigment prediction.
"""

import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from typing import Optional

from .color_utils import rgb_to_lab, delta_e_76, km_mix_subtractive


# ─────────────────────────── Optimisation helpers ────────────────────────────

def _objective(w: np.ndarray, colors: list, target_lab: np.ndarray) -> float:
    """Objective: ΔE between K-M mixture and target colour."""
    w = np.abs(w)
    total = w.sum()
    if total < 1e-9:
        return 1000.0
    mixed_rgb = km_mix_subtractive(colors, w)
    mixed_lab = rgb_to_lab(mixed_rgb)
    return delta_e_76(mixed_lab, target_lab)


def _optimise_weights(colors: list, target_lab: np.ndarray) -> tuple:
    """
    Find optimal mixing weights for a given set of pigments.

    Tries several starting points (uniform + per-colour dominant) and keeps
    the result with the lowest ΔE.

    Returns:
        (weights np.ndarray, best_delta_e float)
    """
    n = len(colors)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    # Starting guesses: uniform blend + each colour dominant
    starts = [np.full(n, 1.0 / n)]
    for i in range(n):
        w0 = np.full(n, 0.05 / max(n - 1, 1))
        w0[i] = 0.85
        starts.append(w0 / w0.sum())

    best_w = starts[0]
    best_de = float("inf")

    for x0 in starts:
        try:
            res = minimize(
                _objective,
                x0,
                args=(colors, target_lab),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-7},
            )
            if res.fun < best_de:
                best_de = res.fun
                best_w = res.x
        except Exception:
            pass

    best_w = np.abs(best_w)
    s = best_w.sum()
    if s > 0:
        best_w /= s

    return best_w, best_de


# ─────────────────────────── Parts conversion ────────────────────────────────

def _weights_to_parts(weights: np.ndarray, resolution: float = 0.25) -> list:
    """
    Convert fractional weights to painter-friendly part counts.

    Normalises so the dominant colour = 4 parts max, then rounds to
    nearest `resolution` (default 0.25 = ¼ part).
    """
    w = np.array(weights, dtype=float)
    if w.max() < 1e-9:
        return [1.0] * len(w)
    # Scale so largest is 4 parts
    w = w / w.max() * 4.0
    # Round to resolution
    w = np.round(w / resolution) * resolution
    return [max(resolution, float(p)) for p in w]


# ─────────────────────────── Public interface ─────────────────────────────────

def find_best_recipe(
    target_rgb: list,
    inventory: list,
    max_colors: int = 3,
    prefilter_n: int = 7,
) -> Optional[dict]:
    """
    Find the best subtractive mixing recipe from the user's paint inventory.

    Args:
        target_rgb  : [R, G, B] target colour (0-255)
        inventory   : list of paint dicts — each must have 'name', 'hex', 'rgb'
        max_colors  : maximum pigments to mix (1-3)
        prefilter_n : only consider the N closest inventory colours
                      (drastically reduces combinations for large inventories)

    Returns:
        dict with:
            paints       — list of {name, hex, rgb, parts, weight}
            delta_e      — perceptual colour difference (lower = better)
            mixed_rgb    — [R, G, B] of the simulated K-M mixture
            match_quality— "Excellent ✅" / "Good 🟢" / "Fair 🟡" / "Poor 🔴"
        or None if inventory is empty.
    """
    if not inventory or not target_rgb:
        return None

    target_lab = rgb_to_lab(target_rgb)

    # ── Pre-filter: keep the `prefilter_n` closest single-colour matches ──
    scored = []
    for p in inventory:
        de = delta_e_76(rgb_to_lab(p["rgb"]), target_lab)
        scored.append((de, p))
    scored.sort(key=lambda x: x[0])
    candidates = [p for _, p in scored[: min(prefilter_n, len(scored))]]

    inv_colors = [p["rgb"] for p in candidates]

    best_recipe: Optional[dict] = None
    best_de = float("inf")

    for n in range(1, min(max_colors, len(candidates)) + 1):
        for combo_idx in combinations(range(len(candidates)), n):
            combo_colors = [inv_colors[i] for i in combo_idx]

            if n == 1:
                weights = np.array([1.0])
                mixed_rgb = np.asarray(combo_colors[0])
                de = delta_e_76(rgb_to_lab(mixed_rgb.tolist()), target_lab)
            else:
                weights, de = _optimise_weights(combo_colors, target_lab)
                mixed_rgb = km_mix_subtractive(combo_colors, weights)

            if de < best_de:
                best_de = de
                parts = _weights_to_parts(weights)

                paint_list = []
                for rank, cidx in enumerate(combo_idx):
                    paint = candidates[cidx]
                    paint_list.append({
                        "name"  : paint["name"],
                        "hex"   : paint["hex"],
                        "rgb"   : paint["rgb"],
                        "parts" : parts[rank],
                        "weight": float(weights[rank]),
                    })

                if n == 1:
                    final_rgb = list(combo_colors[0])
                else:
                    final_rgb = km_mix_subtractive(combo_colors, weights).tolist()

                if de < 5:
                    quality = "Excellent ✅"
                elif de < 10:
                    quality = "Good 🟢"
                elif de < 20:
                    quality = "Fair 🟡"
                else:
                    quality = "Poor 🔴"

                best_recipe = {
                    "paints"       : paint_list,
                    "delta_e"      : float(best_de),
                    "mixed_rgb"    : final_rgb,
                    "match_quality": quality,
                }

    return best_recipe