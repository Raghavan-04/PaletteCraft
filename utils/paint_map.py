# utils/paint_map.py
import numpy as np
from PIL import Image
from utils.color_utils import rgb_to_lab, delta_e_76

# ─── Constants used elsewhere ──────────────────────────────
NEON_COLOURS = {
    "Hot Pink": [255, 105, 180],
    "Neon Green": [57, 255, 20],
    "Electric Blue": [0, 255, 255],
    "Bright Yellow": [255, 255, 0],
}

def generate_paint_map(image_array, target_rgb, threshold=8.0,
                       overlay_color=[255, 0, 0], overlay_alpha=0.6, dim_factor=0.2):
    """Generate a neon overlay map for a single target colour (used in other parts)."""
    img_lab = rgb_to_lab(image_array)
    target_lab = rgb_to_lab(np.array([[target_rgb]]))[0][0]
    delta_e = delta_e_76(img_lab, target_lab)
    mask = delta_e < threshold

    gray = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
    canvas = np.stack((gray,)*3, axis=-1) * dim_factor
    canvas = canvas.astype(np.uint8)

    overlay = np.array(overlay_color, dtype=np.uint8)
    canvas[mask] = (1 - overlay_alpha) * canvas[mask] + overlay_alpha * overlay
    return Image.fromarray(canvas)

def get_match_percentage(image_array, target_rgb, threshold=8.0):
    """Compute what percentage of the image matches a target colour."""
    img_lab = rgb_to_lab(image_array)
    target_lab = rgb_to_lab(np.array([[target_rgb]]))[0][0]
    delta_e = delta_e_76(img_lab, target_lab)
    mask = delta_e < threshold
    return (np.sum(mask) / mask.size) * 100

def generate_layered_steps(image_array, palette, threshold=8.0):
    """
    Build a sequence of layers starting from a white canvas.
    Each step paints one palette colour (in order) on top of the previous,
    and the final step is the original image.
    """
    canvas = np.full_like(image_array, 255, dtype=np.uint8)
    img_lab = rgb_to_lab(image_array)
    steps = []

    for rgb_color in palette:
        target_lab = rgb_to_lab(np.array([[rgb_color]]))[0][0]
        de = delta_e_76(img_lab, target_lab)
        mask = de < threshold
        canvas[mask] = rgb_color
        steps.append(Image.fromarray(canvas.copy()))

    steps.append(Image.fromarray(image_array))   # final original image
    return steps