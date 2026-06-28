"""
kmeans_palette.py — Dominant-colour extraction via K-Means clustering

Samples pixels from the calibrated image, clusters them in RGB space using
scikit-learn's KMeans, and returns the cluster centres ordered by frequency
(most dominant colour first).
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans

def extract_palette(image_array, n_colors=5):
    """
    Extract n dominant colours using K‑Means.
    Returns colours sorted by frequency (most common first).
    """
    arr = image_array.reshape(-1, 3).astype(np.float32)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(arr)

    labels = kmeans.labels_
    counts = np.bincount(labels)

    # Sort cluster indices by frequency descending
    sorted_idx = np.argsort(counts)[::-1]
    centers = kmeans.cluster_centers_[sorted_idx]

    palette = [tuple(map(int, center)) for center in centers]
    return palette

def get_pixel_color(arr: np.ndarray, x: int, y: int) -> list:
    """
    Return the [R, G, B] colour at pixel coordinate (x, y).

    Coordinates are clamped to valid image bounds.

    Args:
        arr: uint8 RGB image array (H, W, 3)
        x  : column (horizontal) index from left
        y  : row    (vertical)   index from top

    Returns:
        [R, G, B] int list
    """
    h, w = arr.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    return arr[y, x].tolist()

def sort_palette_by_saturation(palette):
    """
    Sort RGB colours by saturation (higher saturation first).
    Saturation = (max(R,G,B) - min(R,G,B)) / max(R,G,B)
    """
    def saturation(rgb):
        r, g, b = rgb
        mx = max(r, g, b)
        mn = min(r, g, b)
        return (mx - mn) / (mx + 1e-6)
    return sorted(palette, key=saturation, reverse=True)