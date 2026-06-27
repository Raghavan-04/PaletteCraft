"""
kmeans_palette.py — Dominant-colour extraction via K-Means clustering

Samples pixels from the calibrated image, clusters them in RGB space using
scikit-learn's KMeans, and returns the cluster centres ordered by frequency
(most dominant colour first).
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def extract_palette(
    arr: np.ndarray,
    n_colors: int = 5,
    sample_size: int = 8000,
    random_state: int = 42,
) -> list:
    """
    Extract the dominant colour palette from an image using K-Means.

    MiniBatchKMeans is used instead of full KMeans for speed on large images
    while producing nearly identical results.

    Args:
        arr         : uint8 RGB image array (H, W, 3)
        n_colors    : number of dominant colours to find
        sample_size : max number of pixels to use (random sample)
        random_state: reproducibility seed

    Returns:
        List of [R, G, B] int lists, sorted by cluster size (largest first).
        Each element is the representative colour of a dominant region.
    """
    pixels = arr.reshape(-1, 3).astype(np.float32)

    # Subsample for performance
    if len(pixels) > sample_size:
        idx = np.random.default_rng(random_state).choice(
            len(pixels), sample_size, replace=False
        )
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels

    n_colors = min(n_colors, len(pixels_sample))

    km = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=random_state,
        n_init=10,
        batch_size=min(1024, len(pixels_sample)),
    )
    km.fit(pixels_sample)

    centers = km.cluster_centers_          # (n_colors, 3)
    labels = km.labels_                    # (n_sample,)
    counts = np.bincount(labels, minlength=n_colors)
    order = np.argsort(-counts)            # descending by cluster size

    return [centers[i].clip(0, 255).astype(int).tolist() for i in order]


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