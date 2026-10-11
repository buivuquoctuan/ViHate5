"""Deterministic, canvas-preserving corruptions for paired robustness tests."""

from numbers import Integral

import numpy as np
from PIL import Image, ImageFilter

CORRUPTION_KINDS = ("gaussian_blur", "perspective", "wave")
CORRUPTION_SPEC = {
    "version": "canvas-preserving-v1",
    "gaussian_blur": {"radius_pixels_by_severity": [0.7, 1.2, 2.0]},
    "perspective": {
        "horizontal_margin_fraction_by_severity": [0.04, 0.08, 0.12],
        "vertical_margin_fraction_by_severity": [0.05, 0.10, 0.15],
        "corner_margin_multipliers": [[1.0, 1.0], [0.25, 1.5], [1.0, 0.25], [0.5, 1.0]],
        "resampling": "bicubic", "fill": "opaque white",
    },
    "wave": {
        "amplitude_height_fraction_by_severity": [0.04, 0.07, 0.10],
        "cycles": 1, "vertical_padding_pixels": 0.5,
        "resampling": "vertical bilinear", "fill": "opaque white",
    },
}


def _white(image: Image.Image) -> int | tuple[int, ...]:
    if image.mode == "L":
        return 255
    return (255,) * len(image.getbands())


def _perspective(image: Image.Image, severity: int) -> Image.Image:
    width, height = image.size
    if width < 3 or height < 3:
        return image.copy()
    mx = (width - 1) * CORRUPTION_SPEC["perspective"]["horizontal_margin_fraction_by_severity"][severity - 1]
    my = (height - 1) * CORRUPTION_SPEC["perspective"]["vertical_margin_fraction_by_severity"][severity - 1]
    source = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    # All four corners are inside the output canvas; no part of the word is cut.
    destination = [(mx, my), (width - 1 - mx * 0.25, my * 1.5),
                   (width - 1 - mx, height - 1 - my * 0.25),
                   (mx * 0.5, height - 1 - my)]
    matrix, target = [], []
    for (x, y), (u, v) in zip(destination, source):
        matrix.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y],
                       [0, 0, 0, x, y, 1, -v * x, -v * y]])
        target.extend([u, v])
    coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(target))
    return image.transform(image.size, Image.Transform.PERSPECTIVE,
                           tuple(coefficients), resample=Image.Resampling.BICUBIC,
                           fillcolor=_white(image))


def _wave(image: Image.Image, severity: int) -> Image.Image:
    width, height = image.size
    if height < 4 or width < 2:
        return image.copy()
    amplitude = (height - 1) * CORRUPTION_SPEC["wave"]["amplitude_height_fraction_by_severity"][severity - 1]
    margin = amplitude + 0.5
    scale = (height - 1 - 2 * margin) / (height - 1)
    # First compress vertically, then displace. The full input stays in canvas.
    displacement = amplitude * np.sin(np.linspace(0.0, 2 * np.pi, width))
    source_y = (np.arange(height)[:, None] - margin - displacement[None, :]) / scale
    valid = (source_y >= 0) & (source_y <= height - 1)
    clipped = np.clip(source_y, 0, height - 1)
    low = clipped.astype(np.intp)
    high = np.minimum(low + 1, height - 1)
    weight = clipped - low
    pixels = np.asarray(image, dtype=np.float32)
    x = np.arange(width)[None, :]
    if pixels.ndim == 3:
        weight = weight[..., None]
        valid = valid[..., None]
    sampled = pixels[low, x] * (1 - weight) + pixels[high, x] * weight
    output = np.where(valid, sampled, 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)


def apply_corruption(image: Image.Image, kind: str, severity: int) -> Image.Image:
    """Apply a reproducible corruption at severity 1, 2, or 3.

    Gaussian radii are 0.7, 1.2, and 2.0 pixels at the original image size.
    Geometric transforms inset all content with an opaque white background.
    Evaluate these synthetic corruptions as paired tests, not natural labels.
    """
    if kind not in CORRUPTION_KINDS:
        raise ValueError(f"Unknown corruption {kind!r}; choose from {CORRUPTION_KINDS}")
    if isinstance(severity, bool) or not isinstance(severity, Integral) or severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    if image.mode not in ("L", "RGB", "RGBA"):
        image = image.convert("RGB")
    if kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(CORRUPTION_SPEC["gaussian_blur"]["radius_pixels_by_severity"][severity - 1]))
    if kind == "perspective":
        return _perspective(image, severity)
    return _wave(image, severity)
