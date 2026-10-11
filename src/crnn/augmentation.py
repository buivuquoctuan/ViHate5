"""Mild stochastic augmentation used only on training images.

Python random and NumPy's global RNG are deliberately used so the existing
trainer/worker seeds control every draw. Rotation expands its canvas and
never crops the input. CTC feasibility is checked by the training dataset.
"""

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


AUGMENTATION_SPEC = {
    "version": "light_v1", "contrast_factor": [0.85, 1.15],
    "blur_probability": 0.25, "blur_radius_pixels": [0.2, 0.7],
    "noise_probability": 0.35, "noise_std_pixels": [1.0, 4.0],
    "rotation_probability": 0.35, "rotation_degrees": [-3.0, 3.0],
    "rotation_expand": True, "rotation_fill": "white", "cropping": False,
    "random_sources": ["python_random", "numpy_global"],
}


class TrainAugmentation:
    """Apply seeded contrast, occasional blur/noise, and expanded rotation."""

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(*AUGMENTATION_SPEC["contrast_factor"]))
        if random.random() < AUGMENTATION_SPEC["blur_probability"]:
            image = image.filter(ImageFilter.GaussianBlur(
                random.uniform(*AUGMENTATION_SPEC["blur_radius_pixels"])))
        if random.random() < AUGMENTATION_SPEC["noise_probability"]:
            std = random.uniform(*AUGMENTATION_SPEC["noise_std_pixels"])
            pixels = np.asarray(image, dtype=np.float32)
            noise = np.random.normal(0.0, std, pixels.shape)
            image = Image.fromarray(np.clip(np.rint(pixels + noise), 0, 255).astype(np.uint8))
        if random.random() < AUGMENTATION_SPEC["rotation_probability"]:
            image = image.rotate(random.uniform(*AUGMENTATION_SPEC["rotation_degrees"]),
                                 resample=Image.Resampling.BICUBIC,
                                 expand=True, fillcolor=(255, 255, 255))
        return image
