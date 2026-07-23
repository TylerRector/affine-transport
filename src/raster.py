from __future__ import annotations

import numpy as np
from PIL import Image

ERROR_SCALE = 0.1
ERROR_RAMP = (
    (0.98, 0.98, 0.97),
    (0.80, 0.83, 0.96),
    (0.42, 0.60, 0.90),
    (0.16, 0.36, 0.67),
    (0.05, 0.13, 0.32),
)


def save_tone_png(path, tone_rgb):
    pixels = np.clip(np.rint(tone_rgb * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def ramp_lookup(values):
    stops = np.asarray(ERROR_RAMP, dtype=np.float64)
    position = np.clip(values, 0.0, 1.0) * (len(stops) - 1)
    low = np.clip(np.floor(position).astype(int), 0, len(stops) - 2)
    weight = (position - low)[..., None]
    return stops[low] * (1.0 - weight) + stops[low + 1] * weight


def save_error_png(path, prediction_tone, reference_tone, scale=ERROR_SCALE):
    magnitude = np.mean(np.abs(prediction_tone - reference_tone), axis=2) / scale
    pixels = np.clip(np.rint(ramp_lookup(magnitude) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)
