from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, sparse
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import spsolve

TILE = 4
SMAPE_EPS = 1e-8
LAYER_RECORD_BYTES = 8
ATOM_PAYLOAD_BYTES = 12
COLOR_AXIS_BYTES = 6
QUANT_LEVELS = 15


def tone(linear_rgb):
    x = np.maximum(linear_rgb, 0.0)
    return np.power(x / (1.0 + x), 1.0 / 2.2)


def untone(tone_rgb):
    t = np.clip(tone_rgb, 0.0, 1.0 - 1e-9)
    p = np.power(t, 2.2)
    return p / (1.0 - p)


def smape(prediction_tone, reference_tone, mask=None):
    if mask is not None and not mask.any():
        return None
    values = (
        2.0
        * np.abs(prediction_tone - reference_tone)
        / (np.abs(prediction_tone) + np.abs(reference_tone) + SMAPE_EPS)
    )
    if mask is not None:
        values = values[mask]
    return float(np.mean(values))


def psnr(prediction_tone, reference_tone, mask=None):
    error = prediction_tone - reference_tone
    if mask is not None:
        error = error[mask]
    mse = float(np.mean(error * error))
    if mse <= 0.0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def hole_mask(shader_rgb):
    return np.max(shader_rgb, axis=2) <= 0.0


def fill_holes(linear_rgb, holes, pad=8):
    filled = linear_rgb.copy()
    labels, count = ndimage.label(holes)
    height, width = holes.shape
    for label in range(1, count + 1):
        component = labels == label
        ys, xs = np.where(component)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(height, int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(width, int(xs.max()) + 1 + pad)

        window = filled[y0:y1, x0:x1]
        window_mask = component[y0:y1, x0:x1]
        unknowns = int(window_mask.sum())
        if unknowns == 0:
            continue

        index = -np.ones(window_mask.shape, dtype=np.int64)
        index[window_mask] = np.arange(unknowns)

        rows, columns, values = [], [], []
        rhs = np.zeros((unknowns, 3), dtype=np.float64)
        for y, x in zip(*np.where(window_mask)):
            equation = int(index[y, x])
            rows.append(equation)
            columns.append(equation)
            values.append(4.0)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                sy = min(max(y + dy, 0), window_mask.shape[0] - 1)
                sx = min(max(x + dx, 0), window_mask.shape[1] - 1)
                if window_mask[sy, sx]:
                    rows.append(equation)
                    columns.append(int(index[sy, sx]))
                    values.append(-1.0)
                else:
                    rhs[equation] += window[sy, sx]

        laplacian = sparse.csr_matrix(
            (values, (rows, columns)), shape=(unknowns, unknowns)
        )
        solution = np.column_stack(
            [spsolve(laplacian, rhs[:, channel]) for channel in range(3)]
        )
        window[window_mask] = solution
        filled[y0:y1, x0:x1] = window
    return filled


def _sample_shifted(linear_rgb, y, x, dx, dy):
    coordinates = np.array([y + dy, x + dx], dtype=np.float64)
    return np.column_stack(
        [
            map_coordinates(
                linear_rgb[..., channel], coordinates, order=1, mode="nearest"
            )
            for channel in range(3)
        ]
    )


def fit_layers(background_linear, oracle_linear, support, max_shift=0):
    core = background_linear.copy()
    oracle_tone = tone(oracle_linear)
    ys, xs = np.where(support)
    if len(xs) == 0:
        return core, 0

    height, width = support.shape
    y0 = (int(ys.min()) // TILE) * TILE
    x0 = (int(xs.min()) // TILE) * TILE
    y1 = min(((int(ys.max()) + TILE) // TILE) * TILE, height)
    x1 = min(((int(xs.max()) + TILE) // TILE) * TILE, width)

    shifts = [(0, 0)]
    if max_shift > 0:
        shifts = [
            (dx, dy)
            for dy in range(-max_shift, max_shift + 1)
            for dx in range(-max_shift, max_shift + 1)
        ]

    layer_count = 0
    for tile_y in range(y0, y1, TILE):
        for tile_x in range(x0, x1, TILE):
            local = support[
                tile_y : min(tile_y + TILE, height),
                tile_x : min(tile_x + TILE, width),
            ]
            if not np.any(local):
                continue

            local_y, local_x = np.where(local)
            gy = local_y + tile_y
            gx = local_x + tile_x
            target = oracle_linear[gy, gx]
            target_tone = oracle_tone[gy, gx]
            target_mean = target.mean(axis=0)

            best_error = None
            best_prediction = None
            for dx, dy in shifts:
                samples = _sample_shifted(
                    background_linear, gy.astype(np.float64), gx.astype(np.float64), dx, dy
                )
                sample_mean = samples.mean(axis=0)
                centered_samples = samples - sample_mean
                centered_target = target - target_mean

                denominator = float(np.sum(centered_samples * centered_samples))
                transmission = 0.0
                if denominator > 1e-12:
                    transmission = float(
                        np.sum(centered_samples * centered_target) / denominator
                    )
                transmission = float(np.float16(np.clip(transmission, 0.0, 1.0)))
                emission = np.asarray(
                    target_mean - transmission * sample_mean, dtype=np.float16
                ).astype(np.float64)

                prediction = np.maximum(emission + transmission * samples, 0.0)
                error = smape(tone(prediction), target_tone)
                if best_error is None or error < best_error:
                    best_error = error
                    best_prediction = prediction

            core[gy, gx] = best_prediction
            layer_count += 1
    return core, layer_count


def color_axis(core_tone, oracle_tone):
    residual = (oracle_tone - core_tone).reshape(-1, 3)
    stride = max(1, residual.shape[0] // 200000)
    sample = residual[::stride]
    _, _, right = np.linalg.svd(sample, full_matrices=False)
    axis = right[0]
    if float(axis.sum()) < 0.0:
        axis = -axis
    axis = np.asarray(axis, dtype=np.float16).astype(np.float64)
    return axis / np.linalg.norm(axis)


def image_to_tiles(image):
    height, width, channels = image.shape
    if height % TILE or width % TILE:
        raise ValueError("image dimensions must be divisible by the tile size")
    return (
        image.reshape(height // TILE, TILE, width // TILE, TILE, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, TILE * TILE, channels)
    )


def tiles_to_image(tiles, height, width):
    channels = tiles.shape[-1]
    return (
        tiles.reshape(height // TILE, width // TILE, TILE, TILE, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(height, width, channels)
    )


def build_atoms(core_tone, oracle_tone, axis):
    core_tiles = image_to_tiles(core_tone)
    oracle_tiles = image_to_tiles(oracle_tone)

    amplitudes = (oracle_tiles - core_tiles) @ axis
    scales = np.asarray(
        np.max(np.abs(amplitudes), axis=1) / QUANT_LEVELS, dtype=np.float16
    ).astype(np.float64)

    quantized = np.zeros_like(amplitudes)
    active = scales > 0.0
    quantized[active] = np.clip(
        np.rint(amplitudes[active] / scales[active, None]), -QUANT_LEVELS, QUANT_LEVELS
    )
    corrections = (
        quantized[..., None] * scales[:, None, None] * axis.reshape(1, 1, 3)
    )
    corrected = np.clip(core_tiles + corrections, 0.0, 1.0)

    def tile_cost(tiles):
        return np.sum(
            2.0
            * np.abs(tiles - oracle_tiles)
            / (np.abs(tiles) + np.abs(oracle_tiles) + SMAPE_EPS),
            axis=(1, 2),
        )

    gains = tile_cost(core_tiles) - tile_cost(corrected)
    order = np.argsort(-gains, kind="stable")
    return order, corrections.astype(np.float32), gains


def apply_atoms(core_tone, order, corrections, count):
    tiles = image_to_tiles(core_tone).copy()
    selected = order[:count]
    tiles[selected] = np.clip(tiles[selected] + corrections[selected], 0.0, 1.0)
    return tiles_to_image(tiles, *core_tone.shape[:2])


def elias_fano_bytes(universe, count):
    if count <= 0:
        return 0
    low_bits = max(0, int(math.floor(math.log2(universe / count))))
    bits = count * low_bits + count + math.ceil(universe / (2**low_bits))
    return math.ceil(bits / 8)


def encode(shader_linear, oracle_linear, fractions, max_shift=0):
    holes = hole_mask(shader_linear)
    background = fill_holes(shader_linear, holes) if holes.any() else shader_linear.copy()
    core_linear, layer_count = fit_layers(
        background, oracle_linear, holes, max_shift=max_shift
    )

    shader_tone = tone(shader_linear)
    oracle_tone = tone(oracle_linear)
    core_tone = tone(core_linear)

    axis = color_axis(core_tone, oracle_tone)
    order, corrections, gains = build_atoms(core_tone, oracle_tone, axis)

    universe = order.shape[0]
    core_bytes = layer_count * LAYER_RECORD_BYTES + COLOR_AXIS_BYTES

    rows = []
    images = {}
    for fraction in sorted(set(fractions)):
        count = int(round(fraction * universe))
        count = max(0, min(count, universe))
        output_tone = apply_atoms(core_tone, order, corrections, count)
        total_bytes = (
            core_bytes + ATOM_PAYLOAD_BYTES * count + elias_fano_bytes(universe, count)
        )
        rows.append(
            {
                "fraction": fraction,
                "atoms": count,
                "bytes": total_bytes,
                "bytes_per_megapixel": total_bytes
                / (shader_linear.shape[0] * shader_linear.shape[1] / 1e6),
                "smape_full": smape(output_tone, oracle_tone),
                "smape_hole": smape(output_tone, oracle_tone, holes),
                "smape_shaded": smape(output_tone, oracle_tone, ~holes),
                "psnr_full_db": psnr(output_tone, oracle_tone),
            }
        )
        images[fraction] = output_tone

    baseline = {
        "smape_full": smape(shader_tone, oracle_tone),
        "smape_hole": smape(shader_tone, oracle_tone, holes),
        "smape_shaded": smape(shader_tone, oracle_tone, ~holes),
        "psnr_full_db": psnr(shader_tone, oracle_tone),
        "bytes": 0,
    }

    return {
        "rows": rows,
        "images": images,
        "baseline": baseline,
        "shader_tone": shader_tone,
        "oracle_tone": oracle_tone,
        "core_tone": core_tone,
        "holes": holes,
        "axis": axis.tolist(),
        "layers": layer_count,
        "tiles": universe,
        "core_bytes": core_bytes,
        "gains": gains,
    }
