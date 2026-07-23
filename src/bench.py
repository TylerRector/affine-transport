from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from codec import encode
from raster import save_error_png, save_tone_png

FRACTIONS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.0926, 0.1235, 0.2)
SHIP_FRACTION = 0.0926
SCENES = ("kitchen", "cornell")


def run_scene(scene, data_dir, figure_dir, fractions):
    shader = np.load(data_dir / f"{scene}_shader_rgb_f32.npy").astype(np.float64)
    oracle = np.load(data_dir / f"{scene}_oracle_rgb_f32.npy").astype(np.float64)

    started = time.time()
    result = encode(shader, oracle, fractions)
    elapsed = time.time() - started

    shipped = result["images"][SHIP_FRACTION]
    save_tone_png(figure_dir / f"{scene}_shader.png", result["shader_tone"])
    save_tone_png(figure_dir / f"{scene}_core.png", result["core_tone"])
    save_tone_png(figure_dir / f"{scene}_model.png", shipped)
    save_tone_png(figure_dir / f"{scene}_oracle.png", result["oracle_tone"])
    save_error_png(
        figure_dir / f"{scene}_error_shader.png",
        result["shader_tone"],
        result["oracle_tone"],
    )
    save_error_png(
        figure_dir / f"{scene}_error_model.png", shipped, result["oracle_tone"]
    )

    height, width = shader.shape[:2]
    return {
        "scene": scene,
        "resolution": [int(width), int(height)],
        "megapixels": width * height / 1e6,
        "hole_fraction": float(result["holes"].mean()),
        "tiles": int(result["tiles"]),
        "layers": int(result["layers"]),
        "core_bytes": int(result["core_bytes"]),
        "color_axis": result["axis"],
        "encode_seconds": elapsed,
        "baseline": result["baseline"],
        "rate_curve": result["rows"],
        "shipped": next(
            row for row in result["rows"] if row["fraction"] == SHIP_FRACTION
        ),
    }


def main():
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--figures", type=Path, default=root / "figures")
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    report = {
        "metric": "mean(2*|p-r|/(|p|+|r|+1e-8)) on tone((x/(1+x))^(1/2.2))",
        "ship_fraction": SHIP_FRACTION,
        "fractions": list(FRACTIONS),
        "scenes": [],
    }
    for scene in args.scenes:
        entry = run_scene(scene, args.data, args.figures, FRACTIONS)
        report["scenes"].append(entry)
        print(
            scene,
            "hole",
            round(entry["hole_fraction"], 4),
            "shader sMAPE",
            round(entry["baseline"]["smape_full"], 4),
            "model sMAPE",
            round(entry["shipped"]["smape_full"], 4),
            "bytes",
            entry["shipped"]["bytes"],
        )

    (args.results / "benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "scene,fraction,atoms,bytes,bytes_per_megapixel,smape_full,smape_hole,smape_shaded,psnr_full_db"
    ]
    for entry in report["scenes"]:
        for row in entry["rate_curve"]:
            lines.append(
                f"{entry['scene']},{row['fraction']},{row['atoms']},{row['bytes']},"
                f"{row['bytes_per_megapixel']:.1f},{row['smape_full']:.6f},"
                f"{row['smape_hole']:.6f},{row['smape_shaded']:.6f},"
                f"{row['psnr_full_db']:.4f}"
            )
    (args.results / "rate_curve.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
