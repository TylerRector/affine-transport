from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, ScalarFormatter
from PIL import Image

from raster import ERROR_RAMP, ERROR_SCALE

SERIES = {"kitchen": "#2a78d6", "cornell": "#eb6834"}
INK = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e2e1dc"
COLUMNS = ("model", "oracle", "error_model")
COLUMN_TITLES = ("Model", "Ray-traced oracle", "Absolute difference")
ERROR_CMAP = LinearSegmentedColormap.from_list("error", ERROR_RAMP)


def load_panel(figure_dir, scene, name):
    return np.asarray(Image.open(figure_dir / f"{scene}_{name}.png"))


def style_axis(axis, title=None):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    if title:
        axis.set_title(title, fontsize=11, color=INK, pad=6)


def add_error_bar(figure, axis):
    bar = figure.colorbar(
        plt.cm.ScalarMappable(cmap=ERROR_CMAP), ax=axis, fraction=0.04, pad=0.02
    )
    bar.outline.set_visible(False)
    bar.set_ticks([0.0, 1.0])
    bar.set_ticklabels(["0", f"{ERROR_SCALE:g}"])
    bar.ax.tick_params(labelsize=8, colors=MUTED, length=0)


def scene_comparison(figure_dir, scene, out_path):
    panels = [load_panel(figure_dir, scene, name) for name in COLUMNS]
    height, width = panels[0].shape[:2]
    figure, axes = plt.subplots(
        1, 3, figsize=(13.5, 13.5 / 3 * height / width + 0.6), facecolor="white"
    )
    for axis, panel, title in zip(axes, panels, COLUMN_TITLES):
        axis.imshow(panel)
        style_axis(axis, title)
    add_error_bar(figure, axes[2])
    figure.suptitle(scene, fontsize=13, color=INK, x=0.012, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def paired_comparison(figure_dir, scenes, out_path):
    ratios = []
    for scene in scenes:
        panel = load_panel(figure_dir, scene, "model")
        ratios.append(panel.shape[0] / panel.shape[1])

    width = 13.5
    figure, axes = plt.subplots(
        len(scenes),
        3,
        figsize=(width, width / 3 * sum(ratios) + 1.1),
        height_ratios=ratios,
        facecolor="white",
    )
    for row, scene in enumerate(scenes):
        for column, name in enumerate(COLUMNS):
            axis = axes[row][column]
            axis.imshow(load_panel(figure_dir, scene, name))
            style_axis(axis, COLUMN_TITLES[column] if row == 0 else None)
            if column == 0:
                axis.set_ylabel(scene, fontsize=11, color=INK, labelpad=8)
                axis.yaxis.set_visible(True)
                axis.set_yticks([])
        add_error_bar(figure, axes[row][2])
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def rate_distortion(report, out_path):
    figure, axis = plt.subplots(figsize=(8.5, 5.2), facecolor="white")
    for entry in report["scenes"]:
        scene = entry["scene"]
        color = SERIES.get(scene, INK)
        rows = entry["rate_curve"]
        x = [row["bytes_per_megapixel"] / 1024.0 for row in rows]
        y = [row["smape_full"] for row in rows]
        axis.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=scene)
        axis.axhline(
            entry["baseline"]["smape_full"],
            color=color,
            linewidth=1.2,
            linestyle=":",
            alpha=0.9,
        )
        axis.annotate(
            f"{scene} shader baseline",
            (x[0], entry["baseline"]["smape_full"]),
            textcoords="offset points",
            xytext=(0, 5),
            fontsize=9,
            color=MUTED,
        )
        shipped = entry["shipped"]
        axis.plot(
            [shipped["bytes_per_megapixel"] / 1024.0],
            [shipped["smape_full"]],
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=2,
        )
        axis.annotate(
            f"{shipped['atoms']:,} atoms",
            (shipped["bytes_per_megapixel"] / 1024.0, shipped["smape_full"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color=MUTED,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks([30, 50, 70, 100, 150, 220])
    axis.set_yticks([0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.25])
    for formatter_axis in (axis.xaxis, axis.yaxis):
        formatter_axis.set_major_formatter(ScalarFormatter())
        formatter_axis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("payload (KiB per megapixel, log scale)", fontsize=10, color=INK)
    axis.set_ylabel("tone-domain sMAPE vs oracle", fontsize=10, color=INK)
    axis.set_title(
        "Rate-distortion, shared codec on both scenes", fontsize=12, color=INK, pad=10
    )
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    legend = axis.legend(frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def main():
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--figures", type=Path, default=root / "figures")
    args = parser.parse_args()

    report = json.loads(
        (args.results / "benchmark.json").read_text(encoding="utf-8")
    )
    scenes = [entry["scene"] for entry in report["scenes"]]
    for scene in scenes:
        scene_comparison(args.figures, scene, args.figures / f"compare_{scene}.png")
    paired_comparison(args.figures, scenes, args.figures / "compare_both.png")
    rate_distortion(report, args.figures / "rate_distortion.png")


if __name__ == "__main__":
    main()
