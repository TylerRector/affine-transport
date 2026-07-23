from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CORNELL_SIZE = 1024
CORNELL_ORACLE_SPP = 4096
CORNELL_SHADER_SPP = 1024
CORNELL_PASS_SPP = 128
CORNELL_ORACLE_DEPTH = 12
CORNELL_SHADER_DEPTH = 3
GLASS_RADIUS = 0.32
GLASS_CENTER = (-0.33, -0.55, 0.2)


def _cornell_dict(mi, size):
    scene = mi.cornell_box()
    scene["sensor"]["film"]["width"] = size
    scene["sensor"]["film"]["height"] = size
    scene["sensor"]["film"]["rfilter"] = {"type": "box"}
    return scene


def _glass(mi):
    return {
        "type": "sphere",
        "to_world": mi.ScalarTransform4f()
        .translate(list(GLASS_CENTER))
        .scale(GLASS_RADIUS),
        "bsdf": {"type": "dielectric", "int_ior": 1.5},
    }


def _depth(mi, size, with_glass):
    scene = _cornell_dict(mi, size)
    if with_glass:
        scene["glass"] = _glass(mi)
    scene["integrator"] = {"type": "aov", "aovs": "d:depth"}
    return np.array(mi.render(mi.load_dict(scene), spp=1))[..., 0]


def _accumulate(mi, scene_dict, total_spp, pass_spp):
    scene = mi.load_dict(scene_dict)
    passes = max(1, total_spp // pass_spp)
    total = None
    for index in range(passes):
        frame = np.array(
            mi.render(scene, spp=pass_spp, seed=index), dtype=np.float64
        )
        total = frame if total is None else total + frame
    return (total / passes).astype(np.float32)


def build_cornell(out_dir, size=CORNELL_SIZE):
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("llvm_ad_rgb")

    glass_pixels = np.abs(_depth(mi, size, True) - _depth(mi, size, False)) > 1e-4

    oracle_scene = _cornell_dict(mi, size)
    oracle_scene["glass"] = _glass(mi)
    oracle_scene["integrator"] = {"type": "path", "max_depth": CORNELL_ORACLE_DEPTH}
    oracle = _accumulate(mi, oracle_scene, CORNELL_ORACLE_SPP, CORNELL_PASS_SPP)

    shader_scene = _cornell_dict(mi, size)
    shader_scene["integrator"] = {"type": "path", "max_depth": CORNELL_SHADER_DEPTH}
    shader = _accumulate(mi, shader_scene, CORNELL_SHADER_SPP, CORNELL_PASS_SPP)
    shader[glass_pixels] = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cornell_shader_rgb_f32.npy", np.maximum(shader, 0.0))
    np.save(out_dir / "cornell_oracle_rgb_f32.npy", np.maximum(oracle, 0.0))
    return shader.shape


def build_kitchen(out_dir, dump):
    arrays = np.load(dump)
    out_dir.mkdir(parents=True, exist_ok=True)
    shader = np.maximum(arrays["shader"].astype(np.float32), 0.0)
    oracle = np.maximum(arrays["oracle"].astype(np.float32), 0.0)
    np.save(out_dir / "kitchen_shader_rgb_f32.npy", shader)
    np.save(out_dir / "kitchen_oracle_rgb_f32.npy", oracle)
    return shader.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent.parent / "data")
    parser.add_argument("--kitchen-dump", type=Path)
    parser.add_argument("--size", type=int, default=CORNELL_SIZE)
    parser.add_argument("--skip-cornell", action="store_true")
    args = parser.parse_args()

    if args.kitchen_dump is not None:
        print("kitchen", build_kitchen(args.data, args.kitchen_dump))
    if not args.skip_cornell:
        print("cornell", build_cornell(args.data, args.size))


if __name__ == "__main__":
    main()
