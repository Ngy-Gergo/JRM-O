import argparse
import json
import sys
import traceback
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_scene import (  # noqa: E402
    auto_frame_camera,
    clear_scene,
    set_background,
    setup_camera,
    setup_lighting,
    setup_render_settings,
)
from label_utils import ensure_label_folders, yaw_to_label  # noqa: E402
from metadata_writer import (  # noqa: E402
    append_metadata_row,
    format_vector,
    initialize_metadata,
)
from model_utils import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    get_imported_mesh_objects,
    get_model_bounding_box_corners,
    import_model,
    normalize_model,
    set_model_yaw,
)


DEFAULT_CONFIG = {
    "angle_step": 10,
    "image_size": 512,
    "render_samples": 64,
    "target_model_size": 3.0,
    "front_rear_camera_height": 1.15,
    "front_rear_camera_target_z": 0.8,
    "side_camera_height": 1.15,
    "side_camera_target_z": 0.8,
    "camera_lens": 50,
    "front_rear_margin": 1.10,
    "side_margin": 1.15,
    "light_location": [0, -4, 6],
    "light_energy": 600,
    "light_size": 5,
    "background_color": [1.0, 1.0, 1.0],
}


def parse_args():
    """Parse arguments passed after Blender's -- separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic vehicle orientation dataset."
    )
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--angle-step", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--render-samples", type=int)

    return parser.parse_args(argv)


def load_effective_config(args):
    """Load config defaults, then apply explicit CLI overrides."""
    config = dict(DEFAULT_CONFIG)

    if args.config:
        config_path = resolve_project_path(args.config)
        with config_path.open("r", encoding="utf-8") as config_file:
            loaded_config = json.load(config_file)
        apply_legacy_camera_config(config, loaded_config)
        config.update(loaded_config)

    if args.angle_step is not None:
        config["angle_step"] = args.angle_step
    if args.image_size is not None:
        config["image_size"] = args.image_size
    if args.render_samples is not None:
        config["render_samples"] = args.render_samples

    config["angle_step"] = int(config["angle_step"])
    config["image_size"] = int(config["image_size"])
    config["render_samples"] = int(config["render_samples"])
    config["target_model_size"] = float(config["target_model_size"])
    config["front_rear_camera_height"] = float(config["front_rear_camera_height"])
    config["front_rear_camera_target_z"] = float(config["front_rear_camera_target_z"])
    config["side_camera_height"] = float(config["side_camera_height"])
    config["side_camera_target_z"] = float(config["side_camera_target_z"])
    config["camera_lens"] = float(config["camera_lens"])
    config["front_rear_margin"] = float(config["front_rear_margin"])
    config["side_margin"] = float(config["side_margin"])

    if config["angle_step"] <= 0:
        raise ValueError("--angle-step must be a positive integer.")
    if config["image_size"] <= 0:
        raise ValueError("--image-size must be a positive integer.")
    if config["render_samples"] <= 0:
        raise ValueError("--render-samples must be a positive integer.")
    if config["front_rear_margin"] < 1.0 or config["side_margin"] < 1.0:
        raise ValueError("Camera margins must be greater than or equal to 1.0.")

    return config


def apply_legacy_camera_config(config, loaded_config):
    """Map older fixed-camera config keys to the new auto-frame defaults."""
    _copy_legacy_z(
        config,
        loaded_config,
        "front_rear_camera_location",
        "front_rear_camera_height",
    )
    _copy_legacy_z(
        config,
        loaded_config,
        "front_rear_camera_target",
        "front_rear_camera_target_z",
    )
    _copy_legacy_z(
        config,
        loaded_config,
        "side_camera_location",
        "side_camera_height",
    )
    _copy_legacy_z(
        config,
        loaded_config,
        "side_camera_target",
        "side_camera_target_z",
    )
    _copy_legacy_z(
        config,
        loaded_config,
        "camera_location",
        "front_rear_camera_height",
    )
    _copy_legacy_z(
        config,
        loaded_config,
        "camera_location",
        "side_camera_height",
    )
    _copy_legacy_z(
        config,
        loaded_config,
        "camera_target",
        "front_rear_camera_target_z",
    )
    _copy_legacy_z(
        config,
        loaded_config,
        "camera_target",
        "side_camera_target_z",
    )

    if "camera_lens" not in loaded_config:
        if "side_camera_lens" in loaded_config:
            config["camera_lens"] = loaded_config["side_camera_lens"]
        elif "front_rear_camera_lens" in loaded_config:
            config["camera_lens"] = loaded_config["front_rear_camera_lens"]


def _copy_legacy_z(config, loaded_config, legacy_key, target_key):
    if target_key in loaded_config or legacy_key not in loaded_config:
        return

    legacy_value = loaded_config[legacy_key]
    if isinstance(legacy_value, (list, tuple)) and len(legacy_value) >= 3:
        config[target_key] = legacy_value[2]


def resolve_project_path(path):
    """Resolve relative CLI paths from the project root, not Blender's cwd."""
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path.resolve()
    return (PROJECT_ROOT / resolved_path).resolve()


def find_supported_models(models_dir):
    """Return supported model files in deterministic filename order."""
    if not models_dir.exists():
        print(f"[Warning] Models directory does not exist: {models_dir}")
        return []

    supported_models = []
    for path in sorted(models_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"[Skip] Unsupported file: {path}")
            continue
        supported_models.append(path)

    return supported_models


def process_model(model_path, output_dir, metadata_path, config):
    """Import one model and render every yaw angle."""
    print(f"[Model] Processing {model_path}")

    clear_scene()
    imported_objects = import_model(model_path)
    mesh_objects = get_imported_mesh_objects(imported_objects)
    model_root = normalize_model(
        mesh_objects,
        target_size=config["target_model_size"],
    )

    camera = setup_camera(
        location=(0, -4, config["side_camera_height"]),
        target=(0, 0, config["side_camera_target_z"]),
        lens=config["camera_lens"],
    )
    setup_lighting(
        location=config["light_location"],
        energy=config["light_energy"],
        size=config["light_size"],
    )
    set_background(config["background_color"])
    setup_render_settings(
        image_size=config["image_size"],
        render_samples=config["render_samples"],
    )

    for yaw in range(0, 360, config["angle_step"]):
        label = yaw_to_label(yaw)
        camera_preset = camera_preset_for_label(config, label)
        filename = f"{model_path.stem}_yaw_{yaw:03d}.png"
        relative_path = Path(label) / filename
        output_path = output_dir / relative_path

        set_model_yaw(model_root, yaw)
        bpy.context.view_layer.update()
        bounds_corners = get_model_bounding_box_corners(model_root)
        camera_target, camera_distance = auto_frame_camera(
            camera,
            bounds_corners,
            height=camera_preset["height"],
            target_z=camera_preset["target_z"],
            lens=camera_preset["lens"],
            margin=camera_preset["margin"],
        )
        bpy.context.view_layer.update()

        bpy.context.scene.render.filepath = str(output_path)
        print(
            f"[Render] {model_path.name} yaw={yaw:03d} "
            f"label={label} distance={camera_distance:.3f} "
            f"margin={camera_preset['margin']:.2f}"
        )
        bpy.ops.render.render(write_still=True)

        append_metadata_row(
            metadata_path,
            {
                "filename": filename,
                "relative_path": relative_path.as_posix(),
                "label": label,
                "yaw_deg": yaw,
                "model_name": model_path.name,
                "model_path": model_path.as_posix(),
                "image_width": config["image_size"],
                "image_height": config["image_size"],
                "camera_location": format_vector(camera.location),
                "camera_target": format_vector(camera_target),
            },
        )


def camera_preset_for_label(config, label):
    """Select the deterministic camera preset for a rendered label."""
    if label in {"approaching", "leaving"}:
        return {
            "height": config["front_rear_camera_height"],
            "target_z": config["front_rear_camera_target_z"],
            "lens": config["camera_lens"],
            "margin": config["front_rear_margin"],
        }

    return {
        "height": config["side_camera_height"],
        "target_z": config["side_camera_target_z"],
        "lens": config["camera_lens"],
        "margin": config["side_margin"],
    }


def main():
    args = parse_args()
    config = load_effective_config(args)

    models_dir = resolve_project_path(
        args.models_dir if args.models_dir else Path("assets/models")
    )
    output_dir = resolve_project_path(
        args.output_dir if args.output_dir else Path("generated_dataset")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_label_folders(output_dir)
    metadata_path = initialize_metadata(output_dir)

    models = find_supported_models(models_dir)
    if not models:
        print(f"[Done] No supported model files found in {models_dir}")
        return

    print(f"[Start] Found {len(models)} supported model(s).")
    for model_path in models:
        try:
            process_model(model_path, output_dir, metadata_path, config)
        except Exception as exc:
            print(f"[Error] Failed to process {model_path}: {exc}")
            traceback.print_exc()
            continue

    print(f"[Done] Dataset written to {output_dir}")


if __name__ == "__main__":
    main()
