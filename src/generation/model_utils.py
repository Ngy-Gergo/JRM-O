import math
from pathlib import Path

import bpy
from mathutils import Vector


SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".fbx", ".obj"}


def import_model(filepath):
    """Import a supported model and return the objects added to the scene."""
    model_path = Path(filepath)
    extension = model_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

    before_objects = set(bpy.context.scene.objects)

    if extension in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(model_path))
    elif extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(model_path))
    elif extension == ".obj":
        _import_obj(model_path)

    imported_objects = [
        obj for obj in bpy.context.scene.objects
        if obj not in before_objects
    ]

    if not imported_objects:
        imported_objects = list(bpy.context.selected_objects)

    return imported_objects


def get_imported_mesh_objects(objects=None):
    """Return mesh objects from an imported object list, including descendants."""
    if objects is None:
        source_objects = list(bpy.context.selected_objects)
    elif isinstance(objects, bpy.types.Object):
        source_objects = [objects]
    else:
        source_objects = list(objects)

    mesh_objects = []
    seen = set()
    stack = list(source_objects)

    while stack:
        obj = stack.pop()
        if obj in seen:
            continue
        seen.add(obj)

        if obj.type == "MESH":
            mesh_objects.append(obj)

        stack.extend(obj.children)

    return mesh_objects


def get_model_bounding_box_corners(objects):
    """Return world-space bounding-box corners for all mesh parts in a model."""
    mesh_objects = get_imported_mesh_objects(objects)
    if not mesh_objects:
        raise ValueError("Model does not contain any mesh objects.")

    corners = []
    for obj in mesh_objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))

    return corners


def normalize_model(objects, target_size=3.0):
    """Center imported mesh parts, scale them consistently, and place bottom at z=0."""
    mesh_objects = get_imported_mesh_objects(objects)
    if not mesh_objects:
        raise ValueError("Imported model does not contain any mesh objects.")

    bounds_min, bounds_max = _combined_world_bounds(mesh_objects)
    dimensions = bounds_max - bounds_min
    max_dimension = max(dimensions.x, dimensions.y, dimensions.z)

    if max_dimension <= 0:
        raise ValueError("Imported model has an invalid zero-size bounding box.")

    scale_factor = float(target_size) / max_dimension
    center_xy = Vector((
        (bounds_min.x + bounds_max.x) * 0.5,
        (bounds_min.y + bounds_max.y) * 0.5,
        0.0,
    ))

    root = bpy.data.objects.new("VehicleRoot", None)
    bpy.context.collection.objects.link(root)

    for obj in mesh_objects:
        world_matrix = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world_matrix

    root.scale = (scale_factor, scale_factor, scale_factor)
    root.location = (
        -center_xy.x * scale_factor,
        -center_xy.y * scale_factor,
        -bounds_min.z * scale_factor,
    )

    bpy.context.view_layer.update()
    return root


def set_model_yaw(objects, yaw_deg):
    """Set yaw rotation around Blender's vertical Z axis."""
    target = _resolve_rotation_target(objects)
    target.rotation_euler[2] = math.radians(float(yaw_deg) % 360)


def _import_obj(model_path):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(model_path))
    else:
        bpy.ops.import_scene.obj(filepath=str(model_path))


def _combined_world_bounds(mesh_objects):
    corners = get_model_bounding_box_corners(mesh_objects)

    if not corners:
        raise ValueError("Could not compute a bounding box for the imported model.")

    bounds_min = Vector((
        min(corner.x for corner in corners),
        min(corner.y for corner in corners),
        min(corner.z for corner in corners),
    ))
    bounds_max = Vector((
        max(corner.x for corner in corners),
        max(corner.y for corner in corners),
        max(corner.z for corner in corners),
    ))

    return bounds_min, bounds_max


def _resolve_rotation_target(objects):
    if isinstance(objects, bpy.types.Object):
        return objects

    object_list = list(objects)
    if not object_list:
        raise ValueError("No objects were provided for yaw rotation.")

    vehicle_roots = {
        obj.parent for obj in object_list
        if obj.parent is not None and obj.parent.name.startswith("VehicleRoot")
    }
    if len(vehicle_roots) == 1:
        return vehicle_roots.pop()

    return object_list[0]
