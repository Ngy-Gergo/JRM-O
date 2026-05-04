import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


MIN_AUTO_FRAME_DISTANCE = 0.5
MAX_AUTO_FRAME_DISTANCE = 50.0
AUTO_FRAME_ITERATIONS = 24


def clear_scene():
    """Remove all scene objects and orphaned data blocks from the current file."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for collection_name in ("meshes", "materials", "images", "lights", "cameras"):
        collection = getattr(bpy.data, collection_name, None)
        if collection is None:
            continue
        for data_block in list(collection):
            if data_block.users == 0:
                collection.remove(data_block)


def look_at_point(obj, target):
    """Rotate an object so its local -Z axis points at target."""
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(location=(0, -4.0, 1.15), target=(0, 0, 0.8), lens=50):
    """Create the fixed camera and aim it at the target point."""
    camera_data = bpy.data.cameras.new("FixedCamera")
    camera = bpy.data.objects.new("FixedCamera", camera_data)
    bpy.context.collection.objects.link(camera)

    apply_camera_preset(camera, location, target, lens)
    bpy.context.scene.camera = camera

    return camera


def apply_camera_preset(camera, location, target, lens):
    """Apply a fixed camera preset and aim it at the target point."""
    camera.location = location
    camera.data.lens = float(lens)
    look_at_point(camera, target)


def auto_frame_camera(camera, bounds_corners, height, target_z, lens, margin):
    """Move a fixed-direction camera back until projected bounds fit the frame."""
    target = Vector((0.0, 0.0, float(target_z)))
    camera.data.lens = float(lens)
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0

    high = 1.0
    while high < MAX_AUTO_FRAME_DISTANCE:
        _place_camera_at_distance(camera, high, height, target)
        bpy.context.view_layer.update()
        if _projected_bounds_fit(camera, bounds_corners, margin):
            break
        high *= 1.5

    high = min(high, MAX_AUTO_FRAME_DISTANCE)
    _place_camera_at_distance(camera, high, height, target)
    bpy.context.view_layer.update()

    if _projected_bounds_fit(camera, bounds_corners, margin):
        low = MIN_AUTO_FRAME_DISTANCE
        for _ in range(AUTO_FRAME_ITERATIONS):
            mid = (low + high) * 0.5
            _place_camera_at_distance(camera, mid, height, target)
            bpy.context.view_layer.update()
            if _projected_bounds_fit(camera, bounds_corners, margin):
                high = mid
            else:
                low = mid

    _place_camera_at_distance(camera, high, height, target)
    bpy.context.view_layer.update()
    return target, high


def setup_lighting(location=(0, -4, 6), energy=600, size=5):
    """Create one fixed area light."""
    light_data = bpy.data.lights.new("FixedAreaLight", type="AREA")
    light = bpy.data.objects.new("FixedAreaLight", light_data)
    bpy.context.collection.objects.link(light)

    light.location = location
    light.data.energy = energy
    light.data.size = size

    return light


def setup_render_settings(image_size=512, render_samples=64):
    """Configure deterministic square PNG rendering."""
    scene = bpy.context.scene
    image_size = int(image_size)
    render_samples = int(render_samples)

    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    _set_clean_render_engine(scene, render_samples)

    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
    except TypeError:
        pass


def set_background(color=(1.0, 1.0, 1.0)):
    """Set a fixed plain world background color."""
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("FixedWorld")

    rgb = tuple(float(channel) for channel in color[:3])
    scene.world.color = rgb


def _set_clean_render_engine(scene, render_samples):
    """Prefer Eevee for clean deterministic renders, with Cycles as fallback."""
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except TypeError:
            scene.render.engine = "CYCLES"

    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = render_samples

    if scene.render.engine == "CYCLES" and hasattr(scene, "cycles"):
        scene.cycles.samples = render_samples
        scene.cycles.seed = 0
        scene.cycles.use_denoising = False


def _place_camera_at_distance(camera, distance, height, target):
    camera.location = (0.0, -float(distance), float(height))
    look_at_point(camera, target)


def _projected_bounds_fit(camera, bounds_corners, margin):
    scene = bpy.context.scene
    margin = max(float(margin), 1.0)
    padding = (1.0 - (1.0 / margin)) * 0.5
    min_allowed = padding
    max_allowed = 1.0 - padding

    projected = [
        world_to_camera_view(scene, camera, corner)
        for corner in bounds_corners
    ]

    if any(point.z <= 0 for point in projected):
        return False

    min_x = min(point.x for point in projected)
    max_x = max(point.x for point in projected)
    min_y = min(point.y for point in projected)
    max_y = max(point.y for point in projected)

    return (
        min_x >= min_allowed and
        max_x <= max_allowed and
        min_y >= min_allowed and
        max_y <= max_allowed
    )
