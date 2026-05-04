from pathlib import Path


LABELS = (
    "approaching",
    "leaving",
    "moving_left",
    "moving_right",
)


def yaw_to_label(yaw_deg: float) -> str:
    """Map a Z-axis yaw angle in degrees to an orientation class label."""
    yaw = yaw_deg % 360

    if yaw >= 315 or yaw < 45:
        return "leaving"
    if yaw < 135:
        return "moving_right"
    if yaw < 225:
        return "approaching"
    return "moving_left"


def ensure_label_folders(output_dir):
    """Create the fixed class-label folders used by this dataset."""
    output_path = Path(output_dir)
    for label in LABELS:
        (output_path / label).mkdir(parents=True, exist_ok=True)
