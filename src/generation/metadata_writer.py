import csv
from pathlib import Path


METADATA_COLUMNS = [
    "filename",
    "relative_path",
    "label",
    "yaw_deg",
    "model_name",
    "model_path",
    "image_width",
    "image_height",
    "camera_location",
    "camera_target",
]


def initialize_metadata(output_dir):
    """Create a fresh metadata.csv file with the required header."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "metadata.csv"

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()

    return metadata_path


def append_metadata_row(metadata_path, row):
    """Append one rendered-image metadata row."""
    with Path(metadata_path).open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METADATA_COLUMNS)
        writer.writerow(row)


def format_vector(values):
    """Format vector-like values consistently for CSV output."""
    return "(" + ", ".join(f"{float(value):.6g}" for value in values) + ")"
