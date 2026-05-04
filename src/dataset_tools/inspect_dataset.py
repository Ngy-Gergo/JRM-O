import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


EXPECTED_LABELS = (
    "approaching",
    "leaving",
    "moving_left",
    "moving_right",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
REQUIRED_METADATA_COLUMNS = ("relative_path", "label", "model_name")
IMBALANCE_THRESHOLD = 0.20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a generated vehicle orientation dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to a generated dataset directory containing metadata.csv.",
    )
    return parser.parse_args()


def inspect_dataset(dataset_dir):
    dataset_dir = dataset_dir.resolve()
    metadata_path = dataset_dir / "metadata.csv"

    critical_issues = []
    warnings = []
    metadata_rows = []
    metadata_paths = set()
    missing_metadata_files = []
    invalid_labels = []
    unsafe_paths = []
    label_folder_mismatches = []
    label_counts = Counter({label: 0 for label in EXPECTED_LABELS})
    model_counts = Counter()

    if not metadata_path.exists():
        critical_issues.append(f"Missing metadata file: {metadata_path}")
    else:
        metadata_rows = read_metadata_rows(metadata_path, critical_issues)

    image_paths, image_counts, missing_label_folders = scan_label_folders(dataset_dir)
    for label in missing_label_folders:
        critical_issues.append(f"Missing expected label folder: {label}")

    for row_number, row in enumerate(metadata_rows, start=2):
        label = (row.get("label") or "").strip()
        relative_path_text = (row.get("relative_path") or "").strip()
        model_name = (row.get("model_name") or "").strip()

        if label in EXPECTED_LABELS:
            label_counts[label] += 1
        else:
            invalid_labels.append(format_row_issue(row_number, label or "<empty>"))

        if model_name:
            model_counts[model_name] += 1

        safe_path, reason = validate_relative_path(relative_path_text)
        if not safe_path:
            unsafe_paths.append(format_row_issue(row_number, reason))
            continue

        relative_path = Path(relative_path_text)
        normalized_relative_path = relative_path.as_posix()
        metadata_paths.add(normalized_relative_path)

        if label in EXPECTED_LABELS and not path_stays_in_label_folder(relative_path, label):
            label_folder_mismatches.append(
                f"row {row_number}: relative_path '{relative_path_text}' "
                f"is not inside '{label}/'"
            )

        image_path = dataset_dir / relative_path
        if not image_path.is_file():
            missing_metadata_files.append(
                f"row {row_number}: {normalized_relative_path}"
            )

    orphan_images = sorted(image_paths - metadata_paths)

    for item in unsafe_paths:
        critical_issues.append(f"Unsafe metadata relative_path: {item}")
    for item in invalid_labels:
        critical_issues.append(f"Invalid metadata label: {item}")
    for item in label_folder_mismatches:
        critical_issues.append(f"Label/path mismatch: {item}")
    for item in missing_metadata_files:
        critical_issues.append(f"Metadata row points to missing image: {item}")
    for item in orphan_images:
        critical_issues.append(f"Image exists without metadata: {item}")

    for label in EXPECTED_LABELS:
        if image_counts[label] == 0:
            critical_issues.append(f"Class has zero images: {label}")

    imbalance_warning = detect_imbalance(image_counts)
    if imbalance_warning:
        warnings.append(imbalance_warning)

    report = {
        "dataset_dir": dataset_dir,
        "metadata_path": metadata_path,
        "total_metadata_rows": len(metadata_rows),
        "total_image_files": len(image_paths),
        "missing_metadata_files": missing_metadata_files,
        "orphan_images": orphan_images,
        "label_counts": label_counts,
        "image_counts": image_counts,
        "model_counts": model_counts,
        "warnings": warnings,
        "critical_issues": critical_issues,
    }
    return report


def read_metadata_rows(metadata_path, critical_issues):
    try:
        with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []
            missing_columns = [
                column for column in REQUIRED_METADATA_COLUMNS
                if column not in fieldnames
            ]
            if missing_columns:
                critical_issues.append(
                    "metadata.csv is missing required column(s): "
                    + ", ".join(missing_columns)
                )
            return list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        critical_issues.append(f"Could not read metadata.csv: {exc}")
        return []


def scan_label_folders(dataset_dir):
    image_paths = set()
    image_counts = Counter({label: 0 for label in EXPECTED_LABELS})
    missing_label_folders = []

    for label in EXPECTED_LABELS:
        label_dir = dataset_dir / label
        if not label_dir.is_dir():
            missing_label_folders.append(label)
            continue

        for image_path in sorted(label_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_path = image_path.relative_to(dataset_dir).as_posix()
            image_paths.add(relative_path)
            image_counts[label] += 1

    return image_paths, image_counts, missing_label_folders


def validate_relative_path(relative_path_text):
    if not relative_path_text:
        return False, "missing relative_path"

    relative_path = Path(relative_path_text)
    if relative_path.is_absolute() or relative_path.drive:
        return False, f"absolute path is not allowed: {relative_path_text}"
    if ".." in relative_path.parts:
        return False, f"path contains '..': {relative_path_text}"

    return True, ""


def path_stays_in_label_folder(relative_path, label):
    parts = relative_path.parts
    return bool(parts) and parts[0] == label


def detect_imbalance(image_counts):
    counts = [image_counts[label] for label in EXPECTED_LABELS]
    if any(count == 0 for count in counts):
        return None

    mean_count = sum(counts) / len(counts)
    if mean_count == 0:
        return None

    imbalanced = [
        f"{label}={image_counts[label]}"
        for label in EXPECTED_LABELS
        if abs(image_counts[label] - mean_count) / mean_count > IMBALANCE_THRESHOLD
    ]
    if not imbalanced:
        return None

    return (
        "Label counts differ by more than 20% from the mean "
        f"({mean_count:.2f}): " + ", ".join(imbalanced)
    )


def format_row_issue(row_number, value):
    return f"row {row_number}: {value}"


def print_report(report):
    is_valid = not report["critical_issues"]
    exit_code = 0 if is_valid else 1

    print("Dataset Inspection Report")
    print(f"Dataset directory: {report['dataset_dir']}")
    print(f"Metadata: {report['metadata_path']}")
    print()
    print("Summary")
    print(f"Total metadata rows: {report['total_metadata_rows']}")
    print(f"Total image files: {report['total_image_files']}")
    print(f"Missing files from metadata: {len(report['missing_metadata_files'])}")
    print(f"Images missing from metadata: {len(report['orphan_images'])}")
    print()
    print("Images per label")
    for label in EXPECTED_LABELS:
        print(f"{label}: {report['image_counts'][label]}")
    print()
    print("Metadata rows per label")
    for label in EXPECTED_LABELS:
        print(f"{label}: {report['label_counts'][label]}")
    print()
    print("Images per model")
    if report["model_counts"]:
        for model_name, count in sorted(report["model_counts"].items()):
            print(f"{model_name}: {count}")
    else:
        print("None")
    print()
    print("Warnings")
    print_items(report["warnings"])
    print()
    print("Critical Issues")
    print_items(report["critical_issues"])
    print()
    print("Result")
    print("VALID" if is_valid else "INVALID")
    print(f"Exit code: {exit_code}")


def print_items(items):
    if not items:
        print("None")
        return

    for item in items:
        print(f"- {item}")


def main():
    args = parse_args()
    report = inspect_dataset(args.dataset_dir)
    print_report(report)
    return 0 if not report["critical_issues"] else 1


if __name__ == "__main__":
    sys.exit(main())
