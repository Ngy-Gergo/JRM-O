import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ("approaching", "leaving", "moving_left", "moving_right")
OUTPUT_COLUMNS = ("relative_path", "label", "model_name", "yaw_deg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create deterministic image-level dataset splits."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("generated_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def read_metadata(metadata_path):
    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    for row in rows:
        if row.get("label") not in LABELS:
            raise ValueError(f"Unsupported label in metadata.csv: {row.get('label')}")
        if not row.get("relative_path"):
            raise ValueError("metadata.csv contains a row without relative_path.")

    return rows


def make_stratified_split(rows, seed, train_ratio, val_ratio, test_ratio):
    validate_ratios(train_ratio, val_ratio, test_ratio)
    rng = random.Random(seed)
    rows_by_label = defaultdict(list)

    for row in rows:
        rows_by_label[row["label"]].append(row)

    split_rows = {"train": [], "val": [], "test": []}

    # v0.1 uses image-level splitting for a small sanity-check dataset.
    # Later, with more vehicle models, prefer model-level splitting to measure
    # generalization to unseen vehicle geometry.
    for label in LABELS:
        label_rows = list(rows_by_label[label])
        rng.shuffle(label_rows)
        train_count, val_count = split_counts(len(label_rows), train_ratio, val_ratio)

        split_rows["train"].extend(label_rows[:train_count])
        split_rows["val"].extend(label_rows[train_count:train_count + val_count])
        split_rows["test"].extend(label_rows[train_count + val_count:])

    for split_name in split_rows:
        split_rows[split_name].sort(
            key=lambda row: (row["label"], row.get("model_name", ""), row["relative_path"])
        )

    return split_rows


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must all be positive.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")


def split_counts(total_count, train_ratio, val_ratio):
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)

    if total_count >= 3:
        train_count = max(train_count, 1)
        val_count = max(val_count, 1)
        if train_count + val_count >= total_count:
            train_count = max(total_count - 2, 1)
            val_count = 1

    return train_count, val_count


def write_split_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def print_counts(split_rows):
    print("Dataset split summary")
    for split_name in ("train", "val", "test"):
        rows = split_rows[split_name]
        counts = Counter(row["label"] for row in rows)
        print()
        print(f"{split_name}: {len(rows)} images")
        for label in LABELS:
            print(f"  {label}: {counts[label]}")


def main():
    args = parse_args()
    metadata_path = args.dataset_dir / "metadata.csv"
    rows = read_metadata(metadata_path)
    split_rows = make_stratified_split(
        rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    for split_name, rows_for_split in split_rows.items():
        write_split_csv(args.output_dir / f"{split_name}.csv", rows_for_split)

    print_counts(split_rows)
    print()
    print(f"Splits written to: {args.output_dir}")


if __name__ == "__main__":
    main()
