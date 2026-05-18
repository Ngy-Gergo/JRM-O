import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIXED_CLASS_LABELS = (
    "approaching",
    "leaving",
    "moving_left",
    "moving_right",
)

METRICS_REQUIRED_COLUMNS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
)

COMPARISON_REQUIRED_COLUMNS = (
    "model_name",
    "best_validation_accuracy",
    "final_validation_accuracy",
    "test_accuracy",
)

PER_MODEL_OUTPUTS = (
    "loss",
    "accuracy",
    "overfitting_gap",
    "per_class_accuracy",
    "confusion_matrix",
    "confusion_matrix_normalized",
)

COMPARISON_OUTPUTS = (
    "model_test_accuracy_comparison.png",
    "model_best_validation_accuracy_comparison.png",
    "model_final_val_vs_test_accuracy.png",
    "model_per_class_accuracy_comparison.png",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate static PNG plots from training run outputs."
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_runs") / "figures",
    )
    return parser.parse_args()


def read_metrics(run_dir):
    path = run_dir / "metrics.csv"
    rows, fieldnames = read_csv_dicts(path)
    validate_required_columns(fieldnames, METRICS_REQUIRED_COLUMNS, path)
    if not rows:
        raise ValueError(f"Required file has no metric rows: {path}")

    metrics = []
    for index, row in enumerate(rows, start=2):
        metrics.append(
            {
                "epoch": int(float_value(row, "epoch", path, index)),
                "train_loss": float_value(row, "train_loss", path, index),
                "train_accuracy": float_value(row, "train_accuracy", path, index),
                "val_loss": float_value(row, "val_loss", path, index),
                "val_accuracy": float_value(row, "val_accuracy", path, index),
            }
        )
    return metrics


def read_training_config(run_dir):
    path = run_dir / "training_config.json"
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return data


def read_evaluation(run_dir):
    path = run_dir / "evaluation.json"
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in: {path}")

    for key in ("labels", "per_class_accuracy"):
        if key not in data:
            raise ValueError(f"{path} is missing required field: {key}")

    if not isinstance(data["labels"], list):
        raise ValueError(f"{path} field 'labels' must be a list")
    if not isinstance(data["per_class_accuracy"], dict):
        raise ValueError(f"{path} field 'per_class_accuracy' must be an object")

    missing_labels = [
        label
        for label in FIXED_CLASS_LABELS
        if label not in data["per_class_accuracy"]
    ]
    if missing_labels:
        raise ValueError(
            f"{path} is missing per-class accuracy for: {', '.join(missing_labels)}"
        )

    for label, value in data["per_class_accuracy"].items():
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} has non-numeric per-class accuracy for '{label}'"
            ) from exc

    return data


def read_confusion_matrix(run_dir):
    path = run_dir / "confusion_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Required file is empty: {path}") from exc

        if not header or header[0] != "true_label":
            raise ValueError(f"{path} must start with a 'true_label' column")

        column_labels = header[1:]
        row_labels = []
        matrix = []
        for row_index, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{path} row {row_index} has {len(row)} columns, "
                    f"expected {len(header)}"
                )

            row_labels.append(row[0])
            try:
                matrix.append([int(value) for value in row[1:]])
            except ValueError as exc:
                raise ValueError(
                    f"{path} row {row_index} contains a non-integer count"
                ) from exc

    if not matrix:
        raise ValueError(f"Required file has no confusion matrix rows: {path}")

    row_label_set = set(row_labels)
    column_label_set = set(column_labels)
    if row_label_set != column_label_set:
        missing_from_rows = sorted(column_label_set - row_label_set)
        missing_from_columns = sorted(row_label_set - column_label_set)
        raise ValueError(
            f"{path} row/column label mismatch. "
            f"Missing rows: {', '.join(missing_from_rows) or 'none'}. "
            f"Missing columns: {', '.join(missing_from_columns) or 'none'}."
        )

    missing_rows = [label for label in FIXED_CLASS_LABELS if label not in row_labels]
    missing_columns = [
        label for label in FIXED_CLASS_LABELS if label not in column_labels
    ]
    if missing_rows:
        raise ValueError(
            f"{path} is missing confusion matrix rows for: {', '.join(missing_rows)}"
        )
    if missing_columns:
        raise ValueError(
            f"{path} is missing confusion matrix columns for: "
            f"{', '.join(missing_columns)}"
        )

    return {
        "path": path,
        "row_labels": row_labels,
        "column_labels": column_labels,
        "matrix": matrix,
    }


def read_comparison_csv(path):
    rows, fieldnames = read_csv_dicts(path)
    required_columns = list(COMPARISON_REQUIRED_COLUMNS) + [
        f"test_accuracy_{label}" for label in FIXED_CLASS_LABELS
    ]
    validate_required_columns(fieldnames, required_columns, path)
    if not rows:
        raise ValueError(f"Required file has no comparison rows: {path}")
    return rows


def get_model_name(run_dir, training_config=None, evaluation=None):
    if training_config and training_config.get("model_name"):
        return str(training_config["model_name"])
    if evaluation and evaluation.get("model_name"):
        return str(evaluation["model_name"])
    return run_dir.name


def sanitize_filename_part(value):
    sanitized = str(value).strip()
    sanitized = sanitized.replace("/", "_").replace("\\", "_")
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "model"


def validate_required_columns(fieldnames, required_columns, source_path):
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        raise ValueError(
            f"{source_path} is missing required columns: {', '.join(missing)}"
        )


def validate_comparison_values(rows):
    numeric_columns = list(COMPARISON_REQUIRED_COLUMNS[1:]) + [
        f"test_accuracy_{label}" for label in comparison_class_labels(rows)
    ]
    for row_index, row in enumerate(rows, start=2):
        for column in numeric_columns:
            float_value(row, column, "comparison CSV", row_index)


def validate_inputs(args):
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {args.output_dir}")

    runs = []
    safe_model_names = {}
    for run_dir in args.runs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
        if not run_dir.is_dir():
            raise ValueError(f"Run path is not a directory: {run_dir}")

        training_config = read_training_config(run_dir)
        evaluation = read_evaluation(run_dir)
        metrics = read_metrics(run_dir)
        confusion_matrix = read_confusion_matrix(run_dir)
        model_name = get_model_name(run_dir, training_config, evaluation)
        safe_model_name = sanitize_filename_part(model_name)

        if safe_model_name in safe_model_names:
            previous_run = safe_model_names[safe_model_name]
            raise ValueError(
                "Multiple runs resolve to the same sanitized model name "
                f"'{safe_model_name}': {previous_run} and {run_dir}"
            )
        safe_model_names[safe_model_name] = run_dir

        runs.append(
            {
                "run_dir": run_dir,
                "model_name": model_name,
                "safe_model_name": safe_model_name,
                "training_config": training_config,
                "evaluation": evaluation,
                "metrics": metrics,
                "confusion_matrix": confusion_matrix,
            }
        )

    comparison_rows = read_comparison_csv(args.comparison)
    comparison_by_model = {}
    duplicate_comparison_models = []
    for row in comparison_rows:
        model_name = row.get("model_name", "")
        if model_name in comparison_by_model:
            duplicate_comparison_models.append(model_name)
        comparison_by_model[model_name] = row

    requested_model_names = [run["model_name"] for run in runs]
    duplicate_requested = sorted(
        {
            model_name
            for model_name in requested_model_names
            if requested_model_names.count(model_name) > 1
        }
    )
    if duplicate_requested:
        raise ValueError(
            "Duplicate model names were provided: "
            f"{', '.join(duplicate_requested)}"
        )

    duplicated_requested_rows = [
        model_name
        for model_name in duplicate_comparison_models
        if model_name in requested_model_names
    ]
    if duplicated_requested_rows:
        raise ValueError(
            "Comparison CSV has duplicate rows for: "
            f"{', '.join(sorted(duplicated_requested_rows))}"
        )

    missing_models = [
        model_name
        for model_name in requested_model_names
        if model_name not in comparison_by_model
    ]
    if missing_models:
        raise ValueError(
            "Comparison CSV is missing requested model_name values: "
            f"{', '.join(missing_models)}"
        )

    selected_comparison_rows = [
        comparison_by_model[run["model_name"]] for run in runs
    ]
    validate_comparison_values(selected_comparison_rows)
    planned_paths = planned_output_paths(args.output_dir, runs)
    ensure_unique_paths(planned_paths)

    return {
        "runs": runs,
        "comparison_rows": selected_comparison_rows,
        "planned_paths": planned_paths,
    }


def ordered_class_labels(labels):
    label_set = set(labels)
    ordered = [label for label in FIXED_CLASS_LABELS if label in label_set]
    extras = sorted(label_set.difference(FIXED_CLASS_LABELS))
    return ordered + extras


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if height is None:
            continue

        if height >= 0.95:
            y_position = height - 0.04
            vertical_alignment = "top"
        else:
            y_position = height + 0.015
            vertical_alignment = "bottom"

        rotation = 90 if bar.get_width() < 0.18 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_position,
            f"{height:.3f}",
            ha="center",
            va=vertical_alignment,
            fontsize=8,
            rotation=rotation,
        )


def save_figure(fig, path):
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def plot_loss_curve(run, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = run["metrics"]
    epochs = [row["epoch"] for row in metrics]
    ax.plot(epochs, [row["train_loss"] for row in metrics], marker="o", label="train_loss")
    ax.plot(epochs, [row["val_loss"] for row in metrics], marker="o", label="val_loss")
    ax.set_title(f"{run['model_name']} Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir / f"{run['safe_model_name']}_loss.png")


def plot_accuracy_curve(run, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = run["metrics"]
    epochs = [row["epoch"] for row in metrics]
    ax.plot(
        epochs,
        [row["train_accuracy"] for row in metrics],
        marker="o",
        label="train_accuracy",
    )
    ax.plot(
        epochs,
        [row["val_accuracy"] for row in metrics],
        marker="o",
        label="val_accuracy",
    )
    ax.set_title(f"{run['model_name']} Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir / f"{run['safe_model_name']}_accuracy.png")


def plot_overfitting_gap(run, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = run["metrics"]
    epochs = [row["epoch"] for row in metrics]
    gaps = [row["train_accuracy"] - row["val_accuracy"] for row in metrics]
    ax.plot(epochs, gaps, marker="o", label="train_accuracy - val_accuracy")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_title(f"{run['model_name']} Overfitting Gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy gap")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(
        fig,
        output_dir / f"{run['safe_model_name']}_overfitting_gap.png",
    )


def plot_per_class_accuracy(run, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    per_class_accuracy = run["evaluation"]["per_class_accuracy"]
    labels = ordered_class_labels(per_class_accuracy.keys())
    values = [float(per_class_accuracy[label]) for label in labels]
    bars = ax.bar(labels, values)
    annotate_bars(ax, bars)
    ax.set_title(f"{run['model_name']} Per-Class Accuracy")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return save_figure(
        fig,
        output_dir / f"{run['safe_model_name']}_per_class_accuracy.png",
    )


def plot_confusion_matrix(run, output_dir, normalized=False):
    confusion = run["confusion_matrix"]
    labels, matrix = ordered_confusion_matrix(confusion)
    values = normalize_matrix(matrix) if normalized else matrix

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0 if normalized else None)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Proportion" if normalized else "Count")

    ax.set_title(
        f"{run['model_name']} "
        f"{'Normalized ' if normalized else ''}Confusion Matrix"
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    threshold = matrix_threshold(values)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            text = f"{value:.2f}" if normalized else str(int(value))
            color = "white" if value > threshold else "black"
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color=color,
            )

    suffix = "confusion_matrix_normalized" if normalized else "confusion_matrix"
    return save_figure(fig, output_dir / f"{run['safe_model_name']}_{suffix}.png")


def plot_test_accuracy_comparison(rows, output_dir):
    return plot_single_metric_comparison(
        rows,
        "test_accuracy",
        "Test Accuracy",
        "Model Test Accuracy Comparison",
        output_dir / "model_test_accuracy_comparison.png",
    )


def plot_best_validation_accuracy_comparison(rows, output_dir):
    return plot_single_metric_comparison(
        rows,
        "best_validation_accuracy",
        "Best Validation Accuracy",
        "Model Best Validation Accuracy Comparison",
        output_dir / "model_best_validation_accuracy_comparison.png",
    )


def plot_final_val_vs_test_accuracy(rows, output_dir):
    model_names = [row["model_name"] for row in rows]
    x_positions = list(range(len(rows)))
    width = 0.35
    final_values = [
        float_value(row, "final_validation_accuracy", "comparison CSV")
        for row in rows
    ]
    test_values = [float_value(row, "test_accuracy", "comparison CSV") for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.5), 5))
    final_bars = ax.bar(
        [position - width / 2 for position in x_positions],
        final_values,
        width,
        label="final_validation_accuracy",
    )
    test_bars = ax.bar(
        [position + width / 2 for position in x_positions],
        test_values,
        width,
        label="test_accuracy",
    )
    annotate_bars(ax, final_bars)
    annotate_bars(ax, test_bars)
    ax.set_title("Final Validation Accuracy vs Test Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_positions, model_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir / "model_final_val_vs_test_accuracy.png")


def plot_per_class_accuracy_comparison(rows, output_dir):
    class_labels = comparison_class_labels(rows)
    x_positions = list(range(len(class_labels)))
    width = min(0.8 / len(rows), 0.22)

    fig, ax = plt.subplots(figsize=(max(9, len(class_labels) * 1.8), 5))
    for model_index, row in enumerate(rows):
        offset = (model_index - (len(rows) - 1) / 2) * width
        values = [
            float_value(row, f"test_accuracy_{label}", "comparison CSV")
            for label in class_labels
        ]
        bars = ax.bar(
            [position + offset for position in x_positions],
            values,
            width,
            label=row["model_name"],
        )
        annotate_bars(ax, bars)

    ax.set_title("Per-Class Accuracy Comparison")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_positions, class_labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir / "model_per_class_accuracy_comparison.png")


def print_summary(output_dir, figure_paths):
    print("Plot generation complete.")
    print(f"Output directory: {output_dir}")
    print(f"Figures created: {len(figure_paths)}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"- {path}")


def read_csv_dicts(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Required file is empty: {path}")
        rows = list(reader)

    return rows, fieldnames


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def float_value(row, column, source_path, row_index=None):
    try:
        return float(row[column])
    except (TypeError, ValueError) as exc:
        location = f" row {row_index}" if row_index is not None else ""
        raise ValueError(
            f"{source_path}{location} has non-numeric value for '{column}'"
        ) from exc


def planned_output_paths(output_dir, runs):
    paths = []
    for run in runs:
        for output_name in PER_MODEL_OUTPUTS:
            paths.append(output_dir / f"{run['safe_model_name']}_{output_name}.png")
    paths.extend(output_dir / filename for filename in COMPARISON_OUTPUTS)
    return paths


def ensure_unique_paths(paths):
    seen = {}
    for path in paths:
        key = path.resolve()
        if key in seen:
            raise ValueError(f"Duplicate planned output path: {path}")
        seen[key] = path


def ordered_confusion_matrix(confusion):
    labels = ordered_class_labels(
        set(confusion["row_labels"]).union(confusion["column_labels"])
    )
    row_lookup = {
        label: index for index, label in enumerate(confusion["row_labels"])
    }
    column_lookup = {
        label: index for index, label in enumerate(confusion["column_labels"])
    }

    ordered_matrix = []
    for row_label in labels:
        source_row = row_lookup[row_label]
        ordered_row = []
        for column_label in labels:
            source_column = column_lookup[column_label]
            ordered_row.append(confusion["matrix"][source_row][source_column])
        ordered_matrix.append(ordered_row)

    return labels, ordered_matrix


def normalize_matrix(matrix):
    normalized = []
    for row in matrix:
        total = sum(row)
        if total == 0:
            normalized.append([0.0 for _ in row])
        else:
            normalized.append([value / total for value in row])
    return normalized


def matrix_threshold(matrix):
    values = [value for row in matrix for value in row]
    return max(values) / 2 if values else 0.0


def plot_single_metric_comparison(rows, metric_column, ylabel, title, output_path):
    model_names = [row["model_name"] for row in rows]
    values = [float_value(row, metric_column, "comparison CSV") for row in rows]
    x_positions = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.5), 5))
    bars = ax.bar(x_positions, values)
    annotate_bars(ax, bars)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_positions, model_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    return save_figure(fig, output_path)


def comparison_class_labels(rows):
    labels = {
        column.replace("test_accuracy_", "", 1)
        for row in rows
        for column in row.keys()
        if column.startswith("test_accuracy_")
    }
    return ordered_class_labels(labels)


def main():
    args = parse_args()
    try:
        context = validate_inputs(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        figure_paths = []
        for run in context["runs"]:
            figure_paths.append(plot_loss_curve(run, args.output_dir))
            figure_paths.append(plot_accuracy_curve(run, args.output_dir))
            figure_paths.append(plot_overfitting_gap(run, args.output_dir))
            figure_paths.append(plot_per_class_accuracy(run, args.output_dir))
            figure_paths.append(plot_confusion_matrix(run, args.output_dir))
            figure_paths.append(
                plot_confusion_matrix(run, args.output_dir, normalized=True)
            )

        comparison_rows = context["comparison_rows"]
        figure_paths.append(plot_test_accuracy_comparison(comparison_rows, args.output_dir))
        figure_paths.append(
            plot_best_validation_accuracy_comparison(comparison_rows, args.output_dir)
        )
        figure_paths.append(plot_final_val_vs_test_accuracy(comparison_rows, args.output_dir))
        figure_paths.append(
            plot_per_class_accuracy_comparison(comparison_rows, args.output_dir)
        )

        print_summary(args.output_dir, figure_paths)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
