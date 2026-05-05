import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare trained vehicle orientation model runs."
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_metrics(metrics_path):
    with metrics_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    for row in rows:
        row["val_accuracy"] = float(row["val_accuracy"])

    return rows


def read_json(path):
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def summarize_run(run_dir):
    metrics = read_metrics(run_dir / "metrics.csv")
    evaluation = read_json(run_dir / "evaluation.json")
    config = read_json(run_dir / "training_config.json")

    best_val_accuracy = max(row["val_accuracy"] for row in metrics) if metrics else 0.0
    final_val_accuracy = metrics[-1]["val_accuracy"] if metrics else 0.0

    return {
        "run_dir": str(run_dir),
        "model_name": config.get("model_name", evaluation.get("model_name", "")),
        "best_validation_accuracy": best_val_accuracy,
        "final_validation_accuracy": final_val_accuracy,
        "test_accuracy": float(evaluation.get("test_accuracy", 0.0)),
        "per_class_accuracy": evaluation.get("per_class_accuracy", {}),
    }


def write_csv(path, summaries):
    path.parent.mkdir(parents=True, exist_ok=True)
    class_labels = sorted(
        {
            label
            for summary in summaries
            for label in summary["per_class_accuracy"].keys()
        }
    )
    fieldnames = [
        "model_name",
        "run_dir",
        "best_validation_accuracy",
        "final_validation_accuracy",
        "test_accuracy",
    ] + [f"test_accuracy_{label}" for label in class_labels]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {
                "model_name": summary["model_name"],
                "run_dir": summary["run_dir"],
                "best_validation_accuracy": summary["best_validation_accuracy"],
                "final_validation_accuracy": summary["final_validation_accuracy"],
                "test_accuracy": summary["test_accuracy"],
            }
            for label in class_labels:
                row[f"test_accuracy_{label}"] = summary["per_class_accuracy"].get(label, "")
            writer.writerow(row)


def write_json(path, summaries):
    json_path = path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(summaries, json_file, indent=2)


def print_table(summaries):
    print("Model comparison")
    print(
        "model_name,best_validation_accuracy,"
        "final_validation_accuracy,test_accuracy,per_class_accuracy"
    )
    for summary in summaries:
        per_class = "; ".join(
            f"{label}={accuracy:.4f}"
            for label, accuracy in sorted(summary["per_class_accuracy"].items())
        )
        print(
            f"{summary['model_name']},"
            f"{summary['best_validation_accuracy']:.4f},"
            f"{summary['final_validation_accuracy']:.4f},"
            f"{summary['test_accuracy']:.4f},"
            f"{per_class}"
        )


def main():
    args = parse_args()
    summaries = [summarize_run(run_dir) for run_dir in args.runs]
    write_csv(args.output, summaries)
    write_json(args.output, summaries)
    print_table(summaries)
    print()
    print(f"Comparison CSV written to: {args.output}")
    print(f"Comparison JSON written to: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
